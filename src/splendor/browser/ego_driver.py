"""
EgoBrowserDriver: a BrowserDriver implementation over the ``ego-browser``
CLI (Chromium-based, agent-friendly browser).

This is the *reference adapter* proving the driver abstraction is really
tool-agnostic: every method shells out to one ``ego-browser nodejs``
invocation whose script re-uses the logged-in task space and prints a single
``__RESULT__<json>`` line. Consequences, documented on purpose:

* **Latency**: each call pays one Node runtime start (~0.5-1s). That is fine
  for the executor's human-paced clicks and the env's turn polling (which is
  throttled to etiquette intervals anyway); a CDP/playwright adapter is the
  drop-in fast path when latency ever matters.
* **JS transport**: caller JS is embedded as an ensure_ascii JSON string
  literal inside a fixed ``(() => eval(...))()`` wrapper - ``js()`` returns
  ``undefined`` for some verbatim multi-line sources (see ``_run_js``), and
  the wrapper is the measured-safe passthrough. The ``js_files`` temp-file
  channel remains available in ``_run`` for node-side payloads.
* The click helpers (``click_labelled`` / ``click_card_button``) mirror the
  MockBrowserDriver's matching semantics in page-side JS, so offline and
  live behaviour stay aligned.
"""

import base64
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

_RESULT_MARKER = "__RESULT__"
_DEFAULT_CALL_TIMEOUT = 90.0


class EgoBrowserDriver:
    """BrowserDriver over one ego-browser task space.

    ``profile_id`` pins the task space to an ego-browser browser profile
    (id from ``profiles()``). Task spaces inside one profile share cookies
    and localStorage; distinct profiles give distinct logins, which is how
    multi-bot deployments hold two accounts at once. The profile only takes
    effect at space creation, so the space name embeds it - a profile
    change then addresses a fresh space instead of silently reusing one
    created under the old profile.
    """

    def __init__(
        self,
        task_space: str,
        room_url: str | None = None,
        profile_id: str | None = None,
    ) -> None:
        self._task_space = task_space
        self._room_url = room_url
        self._profile_id = profile_id

    # ----- plumbing -----------------------------------------------------------
    def _run(
        self,
        body: str,
        js_files: dict[str, str] | None = None,
        timeout: float = _DEFAULT_CALL_TIMEOUT,
    ) -> Any:  # noqa: ANN401
        """
        Run one ``ego-browser nodejs`` script and decode its result line.

        :param body: node script; must end by printing the result line.
        :param js_files: mapping of node variable name -> JS source, written
                         to temp files and read inside the node script (this
                         is how caller JS travels without an escaping layer).
        """
        with tempfile.TemporaryDirectory() as tmp:
            reads = "".join(
                f"const {name} = fs.readFileSync('{Path(tmp, name)}', 'utf8');\n"
                for name in js_files or {}
            )
            for name, source in (js_files or {}).items():
                Path(tmp, name).write_text(source, encoding="utf-8")
            script = "const fs = await import('node:fs');\n" + reads + body
            completed = subprocess.run(
                ["ego-browser", "nodejs"],
                input=script,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        # cliLog writes to the process' stderr; the searched stream is both.
        output = completed.stdout + "\n" + completed.stderr
        marker_index = output.rfind(_RESULT_MARKER)
        if marker_index < 0:
            raise RuntimeError(
                f"ego-browser call produced no result (exit {completed.returncode}):\n"
                f"{completed.stderr[-500:] or output[-500:]}"
            )
        result_line = output[marker_index + len(_RESULT_MARKER):].splitlines()[0]
        decoded = json.loads(result_line)
        if isinstance(decoded, dict) and decoded.get("__error__"):
            raise ValueError(str(decoded["__error__"]))
        return decoded.get("value") if isinstance(decoded, dict) else decoded

    @staticmethod
    def _emit(value_js: str) -> str:
        """Node statements that safely emit one JSON result line."""
        return (
            "try {\n"
            f"  const value = {value_js};\n"
            f"  cliLog('{_RESULT_MARKER}' + JSON.stringify({{value}}));\n"
            "} catch (error) {\n"
            f"  cliLog('{_RESULT_MARKER}' + JSON.stringify({{__error__: String(error)}}));\n"
            "}\n"
        )

    def _select_task_space(self) -> str:
        space = json.dumps(self._space_name())
        if self._profile_id is None:
            return f"await useOrCreateTaskSpace({space});\n"
        profile = json.dumps(self._profile_id)
        # profileId only applies at creation: reusing by name while passing
        # it raises "already exists" (measured 2026-09-10), so find first
        # and create only when absent. listTaskSpaces exposes ``id``.
        return (
            "const existing = (await listTaskSpaces())"
            f".find((s) => s.name === {space});\n"
            "await (existing"
            " ? taskSpace(existing.id)"
            f" : taskSpace({space}, {{ profileId: {profile} }}));\n"
        )

    def _space_name(self) -> str:
        if self._profile_id is None:
            return self._task_space
        return f"{self._task_space}@{self._profile_id}"

    def _run_js(self, page_js: str, prologue: str = "") -> Any:  # noqa: ANN401
        """
        Select the task space, run one page-side JS, decode the result.

        The whole body is wrapped in an async IIFE: piped stdin scripts are
        evaluated without top-level-await support (unlike interactive runs).

        The caller JS travels as a JSON string literal inside a fixed
        ``(() => eval(...))()`` wrapper rather than being handed to ``js()``
        verbatim. Measured 2026-09-08: ``js()`` returns ``undefined`` for some
        multi-line sources (EXTRACT_SNAPSHOT_JS reproduces it 100%), which
        then explodes node-side as "Cannot convert undefined or null to
        object" - while the identical script through this wrapper, and its
        identical result object, both pass. The wrapper shape is the only
        form js() ever sees; the literal is ensure_ascii JSON, so no
        character of the caller JS can break out of the string.
        """
        wrapper = f"(() => eval({json.dumps(page_js)}))()"
        body = (
            "(async () => {\n"
            + self._select_task_space()
            + prologue
            + self._emit(f"await js({json.dumps(wrapper)})")
            + "})()\n"
        )
        return self._run(body)

    # ----- BrowserDriver protocol --------------------------------------------
    def evaluate(self, js: str) -> Any:  # noqa: ANN401  # JSON payload by design
        return self._run_js(js)

    def click(self, selector: str, index: int = 0) -> None:
        page_js = f"""(() => {{
  const selector = {json.dumps(selector)};
  const index = {int(index)};
  const matches = [...document.querySelectorAll(selector)];
  if (!matches.length) return {{ __error__: 'no element matches ' + selector }};
  if (index >= matches.length)
    return {{ __error__: 'index ' + index + ' of ' + matches.length }};
  matches[index].click();
  return {{ ok: true }};
}})()"""
        self._run_js(page_js)

    def click_labelled(
        self,
        label: str,
        *,
        exact: bool = True,
        index: int = 0,
        container_selector: str | None = None,
        container_index: int = 0,
    ) -> None:
        page_js = f"""(() => {{
  const label = {json.dumps(label)};
  const exact = {str(bool(exact)).lower()};
  const index = {int(index)};
  const containerSelector = {json.dumps(container_selector)};
  const containerIndex = {int(container_index)};
  const scope = containerSelector
    ? document.querySelectorAll(containerSelector)[containerIndex]
    : document;
  if (!scope) return {{ __error__: 'container not found: ' + containerSelector }};
  const trimmed = el => (el.textContent || '').trim();
  const matches = [...scope.querySelectorAll('*')].filter(el =>
    exact ? trimmed(el) === label : trimmed(el).includes(label));
  if (!matches.length) return {{ __error__: 'label not found: ' + label }};
  if (index >= matches.length)
    return {{ __error__: 'label index ' + index + ' of ' + matches.length }};
  matches[index].click();
  return {{ ok: true }};
}})()"""
        self._run_js(page_js)

    def click_card_button(
        self,
        container_selector: str,
        container_index: int,
        card_index: int,
        label: str,
    ) -> None:
        page_js = f"""(() => {{
  const containerSelector = {json.dumps(container_selector)};
  const containerIndex = {int(container_index)};
  const cardIndex = {int(card_index)};
  const label = {json.dumps(label)};
  const containers = [...document.querySelectorAll(containerSelector)];
  const scope = containers[containerIndex];
  if (!scope) return {{ __error__: 'container not found: ' + containerSelector }};
  const cards = [...scope.querySelectorAll('.ccbs-card')];
  const card = cards[cardIndex];
  if (!card) return {{ __error__: 'card index ' + cardIndex + ' of ' + cards.length }};
  const button = [...card.querySelectorAll('button')]
    .find(el => (el.textContent || '').trim() === label);
  if (!button) return {{ __error__: 'no ' + label + ' button in card ' + cardIndex }};
  button.click();
  return {{ ok: true }};
}})()"""
        self._run_js(page_js)

    def wait_for(self, condition_js: str, timeout: float) -> None:
        # The env already throttles its own polling loop; a single check per
        # call keeps the subprocess cost bounded and the semantics simple.
        self.evaluate(condition_js)

    def navigate(self, url: str) -> None:
        # gotoAndWait is a NODE-side helper, so navigation runs in the node
        # body, not through the page-side js() evaluator.
        body = (
            "(async () => {\n"
            + self._select_task_space()
            + f"const url = {json.dumps(url)};\n"
            + self._emit(
                "(await gotoAndWait(url, { timeout: 30 }), await js('location.href'))"
            )
            + "})()\n"
        )
        self._run(body)

    def get_cookies(self, domain: str) -> list[dict]:
        body = (
            "(async () => {\n"
            + self._select_task_space()
            + f"const domain = {json.dumps(domain)};\n"
            + self._emit(
                "(await cdp('Network.getCookies',"
                " { urls: ['https://' + domain] })).cookies"
            )
            + "})()\n"
        )
        return self._run(body)

    def set_cookie(self, cookie: dict) -> None:
        body = (
            "(async () => {\n"
            + self._select_task_space()
            + f"const cookie = {json.dumps(cookie)};\n"
            + self._emit("await cdp('Network.setCookie', cookie)")
            + "})()\n"
        )
        self._run(body)

    def delete_cookies(self, name: str, domain: str) -> None:
        body = (
            "(async () => {\n"
            + self._select_task_space()
            + f"const name = {json.dumps(name)};\n"
            + f"const domain = {json.dumps(domain)};\n"
            + self._emit("await cdp('Network.deleteCookies', { name, domain })")
            + "})()\n"
        )
        self._run(body)

    def screenshot(self, path: str) -> None:
        body = (
            "(async () => {\n"
            + self._select_task_space()
            + self._emit(
                "(await cdp('Page.captureScreenshot', { format: 'png' })).data"
            )
            + "})()\n"
        )
        encoded = self._run(body)
        Path(path).write_bytes(base64.b64decode(encoded))
