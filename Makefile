SOURCE_DIR := src/splendor
DOCS_SOURCE_DIR := docs/source
DOCS_BUILD_DIR := docs/_build/html
COMMIT_ID := $(shell git rev-parse HEAD)
COMMIT_MSG := $(addprefix Documentation for commit ,$(COMMIT_ID))

.PHONY: format
format:
	ruff format .

.PHONY: lint
lint:
	# No --preview here: CI runs plain `ruff check`, and preview rules change
	# between ruff releases. When the two disagree, a `# noqa: <preview rule>`
	# is simultaneously "required" (preview) and "unused" (CI), so `ruff --fix`
	# and `make lint` undo each other. Keep local == CI.
	ruff check .
	mypy .
	pylint src/

.PHONY: pre-commit
pre-commit:
	pre-commit install
	pre-commit run --all-files

.PHONY: docs
docs:
	splendor --version  # ensure splendor is installed.
	sphinx-apidoc --output-dir $(DOCS_SOURCE_DIR) $(SOURCE_DIR) --force
	sphinx-build $(DOCS_SOURCE_DIR) $(DOCS_BUILD_DIR)

.PHONY: publish-docs
publish-docs: docs
	pre-commit uninstall
	cd $(DOCS_BUILD_DIR) && git add -A . && git commit -sm "$(COMMIT_MSG)." && git push origin gh-pages
	pre-commit install

.PHONY: clean
clean:
	make -C docs/ clean

# Repo venv first (uv-built Pythons lack tkinter, and system Python may miss
# the deps entirely); override with `make PYTHON=python3`.
PYTHON ?= $(shell if [ -x .venv/bin/python ]; then echo .venv/bin/python; else echo python3; fi)

.PHONY: test
test:
	$(PYTHON) -m pytest tests/

# The parity gate: mandatory before any web deployment after touching the
# engine's mask logic, features.py, or the browser extraction layer.
.PHONY: parity
parity:
	$(PYTHON) -m pytest tests/test_feature_parity.py tests/test_browser_adapter.py -v

.PHONY: train-dqn
train-dqn:
	$(PYTHON) -m splendor.agents.our_agents.dqn.dqn --help && \
	echo "usage: dqn -o random --test-opponent minimax [see plan/phase-1]"

.PHONY: play-web
play-web:
	$(PYTHON) -m splendor.play_web --help && \
	echo "usage: play-web --checkpoint <path> --games 50 [see plan/phase-3]"

# Phase-6: browser control local, inference remote (see docs/REMOTE_DEPLOYMENT_GUIDE.md).
.PHONY: serve-inference
serve-inference:
	$(PYTHON) -m splendor.remote.server --help && \
	echo "usage: inference-server --model <name>=<path.pth> [--model ...] [--port 8765]"

.PHONY: play-web-remote
play-web-remote:
	$(PYTHON) -m splendor.play_remote --help && \
	echo "usage: play-web-remote --server <host>:<port> --model <name> --bots 1 [see docs/REMOTE_DEPLOYMENT_GUIDE.md]"

.PHONY: dashboard
dashboard:
	$(PYTHON) -m splendor.remote.dashboard --help && \
	echo "usage: play-dashboard --events-dir web_events --port 8899"
