"""Task-1 tests for event-keyed, actor-local random streams."""

import os
import pickle
import random
import subprocess
import sys
import time

import numpy as np
import pytest
import torch

from splendor.agents.our_agents.policy_imitation import protocol as protocol_module
from splendor.agents.our_agents.policy_imitation.protocol import (
    FormalGameRng,
    RngBundle,
    RngKey,
    configure_formal_torch_determinism,
    derive_seed,
    inverse_cdf_index,
    require_formal_spawn_context,
    run_formal_spawn_jobs,
)


def _spawn_probe(job: tuple[int, float]) -> tuple[int, str, str | None]:
    """Return after an intentionally different duration in a spawn worker."""
    index, delay = job
    time.sleep(delay)
    import multiprocessing as mp

    return (
        index,
        mp.current_process().name,
        os.environ.get("SPLENDOR_FORMAL_WORKER_COUNT"),
    )


def test_rng_protocol_known_vector() -> None:
    lineage = derive_seed({"stream_name": "policy_action", "update": 0})
    assert lineage.canonical_key_json == '{"stream_name":"policy_action","update":0}'
    assert (
        lineage.digest_hex
        == "2e72c022858541b7b24005eec63c2004b999a77a810949d2d556525b2241b7f1"
    )
    assert lineage.seed63 == 3346948727591223735
    assert lineage.u53 == 0.6962894161172106


def test_coupling_group_controls_treatment_sharing() -> None:
    def make_key(treatment_id: str, coupling_group: str) -> RngKey:
        return RngKey(
            stream_name="policy_action",
            experiment_id="task1",
            phase="T1.1",
            coupling_group=coupling_group,
            replicate_id=2,
            treatment_id=treatment_id,
            scenario_id="scenario-9",
            seat=1,
            update=4,
            game_index=3,
            focal_step=8,
        )

    paired_a = derive_seed(make_key("A", "replicate-2"))
    paired_b = derive_seed(make_key("B", "replicate-2"))
    independent = derive_seed(make_key("B", "B-replicate-2"))

    assert isinstance(paired_a.key, RngKey)
    assert isinstance(paired_b.key, RngKey)
    assert paired_a.key.as_dict()["treatment_id"] == "A"
    assert paired_b.key.as_dict()["treatment_id"] == "B"
    assert paired_a.digest_hex == paired_b.digest_hex
    assert independent.digest_hex != paired_a.digest_hex


def test_mapping_keys_use_the_same_explicit_coupling_semantics() -> None:
    common = {
        "stream_name": "policy_action",
        "coupling_group": "replicate-2",
        "replicate_id": 2,
        "focal_step": 8,
    }
    paired_a = derive_seed({**common, "treatment_id": "A"})
    paired_b = derive_seed({**common, "treatment_id": "B"})

    assert paired_a.digest_hex == paired_b.digest_hex
    assert paired_a.as_dict()["key"]["treatment_id"] == "A"  # type: ignore[index]
    assert "treatment_id" not in paired_a.canonical_key_json
    with pytest.raises(ValueError, match="explicit coupling_group"):
        derive_seed({"stream_name": "policy_action", "treatment_id": "A"})


def test_local_rng_bundle_does_not_touch_global_rngs() -> None:
    random.seed(77)
    np.random.seed(77)
    torch.manual_seed(77)
    python_before = random.getstate()
    numpy_before = pickle.dumps(np.random.get_state())
    torch_before = torch.get_rng_state().clone()

    lineage = derive_seed(RngKey(stream_name="worker", replicate_id=4))
    bundle = RngBundle.from_lineage(lineage)
    assert 0.0 <= bundle.python.random() < 1.0
    assert 0.0 <= float(bundle.numpy.random()) < 1.0
    assert 0.0 <= float(torch.rand((), generator=bundle.torch_cpu)) < 1.0

    assert random.getstate() == python_before
    assert pickle.dumps(np.random.get_state()) == numpy_before
    assert torch.equal(torch.get_rng_state(), torch_before)


def test_streams_are_independent_and_u53_is_open_interval() -> None:
    context = FormalGameRng(
        experiment_id="task1",
        phase="T1.1",
        coupling_group="replicate-0",
        replicate_id=0,
        treatment_id="O",
        scenario_id="scenario-0",
        seat=0,
        update=1,
        game_index=0,
    )
    lineages = {
        name: context.lineage(name)
        for name in ("scenario_source", "pool_draw", "policy_action", "minibatch")
    }
    assert len({lineage.digest_hex for lineage in lineages.values()}) == 4
    assert all(0 <= lineage.seed63 < 2**63 for lineage in lineages.values())
    assert all(0.0 < lineage.u53 < 1.0 for lineage in lineages.values())


def test_rng_key_rejects_ambiguous_numeric_or_protocol_inputs() -> None:
    with pytest.raises(ValueError, match="non-negative integer"):
        RngKey(stream_name="policy_action", update=True)
    with pytest.raises(ValueError, match="strings, integers, or null"):
        derive_seed({"stream_name": "policy_action", "update": 1.0})
    with pytest.raises(ValueError, match="unsupported RNG protocol"):
        derive_seed({"protocol_version": "splendor-rng-v2", "stream_name": "worker"})


@pytest.mark.parametrize(
    ("u53", "expected"),
    [(0.0, 0), (0.199999, 0), (0.2, 1), (0.999999, 2)],
)
def test_inverse_cdf_has_locked_boundary_semantics(u53: float, expected: int) -> None:
    assert inverse_cdf_index([1.0, 1.0, 3.0], u53) == expected


def test_inverse_cdf_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="positive mass"):
        inverse_cdf_index([0.0, 0.0], 0.5)
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        inverse_cdf_index([1.0], 1.0)


def test_formal_workers_require_startup_hash_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    with pytest.raises(RuntimeError, match="before Python starts"):
        require_formal_spawn_context()
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    monkeypatch.setattr(protocol_module, "_started_with_zero_hash_seed", lambda: True)
    assert require_formal_spawn_context().get_start_method() == "spawn"


def test_runtime_env_change_cannot_forge_startup_hash_seed() -> None:
    command = (
        "import os; "
        "os.environ['PYTHONHASHSEED']='0'; "
        "from splendor.agents.our_agents.policy_imitation.protocol "
        "import require_formal_spawn_context; "
        "require_formal_spawn_context()"
    )
    environment = dict(os.environ)
    environment["PYTHONHASHSEED"] = "random"
    result = subprocess.run(
        [sys.executable, "-c", command],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert result.returncode != 0
    assert "before Python starts" in result.stderr


def test_formal_spawn_executor_preserves_input_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    monkeypatch.setattr(protocol_module, "_started_with_zero_hash_seed", lambda: True)
    results = run_formal_spawn_jobs(
        ((0, 0.03), (1, 0.0), (2, 0.01)),
        worker=_spawn_probe,
        worker_count=2,
    )
    assert [result[0] for result in results] == [0, 1, 2]
    assert all(name != "MainProcess" for _, name, _ in results)
    assert all(count == "2" for _, _, count in results)


def test_formal_cuda_never_falls_back_to_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    monkeypatch.setattr(protocol_module, "_started_with_zero_hash_seed", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        configure_formal_torch_determinism("cuda")
