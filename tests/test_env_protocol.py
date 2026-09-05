"""
Phase-0 acceptance test for the unified environment protocol
(plan/phase-0 §4: A0.4).

``SplendorEnvBase`` is the "single interface, two environments" contract:
the local SplendorEnv must satisfy it structurally, without changing its
inheritance tree.
"""

import numpy as np

import splendor.splendor.gym  # noqa: F401  # registers the splendor-v1 env
from splendor.splendor.gym.base import SplendorEnvBase
from splendor.splendor.gym.envs.splendor_env import SplendorEnv


def test_splendor_env_satisfies_protocol():
    env = SplendorEnv(agents=[])
    assert isinstance(env, SplendorEnvBase)


def test_step_accepts_forward_compatible_payment_argument():
    env = SplendorEnv(agents=[])
    obs, info = env.reset(seed=1234)
    assert obs.shape == (265,)
    assert "my_id" in info

    mask = env.get_legal_actions_mask()
    legal_action = int(np.flatnonzero(mask)[0])

    # the payment argument is accepted (and ignored) by the tier-1 env.
    result = env.step(legal_action, payment=None)
    assert len(result) == 5
    next_obs, reward, terminated, truncated, extra = result
    assert next_obs.shape == (265,)
    # the engine's reward is a score delta, so int values are expected too.
    assert isinstance(reward, (int, float))
    assert isinstance(terminated, bool)
    assert truncated is False
    assert extra == {}


def test_get_payment_options_is_none_for_tier1():
    env = SplendorEnv(agents=[])
    assert env.get_payment_options(0) is None
