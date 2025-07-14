"""
env.py
Small helper to construct the Abel-RL equation-solver environment
(or fall back to any Box-obs / Discrete-act Gymnasium env).

The only requirement for the SAC agent is:
    * obs  ∼ gym.spaces.Box
    * act  ∼ gym.spaces.Discrete
    * (optional) info["action_mask"]  → boolean array of valid actions
"""

from __future__ import annotations
import gymnasium as gym
from envs.multi_eqn import multiEqn   # <- your Abel-RL env
from envs.single_eqn import singleEqn   # <- your Abel-RL env
from envs.single_eqn_fixed_action_dim import singleEqnFixedAction   # <- your Abel-RL env

from sb3_contrib.common.wrappers import ActionMasker


def _get_mask(env):
    return getattr(env, "action_mask")          # bool [n_actions]


def make_abel_rl(level=2, generalization="poesia",
                 state_rep="integer_1d", normalize=True,
                 mask=False) -> gym.Env:
    """Return a *single* wrapped Abel-RL environment."""
    # env = multiEqn(
    #     normalize_rewards=normalize,
    #     state_rep=state_rep,
    #     level=level,
    #     generalization=generalization,
    # )
    env = singleEqnFixedAction()
    if mask:
        env = ActionMasker(env, _get_mask)
    return env


def make_env(env_id: str, **kwargs) -> gym.Env:
    """
    Generic factory.
    If `env_id == "abel_rl"` we build the custom env; otherwise we call gym.make().
    """
    if env_id == "single_fixed":
        return singleEqnFixedAction()
    elif env_id == 'single':
        return singleEqn()
    elif env_id == 'multi':
        env = multiEqn(
         generalization='poesia'
        )
        return env 
    else:
        return gym.make(env_id)

