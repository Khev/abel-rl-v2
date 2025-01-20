import torch
import multiprocessing as mp

from sb3_contrib import MaskablePPO
from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.vec_env import VecEnv

from rllte.xplore.reward import E3B, ICM, RIDE

def get_device():
    if torch.cuda.is_available():
        print('Found cuda: using')
        cur_proc_identity = mp.current_process()._identity
        if cur_proc_identity:
            return (cur_proc_identity[0] - 1) % th.cuda.device_count()
        else:
            return 0
    else:
        #print('Did not find cuda: using cpu')
        return 'cpu'


def get_agent(agent_type, env, policy="MlpPolicy", **kwargs):
    """Returns the appropriate RL agent, including PPO-Mask from sb3-contrib."""
    agents = {
        "dqn": DQN,
        "ppo": PPO,
        "ppo-mask": MaskablePPO,
        "a2c": A2C,
    }
    
    if agent_type not in agents:
        raise ValueError(f"Unsupported agent type: {agent_type}. Choose from {list(agents.keys())}")

    # Set ent_coef only for PPO-Mask
    #if agent_type == "ppo-mask":
    #    kwargs.setdefault("ent_coef", 0.05)

    model = agents[agent_type](policy, env, **kwargs)
    return model



def get_intrinsic_reward(intrinsic_reward, vec_env):
    device = get_device()
    if intrinsic_reward == 'ICM':
        return ICM(vec_env, device=device)
    elif intrinsic_reward == 'E3B':
        return E3B(vec_env, device=device)
    elif intrinsic_reward == 'RIDE':
        return RIDE(vec_env, device=device)
    else:
        return None