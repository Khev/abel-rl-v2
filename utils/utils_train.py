import torch
import multiprocessing as mp
import gym  # ✅ Needed for observation spaces
import torch.nn as nn
import torch.nn.functional as F  # ✅ Needed for activation functions

from sb3_contrib import MaskablePPO
from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.policies import ActorCriticPolicy  # ✅ Needed for CustomCNNPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor  # ✅ Needed for EquationCNN
from stable_baselines3.common.utils import get_device  # ✅ Not needed anymore (you define your own `get_device`)

from rllte.xplore.reward import E3B, ICM, RIDE


def get_device():
    """Returns the appropriate device (CUDA or CPU)."""
    if torch.cuda.is_available():
        print('Found CUDA: using GPU')
        cur_proc_identity = mp.current_process()._identity
        if cur_proc_identity:
            return (cur_proc_identity[0] - 1) % torch.cuda.device_count()
        else:
            return 0
    else:
        print('CUDA not found: using CPU')
        return 'cpu'


def get_agent(agent_type, env, policy="MlpPolicy", **kwargs):
    """Returns the appropriate RL agent, including PPO-Mask from sb3-contrib and PPO with CNN."""

    agents = {
        "dqn": DQN,
        "ppo": PPO,
        "a2c": A2C,
        "ppo-mask": MaskablePPO,
        "ppo-cnn": lambda policy, env, **kwargs: PPO(
            CustomCNNPolicy, env, **kwargs  # ✅ Use CustomCNNPolicy for CNN-based agent
        ),
    }
    
    if agent_type not in agents:
        raise ValueError(f"Unsupported agent type: {agent_type}. Choose from {list(agents.keys())}")

    model = agents[agent_type](policy, env, **kwargs)
    return model


def get_intrinsic_reward(intrinsic_reward, vec_env):
    """Returns an intrinsic reward module from rllte.xplore."""
    device = get_device()
    if intrinsic_reward == 'ICM':
        return ICM(vec_env, device=device)
    elif intrinsic_reward == 'E3B':
        return E3B(vec_env, device=device)
    elif intrinsic_reward == 'RIDE':
        return RIDE(vec_env, device=device)
    else:
        return None


class EquationCNN(BaseFeaturesExtractor):
    """
    Custom CNN Feature Extractor for equations encoded as (seq_length, 2).

    The network learns to extract meaningful representations from symbolic math sequences.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        super(EquationCNN, self).__init__(observation_space, features_dim)

        n_input_channels = 2  # ✅ Encoding has 2 channels (feature type + index)

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=n_input_channels, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute output size after CNN layers
        with torch.no_grad():
            sample_input = torch.zeros(1, n_input_channels, observation_space.shape[0])
            cnn_output_size = self.cnn(sample_input).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(cnn_output_size, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # 🔄 Swap axis: (batch, seq_len, 2) → (batch, 2, seq_len) for 1D CNN
        observations = observations.permute(0, 2, 1)
        cnn_out = self.cnn(observations)
        return self.fc(cnn_out)


# Custom Policy using EquationCNN
class CustomCNNPolicy(ActorCriticPolicy):
    """Custom CNN-based Policy for symbolic equation solving."""
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=EquationCNN,  # 🔥 Use our custom CNN
            features_extractor_kwargs={"features_dim": 128},
            **kwargs
        )
