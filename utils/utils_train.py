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
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

from torch_geometric.nn import GCNConv, global_mean_pool

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
    """Returns the appropriate RL agent, including PPO-Mask from sb3-contrib, PPO with CNN, and PPO with GNN."""

    agents = {
        "dqn": DQN,
        "ppo": PPO,
        "a2c": A2C,
        "ppo-mask": MaskablePPO,
        "ppo-cnn": lambda policy, env, **kwargs: MaskablePPO(CustomCNNPolicy, env, **kwargs),
        "ppo-gnn": lambda policy, env, **kwargs: MaskablePPO(CustomGNNPolicy, env, **kwargs)
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


##################  Custom PPO + CNN ######################################

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
class CustomCNNPolicy(MaskableActorCriticPolicy):
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


##################  Custom PPO + GNN ######################################



class GNNFeatureExtractor(BaseFeaturesExtractor):
    """
    Simple GNN Feature Extractor for SB3 PPO with multi-graph batching.
    """

    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)

        # GNN layers
        in_channels = observation_space["node_features"].shape[1]  # e.g. 2
        self.conv1 = GCNConv(in_channels=in_channels, out_channels=64)
        self.conv2 = GCNConv(in_channels=64, out_channels=128)

        # Final linear layer
        self.fc = nn.Linear(128, features_dim)

    def forward(self, obs_batch):
        """
        obs_batch is a dict of Tensors, each with shape (batch_size, ...)
        We'll parse it item by item to build a single large "batch" graph.
        """
        # Unpack
        node_features_b = obs_batch["node_features"]  # (batch_size, max_nodes, feat_dim)
        edge_index_b    = obs_batch["edge_index"]     # (batch_size, 2, max_edges)
        node_mask_b     = obs_batch["node_mask"]      # (batch_size, max_nodes)
        edge_mask_b     = obs_batch["edge_mask"]      # (batch_size, max_edges)

        batch_size = node_features_b.shape[0]

        # We'll accumulate all nodes from each environment/time-step
        # into a single big graph. We must offset node indices for each item.
        all_x = []          # List of [num_nodes_i, feat_dim]
        all_edge_index = [] # List of [2, num_edges_i]
        all_batch_idx = []  # Which graph each node belongs to
        node_offset = 0     # how many nodes we've added so far

        for i in range(batch_size):
            # 1) Extract arrays for item i
            node_features_i = node_features_b[i]  # shape (max_nodes, feat_dim)
            edge_index_i    = edge_index_b[i]     # shape (2, max_edges)
            node_mask_i     = node_mask_b[i]      # shape (max_nodes,)
            edge_mask_i     = edge_mask_b[i]      # shape (max_edges,)

            # 2) Convert to PyTorch types
            node_features_i = node_features_i.float()
            node_mask_i     = node_mask_i.bool()
            edge_mask_i     = edge_mask_i.bool()
            edge_index_i    = edge_index_i.long()

            # 3) Filter out edges referencing invalid nodes
            src = edge_index_i[0]
            dst = edge_index_i[1]
            valid_edges_i = edge_mask_i & node_mask_i[src] & node_mask_i[dst]
            edge_index_i = edge_index_i[:, valid_edges_i]

            # (Optional) If you want to remove the unused nodes from node_features_i, you can do:
            # valid_nodes_i = node_mask_i.nonzero(as_tuple=True)[0]
            # But let's keep them all at their indices 0..max_nodes-1
            # so we do not do advanced indexing.

            # 4) Offset the node indices by node_offset
            edge_index_i = edge_index_i + node_offset

            # 5) Accumulate node features
            # We keep all nodes, ignoring those that are masked-out (they can remain as dummy features).
            all_x.append(node_features_i)

            # 6) Accumulate edge_index
            all_edge_index.append(edge_index_i)

            # 7) Build the "batch" array that says "these nodes belong to graph i"
            #    It's length = max_nodes (or number of valid nodes if you filtered them).
            #    We'll just mark them all with i, even the masked-out ones.
            #    If you filtered nodes, you'd make it shape (num_valid_nodes_i,).
            n_nodes_i = node_features_i.shape[0]  # e.g. max_nodes
            batch_i = torch.full((n_nodes_i,), i, dtype=torch.long)

            all_batch_idx.append(batch_i)

            # 8) Increase offset for the next item
            node_offset += n_nodes_i

        # Concatenate across all items
        x = torch.cat(all_x, dim=0)  # shape (sum_of_nodes, feat_dim)
        edge_index = torch.cat(all_edge_index, dim=1)  # shape (2, sum_of_edges)
        batch_idx  = torch.cat(all_batch_idx, dim=0)   # shape (sum_of_nodes,)

        # Move everything to the same device
        device = x.device
        x = x.to(device)
        edge_index = edge_index.to(device)
        batch_idx = batch_idx.to(device)

        # 9) GCN forward
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))

        # 10) Global mean pool => shape (batch_size, 128)
        x = global_mean_pool(x, batch_idx)

        return self.fc(x)


class CustomGNNPolicy(MaskableActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=GNNFeatureExtractor,
            features_extractor_kwargs={"features_dim": 128},
            **kwargs
        )