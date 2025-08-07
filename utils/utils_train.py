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
from stable_baselines3.common.buffers import ReplayBuffer, ReplayBufferSamples
from collections import namedtuple
import numpy as np

from torch_geometric.nn import GCNConv, global_mean_pool

from rllte.xplore.reward import E3B, ICM, RIDE

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, BatchNorm, global_mean_pool
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


# For PPO-mask
def get_action_mask(env):
    return env.action_mask

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


# def get_agent(agent_type, env, policy="MlpPolicy", **kwargs):
#     """Returns the appropriate RL agent, including PPO-Mask from sb3-contrib, PPO with CNN, and PPO with GNN."""

#     agents = {
#         "dqn": DQN,
#         "ppo": PPO,
#         "a2c": A2C,
#         "ppo-mask": MaskablePPO,
#         "ppo-cnn": lambda policy, env, **kwargs: MaskablePPO(CustomCNNPolicy, env, **kwargs),
#         "ppo-gnn": lambda policy, env, **kwargs: MaskablePPO(CustomGNNPolicy, env, **kwargs),
#         # "ppo-gnn": lambda policy, env, **kwargs: MaskablePPO(CustomGNNPolicy, env, ent_coef=0, **kwargs),
#         #"ppo-gnn1": lambda policy, env, **kwargs: MaskablePPO(PPOGNN1Policy, env, ent_coef=0.1, **kwargs)
#     }


#         model = DQN(
#             "MlpPolicy", env,
#             buffer_size            = 100_000,
#             replay_buffer_class    = PERBuffer,
#             replay_buffer_kwargs   = dict(alpha=0.6, beta=0.4, eps=1e-5),
#             learning_starts        = 1_000,
#             batch_size             = 256,
#             tensorboard_log        = tb.log_dir,
#             seed                   = args.seed,
#             device          = device,
#         )
    
#     if agent_type not in agents:
#         raise ValueError(f"Unsupported agent type: {agent_type}. Choose from {list(agents.keys())}")

#     model = agents[agent_type](policy, env, **kwargs)
#     return model

# Extend the SB3 replay samples tuple with weights and indices
PrioritizedReplayBufferSamples = namedtuple(
    "PrioritizedReplayBufferSamples",
    ReplayBufferSamples._fields + ("weights", "indices")
)

class PERBuffer(ReplayBuffer):
    """
    Prioritised Experience Replay for SB3 DQN.
    Usage in DQN(...) constructor:
        buffer_size=100_000,
        replay_buffer_class=PERBuffer,
        replay_buffer_kwargs=dict(alpha=0.6, beta=0.4, eps=1e-5)
    """
    def __init__(
        self,
        buffer_size: int,
        observation_space,
        action_space,
        device: torch.device,
        alpha: float = 0.6,
        beta: float = 0.4,
        eps: float = 1e-5,
        **kwargs
    ):
        super().__init__(buffer_size, observation_space, action_space, device, **kwargs)
        self.alpha = alpha
        self.beta  = beta
        self.eps   = eps
        self.priorities = np.zeros((self.buffer_size,), dtype=np.float32)

    def add(self, *args, **kwargs):
        idx = self.pos
        super().add(*args, **kwargs)
        # New transitions get max priority so they’re sampled at least once
        self.priorities[idx] = self.priorities.max() if (self.full or idx > 0) else 1.0

    def sample(self, batch_size: int, env=None) -> PrioritizedReplayBufferSamples:
        # 1. Compute sampling probabilities
        if self.full:
            probs = self.priorities ** self.alpha
        else:
            probs = self.priorities[: self.pos] ** self.alpha
        probs /= probs.sum()

        # 2. Draw indices according to probabilities
        indices = np.random.choice(len(probs), batch_size, p=probs)

        # 3. Get the base samples (obs, actions, ...)
        base_samples = super()._get_samples(indices, env)

        # 4. Compute importance sampling weights
        weights = (len(probs) * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()  # normalize to [0,1]

        # 5. Return extended namedtuple
        return PrioritizedReplayBufferSamples(
            *base_samples,
            torch.as_tensor(weights, device=self.device, dtype=torch.float32),
            indices,
        )

    def update_priorities(self, indices, td_errors):
        # After learning, call model.replay_buffer.update_priorities(...)
        self.priorities[indices] = np.abs(td_errors) + self.eps
    

def get_agent(agent_type, env, policy="MlpPolicy", **kwargs):
    """Returns the appropriate RL agent, including PPO-Mask from sb3-contrib, PPO with CNN, and PPO with GNN."""

    agents = {
        "dqn": lambda policy, env, **kwargs: DQN(
            policy, env,
            replay_buffer_class=PERBuffer,
            replay_buffer_kwargs=dict(alpha=0.6, beta=0.4, eps=1e-5),
            learning_starts=10**4,
            **kwargs
        ),
        "ppo": PPO,
        "a2c": A2C,
        "ppo-mask": MaskablePPO,
        "ppo-cnn": lambda policy, env, **kwargs: MaskablePPO(CustomCNNPolicy, env, **kwargs),
        "ppo-gnn": lambda policy, env, **kwargs: MaskablePPO(CustomGNNPolicy, env, **kwargs),
        # Add other custom agents as needed
    }

    if agent_type not in agents:
        raise ValueError(f"Unsupported agent type: {agent_type}. Choose from {list(agents.keys())}")

    return agents[agent_type](policy, env, **kwargs)



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



class GNNFeatureExtractorOld(BaseFeaturesExtractor):
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


class GNNFeatureExtractor(BaseFeaturesExtractor):
    """
    Wider GNN Feature Extractor for SB3 PPO with multi-graph batching.
    conv1: in -> 256
    conv2: 256 -> 512
    head : 512 -> features_dim (default 512)
    """

    def __init__(self, observation_space, features_dim: int = 512):
        super().__init__(observation_space, features_dim)

        in_channels = observation_space["node_features"].shape[1]  # e.g. 2

        # Wider GCN stack
        self.conv1   = GCNConv(in_channels=in_channels, out_channels=256)
        self.conv2   = GCNConv(in_channels=256, out_channels=512)

        # Light regularization
        self.dropout = nn.Dropout(p=0.2)

        # Final projection
        self.fc = nn.Linear(512, features_dim)

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

        # Accumulate nodes/edges from each item into one big graph
        all_x = []          # List of [num_nodes_i, feat_dim]
        all_edge_index = [] # List of [2, num_edges_i]
        all_batch_idx = []  # Which graph each node belongs to
        node_offset = 0

        for i in range(batch_size):
            node_features_i = node_features_b[i]  # (max_nodes, feat_dim)
            edge_index_i    = edge_index_b[i]     # (2, max_edges)
            node_mask_i     = node_mask_b[i]      # (max_nodes,)
            edge_mask_i     = edge_mask_b[i]      # (max_edges,)

            # Dtypes
            node_features_i = node_features_i.float()
            node_mask_i     = node_mask_i.bool()
            edge_mask_i     = edge_mask_i.bool()
            edge_index_i    = edge_index_i.long()

            # Keep only edges whose endpoints are valid nodes and the edge itself is valid
            src = edge_index_i[0]
            dst = edge_index_i[1]
            valid_edges_i = edge_mask_i & node_mask_i[src] & node_mask_i[dst]
            edge_index_i = edge_index_i[:, valid_edges_i]

            # Offset node indices for concatenation
            edge_index_i = edge_index_i + node_offset

            # Accumulate
            all_x.append(node_features_i)
            all_edge_index.append(edge_index_i)

            n_nodes_i = node_features_i.shape[0]
            batch_i = torch.full((n_nodes_i,), i, dtype=torch.long)
            all_batch_idx.append(batch_i)

            node_offset += n_nodes_i

        # Concatenate across the batch
        x = torch.cat(all_x, dim=0)               # (sum_nodes, feat_dim)
        edge_index = torch.cat(all_edge_index, 1) # (2, sum_edges)
        batch_idx  = torch.cat(all_batch_idx, 0)  # (sum_nodes,)

        # Ensure tensors share device
        device = x.device
        edge_index = edge_index.to(device)
        batch_idx  = batch_idx.to(device)

        # GCN forward (wider)
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)

        # Pool per-graph
        x = global_mean_pool(x, batch_idx)        # (batch_size, 512)

        return self.fc(x)                          # (batch_size, features_dim)



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



class ImprovedGNNFeatureExtractor(BaseFeaturesExtractor):
    """
    Improved GNN Feature Extractor for SB3 PPO with:
      - GATConv for attention-based aggregation
      - Batch normalization for stabilization
      - Residual connections for deep learning stability (with projection)
      - Increased embedding dimension (512)
      - Dropout for better generalization
      - LeakyReLU instead of ReLU for smoother training
    """

    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)
        in_channels = observation_space["node_features"].shape[1]  # e.g., 2

        # GAT layers with BatchNorm
        self.conv1 = GATConv(in_channels, 256, heads=4, concat=False)
        self.bn1 = BatchNorm(256)

        self.conv2 = GATConv(256, 512, heads=4, concat=False)
        self.bn2 = BatchNorm(512)

        self.conv3 = GATConv(512, 512, heads=4, concat=False)
        self.bn3 = BatchNorm(512)

        self.dropout = nn.Dropout(p=0.2)

        # Projection for the residual connection: project 256-dim (from conv1) to 512-dim
        self.res_conv1 = nn.Linear(256, 512)

        # Final MLP for feature extraction
        self.fc = nn.Sequential(
            nn.Linear(512, features_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=0.2)
        )

    def forward(self, obs_batch):
        """
        Expects obs_batch to be a dict with at least:
          - "node_features": Tensor of shape (B, max_nodes, feat_dim)
          - "edge_index": Tensor of shape (B, 2, max_edges)
        If "batch_idx" is not provided, it is computed from the batch dimension.
        """
        # Unpack node features and edge indices
        node_features_b = obs_batch["node_features"]  # (B, max_nodes, feat_dim)
        edge_index_b = obs_batch["edge_index"]          # (B, 2, max_edges)
        B, max_nodes, feat_dim = node_features_b.shape
        device = node_features_b.device

        # Compute batch indices if not provided
        if "batch_idx" in obs_batch:
            batch_idx_b = obs_batch["batch_idx"]
        else:
            batch_idx_b = torch.arange(B, device=device).unsqueeze(1).repeat(1, max_nodes).view(-1)

        # Flatten node features: (B * max_nodes, feat_dim)
        x_all = node_features_b.view(-1, feat_dim)

        # Build a global edge_index by offsetting each graph's indices
        all_edge_index = []
        node_offset = 0
        for i in range(B):
            # Ensure edge_index is an integer tensor
            edge_index_i = edge_index_b[i].long()  
            edge_index_i = edge_index_i + node_offset
            all_edge_index.append(edge_index_i)
            node_offset += max_nodes  # Increase offset by the number of nodes per graph
        edge_index = torch.cat(all_edge_index, dim=1)

        # Ensure tensors are on the same device
        x_all = x_all.to(device)
        edge_index = edge_index.to(device)
        batch_idx_b = batch_idx_b.to(device)

        # --- GAT layers with residual connections ---
        x1 = F.leaky_relu(self.bn1(self.conv1(x_all, edge_index)), negative_slope=0.01)
        x1 = self.dropout(x1)
        x2 = F.leaky_relu(self.bn2(self.conv2(x1, edge_index)), negative_slope=0.01)
        # Project x1 from 256-dim to 512-dim before adding
        x = self.dropout(x2 + self.res_conv1(x1))
        x3 = F.leaky_relu(self.bn3(self.conv3(x, edge_index)), negative_slope=0.01)
        x = x3 + x

        # Global mean pooling over nodes for each graph in the batch
        x = global_mean_pool(x, batch_idx_b)

        # Final MLP for feature extraction
        return self.fc(x)




class PPOGNN1Policy(MaskableActorCriticPolicy):
    """ Custom PPO-GNN-1 Policy using ImprovedGNNFeatureExtractor """
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=ImprovedGNNFeatureExtractor,
            features_extractor_kwargs={"features_dim": 512},
            **kwargs
        )

