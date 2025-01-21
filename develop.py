import numpy as np
from sympy import symbols, E, I, pi, zoo, sympify, Basic,  Integer, Float
from envs.single_eqn import singleEqn

import torch, os
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env

from colorama import Fore, Style


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



class CustomGNNPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=GNNFeatureExtractor,
            features_extractor_kwargs={"features_dim": 128},
            **kwargs
        )


class TrainingLogger(BaseCallback):
    """
    Callback for logging reward statistics, evaluating the agent, and saving checkpoints at regular intervals.
    """

    def __init__(self, log_interval=1000, eval_interval=10000, save_dir=".", verbose=1, eval_env=None):
        super(TrainingLogger, self).__init__(verbose)
        self.log_interval = log_interval
        self.eval_interval = log_interval
        self.save_dir = save_dir
        self.eval_env = eval_env  # Separate evaluation environment
        self.rewards_ext = []  # External rewards
        self.T_solve = None
        self.T_converge = None
        self.early_stopping = True

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def _on_step(self) -> bool:
        """
        This function is called at each step of training.
        """
        # Get latest external reward from the environment
        reward_ext = self.locals["rewards"][0]
        self.rewards_ext.append(reward_ext)
        info = self.locals["infos"][0]

        if info['is_solved']:
            main_eqn, lhs, rhs = info['main_eqn'], info['lhs'], info['rhs']
            #print(Fore.GREEN + f'\nSolved {main_eqn} = 0 ==> {lhs} = {rhs} at Nstep = {self.num_timesteps}' + Style.RESET_ALL)
            self.T_solve = self.num_timesteps

            # Stop training early if enabled
            if self.early_stopping:
                print(Fore.YELLOW + f"{main_eqn}: Early stopping for triggered at step {self.num_timesteps}!" + Style.RESET_ALL)
                return False  

        # Logging reward statistics
        if self.n_calls % self.log_interval == 0:
            mean_reward_ext = np.mean(self.rewards_ext[-self.log_interval:])
            min_reward_ext = np.min(self.rewards_ext[-self.log_interval:])
            max_reward_ext = np.max(self.rewards_ext[-self.log_interval:])

            main_eqn = info['main_eqn']
            print(f"{main_eqn}: Step {self.num_timesteps}: "
                  f"(min, mean, max)_reward_external = ({min_reward_ext:.2f}, {mean_reward_ext:.2f}, {max_reward_ext:.2f})")

        # Evaluation at intervals
        # if self.eval_env and self.n_calls % self.eval_interval == 0:
        #     print("\nRunning evaluation...")
        #     mean_eval_reward = evaluate_trained_agent(self.model, self.eval_env, n_eval_episodes=1, deterministic=False)
        #     print(f"Evaluation Mean Reward: {mean_eval_reward:.2f}\n")

        # # Save model checkpoint at intervals
        # if self.n_calls % self.eval_interval == 0:
        #     save_path = os.path.join(self.save_dir, f"checkpoint_{self.num_timesteps}.zip")
        #     self.model.save(save_path)
        #     print(f"Checkpoint saved: {save_path}")

        return True  # Continue training


# Environment
def make_single_env():
    return singleEqn(state_rep='graph_integer_2d', main_eqn='a/x+b')

# Args
Ntrain = 10**4

# Env
env = make_vec_env(make_single_env, n_envs=1)
state = env.reset()

# Model
model = PPO(CustomGNNPolicy, env, device="cuda" if torch.cuda.is_available() else "cpu")

# Callback
callback = TrainingLogger(log_interval=int(0.1*Ntrain))

# Train
model.learn(total_timesteps=Ntrain,callback=callback)