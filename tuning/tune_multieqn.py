import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch, time
import argparse, json, pickle, time
import numpy as np
import torch as th
import pandas as pd
import matplotlib.pyplot as plt

from envs.multi_eqn import multiEqn
from itertools import product

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

import torch.nn as nn
import torch.nn.functional as F  # ✅ Needed for activation functions
from torch_geometric.nn import GCNConv, global_mean_pool


from utils.utils_train import get_agent, get_intrinsic_reward, get_device
from utils.custom_functions import operation_names
from utils.utils_general import print_parameters, print_header
from colorama import Fore, Style

def print_eval_results(test_results, label=""):
    """
    test_results is a dict: { eqn_string : success_rate_float, ... }
    """
    print(f"{label} Equations")
    # Convert dict to a DataFrame with columns = ['Eqn', 'Win%']
    df = pd.DataFrame(
        [{'Eqn': eqn, 'Win%': f"{winpct:.1f}%"} for eqn, winpct in test_results.items()]
    )
    print(df.to_string(index=False))
    print()


def evaluate_agent(agent, env, equation_list, n_eval_episodes=10):
    """
    Evaluate agent on each eqn in equation_list for n_eval_episodes per eqn.
    Returns a dict eqn -> success percentage.
    """
    results = {}

    for eqn in equation_list:
        eqn_successes = 0
        for ep in range(n_eval_episodes):
            # Ensure we reset correctly in DummyVecEnv
            #obs, _ = env.get_attr('reset')[0]() if hasattr(env, 'get_attr') else env.reset()
            obs = env.reset()
            env.env_method('set_equation', eqn)
            #breakpoint()
            done = False
            while not done:
                action, _ = agent.predict(obs, deterministic=False)
                obs, reward, done, info = env.step(action)
                if info[0].get('is_solved', False):
                    eqn_successes += 1
                    break
        success_rate = eqn_successes / n_eval_episodes * 100.0
        results[eqn] = success_rate

    return results

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
        self.early_stopping = False

        # We'll defer initialization of results_train until training actually starts:
        self.results_train = {}
        self.results_test = {}

        # Learning states
        self.logged_steps = []
        self.train_accuracy = []
        self.test_accuracy = []
        self.test_acc_max = 0

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def _on_training_start(self):
        """
        Called once the training starts, when self.training_env is available.
        """
        # 'train_eqns' presumably is an attribute in your env that lists all training eqns
        # Because we are inside a DummyVecEnv, we can do get_attr('train_eqns')[0].
        eqns_attr = self.training_env.get_attr('train_eqns')
        if eqns_attr:
            train_eqns = eqns_attr[0]
            self.results_train = {eqn: None for eqn in train_eqns}
        else:
            print("Warning: no 'train_eqns' found in env; results_train will remain empty.")

        # If you have test_eqns, you can do the same for results_test (though you might
        # want to populate it after your final evaluation, see below).
        test_eqns_attr = self.training_env.get_attr('test_eqns')
        if test_eqns_attr:
            test_eqns = test_eqns_attr[0]
            self.results_test = {eqn: None for eqn in test_eqns}
        else:
            print("No 'test_eqns' found in env. results_test will remain empty.")

        # if self.eval_env:
        #     print("\nInitial evaluation (t=0)...")
        #     train_results = evaluate_agent(self.model, self.eval_env, train_eqns, n_eval_episodes=10)
        #     print_eval_results(train_results, label="Train")

        #     test_results = evaluate_agent(self.model, self.eval_env, test_eqns, n_eval_episodes=10)
        #     print_eval_results(test_results, label="Test")

        #     self.logged_steps.append(0)  # Log step 0
        #     self.train_accuracy.append(np.mean(list(train_results.values())))
        #     self.test_accuracy.append(np.mean(list(test_results.values())))

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
            #print(Fore.YELLOW + f'\nSolved {main_eqn} = 0 ==> {lhs} = {rhs} at Nstep = {self.num_timesteps}' + Style.RESET_ALL)
            self.T_solve = self.num_timesteps

            if self.results_train[main_eqn] == None:
                self.results_train[main_eqn] = self.num_timesteps
                #print(Fore.YELLOW + f'\nSolved {main_eqn} = 0 ==> {lhs} = {rhs} at Nstep = {self.num_timesteps}' + Style.RESET_ALL)

            # Stop training early if enabled
            if self.early_stopping:
                #print(Fore.YELLOW + f"{main_eqn}: Early stopping for triggered at step {self.num_timesteps}!" + Style.RESET_ALL)
                return False  

        # Logging reward statistics
        # if self.n_calls % self.log_interval == 0:
        #     mean_reward_ext = np.mean(self.rewards_ext[-self.log_interval:])
        #     min_reward_ext = np.min(self.rewards_ext[-self.log_interval:])
        #     max_reward_ext = np.max(self.rewards_ext[-self.log_interval:])

        #     main_eqn = info['main_eqn']
        #     print(f"{main_eqn}: Step {self.num_timesteps}: "
        #           f"(min, mean, max)_reward_external = ({min_reward_ext:.2f}, {mean_reward_ext:.2f}, {max_reward_ext:.2f})")

        #Evaluation at intervals
        if self.eval_env and self.n_calls % self.eval_interval == 0:

            #print("\nRunning evaluation...")
            train_results = evaluate_agent(self.model, self.eval_env, self.eval_env.get_attr('train_eqns')[0], n_eval_episodes=10)
            #print_eval_results(train_results, label='Train')

            test_results = evaluate_agent(self.model, self.eval_env, self.eval_env.get_attr('test_eqns')[0], n_eval_episodes=10)
            #print_eval_results(test_results, label='Test')


            self.logged_steps.append(self.num_timesteps)
            train_acc = np.mean(list(train_results.values()))
            test_acc = np.mean(list(test_results.values()))
            self.train_accuracy.append(np.mean(list(train_results.values())))
            self.test_accuracy.append(np.mean(list(test_results.values())))

            self.test_acc_max = max(self.test_acc_max, test_acc)
            #print(f'here = {self.test_acc_max}')


            # Early stopping
            if test_acc == 100.0 and train_acc == 100.0:
                print(Fore.YELLOW + f"'train_acc = test_acc = 100. Early stopping step {self.num_timesteps}!" + Style.RESET_ALL)
                return False

        return True  # Continue training


    def _on_training_end(self) -> None:
        """
        Called when training finishes. Plots train vs test accuracy curves.
        """
        print("\nFinal Training Completed. Plotting Learning Curves...")

        plt.figure(figsize=(10, 6))
        plt.plot(self.logged_steps, self.train_accuracy, label="Train Accuracy", marker='o', linestyle='-')
        plt.plot(self.logged_steps, self.test_accuracy, label="Test Accuracy", marker='s', linestyle='--')

        plt.xlabel("Training Steps")
        plt.ylabel("Success Rate (%)")
        plt.title("Train vs. Test Learning Progress")
        plt.legend()
        plt.grid()
        plt.ylim([0,105])
        plt.savefig(os.path.join(self.save_dir, "learning_curve.png"))
        #plt.show()  # Only if running interactively
        plt.close()


        # Save learning curve data
        save_path = os.path.join(self.save_dir, "learning_progress.pkl")
        with open(save_path, "wb") as f:
            pickle.dump({
                "steps": self.logged_steps,
                "train_success": self.train_accuracy,
                "test_success": self.test_accuracy
            }, f)
        print(f"Saved learning progress to {save_path}")




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


def main(ent_coeff, hidden_dim, n_layers, features_dim, save_dir='.'):

    # env 
    def get_action_mask(env):
        return env.action_mask
    env = multiEqn(state_rep='graph_integer_2d', level=2)
    env = ActionMasker(env, get_action_mask)
    env = DummyVecEnv([lambda: Monitor(env)])

    # agent
    policy_kwargs = dict(
        activation_fn=th.nn.ReLU,  # ✅ ReLU activation function
        net_arch=dict(
            pi=[hidden_dim] * n_layers,  # ✅ Policy network architecture
            vf=[hidden_dim] * n_layers   # ✅ Value function architecture
        ),
        # features_extractor_class=GNNFeatureExtractor,  # ✅ Custom GNN feature extractor
        # features_extractor_kwargs={"features_dim": features_dim},  # ✅ Feature dimension
    )

    kwargs = {
        'ent_coef': ent_coeff,  # ✅ Correct SB3 argument
        'learning_rate': 1e-4,  # ✅ Ensure learning_rate is set correctly
        'n_steps': 2048,
        'batch_size': batch_size,
        'gamma': 0.99,
        'gae_lambda': 0.99,
        'clip_range':0.05           # Smaller step sizes to prevent oscillations
        # 'policy_kwargs': policy_kwargs,  # ✅ Correctly pass policy settings here
    }
    agent = MaskablePPO(CustomGNNPolicy, env, policy_kwargs=policy_kwargs, **kwargs)

    # Callback
    Ntrain = 5*10**6
    #print(f"\nTraining for {Ntrain} step")
    log_interval = int(0.1*Ntrain)
    callback = TrainingLogger(log_interval=log_interval, eval_env=env, save_dir=save_dir)

    # train
    agent.learn(total_timesteps=Ntrain, callback=callback)

    # get results
    train_results = evaluate_agent(agent, env, env.get_attr('train_eqns')[0], n_eval_episodes=10)
    test_results = evaluate_agent(agent, env, env.get_attr('test_eqns')[0], n_eval_episodes=10)

    print(f'\n\nEnt_coeff = {ent_coeff}')
    for eqn, result in test_results.items():
        print(f'{eqn}: {result}')

    acc_train =  np.mean([acc for acc in train_results.values()])
    acc_test  = np.mean([acc for acc in test_results.values()])

    acc_test_max = callback.test_acc_max

    return acc_train, acc_test, acc_test_max


# Set device
device_type = 'mps'
batch_size = 512
## FILL IN CODE HERE

# Save data
save_dir = 'data/tuning/multieqn'
ent_coeffs = [0.01]
hidden_dims = [256]
n_layers = [2,3,4]
feature_dims = [128]
param_combinations = list(product(ent_coeffs, hidden_dims, n_layers, feature_dims))

df_results = []
t1 = time.time()
for ent_coeff, hidden_dim, n_layers, feature_dim in param_combinations:
    print(f'Ent_coeff: {ent_coeff}, Hidden_dim: {hidden_dim}, N_layers: {n_layers}, Features_dim: {feature_dim}')
    acc_train, acc_test, acc_test_max = main(ent_coeff, hidden_dim, n_layers, feature_dim, save_dir=save_dir)
    print(f'Test_acc: {acc_test:.3f}')
    
    df_results.append({
        "entropy_coeff": ent_coeff,
        "hidden_dim": hidden_dim,
        "n_layers": n_layers,
        "feature_dim": feature_dim,
        "test_acc": acc_test,
        "test_acc_max": acc_test_max
    })

# Save results
df_results = pd.DataFrame(df_results)
df_results.to_csv("tuning_results.csv", index=False)
print("✅ Grid Search Completed! Results saved.")


# Order results by test accuracy and print best parameters
print("\n===============================")
print("🎯 Grid Search Completed! Sorting Best Configurations")
print("===============================\n")

best_results = df_results.sort_values(by="test_acc_max", ascending=False).head(5)

print("🏆 Top 5 Best Configurations (Sorted by Test Accuracy):")
print(best_results.to_string(index=False))

# Save sorted results
sorted_filename = os.path.join(save_dir, "sorted_grid_search_results.csv")
best_results.to_csv(sorted_filename, index=False)
print(f"📈 Sorted results saved to: {sorted_filename}")
print("\n✅ All done! 🚀")

# Calculate elapsed time
t2 = time.time()
elapsed_time = t2 - t1
hours, rem = divmod(elapsed_time, 3600)
minutes, seconds = divmod(rem, 60)

print(f"⏳ Total time taken: {int(hours)}h {int(minutes)}m {int(seconds)}s")


