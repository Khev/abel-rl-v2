#!/usr/bin/env python
"""
train.py – one script for PPO, A2C, DQN-PER *or* custom SAC.

Example
-------
python train.py --env-id abel_rl --algo sac  --Ntrain 300000
"""
from __future__ import annotations
import time, argparse, os, numpy as np, torch, gymnasium as gym
from torch.utils.tensorboard import SummaryWriter

from env   import make_env
from agent import SACDiscrete
from stable_baselines3         import PPO, A2C, DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor    import Monitor
from stable_baselines3.common.vec_env    import DummyVecEnv
from stable_baselines3.common.buffers    import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.evaluation    import evaluate_policy
from collections import namedtuple
import numpy as np
import torch as th
from stable_baselines3.common.buffers import ReplayBuffer, ReplayBufferSamples
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.env_util import make_vec_env

# Add for RLeXplore (install via: pip install rllte-core)
from rllte.xplore.reward import ICM, E3B

def get_main_eqn(env):
    """Drill through Monitor / DummyVecEnv / VectorWrapper to reach the raw env."""
    # 1. If it is a VecEnv (has .envs list), grab the first sub-env
    if hasattr(env, "envs"):           # DummyVecEnv, SubprocVecEnv, VecMonitor, …
        return get_main_eqn(env.envs[0])

    # 2. If it is a Gymnasium Wrapper (has .env), dive one level
    if hasattr(env, "env"):            # Monitor, TimeLimit, custom wrappers …
        return get_main_eqn(env.env)

    # 3. Base env reached; return attribute if present
    return getattr(env, "main_eqn", "N/A")

class SolveLoggerCallback(BaseCallback):
    """
    Prints a message whenever an episode is solved (info['is_solved']==True),
    then runs one deterministic evaluation episode in a separate env and
    stops training if it also solves.
    """
    def __init__(self, eval_env, algo, verbose: int = 0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.Tstop = -1
        self.Tfirst = -1
        self.algo = algo

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if info.get("is_solved", False):
                # log the solve event
                if self.Tfirst == -1:
                    self.Tfirst = self.num_timesteps
                env = self.training_env.envs[0]
                print(
                    f"{self.algo}: Solved at T = {self.num_timesteps}: "
                    f"{env.main_eqn} ==> {info['lhs']} = {info['rhs']}"
                )

                # run one deterministic eval in the separate eval_env
                obs, _ = self.eval_env.reset()
                done, solved = False, False
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, _, term, trunc, info_eval = self.eval_env.step(action)
                    solved |= info_eval.get("is_solved", False)
                    done = term or trunc

                if solved:
                    print(f"\n\nEarly stopping: at T = {self.num_timesteps}\n\n")
                    self.Tstop = self.num_timesteps
                    return False
        return True


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



def evaluate_agent(model, env: gym.Env, n_eval_episodes: int = 10):
    """
    Runs n_eval_episodes and returns:
      - mean_reward
      - std_reward
      - solve_rate  (fraction of episodes where info['is_solved']==True)
    """
    total_rewards = []
    solves = 0

    #print('\nEVALUATE AGENT')
    #print('-------------------\n')
    for _ in range(n_eval_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        ep_solved = False

        while not done:
            if isinstance(model, SACDiscrete):
                action = model.act(obs, greedy=True)
            else:
                action, _ = model.predict(obs, deterministic=False)

            obs, r, term, trunc, info = env.step(action)
            ep_reward += r
            done = term or trunc
            #print(f'{env.main_eqn} || {env.actions[action]} -> {info["lhs"]} = {info["rhs"]}')

            if info.get("is_solved", False):
                ep_solved = True

        total_rewards.append(ep_reward)
        if ep_solved:
            solves += 1

    mean_r = float(np.mean(total_rewards))
    std_r  = float(np.std(total_rewards))
    solve_rate = solves / n_eval_episodes
    #print('\n')

    return mean_r, std_r, solve_rate


# ───────────── helpers ─────────────
def make_vec(env_id, seed):
    def thunk(): return Monitor(make_env(env_id))
    venv = DummyVecEnv([thunk])
    venv.seed(seed)
    return venv


class RLeXploreWithOnPolicyRL(BaseCallback):
    """
    A custom callback for combining RLeXplore and on-policy algorithms from SB3.
    """
    def __init__(self, irs, verbose=0):
        super(RLeXploreWithOnPolicyRL, self).__init__(verbose)
        self.irs = irs
        self.buffer = None

    def init_callback(self, model: BaseAlgorithm) -> None:
        super().init_callback(model)
        self.buffer = self.model.rollout_buffer

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        observations = self.locals["obs_tensor"].float()
        device = observations.device
        actions = th.as_tensor(self.locals["actions"], device=device).float()
        rewards = th.as_tensor(self.locals["rewards"], device=device).float()
        dones = th.as_tensor(self.locals["dones"], device=device).float()
        next_observations = th.as_tensor(self.locals["new_obs"], device=device).float()

        # ===================== watch the interaction ===================== #
        self.irs.watch(observations, actions, rewards, dones, dones, next_observations)
        # ===================== watch the interaction ===================== #
        return True

    def _on_rollout_end(self) -> None:
        # ===================== compute the intrinsic rewards ===================== #
        # prepare the data samples
        obs = th.as_tensor(self.buffer.observations).float()
        # get the new observations
        new_obs = obs.clone()
        new_obs[:-1] = obs[1:]
        new_obs[-1] = th.as_tensor(self.locals["new_obs"]).float()
        actions = th.as_tensor(self.buffer.actions).float()
        rewards = th.as_tensor(self.buffer.rewards).float()
        dones = th.as_tensor(self.buffer.episode_starts).float()
        # compute the intrinsic rewards
        intrinsic_rewards = self.irs.compute(
            samples=dict(observations=obs, actions=actions, 
                         rewards=rewards, terminateds=dones, 
                         truncateds=dones, next_observations=new_obs),
            sync=True)
        # add the intrinsic rewards to the buffer
        self.buffer.advantages += intrinsic_rewards.cpu().numpy()
        self.buffer.returns += intrinsic_rewards.cpu().numpy()
        # ===================== compute the intrinsic rewards ===================== #

class RLeXploreWithOffPolicyRL(BaseCallback):
    """
    A custom callback for combining RLeXplore and off-policy algorithms from SB3. 
    """
    def __init__(self, irs, verbose=0):
        super(RLeXploreWithOffPolicyRL, self).__init__(verbose)
        self.irs = irs
        self.buffer = None

    def init_callback(self, model: BaseAlgorithm) -> None:
        super().init_callback(model)
        self.buffer = self.model.replay_buffer
        

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        device = self.irs.device
        obs = th.as_tensor(self.locals['self']._last_obs, device=device).float()
        actions = th.as_tensor(self.locals["actions"], device=device).float()
        rewards = th.as_tensor(self.locals["rewards"], device=device).float()
        dones = th.as_tensor(self.locals["dones"], device=device).float()
        next_obs = th.as_tensor(self.locals["new_obs"], device=device).float()

        # ===================== watch the interaction ===================== #
        self.irs.watch(obs, actions, rewards, dones, dones, next_obs)
        # ===================== watch the interaction ===================== #
        
        # ===================== compute the intrinsic rewards ===================== #
        intrinsic_rewards = self.irs.compute(samples={'observations':obs.unsqueeze(0), 
                                            'actions':actions.unsqueeze(0), 
                                            'rewards':rewards.unsqueeze(0),
                                            'terminateds':dones.unsqueeze(0),
                                            'truncateds':dones.unsqueeze(0),
                                            'next_observations':next_obs.unsqueeze(0)}, 
                                            sync=False)
        # ===================== compute the intrinsic rewards ===================== #

        try:
            # add the intrinsic rewards to the original rewards
            self.locals['rewards'] += intrinsic_rewards.cpu().numpy().squeeze()
            # update the intrinsic reward module
            replay_data = self.buffer.sample(batch_size=self.irs.batch_size)
            self.irs.update(samples={'observations': th.as_tensor(replay_data.observations).unsqueeze(1).to(device).float(), # (n_steps, n_envs, *obs_shape)
                                     'actions': th.as_tensor(replay_data.actions).unsqueeze(1).to(device).float(),
                                     'rewards': th.as_tensor(replay_data.rewards).to(device).float(),
                                     'terminateds': th.as_tensor(replay_data.dones).to(device).float(),
                                     'truncateds': th.as_tensor(replay_data.dones).to(device).float(),
                                     'next_observations': th.as_tensor(replay_data.next_observations).unsqueeze(1).to(device).float()
                                     })
        except:
            pass

        return True

    def _on_rollout_end(self) -> None:
        pass

# ───────────── main ─────────────
def main(args):
    run_name = f"{args.env_id}_{args.algo}_{int(time.time())}"
    tb       = SummaryWriter(os.path.join("runs", run_name))

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available()
                          else "cpu")
    np.random.seed(args.seed); torch.manual_seed(args.seed)

    eval_env = Monitor(make_env(args.env_id))
    eval_cb  = EvalCallback(eval_env, eval_freq=args.eval_freq,
                            deterministic=True, verbose=0)
    solve_logger = SolveLoggerCallback(eval_env, args.algo)
    callback = [solve_logger, eval_cb]

    # ╭──────────────── select & build model ───────────────╮
    if args.algo in {"ppo", "a2c"}:
        Algo = PPO if args.algo == "ppo" else A2C
        env  = make_vec(args.env_id, args.seed)
        model = Algo("MlpPolicy", env,
                     tensorboard_log = tb.log_dir,
                     seed            = args.seed,
                     device          = device)

    elif args.algo == "dqn":
        env = make_vec(args.env_id, args.seed)
        model =  DQN("MlpPolicy", env)

    elif args.algo == 'dqn-per':
        env = make_vec(args.env_id, args.seed)
        model = DQN(
            "MlpPolicy", env,
            buffer_size            = 100_000,
            replay_buffer_class    = PERBuffer,
            replay_buffer_kwargs   = dict(alpha=0.6, beta=0.4, eps=1e-5),
            learning_starts        = 1_000,
            batch_size             = 256,
            tensorboard_log        = tb.log_dir,
            seed                   = args.seed,
            device          = device,
        )

    else:  # SAC
        env   = Monitor(make_env(args.env_id))
        model = SACDiscrete(env, device,
                            autotune   = True,
                            alpha_init = 0.05)
    # ╰──────────────────────────────────────────────────────╯

    if args.curiosity is not None:
        if args.curiosity == 'ICM':
            irs = ICM(env, device=device)
        elif args.curiosity == 'E3B':
            irs = E3B(env, device=device)
        else:
            raise ValueError(f"Unknown curiosity method: {args.curiosity}")

        if args.algo in {"ppo", "a2c"}:
            irs_callback = RLeXploreWithOnPolicyRL(irs)
        else:
            irs_callback = RLeXploreWithOffPolicyRL(irs)
        callback.append(irs_callback)


    # --- evaluate before training ---
    mean_pre, std_pre, win_rate_pre = evaluate_agent(model, eval_env, n_eval_episodes=5)

    # --- unified .learn() call ---
    model.learn(total_timesteps = args.Ntrain,
                callback        = callback,
                progress_bar    = True)

    mean_post, std_post, win_rate_post = evaluate_agent(model, eval_env, n_eval_episodes=5)


    print(f'{args.algo} | Eqn: {get_main_eqn(env.envs[0])}')
    print(f"win_rate_before = {win_rate_pre:.2f}")
    print(f"win_rate_after = {win_rate_post:.2f}")

    eval_env.close(); tb.close()
    print("Finished training! ✅")


    Tfirst, Tstop = solve_logger.Tfirst, solve_logger.Tstop
    return Tfirst, Tstop


if __name__ == "__main__":

    p = argparse.ArgumentParser()
    p.add_argument("--env-id",           default="single")
    p.add_argument("--algo",             default="ppo",
                   choices=["ppo", "a2c", "dqn", "dqn-per", "sac"])
    p.add_argument("--Ntrain",  type=int, default=3*10**6)
    p.add_argument("--seed",             type=int, default=1)
    p.add_argument("--cuda",             action="store_true")
    p.add_argument("--eval-freq",        type=int, default=10_000)
    p.add_argument("--curiosity",        type=str, default='ICM', choices=[None, "ICM", "E3B"])
    args = p.parse_args()

    algos = ["ppo", "a2c", "dqn", "dqn-per"]
    results = {}
    for algo in algos:
        args.algo = algo
        Tfirst, Tstop = main(args)
        results[algo] = [Tfirst,Tstop]
    
    print(f'\n\nFinal results: env = {args.env_id}, curiosity = {args.curiosity}')
    print('----------------------------------------')
    for key,val in results.items():
        print(f'{key}: Tsolve, Tconverge = {val[0]}, {val[1]}')