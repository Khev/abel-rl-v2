"""
agent.py
Discrete-action Soft-Actor-Critic compatible with the SB3 `.learn()` API.
"""
from __future__ import annotations
from typing import Tuple, Optional, Callable, Any
import time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical

# ───────────────────────── helpers ──────────────────────────
def _init(layer: nn.Linear) -> nn.Linear:
    nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
    nn.init.constant_(layer.bias, 0.0)
    return layer


def _apply_mask(logits: torch.Tensor, mask: torch.Tensor | None):
    if mask is None:
        return logits
    if mask.ndim == 1:
        mask = mask.unsqueeze(0)
    mask = mask.bool().to(logits.device)
    safe = torch.where((~mask).all(1, keepdim=True), torch.ones_like(mask), mask)
    return logits.masked_fill(~safe, -1e9)


# ────────────────────── Replay Buffer ───────────────────────
class Replay:
    def __init__(self, cap: int, obs_shape: Tuple[int, ...],
                 n_act: int, device: torch.device):
        self.device = device
        self.cap    = cap
        self.obs    = np.zeros((cap, *obs_shape), np.float32)
        self.nobs   = np.zeros_like(self.obs)
        self.act    = np.zeros((cap, 1), np.int64)
        self.rew    = np.zeros((cap, 1), np.float32)
        self.done   = np.zeros((cap, 1), np.float32)
        self.mask   = np.ones((cap, n_act), np.bool_)
        self.ptr, self.full = 0, False

    def add(self, o, no, a, r, d, m):
        self.obs[self.ptr],  self.nobs[self.ptr] = o, no
        self.act[self.ptr],  self.rew[self.ptr]  = a, r
        self.done[self.ptr], self.mask[self.ptr] = d, m
        self.ptr = (self.ptr + 1) % self.cap
        self.full |= self.ptr == 0

    def sample(self, batch: int):
        idx  = np.random.randint(0, self.cap if self.full else self.ptr, batch)
        to_t = lambda x: torch.as_tensor(x, device=self.device)
        return tuple(map(to_t, (self.obs[idx], self.nobs[idx],
                                self.act[idx], self.rew[idx],
                                self.done[idx], self.mask[idx])))


# ───────────────────── networks / policy ────────────────────
class QNet(nn.Module):
    def __init__(self, obs_dim: int, n_act: int, hid: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            _init(nn.Linear(obs_dim, hid)), nn.ReLU(),
            _init(nn.Linear(hid, hid)),     nn.ReLU(),
            _init(nn.Linear(hid, n_act)))
    def forward(self, x): return self.net(x)


class Policy(nn.Module):
    def __init__(self, obs_dim: int, n_act: int, hid: int = 256):
        super().__init__()
        self.body = nn.Sequential(
            _init(nn.Linear(obs_dim, hid)), nn.ReLU(),
            _init(nn.Linear(hid, hid)),     nn.ReLU())
        self.head = _init(nn.Linear(hid, n_act))

    def _logits(self, x, mask=None):
        return _apply_mask(self.head(self.body(x)), mask)

    def sample(self, x, mask=None):
        logits = self._logits(x, mask)
        dist   = Categorical(logits=logits)
        a      = dist.sample()
        logp   = F.log_softmax(logits, dim=-1)
        probs  = dist.probs
        return a, logp, probs

    def greedy(self, x, mask=None):
        return self._logits(x, mask).argmax(-1)


# ────────────────────────── Agent ────────────────────────────
class SACDiscrete:
    """
    A minimal SAC implementation whose **learn()** matches SB3’s signature:

        model.learn(total_timesteps, callback=None, progress_bar=False)

    Pass the environment in the constructor so learn() doesn’t need
    its own loop outside.
    """
    def __init__(self,
                 env,
                 device: torch.device,
                 *,
                 gamma      = 0.99,
                 tau        = 0.005,
                 batch_size = 256,
                 buffer_size= 200_000,
                 alpha_init = 0.05,
                 alpha_min  = 0.01,
                 autotune   = True):
        self.env      = env
        self.device   = device
        self.gamma, self.tau = gamma, tau
        self.batch    = batch_size
        self.alpha_min= alpha_min

        obs_dim = env.observation_space.shape[0]
        n_act   = env.action_space.n

        # networks
        self.q1  = QNet(obs_dim, n_act).to(device)
        self.q2  = QNet(obs_dim, n_act).to(device)
        self.tq1 = QNet(obs_dim, n_act).to(device)
        self.tq2 = QNet(obs_dim, n_act).to(device)
        self.tq1.load_state_dict(self.q1.state_dict())
        self.tq2.load_state_dict(self.q2.state_dict())
        self.pi  = Policy(obs_dim, n_act).to(device)

        self.opt_q  = optim.Adam(
            (*self.q1.parameters(), *self.q2.parameters()), lr=3e-4)
        self.opt_pi = optim.Adam(self.pi.parameters(), lr=3e-4)

        self.autotune = autotune
        if autotune:
            self.log_alpha  = torch.tensor(np.log(alpha_init),
                                           device=device, requires_grad=True)
            self.target_ent = -np.log(1.0 / n_act)
            self.opt_alpha  = optim.Adam([self.log_alpha], lr=3e-4)
        self.alpha = alpha_init

        self.rb = Replay(buffer_size, env.observation_space.shape,
                         n_act, device)

    # ───── interaction helpers ─────
    def _mask(self):
        mask = getattr(self.env, "action_mask", None)
        return mask if mask is not None else np.ones(
            self.env.action_space.n, dtype=bool)

    def act(self, obs, greedy=False):
        x    = torch.as_tensor(obs, device=self.device,
                               dtype=torch.float32).unsqueeze(0)
        mask = torch.as_tensor(self._mask(), device=self.device)
        with torch.no_grad():
            return (self.pi.greedy(x, mask) if greedy
                    else self.pi.sample(x, mask)[0]).item()

    def observe(self, *transition):
        self.rb.add(*transition)

    def ready(self):
        return self.rb.full or self.rb.ptr >= self.batch

    # ───── one gradient step ─────
    def _update(self):
        obs, nobs, act, rew, done, mask = self.rb.sample(self.batch)

        with torch.no_grad():
            na, nlogp, nap = self.pi.sample(nobs, mask)
            q1n, q2n = self.tq1(nobs), self.tq2(nobs)
            min_q_next = (nap * (torch.min(q1n, q2n) -
                                 self.alpha * nlogp)).sum(-1, keepdim=True)
            target = rew + (1 - done) * self.gamma * min_q_next

        q1_a = self.q1(obs).gather(1, act)
        q2_a = self.q2(obs).gather(1, act)
        q_loss = F.mse_loss(q1_a, target) + F.mse_loss(q2_a, target)
        self.opt_q.zero_grad(); q_loss.backward(); self.opt_q.step()

        a, logp, prob = self.pi.sample(obs, mask)
        q_min = torch.min(self.q1(obs), self.q2(obs)).detach()
        pi_loss = (prob * (self.alpha * logp - q_min)).sum(-1).mean()
        self.opt_pi.zero_grad(); pi_loss.backward(); self.opt_pi.step()

        if self.autotune:
            a_loss = -(self.log_alpha * (logp + self.target_ent).detach()).mean()
            self.opt_alpha.zero_grad(); a_loss.backward(); self.opt_alpha.step()
            self.alpha = max(self.log_alpha.exp().item(), self.alpha_min)

        with torch.no_grad():
            for p, tp in zip(self.q1.parameters(), self.tq1.parameters()):
                tp.data.mul_(1 - self.tau).add_(self.tau * p.data)
            for p, tp in zip(self.q2.parameters(), self.tq2.parameters()):
                tp.data.mul_(1 - self.tau).add_(self.tau * p.data)

    # ───── SB3-style train loop ─────
    def learn(self,
              total_timesteps: int,
              callback:  Optional[Callable] = None,
              progress_bar: bool = False,
              log_interval: int = 1_000,
              ) -> "SACDiscrete":

        obs, _  = self.env.reset()
        mask    = self._mask()
        start_t = time.time()

        for step in range(1, total_timesteps + 1):
            act = (self.env.action_space.sample()
                   if step < 10_000 else self.act(obs))
            nxt, rew, term, trunc, info = self.env.step(act)
            done = term or trunc
            self.observe(obs, nxt, act, rew, done, mask)
            obs  = nxt if not done else self.env.reset()[0]
            mask = self._mask()

            if self.ready() and step % 4 == 0:
                self._update()

            if progress_bar and step % log_interval == 0:
                sps = int(step / (time.time() - start_t))
                print(f"{step:,} | α={self.alpha:.3f} | SPS {sps}")

            if callback is not None and step % log_interval == 0:
                callback(self)
        return self
