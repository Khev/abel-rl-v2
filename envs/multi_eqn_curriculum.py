import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
import numpy as np
from sympy import sympify
from gymnasium import spaces, Env
from utils.custom_functions import *
from operator import add, sub, mul, truediv
from utils.utils_env import *
from collections import defaultdict, deque
import faiss                    # pip install faiss-cpu

logger = logging.getLogger(__name__)

import hashlib
import numpy as np
import faiss                     # pip install faiss-cpu


def _is_effective(lhs_old, rhs_old, lhs_new, rhs_new) -> bool:
    """True ↔ (lhs,rhs) actually changed."""
    return (lhs_old != lhs_new) or (rhs_old != rhs_new)

def fmt_action(act):
    """Readable string for an (operation, term) tuple."""
    op, term = act
    op_name = operation_names.get(op, op.__name__)
    return f"({op_name}, {term})"


def _fmt_step(lhs, rhs, op, term) -> str:
    """`a*x = b  --add(c)-->`  (one step)."""
    name = operation_names.get(op, op.__name__)
    t    = "" if term is None else str(term)
    return f"{lhs} = {rhs}  --{name}({t})-->"

def fmt_traj(traj_readable) -> str:
    """
    Convert the *readable* trajectory list to a multi-line string.
    Each item is (lhs, rhs, operation, term).
    """
    lines = []
    for lhs, rhs, op, term in traj_readable:
        lines.append(_fmt_step(lhs, rhs, op, term))
    return "\n".join(lines)


class SolveMemory:
    """
    Episodic nearest-neighbour memory used for imitation-style hints.

    * Every **successful** episode supplies (obs, action) pairs.
    * At query-time we return the action of the single L2-nearest state
      together with a similarity score in (0, 1].

    ---------- NEW vs. previous version ----------
    1. **De-duplication** – identical observations are stored only once
       (hash-based guard).  This keeps the index small and unbiased.
    2. **Capacity clamp** unchanged (FIFO trimming when full).
    3. **Doc-strings / typing** added for clarity.
    """

    def __init__(self, capacity: int = 50_000, k: int = 1, dedup: bool = True):
        self.capacity  : int  = capacity         # max transitions kept
        self.k         : int  = k                # # neighbours to retrieve
        self.dedup     : bool = dedup            # skip exact duplicates
        self.index     = None                    # FAISS index (built lazily)

        self.obs_store: list[np.ndarray] = []    # flattened observations
        self.act_store: list[int]        = []    # matching discrete actions
        self._hash_set: set[str]         = set() # for de-duplication

    # ------------------------------------------------------------------ #
    # utilities
    # ------------------------------------------------------------------ #
    @staticmethod
    def _vec_hash(vec: np.ndarray) -> str:
        """MD5 hash of the raw bytes of a NumPy vector (fast & short)."""
        return hashlib.md5(vec.tobytes()).hexdigest()

    # ------------------------------------------------------------------ #
    # ingestion
    # ------------------------------------------------------------------ #
    def add_episode(self, episode):
        """
        Parameters
        ----------
        episode : list
            Sequence of `(obs, action, ...)` tuples from a *successful* run.
            Only the first two fields are used.
        """
        for obs, act, *_ in episode:
            flat = obs.flatten()

            # ---------- de-duplication guard ----------
            if self.dedup:
                h = self._vec_hash(flat)
                if h in self._hash_set:
                    continue                     # identical state already stored
                self._hash_set.add(h)
            # ----------------------------------------- #

            self.obs_store.append(flat)
            self.act_store.append(act)           # (operation, term) tuple is fine

        # FIFO trim to capacity
        overflow = len(self.obs_store) - self.capacity
        if overflow > 0:
            self.obs_store = self.obs_store[overflow:]
            self.act_store = self.act_store[overflow:]
            if self.dedup:                        # drop hashes of removed items
                self._hash_set = {
                    self._vec_hash(v) for v in self.obs_store
                }

        self.index = None                         # FAISS index stale

    # ------------------------------------------------------------------ #
    # internal: (re)build FAISS index on demand
    # ------------------------------------------------------------------ #
    def _maybe_build(self):
        if self.index is not None or not self.obs_store:
            return
        dim = self.obs_store[0].shape[0]
        self.index = faiss.IndexFlatL2(dim)       # exact L2 search
        self.index.add(np.stack(self.obs_store).astype(np.float32))

    # ------------------------------------------------------------------ #
    # query
    # ------------------------------------------------------------------ #
    def query(self, obs_vec: np.ndarray):
        """
        Parameters
        ----------
        obs_vec : np.ndarray
            Observation **row-vector** with shape `(1, dim)` or `(dim,)`.

        Returns
        -------
        (action, similarity) :
            * `action`  – int or **None** if memory empty  
            * `similarity` – mapped to (0, 1],   `1 / (1 + L2²)`
        """
        self._maybe_build()
        if self.index is None:
            return None, 0.0

        # FAISS expects (n, dim) float32
        q = obs_vec.astype(np.float32, copy=False).reshape(1, -1)
        dists, idxs = self.index.search(q, self.k)
        best_id   = int(idxs[0, 0])
        best_dist = float(dists[0, 0])            # squared-L2
        similarity = 1.0 / (1.0 + best_dist)      # (0, ∞) → (1, 0]

        return self.act_store[best_id], similarity



class multiEqn(Env):
    """
    Environment for solving multiple equations using RL, 
    with a simple curriculum that samples equations inversely 
    proportional to how often they've been solved.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, 
                 state_rep='integer_1d', 
                 normalize_rewards=True, 
                 verbose=False,
                 cache=False, 
                 level=4, 
                 generalization='structural') -> None:
        super().__init__()

        # Static parts
        self.max_expr_length = 20
        self.max_steps = 10
        self.action_dim = 50
        self.observation_dim = 2*self.max_expr_length + 1

        # Rewards
        self.reward_solved = +100
        self.reward_invalid_equation = -100
        self.reward_illegal_action = -1

        # Optimizing
        self.cache = cache
        if self.cache:
            self.action_cache = {}

        # Pars
        self.normalize_rewards = normalize_rewards
        self.state_rep = state_rep
        self.verbose = verbose

        # Load train/test equations
        self.level = level
        self.generalization = generalization
        eqn_dirn = f"equation_templates"
        self.train_eqns, self.test_eqns = load_train_test_equations(
            eqn_dirn, level, generalization=generalization
        )

        # Tracking how many times we've solved each eqn
        # Use a dict with keys = the actual sympy expression or string
        self.solve_counts = defaultdict(int)
        self.sample_counts = defaultdict(int)

        # Solution memories
        self.mem = SolveMemory(capacity=1000)
        self.traj = []
        self.traj_readable = []   # human-readable    (lhs , rhs , op , term)

        # Convert each eqn to a canonical string so we can store counts easily
        self.train_eqns_str = [str(eq) for eq in self.train_eqns]
        self.test_eqns_str = [str(eq) for eq in self.test_eqns]


        # Random initial eqn
        eqn_str = np.random.choice(self.train_eqns_str)
        self.main_eqn = sympify(eqn_str)
        self.lhs = self.main_eqn
        self.rhs = 0
        self.x = symbols('x')

        #  Make feature_dict, actions etc
        self.setup()

        # RL env setup
        self.state = self.to_vec(self.lhs, self.rhs)
        self.action_space = spaces.Discrete(self.action_dim)

        if state_rep == 'integer_1d':
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, 
                shape=(self.observation_dim,), 
                dtype=np.float64
            )
        elif state_rep == 'integer_2d':
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, 
                shape=(self.observation_dim, 2), 
                dtype=np.float64
            )
        elif state_rep in ['graph_integer_1d', 'graph_integer_2d']:
            self.observation_space = spaces.Dict({
                "node_features": spaces.Box(
                    low=-np.inf, high=np.inf, 
                    shape=(self.observation_dim, 2), 
                    dtype=np.float64
                ),
                "edge_index": spaces.Box(
                    low=0, high=self.observation_dim, 
                    shape=(2, 2*self.observation_dim), 
                    dtype=np.int32
                ),
                "node_mask": spaces.Box(
                    low=0, high=1, 
                    shape=(self.observation_dim,), 
                    dtype=np.int32
                ),
                "edge_mask": spaces.Box(
                    low=0, high=1, 
                    shape=(2*self.observation_dim,), 
                    dtype=np.int32
                ),
            })
        else:
            raise ValueError(f"Unsupported state representation: {state_rep}")


    def setup(self):
        # Build feature dict from all train eqns
        self.feature_dict = make_feature_dict_multi(
            self.train_eqns, self.test_eqns, self.state_rep
        )

        # Define some fixed 'global' transformations
        self.actions_fixed = [
            (custom_expand, None),
            (custom_factor, None),
            (custom_collect, self.x),
            (custom_together, None),
            (custom_ratsimp, None),
            (custom_square, None),
            (custom_sqrt, None),
            (mul, -1),
        ]

        if self.cache:
            self.actions, self.action_mask = make_actions_cache(
                self.lhs, self.rhs, self.actions_fixed, 
                self.action_dim, self.action_cache
            )
        else:
            self.actions, self.action_mask = make_actions(
                self.lhs, self.rhs, self.actions_fixed, self.action_dim
            )


    def step(self, action_index: int):
        lhs_old, rhs_old, obs_old = self.lhs, self.rhs, self.state

        # ── (re)build the current action list ────────────────────────────────────
        if self.cache:
            action_list, action_mask = make_actions_cache(
                lhs_old, rhs_old, self.actions_fixed,
                self.action_dim, self.action_cache
            )
        else:
            action_list, action_mask = make_actions(
                lhs_old, rhs_old, self.actions_fixed, self.action_dim
            )

        self.actions, self.action_mask = action_list, action_mask

        # ── memory-guided imitation ------------------------------------------------
        mem_tuple, sim = self.mem.query(obs_old.flatten()[None, :])      # ← (op, term) or None
        if mem_tuple is not None:
            try:
                idx_in_current = action_list.index(mem_tuple)
                if np.random.rand() < sim:
                    # print(
                    #     f"Overrode action: {lhs_old} = {rhs_old} | "
                    #     f"{fmt_action(action_list[action_index])}  ==>  {fmt_action(mem_tuple)}"
                    # )
                    action_index = idx_in_current
            except ValueError:
                pass

        # ── apply chosen action ----------------------------------------------------
        operation, term = action_list[action_index]
        lhs_new, rhs_new = operation(lhs_old, term), operation(rhs_old, term)
        obs_new, _ = self.to_vec(lhs_new, rhs_new)

        # ── record transition for potential memory storage ------------------------
        # (skip no-op to keep the buffer useful – optional but recommended)
        if _is_effective(lhs_old, rhs_old, lhs_new, rhs_new):
            self.traj.append((obs_old.copy(), (operation, term)))  # CHANGED: store *tuple*, not index
            self.traj_readable.append((lhs_old, rhs_old, operation, term))


        # ── environment bookkeeping ------------------------------------------------
        is_valid_eqn, lhs_new, rhs_new = check_valid_eqn(lhs_new, rhs_new)
        is_solved = check_eqn_solved(lhs_new, rhs_new, self.main_eqn)

        reward = self.find_reward(lhs_old, rhs_old, lhs_new, rhs_new,
                                is_valid_eqn, is_solved)

        too_many_steps = (self.current_steps >= self.max_steps)
        terminated = bool(is_solved or too_many_steps or not is_valid_eqn)
        truncated = False

        if is_solved:
            eqn_str = str(self.main_eqn)
            self.solve_counts[eqn_str] += 1
            self.mem.add_episode(self.traj)       # CHANGED: episode stores tuples
            #print("Episode stored in memory:\n" + fmt_traj(self.traj_readable))
            #breakpoint()
            self.traj.clear()
            self.traj_readable.clear()

        if terminated or truncated:
            self.traj.clear()                     # reset trajectory buffer
            self.traj_readable.append((lhs_old, rhs_old, operation, term))

        # update state
        self.lhs, self.rhs, self.state = lhs_new, rhs_new, obs_new
        self.current_steps += 1

        info = {
            "is_solved":      is_solved,
            "is_valid_eqn":   is_valid_eqn,
            "too_many_steps": too_many_steps,
            "lhs":            self.lhs,
            "rhs":            self.rhs,
            "main_eqn":       self.main_eqn,
            "action_mask":    self.action_mask,
        }

        if self.verbose:
            print(f"{self.lhs} = {self.rhs}. "
                f"(Operation, term): ({operation_names[operation]}, {term})")

        return obs_new, reward, terminated, truncated, info



    def reset(self, seed=None, options=None):

        # Sample eqn in a 'curriculum' fashion:
        # pick eqn with probability ~ 1/(1+solve_counts)
        eqn_probs = []
        if options == None:
            for eqn_str in self.train_eqns_str:
                eqn_probs.append( 1.0 / (1 + self.solve_counts[eqn_str]) )
            eqn_probs = np.array(eqn_probs, dtype=np.float64)
            eqn_probs /= eqn_probs.sum()
            chosen_eqn_str = np.random.choice(self.train_eqns_str, p=eqn_probs)
            self.main_eqn = sympify(chosen_eqn_str)
            self.sample_counts[chosen_eqn_str] += 1
        elif options == 'train':
            chosen_eqn_str = np.random.choice(self.train_eqns_str)
            self.main_eqn = sympify(chosen_eqn_str)
        elif options == 'test':
            chosen_eqn_str = np.random.choice(self.test_eqns_str)
            self.main_eqn = sympify(chosen_eqn_str)

        self.current_steps = 0
        self.lhs, self.rhs = self.main_eqn, 0
        obs, _ = self.to_vec(self.lhs, self.rhs)
        self.state = obs
        self.traj.clear()
        self.traj_readable.clear()

        # Recompute actions, masks, etc.
        self.setup()

        return obs, {}



    def to_vec(self, lhs, rhs):
        """
        Convert (lhs, rhs) to a suitable observation 
        given self.state_rep.
        """
        if self.state_rep == 'integer_1d':
            return integer_encoding_1d(lhs, rhs, self.feature_dict, self.max_expr_length)
        elif self.state_rep == 'integer_2d':
            return integer_encoding_2d(lhs, rhs, self.feature_dict, self.max_expr_length)
        elif self.state_rep in ['graph_integer_1d', 'graph_integer_2d']:
            return graph_encoding(lhs, rhs, self.feature_dict, self.max_expr_length)
        else:
            raise ValueError(f"Unknown state_rep: {self.state_rep}")


    def find_reward(self, lhs_old, rhs_old, lhs_new, rhs_new, is_valid_eqn, is_solved):
        """
        Reward = 
          +100 if solved
          -100 if invalid eqn
          else ( oldComplexity - newComplexity )
               optionally normalized to [-1, 1].
        """
        if not is_valid_eqn:
            reward = self.reward_invalid_equation
        elif is_solved:
            reward = self.reward_solved
        else:
            old_complex = get_complexity_expression(lhs_old) + get_complexity_expression(rhs_old)
            new_complex = get_complexity_expression(lhs_new) + get_complexity_expression(rhs_new)
            reward = old_complex - new_complex

        if self.normalize_rewards:
            # rescale reward to [-1, 1]
            # min=-100, max=+100
            min_r, max_r = self.reward_invalid_equation, self.reward_solved
            reward = 2.0 * (reward - min_r) / float(max_r - min_r) - 1.0

        return reward

    def render(self, mode="human"):
        print(f"{self.lhs} = {self.rhs}")

    def get_valid_action_mask(self):
        return self.action_mask

    def set_equation(self,main_eqn):
        self.main_eqn, self.lhs, self.rhs = main_eqn, main_eqn, 0
        obs, _ = self.to_vec(self.lhs, self.rhs)
        self.state = obs

