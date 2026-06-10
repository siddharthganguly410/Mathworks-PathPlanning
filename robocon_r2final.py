"""
ABU Robocon 2026 – "Kung Fu Quest"
Robot 2 (R2) – Meihua Forest Q-Learning Solver
================================================
Grid layout (1-indexed):
    1   2   3
    4   5   6
    7   8   9
   10  11  12

R2 enters from {1,2,3}, exits through {10,11,12}.
"""

import random
import pickle
import time
import math
import json
from collections import defaultdict, deque
import numpy as np
import matplotlib
# Use interactive backend when a display is available (VS Code, desktop),
# fall back to Agg (file-only) in headless / CI environments.
import os as _os
if _os.environ.get("DISPLAY") or _os.name == "nt" or _os.environ.get("TERM_PROGRAM"):
    try:
        matplotlib.use("TkAgg")
    except Exception:
        matplotlib.use("Agg")
else:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec


# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
ROWS, COLS = 4, 3
NUM_BLOCKS = 12

# Block numbering: block n → 0-based index (n-1)
# Row/col from block id (1-indexed)
def block_to_rc(b):      return ((b - 1) // COLS, (b - 1) % COLS)
def rc_to_block(r, c):   return r * COLS + c + 1

ADJACENCY = {}
for b in range(1, NUM_BLOCKS + 1):
    r, c = block_to_rc(b)
    neighbors = []
    if r > 0:           neighbors.append(rc_to_block(r - 1, c))  # UP
    if r < ROWS - 1:    neighbors.append(rc_to_block(r + 1, c))  # DOWN
    if c > 0:           neighbors.append(rc_to_block(r, c - 1))  # LEFT
    if c < COLS - 1:    neighbors.append(rc_to_block(r, c + 1))  # RIGHT
    ADJACENCY[b] = neighbors

ENTRY_BLOCKS = {1, 2, 3}
EXIT_BLOCKS  = {10, 11, 12}

# Actions
UP      = 0
DOWN    = 1
LEFT    = 2
RIGHT   = 3
COLLECT = 4
EXIT    = 5
ACTION_NAMES = ["UP", "DOWN", "LEFT", "RIGHT", "COLLECT", "EXIT"]

# Rewards
R_COLLECT_REAL    = +100
R_COLLECT_ALL     = +500
R_EXIT_SUCCESS    = +300
R_MOVE            = -1
R_EXTRA_MOVE      = -2
R_FAKE_PROXIMITY  = -500
R_COLLECT_FAKE    = -1000
R_COLLECT_R1      = -1000
R_INVALID         = -500
R_WANDER          = -5


# ─────────────────────────────────────────────
#  ENVIRONMENT
# ─────────────────────────────────────────────
class ForestEnvironment:
    """
    Models the 4×3 Meihua Forest for R2.

    State tuple:
        (current_block,
         collected_real_scrolls,   ← frozenset → sorted tuple for hashing
         remaining_real_scrolls,   ← frozenset → sorted tuple
         fake_scroll_location,
         r1_scroll_locations)      ← frozenset → sorted tuple
    """

    def __init__(self, r2_real_scrolls, fake_scroll, r1_scrolls):
        assert len(r2_real_scrolls) == 4, "Must have exactly 4 real R2 scrolls"
        assert fake_scroll not in r2_real_scrolls, "Fake scroll overlaps real scroll"

        self.r2_real_scrolls  = frozenset(r2_real_scrolls)
        self.fake_scroll      = fake_scroll
        self.r1_scrolls       = frozenset(r1_scrolls)

        # Validate all blocks
        all_items = list(r2_real_scrolls) + [fake_scroll] + list(r1_scrolls)
        for b in all_items:
            assert 1 <= b <= NUM_BLOCKS, f"Block {b} out of range"

        self.reset()

    # ── reset ──────────────────────────────────
    def reset(self):
        """Start a new episode; pick entry block with a real scroll first."""
        entry_with_scroll = ENTRY_BLOCKS & self.r2_real_scrolls
        self.current_block     = random.choice(
            list(entry_with_scroll) if entry_with_scroll else list(ENTRY_BLOCKS)
        )
        self.collected         = frozenset()
        self.remaining         = frozenset(self.r2_real_scrolls)
        self.done              = False
        self.steps             = 0
        self.visit_count       = defaultdict(int)
        self.visit_count[self.current_block] += 1
        self.path              = [self.current_block]
        self.total_reward      = 0.0
        return self._get_state()

    # ── state ──────────────────────────────────
    def _get_state(self):
        return (
            self.current_block,
            tuple(sorted(self.collected)),
            tuple(sorted(self.remaining)),
            self.fake_scroll,
            tuple(sorted(self.r1_scrolls)),
        )

    # ── legal actions ──────────────────────────
    def legal_actions(self):
        acts = []
        r, c = block_to_rc(self.current_block)

        if r > 0:           acts.append(UP)
        if r < ROWS - 1:    acts.append(DOWN)
        if c > 0:           acts.append(LEFT)
        if c < COLS - 1:    acts.append(RIGHT)

        # COLLECT is legal if current block or neighbour has a real scroll
        nearby = {self.current_block} | set(ADJACENCY[self.current_block])
        if nearby & self.remaining:
            acts.append(COLLECT)

        # EXIT legal only when all scrolls collected and in exit zone
        if not self.remaining and self.current_block in EXIT_BLOCKS:
            acts.append(EXIT)

        return acts

    # ── step ───────────────────────────────────
    def step(self, action):
        if self.done:
            raise RuntimeError("Episode finished – call reset()")

        reward = 0.0
        info   = {}
        r, c   = block_to_rc(self.current_block)

        # ── movement actions ──
        if action in (UP, DOWN, LEFT, RIGHT):
            dr = {UP: -1, DOWN: 1, LEFT: 0, RIGHT: 0}[action]
            dc = {UP: 0,  DOWN: 0, LEFT: -1, RIGHT: 1}[action]
            nr, nc = r + dr, c + dc

            if 0 <= nr < ROWS and 0 <= nc < COLS:
                new_block = rc_to_block(nr, nc)

                # Penalty for stepping toward fake scroll
                if new_block == self.fake_scroll:
                    reward += R_FAKE_PROXIMITY

                self.current_block = new_block
                self.steps += 1
                self.visit_count[new_block] += 1
                self.path.append(new_block)

                # Wandering penalty (re-visited block with no reason)
                if self.visit_count[new_block] > 2 and new_block not in self.remaining:
                    reward += R_WANDER

                reward += R_MOVE
                # Extra cost if we have collected nothing yet and moving away
                if not self.collected:
                    reward += R_EXTRA_MOVE * 0.5

                info["moved_to"] = new_block
            else:
                reward += R_INVALID
                info["error"] = "wall"

        # ── collect ──
        elif action == COLLECT:
            nearby = {self.current_block} | set(ADJACENCY[self.current_block])

            # Check for fake scroll first (heavy penalty)
            if self.fake_scroll in nearby and self.fake_scroll == self.current_block:
                reward += R_COLLECT_FAKE
                self.done = True
                info["error"] = "collected_fake"

            # Check for R1 scroll (heavy penalty)
            elif self.r1_scrolls & nearby and not (self.remaining & nearby):
                reward += R_COLLECT_R1
                info["error"] = "attempted_r1"

            # Collect real scroll
            elif self.remaining & nearby:
                # Prefer collecting from current block, else nearest adjacent
                collectible = self.remaining & nearby
                # Pick the one closest to current block
                if self.current_block in collectible:
                    target = self.current_block
                else:
                    target = min(
                        collectible,
                        key=lambda b: 0 if b == self.current_block else 1
                    )

                self.collected = self.collected | {target}
                self.remaining = self.remaining - {target}
                reward += R_COLLECT_REAL
                info["collected"] = target

                if not self.remaining:
                    reward += R_COLLECT_ALL
                    info["all_collected"] = True

                self.steps += 1
                self.path.append(f"C{target}")

            else:
                reward += R_INVALID
                info["error"] = "nothing_to_collect"

        # ── exit ──
        elif action == EXIT:
            if not self.remaining and self.current_block in EXIT_BLOCKS:
                reward += R_EXIT_SUCCESS
                self.done = True
                info["success"] = True
                self.path.append("EXIT")
            else:
                reward += R_INVALID
                info["error"] = "invalid_exit"

        else:
            reward += R_INVALID
            info["error"] = "unknown_action"

        self.total_reward += reward
        return self._get_state(), reward, self.done, info

    # ── utility ────────────────────────────────
    def render(self, mode="text"):
        grid = {}
        for b in range(1, NUM_BLOCKS + 1):
            symbols = []
            if b == self.current_block:      symbols.append("R2")
            if b in self.remaining:          symbols.append("📜")
            if b in self.collected:          symbols.append("✓")
            if b == self.fake_scroll:        symbols.append("⚠")
            if b in self.r1_scrolls:         symbols.append("R1")
            grid[b] = ",".join(symbols) if symbols else "·"

        lines = ["\n  Meihua Forest"]
        for row in range(ROWS):
            row_str = "  "
            for col in range(COLS):
                b = rc_to_block(row, col)
                cell = f"[{b:2d}:{grid[b]:<5}]"
                row_str += cell
            lines.append(row_str)
        lines.append(f"  Steps: {self.steps}  Collected: {len(self.collected)}/4"
                      f"  Reward: {self.total_reward:.1f}")
        return "\n".join(lines)


# ─────────────────────────────────────────────
#  Q-LEARNING AGENT
# ─────────────────────────────────────────────
class QLearningAgent:
    """
    Tabular Q-Learning with:
      • ε-greedy exploration with decay
      • Optimistic initialisation
      • Per-state learning-rate decay
      • Optional prioritised-replay buffer
    """

    def __init__(
        self,
        alpha=0.15,
        gamma=0.97,
        epsilon_start=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.9995,
        optimistic_init=5.0,
    ):
        self.alpha          = alpha
        self.gamma          = gamma
        self.epsilon        = epsilon_start
        self.epsilon_min    = epsilon_min
        self.epsilon_decay  = epsilon_decay
        self.optimistic_init = optimistic_init

        self.q_table        = defaultdict(lambda: defaultdict(lambda: optimistic_init))
        self.visit_counts   = defaultdict(lambda: defaultdict(int))
        self.episode_rewards = []
        self.episode_steps   = []
        self.episode_success = []

    # ── action selection ───────────────────────
    def select_action(self, state, legal_actions, greedy=False):
        if not greedy and random.random() < self.epsilon:
            return random.choice(legal_actions)
        q_vals = {a: self.q_table[state][a] for a in legal_actions}
        max_q  = max(q_vals.values())
        best   = [a for a, q in q_vals.items() if q == max_q]
        return random.choice(best)

    # ── update ─────────────────────────────────
    def update(self, state, action, reward, next_state, next_legal, done):
        self.visit_counts[state][action] += 1
        n = self.visit_counts[state][action]
        # Harmonic learning-rate schedule: α / (1 + n/200)
        alpha_n = self.alpha / (1.0 + n / 200.0)

        if done:
            target = reward
        else:
            next_q  = max((self.q_table[next_state][a] for a in next_legal),
                          default=0.0)
            target  = reward + self.gamma * next_q

        old_q = self.q_table[state][action]
        self.q_table[state][action] = old_q + alpha_n * (target - old_q)

    # ── epsilon decay ──────────────────────────
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ── save / load ────────────────────────────
    def save(self, path="q_table.pkl"):
        # Convert nested defaultdicts to plain dicts for pickling
        plain_q = {s: dict(a_vals) for s, a_vals in self.q_table.items()}
        data = {
            "q_table":    plain_q,
            "epsilon":    self.epsilon,
            "rewards":    self.episode_rewards,
            "steps":      self.episode_steps,
            "success":    self.episode_success,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"[Agent] Q-table saved → {path}")

    def load(self, path="q_table.pkl"):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.q_table          = defaultdict(lambda: defaultdict(lambda: self.optimistic_init),
                                            {k: defaultdict(lambda: self.optimistic_init, v)
                                             for k, v in data["q_table"].items()})
        self.epsilon          = data["epsilon"]
        self.episode_rewards  = data["rewards"]
        self.episode_steps    = data["steps"]
        self.episode_success  = data["success"]
        print(f"[Agent] Q-table loaded ← {path}")


# ─────────────────────────────────────────────
#  TRAINING
# ─────────────────────────────────────────────
def train(
    env,
    agent,
    episodes=30_000,
    max_steps=200,
    verbose_every=2000,
    patience=3000,
):
    """
    Train the agent. Returns training history dict.
    Early-stopping once success-rate over last `patience` episodes ≥ 98%.
    """
    print(f"\n{'═'*56}")
    print(f"  ABU Robocon 2026 – R2 Q-Learning Training")
    print(f"  Episodes: {episodes:,}   Max steps/ep: {max_steps}")
    print(f"{'═'*56}\n")

    best_reward   = -float("inf")
    best_path     = None
    success_window = deque(maxlen=patience)

    for ep in range(1, episodes + 1):
        state   = env.reset()
        ep_rew  = 0.0
        success = False

        for _ in range(max_steps):
            legal   = env.legal_actions()
            if not legal:
                break
            action              = agent.select_action(state, legal)
            next_state, reward, done, info = env.step(action)
            next_legal          = env.legal_actions()
            agent.update(state, action, reward, next_state, next_legal, done)
            state   = next_state
            ep_rew += reward
            if done:
                success = info.get("success", False)
                break

        agent.decay_epsilon()
        agent.episode_rewards.append(ep_rew)
        agent.episode_steps.append(env.steps)
        agent.episode_success.append(int(success))
        success_window.append(int(success))

        if ep_rew > best_reward and success:
            best_reward = ep_rew
            best_path   = list(env.path)

        if ep % verbose_every == 0:
            recent_n   = min(verbose_every, ep)
            avg_rew    = np.mean(agent.episode_rewards[-recent_n:])
            avg_steps  = np.mean(agent.episode_steps[-recent_n:])
            succ_rate  = np.mean(agent.episode_success[-recent_n:]) * 100
            print(
                f"  Ep {ep:6d}/{episodes}  "
                f"ε={agent.epsilon:.4f}  "
                f"AvgRew={avg_rew:8.1f}  "
                f"AvgSteps={avg_steps:5.1f}  "
                f"SuccessRate={succ_rate:5.1f}%"
            )

        # Early stopping
        if len(success_window) == patience and sum(success_window) / patience >= 0.98:
            print(f"\n  ✔ Early-stop at episode {ep}: "
                  f"success rate ≥ 98% over last {patience} episodes")
            break

    print(f"\n  Best episode reward : {best_reward:.1f}")
    print(f"  Best path           : {best_path}")
    print(f"  Q-table states      : {len(agent.q_table):,}\n")

    return {
        "rewards":    agent.episode_rewards,
        "steps":      agent.episode_steps,
        "success":    agent.episode_success,
        "best_path":  best_path,
        "best_reward": best_reward,
    }


# ─────────────────────────────────────────────
#  EVALUATION – GREEDY ROLLOUT
# ─────────────────────────────────────────────
def evaluate(env, agent, episodes=100, max_steps=100, verbose=True):
    """Run greedy (no exploration) rollouts; return stats."""
    results = []
    for _ in range(episodes):
        state   = env.reset()
        success = False
        for _ in range(max_steps):
            legal  = env.legal_actions()
            if not legal:
                break
            action             = agent.select_action(state, legal, greedy=True)
            state, _, done, info = env.step(action)
            if done:
                success = info.get("success", False)
                break
        results.append({
            "success": success,
            "steps":   env.steps,
            "reward":  env.total_reward,
            "path":    list(env.path),
        })

    successes  = [r for r in results if r["success"]]
    succ_rate  = len(successes) / episodes * 100
    avg_steps  = np.mean([r["steps"] for r in successes]) if successes else float("nan")
    avg_reward = np.mean([r["reward"] for r in successes]) if successes else float("nan")

    if verbose:
        print(f"\n{'─'*50}")
        print(f"  Evaluation over {episodes} greedy episodes:")
        print(f"    Success rate  : {succ_rate:.1f}%")
        print(f"    Avg steps     : {avg_steps:.1f}")
        print(f"    Avg reward    : {avg_reward:.1f}")
        if successes:
            best = min(successes, key=lambda r: r["steps"])
            print(f"    Best path     : {best['path']}")
            print(f"    Best steps    : {best['steps']}")
        print(f"{'─'*50}\n")

    return results


# ─────────────────────────────────────────────
#  PLOTTING
# ─────────────────────────────────────────────
def smooth(data, window=200):
    if len(data) < window:
        return data
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode="valid")


def plot_training(history, eval_results, env_cfg, save_path="training_results.png"):
    rewards  = history["rewards"]
    steps    = history["steps"]
    success  = history["success"]

    fig = plt.figure(figsize=(18, 12), facecolor="#0f1117")
    fig.suptitle(
        "ABU Robocon 2026 – R2 Q-Learning  |  Meihua Forest",
        fontsize=16, fontweight="bold", color="white", y=0.98
    )
    gs  = GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    ACCENT   = "#00e5ff"
    SUCCESS  = "#69ff47"
    WARN     = "#ffaa00"
    DANGER   = "#ff4d6d"
    BG       = "#1a1d2e"
    TEXT     = "#e0e0e0"
    GRID_CLR = "#2a2d3e"

    def style_ax(ax, title):
        ax.set_facecolor(BG)
        ax.set_title(title, color=TEXT, fontsize=10, pad=6)
        ax.tick_params(colors=TEXT, labelsize=8)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_CLR)
        ax.grid(color=GRID_CLR, linestyle="--", linewidth=0.5, alpha=0.7)

    # ── 1. Reward curve ──
    ax1 = fig.add_subplot(gs[0, :2])
    ep  = np.arange(1, len(rewards) + 1)
    ax1.plot(ep, rewards, color=ACCENT, alpha=0.15, linewidth=0.4)
    sm  = smooth(rewards, 300)
    ax1.plot(np.arange(len(sm)) + 150, sm, color=ACCENT, linewidth=1.8,
             label="Smoothed reward")
    ax1.axhline(0, color=WARN, linestyle="--", linewidth=0.7, alpha=0.5)
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total Reward")
    ax1.legend(fontsize=8, facecolor=BG, labelcolor=TEXT)
    style_ax(ax1, "Episode Reward")

    # ── 2. Steps curve ──
    ax2 = fig.add_subplot(gs[1, :2])
    ax2.plot(ep, steps, color=WARN, alpha=0.15, linewidth=0.4)
    sm2 = smooth(steps, 300)
    ax2.plot(np.arange(len(sm2)) + 150, sm2, color=WARN, linewidth=1.8)
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Steps")
    style_ax(ax2, "Steps per Episode")

    # ── 3. Success rate ──
    ax3 = fig.add_subplot(gs[2, :2])
    window = 500
    sr = [np.mean(success[max(0, i - window):i + 1]) * 100
          for i in range(len(success))]
    ax3.plot(ep, sr, color=SUCCESS, linewidth=1.2)
    ax3.fill_between(ep, sr, alpha=0.15, color=SUCCESS)
    ax3.set_ylim(0, 105)
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Success Rate (%)")
    style_ax(ax3, f"Rolling Success Rate (window={window})")

    # ── 4. Forest map ──
    ax4 = fig.add_subplot(gs[0, 2])
    ax4.set_facecolor(BG)
    ax4.set_xlim(-0.1, 3.1)
    ax4.set_ylim(-0.1, 4.1)
    ax4.set_aspect("equal")
    ax4.axis("off")
    ax4.set_title("Forest Layout", color=TEXT, fontsize=10, pad=6)

    SCROLL_C = "#ffd700"
    FAKE_C   = "#ff4d6d"
    R1_C     = "#b48ead"
    EMPTY_C  = "#2a2d3e"

    for b in range(1, NUM_BLOCKS + 1):
        row, col = block_to_rc(b)
        y = ROWS - 1 - row  # flip y so row0 is top
        if b in env_cfg["r2_real"]:    color = SCROLL_C
        elif b == env_cfg["fake"]:     color = FAKE_C
        elif b in env_cfg["r1"]:       color = R1_C
        else:                          color = EMPTY_C

        rect = mpatches.FancyBboxPatch(
            (col + 0.05, y + 0.05), 0.9, 0.9,
            boxstyle="round,pad=0.05",
            facecolor=color, edgecolor="#555", linewidth=1
        )
        ax4.add_patch(rect)
        ax4.text(col + 0.5, y + 0.55, str(b),
                 ha="center", va="center", fontsize=9,
                 fontweight="bold", color="white")
        # symbol
        sym = ""
        if b in env_cfg["r2_real"]:  sym = "📜"
        elif b == env_cfg["fake"]:   sym = "⚠"
        elif b in env_cfg["r1"]:     sym = "R1"
        if sym:
            ax4.text(col + 0.5, y + 0.2, sym,
                     ha="center", va="center", fontsize=7, color="white")

    legend_items = [
        mpatches.Patch(color=SCROLL_C, label="R2 Scroll"),
        mpatches.Patch(color=FAKE_C,   label="Fake Scroll"),
        mpatches.Patch(color=R1_C,     label="R1 Scroll"),
        mpatches.Patch(color=EMPTY_C,  label="Empty"),
    ]
    ax4.legend(handles=legend_items, loc="upper right",
               fontsize=7, facecolor="#0f1117", labelcolor=TEXT,
               framealpha=0.8)

    # ── 5. Best path overlay ──
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.set_facecolor(BG)
    ax5.set_xlim(-0.1, 3.1)
    ax5.set_ylim(-0.1, 4.1)
    ax5.set_aspect("equal")
    ax5.axis("off")
    ax5.set_title("Best Greedy Path", color=TEXT, fontsize=10, pad=6)

    successes = [r for r in eval_results if r["success"]]
    best_path_blocks = []
    if successes:
        best_run  = min(successes, key=lambda r: r["steps"])
        raw_path  = best_run["path"]
        best_path_blocks = [p for p in raw_path if isinstance(p, int)]

    for b in range(1, NUM_BLOCKS + 1):
        row, col = block_to_rc(b)
        y = ROWS - 1 - row
        color = EMPTY_C
        if b in env_cfg["r2_real"]:  color = SCROLL_C
        elif b == env_cfg["fake"]:   color = FAKE_C
        elif b in env_cfg["r1"]:     color = R1_C
        rect = mpatches.FancyBboxPatch(
            (col + 0.05, y + 0.05), 0.9, 0.9,
            boxstyle="round,pad=0.05",
            facecolor=color, edgecolor="#555", linewidth=1,
            alpha=0.5
        )
        ax5.add_patch(rect)
        ax5.text(col + 0.5, y + 0.5, str(b),
                 ha="center", va="center", fontsize=9,
                 fontweight="bold", color="white")

    # Draw path arrows
    for i in range(len(best_path_blocks) - 1):
        b1, b2 = best_path_blocks[i], best_path_blocks[i + 1]
        r1c, c1c = block_to_rc(b1)
        r2c, c2c = block_to_rc(b2)
        y1, x1 = ROWS - 1 - r1c, c1c
        y2, x2 = ROWS - 1 - r2c, c2c
        ax5.annotate(
            "", xy=(x2 + 0.5, y2 + 0.5), xytext=(x1 + 0.5, y1 + 0.5),
            arrowprops=dict(arrowstyle="-|>", color=ACCENT,
                            lw=1.2, mutation_scale=10)
        )

    # ── 6. Eval stats ──
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.set_facecolor(BG)
    ax6.axis("off")
    ax6.set_title("Evaluation Summary", color=TEXT, fontsize=10, pad=6)

    succ_rate  = len(successes) / len(eval_results) * 100 if eval_results else 0
    avg_steps  = np.mean([r["steps"] for r in successes]) if successes else 0
    min_steps  = min([r["steps"] for r in successes], default=0)
    avg_rew    = np.mean([r["reward"] for r in successes]) if successes else 0

    stats = [
        ("Success Rate",   f"{succ_rate:.1f}%",  SUCCESS),
        ("Avg Steps",      f"{avg_steps:.1f}",   ACCENT),
        ("Min Steps",      f"{min_steps}",        SUCCESS),
        ("Avg Reward",     f"{avg_rew:.1f}",      WARN),
        ("Episodes",       f"{len(eval_results)}", TEXT),
        ("Q-States",       f"{len(agent.q_table):,}", TEXT),
    ]

    for i, (label, val, col) in enumerate(stats):
        y_pos = 0.92 - i * 0.15
        ax6.text(0.05, y_pos, label, transform=ax6.transAxes,
                 fontsize=9, color=TEXT)
        ax6.text(0.65, y_pos, val, transform=ax6.transAxes,
                 fontsize=10, fontweight="bold", color=col)

    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"  Plot saved → {save_path}")
    return fig


# ─────────────────────────────────────────────
#  PATH PRINTER
# ─────────────────────────────────────────────
def print_optimal_path(eval_results, env_config):
    successes = [r for r in eval_results if r["success"]]
    if not successes:
        print("  ✗ No successful paths found in evaluation.\n")
        return

    best = min(successes, key=lambda r: r["steps"])
    print(f"\n{'═'*56}")
    print(f"  OPTIMAL PATH  (steps={best['steps']}, reward={best['reward']:.1f})")
    print(f"{'═'*56}")
    print(f"  R2 Scrolls : {sorted(env_config['r2_real'])}")
    print(f"  Fake Scroll: {env_config['fake']}")
    print(f"  R1 Scrolls : {sorted(env_config['r1'])}")
    print(f"  Entry      : block {best['path'][0]}")

    prev_block = None
    for i, step in enumerate(best["path"]):
        if isinstance(step, int):
            direction = ""
            if prev_block and isinstance(prev_block, int):
                pr, pc = block_to_rc(prev_block)
                cr, cc = block_to_rc(step)
                dr, dc = cr - pr, cc - pc
                direction = {(-1, 0): "↑ UP",
                             (1, 0):  "↓ DOWN",
                             (0, -1): "← LEFT",
                             (0, 1):  "→ RIGHT"}.get((dr, dc), "")
            tag = ""
            if step in env_config["r2_real"]:   tag = " [R2 Scroll]"
            elif step == env_config["fake"]:     tag = " [⚠ FAKE]"
            elif step in env_config["r1"]:       tag = " [R1]"
            print(f"  {i+1:3d}. Block {step:2d}  {direction:<10}{tag}")
            prev_block = step
        elif isinstance(step, str) and step.startswith("C"):
            collected_block = int(step[1:])
            print(f"       → COLLECT scroll from block {collected_block} ✓")
        elif step == "EXIT":
            print(f"       → EXIT ✓")
    print(f"{'═'*56}\n")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import os
    # All output files land next to this script, wherever you run it from
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    random.seed(42)
    np.random.seed(42)

    # ── Configuration ──────────────────────────
    r2_real_scrolls = [2, 7, 5, 11]
    fake_scroll     = 8
    r1_scrolls      = [12, 6, 10]

    env_config = {
        "r2_real": set(r2_real_scrolls),
        "fake":    fake_scroll,
        "r1":      set(r1_scrolls),
    }

    print(f"\n  Forest Config:")
    print(f"    R2 Real Scrolls : {r2_real_scrolls}")
    print(f"    Fake Scroll     : {fake_scroll}")
    print(f"    R1 Scrolls      : {r1_scrolls}")

    # ── Create env & agent ──────────────────────
    env   = ForestEnvironment(r2_real_scrolls, fake_scroll, r1_scrolls)
    agent = QLearningAgent(
        alpha=0.15,
        gamma=0.97,
        epsilon_start=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.9997,
        optimistic_init=10.0,
    )

    # ── Train ───────────────────────────────────
    t0      = time.time()
    history = train(env, agent, episodes=10_000, max_steps=200,
                    verbose_every=500, patience=3000)
    elapsed = time.time() - t0
    print(f"  Training time: {elapsed:.1f}s")

    # ── Evaluate ────────────────────────────────
    eval_results = evaluate(env, agent, episodes=200, max_steps=100)

    # ── Print best path ─────────────────────────
    print_optimal_path(eval_results, env_config)

    # ── Show final forest render ─────────────────
    env.reset()
    for _ in range(50):
        legal  = env.legal_actions()
        if not legal: break
        action = agent.select_action(env._get_state(), legal, greedy=True)
        _, _, done, info = env.step(action)
        if done: break
    print(env.render())

    # ── Save agent & plot ───────────────────────
    pkl_path     = os.path.join(SCRIPT_DIR, "r2_q_table.pkl")
    plot_path    = os.path.join(SCRIPT_DIR, "r2_training_results.png")
    summary_path = os.path.join(SCRIPT_DIR, "r2_summary.json")

    agent.save(pkl_path)

    fig = plot_training(history, eval_results, env_config, save_path=plot_path)
    plt.show()          # pops up the window in VS Code / desktop; no-op in headless
    plt.close(fig)

    # ── Save summary JSON ───────────────────────
    successes   = [r for r in eval_results if r["success"]]
    best_run    = min(successes, key=lambda r: r["steps"]) if successes else {}
    summary = {
        "config": {
            "r2_real_scrolls": r2_real_scrolls,
            "fake_scroll":     fake_scroll,
            "r1_scrolls":      r1_scrolls,
        },
        "training_episodes": len(agent.episode_rewards),
        "training_time_s":   round(elapsed, 2),
        "q_table_states":    len(agent.q_table),
        "eval_success_rate": round(len(successes) / len(eval_results) * 100, 1),
        "eval_avg_steps":    round(float(np.mean([r["steps"] for r in successes])), 1) if successes else None,
        "eval_min_steps":    min([r["steps"] for r in successes], default=None),
        "best_path":         best_run.get("path", []),
        "best_reward":       best_run.get("reward", None),
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary saved → {summary_path}")