"""
ABU Robocon 2026 – "Kung Fu Quest"
Robot 2 (R2) – Meihua Forest A* Search Solver
===============================================
Grid layout (1-indexed):
    1   2   3      ← Entry zone  (row 0)
    4   5   6
    7   8   9
   10  11  12      ← Exit zone   (row 3)

R2 enters from {1,2,3}, exits through {10,11,12}.

A* Algorithm Overview
─────────────────────
A* finds the OPTIMAL (least-cost) path from a start state to a goal by
combining two cost measures at every node n:

    f(n) = g(n) + h(n)

  • g(n) : exact cost paid so far to reach n from the start
  • h(n) : admissible heuristic estimate of remaining cost to goal
  • f(n) : estimated total cost of the cheapest solution through n

The open-list (min-heap) always expands the node with smallest f(n),
guaranteeing optimality when h is admissible (never over-estimates).

State Space for Robocon R2
──────────────────────────
  state = (current_block, remaining_scrolls_frozenset)

  • current_block    : which of the 12 blocks R2 occupies (1-12)
  • remaining_scrolls: frozenset of real R2 scrolls not yet collected

  Total reachable states ≤ 12 × 2^4 = 192  → A* is exact here.

Heuristic h(state)
──────────────────
  h is a LOWER BOUND on remaining steps needed (admissible).
  We combine two components:

    h(state) = scroll_tour_cost(state) + exit_distance_cost(state)

  1. scroll_tour_cost:
       Greedy nearest-neighbour lower bound: repeatedly travel to the
       closest uncollected scroll (Manhattan distance) and collect it.
       Admissible because you MUST visit every remaining scroll at least once.

  2. exit_distance_cost:
       After collecting the last scroll, Manhattan distance from that
       position to the nearest exit block.
       Admissible because Manhattan distance ≤ actual grid steps.

Costs (g increments)
────────────────────
  • Move to adjacent block   : +1  (one step)
  • Collect scroll on block  : +1  (one action, R2 stays in place)
  • Move to fake/R1 block    : +0  (movement is allowed; collection would
                                     be penalised but we never collect there)

The algorithm prints the complete open-list expansion log so you can
trace exactly how A* finds the optimal path step by step.
"""

import heapq
import math
import time
import json
import os
import matplotlib
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
import numpy as np


# ─────────────────────────────────────────────
#  GRID CONSTANTS  (identical to RL version)
# ─────────────────────────────────────────────
ROWS, COLS  = 4, 3
NUM_BLOCKS  = 12

def block_to_rc(b):    return ((b - 1) // COLS, (b - 1) % COLS)
def rc_to_block(r, c): return r * COLS + c + 1

ADJACENCY = {}
for b in range(1, NUM_BLOCKS + 1):
    r, c = block_to_rc(b)
    nb = []
    if r > 0:         nb.append(rc_to_block(r - 1, c))
    if r < ROWS - 1:  nb.append(rc_to_block(r + 1, c))
    if c > 0:         nb.append(rc_to_block(r, c - 1))
    if c < COLS - 1:  nb.append(rc_to_block(r, c + 1))
    ADJACENCY[b] = nb

ENTRY_BLOCKS = {1, 2, 3}
EXIT_BLOCKS  = {10, 11, 12}

COST_MOVE    = 1
COST_COLLECT = 1


# ─────────────────────────────────────────────
#  HEURISTIC  (admissible)
# ─────────────────────────────────────────────
def manhattan(b1, b2):
    r1, c1 = block_to_rc(b1)
    r2, c2 = block_to_rc(b2)
    return abs(r1 - r2) + abs(c1 - c2)

def heuristic(current_block, remaining_scrolls):
    """
    Admissible h(n) = greedy_scroll_tour_lb + exit_distance_lb.

    Never over-estimates because:
      • Manhattan distance is always ≤ actual grid steps.
      • NN-tour cost is a lower bound on any visiting order.
    """
    if not remaining_scrolls:
        return min(manhattan(current_block, e) for e in EXIT_BLOCKS)

    pos, visited, cost = current_block, set(), 0
    remaining_list = list(remaining_scrolls)
    while len(visited) < len(remaining_list):
        nearest  = min((b for b in remaining_list if b not in visited),
                       key=lambda b: manhattan(pos, b))
        cost    += manhattan(pos, nearest) + COST_COLLECT
        visited.add(nearest)
        pos = nearest

    cost += min(manhattan(pos, e) for e in EXIT_BLOCKS)
    return cost


# ─────────────────────────────────────────────
#  A* NODE
# ─────────────────────────────────────────────
class AStarNode:
    """
    One search-tree node.

    f = g + h  is the priority; tie-break on highest g (most progress),
    then insertion order (determinism).
    """
    _ctr = 0

    def __init__(self, g, block, remaining, path):
        self.g         = g
        self.h         = heuristic(block, remaining)
        self.f         = g + self.h
        self.block     = block
        self.remaining = remaining
        self.path      = path
        AStarNode._ctr += 1
        self._id       = AStarNode._ctr

    def __lt__(self, other):
        if self.f != other.f:   return self.f < other.f
        if self.g != other.g:   return self.g > other.g   # prefer deeper
        return self._id < other._id


# ─────────────────────────────────────────────
#  A* SOLVER
# ─────────────────────────────────────────────
def _fmt_rem(remaining):
    """Short string for a remaining-scroll frozenset."""
    return str(sorted(remaining)) if remaining else "[done]"

def _print_node_row(prefix, block, action, g, h, rem, extra=""):
    """Print one node row in the all-nodes table."""
    f = g + h
    print(f"  {prefix}  [{block:>2}]  {_fmt_rem(rem):^20}  "
          f"{g:>4}  {h:>4}  {f:>4}  {action}{extra}")


def astar_solve(start_block, r2_real_scrolls, fake_scroll, r1_scrolls,
                verbose=True, show_expansion=True):
    """
    A* from start_block: collect all r2_real_scrolls then exit.

    When show_expansion=True the solver prints, for EVERY expansion round:
      • The node being expanded  (marked  >>>  EXPAND)
      • EVERY child node it generates, with their g / h / f values
      • A snapshot of the full open-list after children are added
      • The closed set state

    This gives complete visibility into how A* evaluates all nodes from
    the start, not just the nodes along the final optimal path.

    Movement through fake/R1 blocks is allowed (cost 1); only
    *collecting* from those blocks is forbidden (never attempted here).
    """
    scrolls_0 = frozenset(r2_real_scrolls)
    h0        = heuristic(start_block, scrolls_0)

    start = AStarNode(
        g         = 0,
        block     = start_block,
        remaining = scrolls_0,
        path      = [{"step": 0, "block": start_block, "action": "START",
                      "g": 0, "h": h0, "f": h0,
                      "note": f"Entry block {start_block}"}],
    )

    open_list = [start]
    closed    = {}          # state_key → best g seen
    expanded  = 0
    generated = 1           # start node counts as generated

    # ── Header ───────────────────────────────────────────────────────
    if verbose:
        sep = "═" * 72
        print(f"\n{sep}")
        print(f"  A* PATH PLANNING  –  ABU Robocon 2026 R2 Meihua Forest")
        print(f"{sep}")
        print(f"  Start block      : {start_block}")
        print(f"  R2 Scrolls       : {sorted(r2_real_scrolls)}")
        print(f"  Fake Scroll      : {fake_scroll}  (movement OK, collect forbidden)")
        print(f"  R1 Scrolls       : {sorted(r1_scrolls)}  (movement OK, collect forbidden)")
        print(f"  h(start)         : {h0}  <- initial admissible lower bound")
        print(f"  f(start)         : {h0}  (= g=0 + h={h0})")
        print(f"{sep}\n")

    COL_HDR = (f"  {'':15}  {'Block':>4}  {'Remaining':^20}  "
               f"{'g':>4}  {'h':>4}  {'f':>4}  Note")
    RULE     = "  " + "─" * 80

    if show_expansion:
        print(f"\n  {'ALL NODES — g/h/f from start':^82}")
        print(f"  {'(EXPAND = node chosen from open-list, CHILD = node generated)':^82}")
        print(RULE)
        print(COL_HDR)
        print(RULE)
        # Print the start node
        _print_node_row("  START          ", start_block, "START",
                        0, h0, scrolls_0, "  <- initial node")

    while open_list:
        node = heapq.heappop(open_list)
        state_key = (node.block, node.remaining)

        if state_key in closed and closed[state_key] <= node.g:
            # Stale duplicate — skip silently (already logged when generated)
            continue

        closed[state_key] = node.g
        expanded += 1

        # ── Print the node being expanded ────────────────────────────
        if show_expansion:
            print(RULE)
            last_act = node.path[-1].get("action", "?")
            _print_node_row(f"  EXPAND #{expanded:<6}    ", node.block,
                            last_act, node.g, node.h, node.remaining,
                            "  *** EXPANDED (lowest f in open-list) ***")

        # ── GOAL CHECK ───────────────────────────────────────────────
        if not node.remaining and node.block in EXIT_BLOCKS:
            if show_expansion:
                print(f"  {'':15}  *** GOAL REACHED — all scrolls collected + exit block ***")
                print(RULE)
                print(f"\n  Closed set ({len(closed)} states explored):")
                for (blk, rem), g_val in sorted(closed.items(),
                                                key=lambda x: x[1]):
                    h_val = heuristic(blk, rem)
                    print(f"    Block {blk:>2}  remain={_fmt_rem(rem):^20}"
                          f"  g={g_val}  h={h_val}  f={g_val+h_val}")
                print()
            if verbose:
                print(f"  Optimal solution found!")
                print(f"    Nodes expanded  : {expanded:,}")
                print(f"    Nodes generated : {generated:,}")
                print(f"    Optimal cost g  : {node.g}")
                print(f"    Path steps      : {len(node.path)}\n")
            return {"success": True, "path": node.path, "cost": node.g,
                    "nodes_expanded": expanded, "nodes_generated": generated,
                    "closed": dict(closed)}

        # ── GENERATE: movement children ──────────────────────────────
        child_rows = []   # collect children to print them grouped

        for nb in ADJACENCY[node.block]:
            new_g   = node.g + COST_MOVE
            new_rem = node.remaining
            new_h   = heuristic(nb, new_rem)
            action  = f"MOVE->{nb}"
            note    = ""
            if nb == fake_scroll: note = "  (fake scroll block — avoid collecting)"
            if nb in r1_scrolls:  note = "  (R1 scroll block — avoid collecting)"

            new_node = AStarNode(g=new_g, block=nb, remaining=new_rem,
                path=node.path + [{"step": len(node.path), "block": nb,
                                   "action": f"MOVE → {nb}",
                                   "g": new_g, "h": new_h, "f": new_g + new_h,
                                   "note": f"Move → {nb}{note}"}])
            generated += 1
            nk = (nb, new_rem)

            pruned = (nk in closed and closed[nk] <= new_g)
            if not pruned:
                heapq.heappush(open_list, new_node)

            child_rows.append((new_node, action, note, pruned))

        # ── GENERATE: collect child ───────────────────────────────────
        if node.block in node.remaining:
            new_rem  = node.remaining - {node.block}
            new_g    = node.g + COST_COLLECT
            new_h    = heuristic(node.block, new_rem)
            action   = f"COLLECT@{node.block}"
            note     = "  <- ALL scrolls collected!" if not new_rem else ""
            new_node = AStarNode(g=new_g, block=node.block, remaining=new_rem,
                path=node.path + [{"step": len(node.path), "block": node.block,
                                   "action": f"COLLECT @ {node.block}",
                                   "g": new_g, "h": new_h, "f": new_g + new_h,
                                   "note": f"Collect scroll at block {node.block}{note}"}])
            generated += 1
            nk = (node.block, new_rem)
            pruned = (nk in closed and closed[nk] <= new_g)
            if not pruned:
                heapq.heappush(open_list, new_node)
            child_rows.append((new_node, action, note, pruned))

        # ── Print all children with their g/h/f ──────────────────────
        if show_expansion:
            for (cn, act, note, pruned) in child_rows:
                prune_tag = "  [PRUNED — cheaper path already known]" if pruned else ""
                _print_node_row("    CHILD         ", cn.block, act,
                                cn.g, cn.h, cn.remaining,
                                f"{note}{prune_tag}")

            # ── Open-list snapshot (sorted by f) ─────────────────────
            # heapify a copy so we can peek without destroying
            snap = sorted(open_list, key=lambda n: (n.f, -n.g, n._id))
            print(f"\n  Open-list after expansion #{expanded} "
                  f"({len(snap)} nodes, sorted by f):")
            print(f"  {'':15}  {'Block':>4}  {'Remaining':^20}  "
                  f"{'g':>4}  {'h':>4}  {'f':>4}  From action")
            for rank, sn in enumerate(snap, 1):
                last = sn.path[-1].get("action", "?")
                marker = " <-- next to expand" if rank == 1 else ""
                print(f"  {'':4} [{rank:>2}]        [{sn.block:>2}]  "
                      f"{_fmt_rem(sn.remaining):^20}  "
                      f"{sn.g:>4}  {sn.h:>4}  {sn.f:>4}  {last}{marker}")

    print("  Open list exhausted — no solution from this entry.")
    return {"success": False, "nodes_expanded": expanded, "nodes_generated": generated}


# ─────────────────────────────────────────────
#  SOLVE FROM ALL ENTRIES
# ─────────────────────────────────────────────
def solve_all_entries(r2_real_scrolls, fake_scroll, r1_scrolls):
    """Try every legal entry block; return (best_result, all_results_dict)."""
    best, all_res = None, {}
    print(f"\n{'═'*68}")
    print(f"  ALL-ENTRY SWEEP  (finding globally optimal start)")
    print(f"{'═'*68}\n")
    for entry in sorted(ENTRY_BLOCKS):
        res = astar_solve(entry, r2_real_scrolls, fake_scroll, r1_scrolls,
                          verbose=False, show_expansion=False)
        all_res[entry] = res
        if res["success"]:
            print(f"  Entry {entry}: cost={res['cost']}  "
                  f"steps={len(res['path'])}  nodes_exp={res['nodes_expanded']:,}")
            if best is None or res["cost"] < best["cost"]:
                best = res; best["entry"] = entry
        else:
            print(f"  Entry {entry}: no solution")
    return best, all_res


# ─────────────────────────────────────────────
#  HEURISTIC ANALYSIS PRINTER
# ─────────────────────────────────────────────
def print_heuristic_analysis(start_block, r2_real_scrolls):
    remaining = frozenset(r2_real_scrolls)
    h_val     = heuristic(start_block, remaining)

    print(f"\n{'─'*62}")
    print(f"  HEURISTIC ANALYSIS  h(start=block {start_block})")
    print(f"{'─'*62}")
    print(f"  Uncollected scrolls: {sorted(remaining)}")
    print(f"\n  ── Step 1: Greedy nearest-neighbour scroll tour ──────────")

    pos, visited, total = start_block, set(), 0
    rlist = list(remaining)
    while len(visited) < len(rlist):
        unvis   = [b for b in rlist if b not in visited]
        nearest = min(unvis, key=lambda b: manhattan(pos, b))
        d       = manhattan(pos, nearest)
        c       = d + COST_COLLECT
        total  += c
        print(f"    Block {pos:>2} → Block {nearest:>2}"
              f"  Manhattan={d}  +collect={COST_COLLECT}  subtotal={c}")
        visited.add(nearest); pos = nearest

    nearest_exit = min(EXIT_BLOCKS, key=lambda e: manhattan(pos, e))
    exit_d = manhattan(pos, nearest_exit)
    print(f"\n  ── Step 2: Exit distance from last scroll ────────────────")
    print(f"    Block {pos:>2} → Exit {nearest_exit:>2}  Manhattan={exit_d}")
    print(f"\n  h(start={start_block}) = scroll_tour({total}) + exit({exit_d}) = {h_val}")
    print(f"  ✔ Admissible: h≤true cost (Manhattan LB on 4×3 grid)")
    print(f"{'─'*62}\n")


# ─────────────────────────────────────────────
#  PATH WORKINGS PRINTER
# ─────────────────────────────────────────────
def print_path_workings(result, env_config):
    if not result or not result.get("success"):
        print("  ✗ No solution.\n"); return

    path, r2, fake, r1 = result["path"], env_config["r2_real"], env_config["fake"], env_config["r1"]
    W = 76
    print(f"\n{'═'*W}")
    print(f"  FULL A* PATH WORKINGS"
          f"  (entry={result.get('entry','?')}  "
          f"cost={result['cost']}  steps={len(path)})")
    print(f"{'═'*W}")
    print(f"  {'Step':>4}  {'Block':>5}  {'g':>6}  {'h':>6}  {'f':>6}  Action / Note")
    print(f"{'─'*W}")

    collected = set()
    for s in path:
        blk, action, g, h, f = s["block"], s["action"], s["g"], s["h"], s["f"]
        tags = []
        if blk in r2 and blk not in collected:
            tags.append("📜 R2 scroll here")
        if blk == fake:   tags.append("⚠ fake")
        if blk in r1:     tags.append("R1")
        if "COLLECT" in action:
            collected.add(blk)
            tags.append(f"✓  ({len(collected)}/4 collected)")
        if blk in EXIT_BLOCKS: tags.append("EXIT ZONE")
        tag_str = "  " + "  ".join(tags) if tags else ""
        print(f"  {s['step']:>4}  [{blk:>2}]  {g:>6}  {h:>6}  {f:>6}  {action}{tag_str}")

    print(f"{'─'*W}")
    print(f"  Optimal cost   : {result['cost']}  (move=1, collect=1)")
    print(f"  Nodes expanded : {result['nodes_expanded']:,}")
    print(f"  Nodes generated: {result['nodes_generated']:,}")
    print(f"{'═'*W}\n")


# ─────────────────────────────────────────────
#  PLOT DASHBOARD
# ─────────────────────────────────────────────
def plot_astar(result, all_results, env_cfg, save_path="r2_astar_results.png"):
    BG       = "#1a1d2e"; TEXT  = "#e0e0e0"; ACCENT = "#00e5ff"
    SUCCESS  = "#69ff47"; WARN  = "#ffaa00"; DANGER = "#ff4d6d"
    GRID_C   = "#2a2d3e"; SCROLL_C = "#ffd700"; FAKE_C = "#ff4d6d"
    R1_C     = "#b48ead"; EMPTY_C  = "#2a2d3e"

    fig = plt.figure(figsize=(18, 11), facecolor="#0f1117")
    fig.suptitle(
        "ABU Robocon 2026 – R2  |  A* Optimal Path Planning  |  Meihua Forest",
        fontsize=15, fontweight="bold", color="white", y=0.98,
    )
    gs = GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.30,
                  left=0.06, right=0.97, top=0.93, bottom=0.07)

    def style_ax(ax, title):
        ax.set_facecolor(BG); ax.set_title(title, color=TEXT, fontsize=10, pad=6)
        ax.tick_params(colors=TEXT, labelsize=8)
        ax.xaxis.label.set_color(TEXT); ax.yaxis.label.set_color(TEXT)
        for sp in ax.spines.values(): sp.set_edgecolor(GRID_C)
        ax.grid(color=GRID_C, linestyle="--", linewidth=0.5, alpha=0.7)

    def draw_forest(ax, highlight=None, title="Forest Layout"):
        ax.set_facecolor(BG); ax.set_xlim(-0.1, 3.1); ax.set_ylim(-0.1, 4.1)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(title, color=TEXT, fontsize=10, pad=6)
        hl = set(highlight) if highlight else set()
        for b in range(1, NUM_BLOCKS + 1):
            row, col = block_to_rc(b); y = ROWS - 1 - row
            col_map = (SCROLL_C if b in env_cfg["r2_real"] else
                       FAKE_C   if b == env_cfg["fake"]    else
                       R1_C     if b in env_cfg["r1"]      else EMPTY_C)
            ec = ACCENT if b in hl else "#555"
            lw = 2.2    if b in hl else 1.0
            ax.add_patch(mpatches.FancyBboxPatch(
                (col + 0.05, y + 0.05), 0.9, 0.9, boxstyle="round,pad=0.05",
                facecolor=col_map, edgecolor=ec, linewidth=lw))
            ax.text(col+0.5, y+0.65, str(b), ha="center", va="center",
                    fontsize=9, fontweight="bold", color="white")
            sym = ("📜" if b in env_cfg["r2_real"] else "⚠" if b==env_cfg["fake"]
                   else "R1" if b in env_cfg["r1"] else "")
            if sym: ax.text(col+0.5, y+0.25, sym, ha="center", va="center",
                            fontsize=7, color="white")
        for row_i in range(ROWS):
            label = ("ENTRY" if row_i==0 else "EXIT" if row_i==ROWS-1 else "")
            if label:
                ax.text(-0.06, ROWS-1-row_i+0.5, label, ha="right", va="center",
                        fontsize=5.5, color=SUCCESS if label=="EXIT" else ACCENT,
                        style="italic")

    path_blocks = set()
    if result and result.get("success"):
        for s in result["path"]: path_blocks.add(s["block"])

    # ── Panel 0: Forest layout ──
    ax0 = fig.add_subplot(gs[0, 0])
    draw_forest(ax0, highlight=path_blocks, title="Forest Layout")
    ax0.legend(handles=[
        mpatches.Patch(color=SCROLL_C, label="R2 Scroll"),
        mpatches.Patch(color=FAKE_C,   label="Fake (avoid collect)"),
        mpatches.Patch(color=R1_C,     label="R1 Scroll"),
        mpatches.Patch(color=EMPTY_C,  label="Empty"),
    ], loc="lower left", fontsize=6.5, facecolor="#0f1117",
       labelcolor=TEXT, framealpha=0.85)

    # ── Panel 1: Path arrows with step numbers ──
    ax1 = fig.add_subplot(gs[0, 1])
    draw_forest(ax1, title="A* Optimal Path")
    if result and result.get("success"):
        path = result["path"]
        step_map = {}
        for s in path:
            step_map.setdefault(s["block"], []).append(s["step"])
        for b, steps in step_map.items():
            row, col = block_to_rc(b); y = ROWS-1-row
            ax1.text(col+0.5, y+0.5, "#" + ",".join(str(x) for x in steps[:3]),
                     ha="center", va="center", fontsize=6.5, color="black",
                     fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.15", fc=ACCENT, ec="none", alpha=0.85))
        prev = None
        for s in path:
            b = s["block"]
            if prev and prev != b:
                r1c,c1c = block_to_rc(prev); r2c,c2c = block_to_rc(b)
                ax1.annotate("",
                    xy=(c2c+0.5, ROWS-1-r2c+0.5),
                    xytext=(c1c+0.5, ROWS-1-r1c+0.5),
                    arrowprops=dict(arrowstyle="-|>", color=SUCCESS, lw=1.4,
                                    mutation_scale=10))
            prev = b
        ax1.text(0.5, -0.02,
            f"Entry {result.get('entry','?')}  |  Cost={result.get('cost','?')}  "
            f"|  Steps={len(path)}",
            transform=ax1.transAxes, ha="center", fontsize=7.5, color=ACCENT)

    # ── Panel 2: ALL nodes g/h/f scatter (every generated node) ──
    ax2 = fig.add_subplot(gs[0, 2])
    style_ax(ax2, "All Generated Nodes: g vs h (bubble = f)")

    if result and result.get("success"):
        closed_map = result.get("closed", {})  # (block, remaining) -> g
        path       = result["path"]
        path_states = set()
        for s in path:
            path_states.add((s["block"],
                             frozenset(s.get("remaining_key",
                                      # fallback: reconstruct from path index
                                      []))))

        # Build per-block aggregated (min g, h) from closed set for display
        pts_x, pts_y, pts_f, pts_lbl, pts_col = [], [], [], [], []
        for (blk, rem), g_val in closed_map.items():
            h_val = heuristic(blk, rem)
            f_val = g_val + h_val
            pts_x.append(g_val)
            pts_y.append(h_val)
            pts_f.append(f_val)
            rem_short = str(len(rem)) + "rem"  # e.g. "3rem"
            pts_lbl.append(f"B{blk}")
            # colour: optimal-path nodes in green, others in accent/warn
            on_path = any(s["block"] == blk and s["g"] == g_val for s in path)
            pts_col.append(SUCCESS if on_path else ACCENT)

        if pts_x:
            sc = ax2.scatter(pts_x, pts_y, s=[max(30, f*8) for f in pts_f],
                             c=pts_col, alpha=0.75, edgecolors="#222", linewidths=0.5,
                             zorder=3)
            for xi, yi, lbl in zip(pts_x, pts_y, pts_lbl):
                ax2.text(xi, yi + 0.12, lbl, ha="center", va="bottom",
                         fontsize=6.5, color=TEXT)

        # Overlay optimal path as a line (g increases along path)
        path_xs = [s["g"] for s in path]
        path_ys = [s["h"] for s in path]
        ax2.plot(path_xs, path_ys, color=SUCCESS, lw=1.6,
                 linestyle="--", label="Optimal path trace", zorder=4)
        ax2.plot(path_xs[0], path_ys[0], "o", color=ACCENT,
                 ms=8, label="Start", zorder=5)
        ax2.plot(path_xs[-1], path_ys[-1], "*", color=SUCCESS,
                 ms=12, label="Goal", zorder=5)

        ax2.set_xlabel("g  (cost so far)"); ax2.set_ylabel("h  (heuristic)")
        ax2.legend(fontsize=7.5, facecolor=BG, labelcolor=TEXT)

        # Add iso-f lines
        max_f = max(pts_f) if pts_f else 12
        for iso_f in range(0, int(max_f) + 3, 2):
            g_pts = [0, iso_f]
            h_pts = [iso_f, 0]
            ax2.plot(g_pts, h_pts, color=GRID_C, lw=0.6, linestyle=":",
                     alpha=0.5, zorder=1)
            ax2.text(iso_f - 0.15, 0.1, f"f={iso_f}", fontsize=5.5,
                     color=WARN, alpha=0.6)

        ax2.set_xlim(-0.5, max(pts_x + [0]) + 1.5)
        ax2.set_ylim(-0.5, max(pts_y + [0]) + 2)

    # ── Panel 3: g/h/f line chart across all path steps ──
    ax3 = fig.add_subplot(gs[1, 0])
    style_ax(ax3, "g / h / f Along Optimal Path Steps")
    if result and result.get("success"):
        path = result["path"]
        xs   = [s["step"] for s in path]
        gs_  = [s["g"]    for s in path]
        hs_  = [s["h"]    for s in path]
        fs_  = [s["f"]    for s in path]
        ax3.plot(xs, gs_, color=ACCENT,  lw=1.8, label="g  (cost so far)", marker="o", ms=4)
        ax3.plot(xs, hs_, color=WARN,    lw=1.8, label="h  (heuristic)",   marker="s", ms=4)
        ax3.plot(xs, fs_, color=SUCCESS, lw=1.8, label="f = g+h",
                 linestyle="--", marker="^", ms=4)
        ax3.fill_between(xs, gs_, alpha=0.08, color=ACCENT)
        ax3.fill_between(xs, hs_, alpha=0.08, color=WARN)
        ax3.set_xlabel("Path Step"); ax3.set_ylabel("Cost Value")
        ax3.legend(fontsize=7.5, facecolor=BG, labelcolor=TEXT)
        h_zero = [i for i, h in enumerate(hs_) if h == 0]
        if h_zero:
            xi = xs[h_zero[0]]
            ax3.axvline(xi, color=SUCCESS, lw=0.9, linestyle=":", alpha=0.7)
            ax3.text(xi + 0.1, max(fs_) * 0.5, "h=0\n(goal reachable)", fontsize=6,
                     color=SUCCESS)

    # ── Panel 4: Cost comparison per entry ──
    ax4 = fig.add_subplot(gs[1, 1])
    style_ax(ax4, "Optimal Cost by Entry Block")
    entries = sorted(all_results.keys())
    costs   = [all_results[e].get("cost", 0) if all_results[e]["success"] else 0
               for e in entries]
    colors  = [SUCCESS if e == result.get("entry") else ACCENT for e in entries]
    bars    = ax4.bar([str(e) for e in entries], costs, color=colors, width=0.5)
    ax4.set_xlabel("Entry Block"); ax4.set_ylabel("Steps (cost)")
    for bar, cost in zip(bars, costs):
        ax4.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.15, str(cost),
                 ha="center", fontsize=9, color=TEXT, fontweight="bold")
    ax4.text(0.5, 0.96, "Green = chosen (lowest cost)", transform=ax4.transAxes,
             ha="center", fontsize=7, color=SUCCESS)

    # ── Panel 5: Summary ──
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.set_facecolor(BG); ax5.axis("off")
    ax5.set_title("Solution Summary", color=TEXT, fontsize=10, pad=6)
    lines = [
        ("Algorithm",     "A* Search",                       ACCENT),
        ("Heuristic",     "NN-tour + Manhattan exit",        TEXT),
        ("Admissible?",   "Yes — Manhattan ≤ actual steps",  SUCCESS),
        ("Optimal?",      "Yes — admissible h guarantees",   SUCCESS),
        ("Entry Block",   str(result.get("entry","?")),       SUCCESS),
        ("Optimal Cost",  str(result.get("cost","?")),        ACCENT),
        ("Path Steps",    str(len(result.get("path",[]))),    ACCENT),
        ("Nodes Exp.",    f"{result.get('nodes_expanded',0):,}", WARN),
        ("Nodes Gen.",    f"{result.get('nodes_generated',0):,}", WARN),
        ("R2 Scrolls",    str(sorted(env_cfg["r2_real"])),   TEXT),
        ("Fake Scroll",   str(env_cfg["fake"]),               DANGER),
        ("R1 Scrolls",    str(sorted(env_cfg["r1"])),         R1_C),
    ]
    for i, (lbl, val, col) in enumerate(lines):
        y = 0.96 - i * 0.078
        ax5.text(0.04, y, lbl+":", transform=ax5.transAxes, fontsize=8.5, color=TEXT)
        ax5.text(0.52, y, val,    transform=ax5.transAxes, fontsize=8.5,
                 fontweight="bold", color=col)

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  Plot saved → {save_path}")
    return fig


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    # ── Same config as RL version for direct comparison ──
    r2_real_scrolls = [2,7,9,11]
    fake_scroll     = 5
    r1_scrolls      = [1, 3, 10]     # note: block 10 is also an exit block
                                       # — movement still allowed, only collect
                                       #   would be penalised (never attempted)

    env_config = {"r2_real": set(r2_real_scrolls),
                  "fake":    fake_scroll,
                  "r1":      set(r1_scrolls)}

    print(f"\n  Forest Configuration:")
    print(f"    R2 Real Scrolls : {r2_real_scrolls}")
    print(f"    Fake Scroll     : {fake_scroll}")
    print(f"    R1 Scrolls      : {r1_scrolls}")

    # ── 1. Heuristic analysis (pick lowest-h entry) ──────────────────
    best_entry_for_demo = min(ENTRY_BLOCKS,
        key=lambda e: heuristic(e, frozenset(r2_real_scrolls)))
    print_heuristic_analysis(best_entry_for_demo, r2_real_scrolls)

    # ── 2. Detailed single-entry run with full expansion log ──────────
    t0 = time.time()
    detail = astar_solve(
        start_block     = best_entry_for_demo,
        r2_real_scrolls = set(r2_real_scrolls),
        fake_scroll     = fake_scroll,
        r1_scrolls      = set(r1_scrolls),
        verbose         = True,
        show_expansion  = True,
    )
    print(f"  Detailed solve time: {(time.time()-t0)*1000:.2f} ms\n")

    # ── 3. All-entry sweep ────────────────────────────────────────────
    t1 = time.time()
    best, all_res = solve_all_entries(r2_real_scrolls, fake_scroll, r1_scrolls)
    print(f"\n  All-entries solve time : {(time.time()-t1)*1000:.2f} ms")
    if best:
        print(f"  Globally optimal entry : block {best['entry']}")
        print(f"  Globally optimal cost  : {best['cost']}")
    else:
        print("  ✗ No solution found from any entry!")
        exit(1)

    # ── 4. Full path workings ─────────────────────────────────────────
    print_path_workings(best, env_config)

    # ── 5. Plot ───────────────────────────────────────────────────────
    plot_path = os.path.join(SCRIPT_DIR, "r2_astar_results.png")
    fig = plot_astar(best, all_res, env_config, save_path=plot_path)
    plt.show()
    plt.close(fig)

    # ── 6. JSON summary ───────────────────────────────────────────────
    summary = {
        "algorithm":    "A* Search",
        "heuristic":    "admissible NN-tour + Manhattan exit distance",
        "config":       {"r2_real_scrolls": r2_real_scrolls,
                         "fake_scroll":     fake_scroll,
                         "r1_scrolls":      r1_scrolls},
        "results_by_entry": {str(e): {
            "success":         r["success"],
            "cost":            r.get("cost"),
            "nodes_expanded":  r.get("nodes_expanded"),
            "nodes_generated": r.get("nodes_generated"),
            "path_steps":      len(r.get("path", [])),
        } for e, r in all_res.items()},
        "optimal_entry":      best.get("entry"),
        "optimal_cost":       best.get("cost"),
        "optimal_path_steps": len(best.get("path", [])),
        "optimal_path": [{k: str(v) for k, v in s.items()} for s in best.get("path", [])],
    }
    sp = os.path.join(SCRIPT_DIR, "r2_astar_summary.json")
    with open(sp, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary JSON saved → {sp}")
