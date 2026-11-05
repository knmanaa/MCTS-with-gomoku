# PUCT-MCTS Algorithm — Implementation Notes

This document explains, phase by phase, how the Monte Carlo tree search inside
`best_move(state, color)` works — with the **mathematical definition of every
quantity** and the **exact code** that implements it. It follows the
AlphaZero-style **PUCT** (Predictor + Upper Confidence bounds applied to
Trees) scheme with the neural networks replaced by traditional substitutes:

| AlphaZero component | Our substitute |
| --- | --- |
| Policy prior $P(s,a)$ (network) | Heuristic prior: line potential, urgency, stone proximity |
| Value network $V(s)$ | Depth-capped random rollouts + static evaluator at the cutoff |
| Search mechanics | Identical: PUCT selection, expansion, sign-alternating backup |
| Move selection | Most-visited root action ($\tau \to 0$) |

All code lives inside `best_move`'s marked block in `project_skeleton.py` —
required by the project rules ("only modify the three required functions").
Every helper is a *nested* definition that reads `n_rows`, `edgeH_l`,
`C_PUCT`, … from the enclosing scope.

---

## Glossary of terms used in this document

| Term | Meaning |
| --- | --- |
| **ply** | One single stone placed by one player (a half-move). `ROLLOUT_DEPTH = 32` plies = 32 stones total (16 per side) before the static evaluator takes over. Two plies = one move by each side. |
| **run** | A straight sequence of same-colour stones along one direction $d \in D$ where every consecutive pair is joined by an *existing* edge (H/V) or is a diagonal pair (always connected). |
| **maximal (run)** | The run cannot be extended in either direction: the cell just beyond each end is not the same colour joined by a live connection (empty, opponent's, board edge, or a missing edge). Guarantees each run is counted exactly once — from its start cell only. |
| **open 3 / closed 3** | A run of 3 stones whose **both** ends are playable (empty *and* reachable through an existing edge — in our variant a missing edge can close an end) vs. only **one** playable end. An open 3 threatens an *open* 4 next move → two winning cells → the opponent must respond immediately. A closed 3 can only become a *closed* 4 → one winning cell → easily blocked. |
| **open 4 / closed 4** | A run of 4 with both ends playable (an **open 4** = two win-in-1 cells = already decisive) vs. one playable end (**closed 4** = exactly one win-in-1 cell = must be blocked now; the root hard rules do this). |
| **win-in-1** | A cell that, if occupied by colour $p$, completes an exactly-five line for $p$: $\mathcal{W}_p$ at the root, detected per ply in rollouts via the four `line_len` checks. |

---

## The quantities, in mathematics

**State and actions.** A state is the tuple
$s = (\text{board}, \text{edgeH}, \text{edgeV})$, where
$\text{board}(r,c) \in \{-1, 0, +1\}$ ($+1$ black, $-1$ white, $0$ empty),
and every player $p \in \{+1, -1\}$ (Black $= +1$, White $= -1$) is on the
move in alternation. An **action** $a = (r,c)$ is any *empty* grid point, so
the legal action set $\mathcal{A}(s)$ is simply the empty points — placing
is allowed anywhere; the edges only influence winning lines. Actions are
encoded internally as flat indices $i = r \cdot N_C + c$.

Every edge $(s,a)$ of the search tree carries four accumulators:

| Symbol | Name | Meaning | Code |
| --- | --- | --- | --- |
| $N(s,a)$ | visit count | how many simulations traversed the edge | `child.N` |
| $W(s,a)$ | total value | sum of all evaluations backed up through the edge | `child.W` |
| $Q(s,a) = \dfrac{W(s,a)}{N(s,a)}$ | mean action value | average outcome of playing $a$ in $s$ — **from the mover's perspective** | `child.Q` |
| $P(s,a)$ | prior probability | heuristic guess of how good $a$ is, assigned *before* any search | `node.prior[a]` |

and every node carries its own $N(s) = \sum_b N(s,b)$ (total visits, the
$\sqrt{\cdot}$ term of PUCT). Terminal states hold an exact value
$z \in \{-1, +1\}$ instead of children:

$$
z(s) \;=\; \begin{cases}
-1, & \text{the move just played won the game (the player to move lost)} \\
+1, & \text{draw and the player to move is White (a draw counts as a win for White)} \\
-1, & \text{draw and the player to move is Black}
\end{cases}
$$

**What is a "heuristic"?** A *heuristic* is a handcrafted rule of thumb that
approximates what a learned network would output, using domain knowledge and
no training. Here $P(s,a)$ is produced by scoring functions over the board
(defined in Phase 2 below) and $V(s)$ by a line-length evaluator (Phase 3);
both are cheap, deterministic functions of the position — exactly the
"domain-specific heuristic policy / static evaluation function" substitutes
of the no-deep-learning PUCT variant.

---

## Preliminaries: two conventions that make the signs work

**Convention 1 — whose value is stored where.**
Every node represents a *state*; the player to move there is `node.player`.
When a simulation descends an edge `(parent, action)`, the *mover* is
`parent.player`. The child's $N, W, Q$ are therefore stored **from the
mover's perspective**: $Q(s,a)$ answers "how good is $a$ for *me*, the one
moving now".

**Convention 2 — the value $v$.**
Evaluations are always produced **from the perspective of the player to
move at the evaluated state**. Since players alternate, converting between
neighbouring levels is a sign flip: $v_{\text{opponent}} = -v_{\text{me}}$.
The whole backpropagation is just this flip applied repeatedly.

### One-time setup

```python
t0 = time.perf_counter()
TIME_LIMIT    = globals().get('MCTS_TIME_LIMIT', 4.7)
C_PUCT        = globals().get('MCTS_C_PUCT', 1.4)
ROLLOUT_DEPTH = globals().get('MCTS_ROLLOUT_DEPTH', 32)
KAPPA         = globals().get('MCTS_KAPPA', 1.5)
deadline = t0 + TIME_LIMIT

n_rows, n_cols = state[0].shape
n_cells = n_rows * n_cols
WHITE = -1
DIRS4 = ((0, 1), (1, 0), (1, 1), (1, -1))

# python-native flat copies: fast scalar access in the hot loops
board_flat = np.asarray(state[0]).ravel().astype(int).tolist()
edgeH_l = np.asarray(state[1]).astype(bool).tolist()
edgeV_l = np.asarray(state[2]).astype(bool).tolist()

# 8-neighbour flat indices (diagonals included: always connected)
neighbors8 = []
for r in range(n_rows):
    for c in range(n_cols):
        nb = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if (dr or dc) and 0 <= r + dr < n_rows and 0 <= c + dc < n_cols:
                    nb.append((r + dr) * n_cols + (c + dc))
        neighbors8.append(tuple(nb))
```

- The deadline is set **before** anything else; the search loop re-checks
  `time.perf_counter()` every simulation, keeping the move inside the
  5-second rule with a 0.3 s safety margin.
- The input arrays are copied into **flat Python lists** (`board_flat` via
  `ravel().tolist()`). Hot-loop optimisation: millions of scalar reads hit a
  Python list (~30 ns) instead of a NumPy array (~100 ns + wrapper object).
  The copy also guarantees *"this function does not change board[r, c]"*.
- `globals().get(...)` lets the test suite set e.g.
  `project_skeleton.MCTS_TIME_LIMIT = 0.15` for fast benchmark games; when
  the name is undefined, the production default applies.
- `neighbors8[i]` holds the flat indices of each cell's eight neighbours —
  diagonals included because diagonal points are *always* connected, even
  though no diagonal lines are drawn. Used only by the prior heuristics.

---

## The two workhorses: `conn`, `line_len`, `is_winning_move`

Mathematically, for a colour $x$ and a direction
$d \in D = \{(0,1),(1,0),(1,1),(1,-1)\}$, let
$L_x(a, d)$ be the length of the **maximal unbroken line** of colour $x$
through cell $a$ along $d$, counting in both directions, where consecutive
stones must be joined by an *existing* edge (horizontal/vertical) or be
diagonal neighbours (always connected). A missing edge breaks the line.

**Reading the arguments.** $d$ is the **direction** — one of the four
canonical step pairs from the table above — so it selects *which line
orientation* through $a$ is measured, while $a$ selects *which cell*. Since
lines are symmetric, the 4 canonical directions combined with the
**both-ways** scan inside `line_len` cover all eight orientations. Example:
$L_{+1}\left((8,4), (0,1)\right)$ is "how long is the horizontal black line
through $(8,4)$?" — the subscript is the colour, the second argument the
orientation. This is exactly how the two prior weight tables use it:
$w_{\text{ext}}\left(L_p(a,d)\right)$ scores extending **our** line through
$a$, and $w_{\text{blk}}\left(L_{-p}(a,d)\right)$ scores occupying a cell
**the opponent's** line through $a$ wants.

### Coordinate convention and the four directions

A step $d = (dr, dc)$ changes the row index by $dr$ and the column index by
$dc$. Recall the board's convention: the row index $r$ **increases
downward** (it counts from the top) and the column index $c$ **increases
rightward** (it counts from the left). Hence:

| $d = (dr, dc)$ | Meaning | Line type |
| --- | --- | --- |
| $(0, +1)$ | row unchanged, column $+1$ | **horizontal** $\rightarrow$ |
| $(+1, 0)$ | one row down, column unchanged | **vertical** $\downarrow$ |
| $(+1, +1)$ | one row down and one column right | **main diagonal** $\searrow$ |
| $(+1, -1)$ | one row down and one column left | **anti-diagonal** $\swarrow$ |

A direction is just this one step **repeated**: $(0,1)$ sweeps along a row,
$(1,0)$ down a column, $(1,1)$ down the main diagonal, $(1,-1)$ down the
anti-diagonal. Only these **four canonical** directions are needed, not
eight, because every straight line is symmetric: stepping $(0,-1)$ (left)
traces the same horizontal line as $(0,+1)$ (right), only walked in the
opposite order. That is why `line_len` scans **both ways** from the centre
cell (one forward and one backward `while` loop) and still covers all eight
orientations with the four pairs in $D$ — and why the diagonals $(1,1)$ and
$(1,-1)$ fall through `conn` to `return True`, while the horizontal and
vertical steps do real `edgeH_l`/`edgeV_l` lookups.

```python
def conn(r, c, dr, dc):
    """Whether (r, c) and (r+dr, c+dc) are joined by an existing edge.
    Diagonals are always connected."""
    if dr == 0 and dc == 1:  return edgeH_l[r][c]
    if dr == 0 and dc == -1: return edgeH_l[r][c - 1]
    if dr == 1 and dc == 0:  return edgeV_l[r][c]
    if dr == -1 and dc == 0: return edgeV_l[r - 1][c]
    return True   # |dr| == 1 and |dc| == 1: diagonals
```

> **Pitfall (bug we fixed).** An earlier version branched on `dr` alone —
> `elif dr == 1: return edgeV_l[r][c]` — so *diagonal* steps `(1, 1)` and
> `(1, -1)` fell into the vertical branch and returned a **vertical edge
> lookup** instead of `True`. Diagonal lines were then silently broken by
> irrelevant missing vertical edges: diagonal win threats were not blocked,
> rollouts missed diagonal wins, and the evaluator undercounted diagonal
> runs — all without any crash. The correct branch must always test **both**
> components (`dr` *and* `dc`) before touching an edge array; only
> $|dr| = |dc| = 1$ may fall through to `True`. This is exactly the same
> shape as `check_result`'s `connected()`, which had it right from the
> start. Regression test: a diagonal four-threat must be blocked even when
> *every* vertical edge is missing.

`conn` is the connectivity oracle — the one function where the "unbroken
line" rule lives. Horizontal/vertical connections exist only where the
corresponding edge was initialised; **any diagonal falls through to
`return True`**. It is the semantic twin of `connected()` in
`check_result`, just flattened and extended to negative directions.

```python
def line_len(bd, r, c, dr, dc):
    """Length of the maximal unbroken same-colour line through (r, c)
    in direction (dr, dc), counting both ways (connectivity-aware)."""
    color = bd[r * n_cols + c]
    L = 1
    rr, cc = r + dr, c + dc
    while (0 <= rr < n_rows and 0 <= cc < n_cols
           and bd[rr * n_cols + cc] == color
           and conn(rr - dr, cc - dc, dr, dc)):
        L += 1
        rr += dr
        cc += dc
    rr, cc = r - dr, c - dc
    while (0 <= rr < n_rows and 0 <= cc < n_cols
           and bd[rr * n_cols + cc] == color
           and conn(rr, cc, dr, dc)):
        L += 1
        rr -= dr
        cc -= dc
    return L
```

`line_len` computes $L_x(a,d)$: walk forward and backward from $(r,c)$
while the next cell is in bounds, holds the same colour, *and* the edge
from the previous cell exists (`conn(rr-dr, cc-dc, ...)` — the edge is
indexed by the **previous** cell, which is always in bounds; this ordering
is why no IndexError can occur). A missing edge stops the count → the line
is "broken" (Figure 5).

```python
def is_winning_move(bd, r, c):
    """Whether the stone just placed at (r, c) completes an unbroken
    line of exactly five stones (the game would be over)."""
    for dr, dc in DIRS4:
        if line_len(bd, r, c, dr, dc) == 5:
            return True
    return False
```

`is_winning_move` encodes the exactness of the win rule:

$$
\text{win}(a) \iff \exists\, d \in D:\; L_x(a, d) = 5
$$

— *exactly* five, not five or more (Figures 3/4: gaps don't count,
six-in-a-row doesn't count). It is only ever called *right after* a stone
is placed — a win can only be completed by the newest stone — so four line
scans through one cell are equivalent to `check_result`'s full-board scan,
but ~50× cheaper. The referee test (MCTS games refereed by
`check_result`) verifies this equivalence empirically.

Two helpers built on the same machinery serve the open-four threat layer
(see *Root hard rules*):

```python
def run_extent(bd, r, c, dr, dc):
    """For the maximal unbroken line of bd[r, c] through (r, c) along
    (dr, dc), return (L, start, end): the length and the two extreme
    cells of the run (start is the cell furthest along -d)."""
    ...
    # Chebyshev distance: horizontal/vertical runs advance one unit per step,
    # but a diagonal run advances one unit in BOTH coordinates per step.  The
    # former dot product (e-s).(dr,dc) therefore double-counted diagonals
    # (L = 3 reported as 5), so no diagonal four ever reached the threat
    # layer's "L == 4" branch.  max(|dr|,|dc|) is the true step count in all
    # four directions.
    L = max(abs(e_r - s_r), abs(e_c - s_c)) + 1
    return L, (s_r, s_c), (e_r, e_c)

def four_threat_count(bd, a, pl):
    """Exact number of distinct empty cells that become immediate winning
    moves for `pl` after `pl` plays at flat index a.  Every candidate within
    four steps of `a` along the four lines through `a` is validated by
    placing a stone and calling win_check.  This rejects a "win" that would
    make a run of six (an overline, which the rules do not count) and catches
    a broken four such as X . X X X whose gap is the winning cell."""
    r, c = divmod(a, n_cols)
    bd[a] = pl
    found = set()
    for dr, dc in DIRS4:
        cand = [(r + k * dr, c + k * dc)
                for k in (-4, -3, -2, -1, 1, 2, 3, 4)]
        L, s, e = run_extent(bd, r, c, dr, dc)
        if L >= 4:                       # straight four: classic extensions
            cand.append((s[0] - dr, s[1] - dc))
            cand.append((e[0] + dr, e[1] + dc))
        for br, bc in cand:
            if not (0 <= br < n_rows and 0 <= bc < n_cols):
                continue
            b = br * n_cols + bc
            if bd[b] != 0:
                continue
            bd[b] = pl
            if win_check(bd, b):
                found.add(b)
            bd[b] = 0
    bd[a] = 0
    return len(found)
```

`run_extent` is `line_len` plus the run's two extreme cells, with the
Chebyshev step count so diagonals are measured correctly. `four_threat_count`
is the atomic operation of the open-four layer, but it no longer trusts
contiguous run length alone: it places a hypothetical stone and *validates
each candidate winning cell with `win_check`*, so overlines (a run of six)
are rejected and broken fours (`X . X X X`) are counted. This is the "count
distinct, actual exact-five completion cells" formulation — it handles
straight fours, broken fours and intersecting threats consistently.

---

## The node

```python
class Node:
    """Search-tree node.  Edge statistics (N, W, Q) are stored on the
    child and are kept from the perspective of the player who made the
    move leading to it (the parent's player to move)."""
    __slots__ = ('board', 'player', 'children', 'untried', 'prior',
                 'N', 'W', 'Q', 'terminal_value', 'empties')

    def __init__(self, board, player):
        self.board = board          # flat list, player to move = `player`
        self.player = player
        self.children = None        # dict {action: Node}, set on expansion
        self.untried = None         # actions not yet tried, prior-ascending
        self.prior = None           # dict {action: P(s, a)}
        self.N = 0                  # visit count N(s)
        self.W = 0.0                # total action value
        self.Q = 0.0                # mean action value W / N
        self.terminal_value = None  # value for the player to move if over

def create_child(node, a):
    """Apply action `a` for node's player to move; return the child."""
    r, c = divmod(a, n_cols)
    bd = node.board[:]              # copy; the parent's board is reused
    bd[a] = node.player
    child = Node(bd, -node.player)
    if is_winning_move(bd, r, c):
        child.terminal_value = -1.0                          # mover won
    elif 0 not in bd:
        child.terminal_value = 1.0 if child.player == WHITE else -1.0
    return child
```

- `__slots__` removes the per-instance `__dict__` — with thousands of nodes
  this roughly halves memory.
- The edge statistics live **on the child** and are kept from the **mover's
  perspective** (Convention 1): the child of edge $(s,a)$ stores
  $(N(s,a), W(s,a), Q(s,a))$.
- `create_child` copies the parent's flat board (one 221-element list copy
  per simulation) and stamps the terminal value $z$ **at creation**, per the
  case distinction in the table above: a win by the mover means the child's
  player to move has lost ($z = -1$); a full board is a draw
  ($z = +1$ for White to move, $-1$ for Black). Terminal children are never
  expanded — the selection loop breaks on them instead.

---

## Phase 2 — Expansion: the heuristic prior $P(s, a)$

The prior is a normalised heuristic score over the legal actions. For the
player $p$ to move at state $s$ and the empty cell $a$:

$$
\underbrace{\text{occ}(a) = \left|\{\, j \in \mathcal{N}_8(a) : \text{board}(j) \neq 0 \,\}\right|}_{\text{stone proximity}}
$$

$$
\text{score}(a) \;=\; 0.6 \cdot \text{occ}(a)
\;+\; \sum_{d \in D} w_{\text{ext}}\left(L_p(a, d)\right)
\;+\; \sum_{d \in D} w_{\text{blk}}\left(L_{-p}(a, d)\right)
$$

where $L_p(a,d)$ is measured with $a$ *hypothetically occupied* by colour
$p$ (the first sum rewards extending **our own** lines, the second rewards
occupying cells the **opponent's** lines want), and the weights are

$$
w_{\text{ext}}(L) = \begin{cases}2 & L = 2\\ 8 & L = 3\\ 30 & L = 4\\ 0 & \text{else}\end{cases}
\qquad
w_{\text{blk}}(L) = \begin{cases}1 & L = 2\\ 4 & L = 3\\ 20 & L = 4\\ 0 & \text{else}\end{cases}
$$

The prior is then the normalised score:

$$
P(s,a) \;=\; \frac{\text{score}(a) + \varepsilon}{\sum_{a' \in \mathcal{A}(s)} \left(\text{score}(a') + \varepsilon\right)},
\qquad \varepsilon = 10^{-9}.
$$

```python
def rich_scores(bd, player, actions):
    """Heuristic scores for the root prior: urgency of extending own
    lines / occupying cells the opponent wants, plus stone proximity."""
    EXT_W   = {2: 2.0, 3: 8.0, 4: 30.0}
    BLOCK_W = {2: 1.0, 3: 4.0, 4: 20.0}
    scores = []
    for a in actions:
        r, c = divmod(a, n_cols)
        s = 0.6 * sum(1 for j in neighbors8[a] if bd[j] != 0)
        for pl, table in ((player, EXT_W), (-player, BLOCK_W)):
            bd[a] = pl
            for dr, dc in DIRS4:
                s += table.get(line_len(bd, r, c, dr, dc), 0.0)
            bd[a] = 0
        scores.append(s)
    return scores

def expand(node, actions, rich):
    """Create the action set of `node` with heuristic priors P(s, a)."""
    if actions is None:
        actions = [i for i in range(n_cells) if node.board[i] == 0]
    if rich:
        scores = rich_scores(node.board, node.player, actions)
    else:
        scores = [1.0 + 0.5 * sum(1 for j in neighbors8[a] if node.board[j] != 0)
                  for a in actions]
    total = sum(scores) + 1e-9
    node.prior = {a: (s + 1e-9) / total for a, s in zip(actions, scores)}
    # ascending by prior, so that list.pop() yields the best-prior action
    node.untried = [a for _, a in sorted(zip(scores, actions))]
    node.children = {}
```

Two moments:

1. **`rich_scores` (root only)** implements the full score above — the
   hypothetical placements are the lines `bd[a] = pl ... bd[a] = 0`, always
   undone. A move that creates a connected four-line is near-winning
   ($+30$); occupying a cell the opponent's line wants is worth up to
   $+20$. This is the stand-in for the policy network.
2. **`expand`** — deeper nodes use the **cheap prior** instead:

   $$
   \text{score}(a) = 1 + 0.5 \cdot \text{occ}(a)
   $$

   A full rich pass over ~200 cells would cost 2–4 ms and we expand once
   *per simulation*; proximity alone is ~10× cheaper and keeps exploration
   near the action. `untried` is stored **sorted ascending by prior** so
   that `list.pop()` — O(1) — yields the *highest*-prior action.

---

## Phase 1 — Selection: the PUCT formula

At every fully-expanded node $s$ the simulation descends by

$$
a^* \;=\; \arg\max_{a \in \mathcal{A}(s)} \left( Q(s,a) + U(s,a) \right),
\qquad
U(s,a) \;=\; c_{\text{puct}} \cdot P(s,a) \cdot \frac{\sqrt{\sum_b N(s,b)}}{1 + N(s,a)}
$$

Term by term: $Q(s,a)$ is **exploitation** (the average outcome from the
mover's perspective); $P(s,a)$ is **guided exploration** (moves that extend
own lines, block the opponent, or sit near stones are tried more often);
the ratio is optimism that decays as the action is tried — a never-tried
action receives the full $\sqrt{N(s)}$ boost.

```python
def puct_select(node):
    """PUCT: argmax_a Q(s, a) + c_puct * P(s, a) * sqrt(N(s)) / (1 + N(s, a))."""
    sq_n = node.N ** 0.5
    best_a, best_s = None, -1e18
    for a, ch in node.children.items():
        s = ch.Q + C_PUCT * node.prior[a] * sq_n / (1 + ch.N)
        if s > best_s:
            best_s, best_a = s, a
    return best_a
```

$\sqrt{N(s)}$ is hoisted out of the loop; the argmax is a single linear
scan with a running maximum — no temporary lists.

The descent itself is the simulation's outer walk:

```python
node, path = root, []
while True:
    if node.terminal_value is not None:
        break                            # game already decided at `node`
    if node.children is None:            # first visit: expand
        expand(node, None, rich=False)
    if node.untried:                     # expansion phase
        a = node.untried.pop()
        child = create_child(node, a)
        node.children[a] = child
        path.append((node, a))
        node = child
        break
    a = puct_select(node)                # selection phase
    path.append((node, a))
    node = node.children[a]
```

Four exits, in priority order:

1. **Terminal node** — $z(s)$ was stamped at creation; the exact score is
   known, no evaluation needed. The check consults `node.terminal_value`
   itself, so a child that *became* terminal during this descent (a freshly
   created child, or a pre-existing one) is recognised without a separate
   flag.
2. **First visit** — `children is None`: the state has never been seen, so
   build its action set and priors.  **Tactical priority:** for non-root
   nodes the action set is first filtered — if the mover has an immediate
   win, only the winning moves are kept; otherwise, if the opponent has one,
   only the blocking moves are kept.  Candidate cells are restricted to those
   adjacent to a stone (a five must include one), so the filter stays cheap.
   This makes forced sequences real search lines instead of relying on
   proximity priors and random rollouts to notice them.
3. **Untried actions left** — the node is not fully expanded:
   `untried.pop()` takes the **highest-prior untried action** (the list is
   prior-ascending) — the prior-guided equivalent of
   `select_arbitrary_unvisited`. Create the child, record `(node, action)`
   in `path`, descend, and break: this fresh leaf is what gets evaluated.
4. **Fully tried** — PUCT-select and keep descending.

Visits are **not** incremented during descent.  The root is incremented once
per completed simulation and every traversed child exactly once during
backpropagation (see *Phase 4*).  The old code also incremented each node as
it was descended, so non-root visits were counted twice while values were
summed once — which inflated the denominator of $Q$ and the PUCT exploration
term.  Counting only in the backup phase keeps $N(s) = \sum_b N(s,b)$ exact.

Every traversed edge is recorded in `path = [(node, action), ...]` from
root to leaf — exactly the route backpropagation will walk.

---

## Phase 3 — Evaluation

### Option B: the depth-capped rollout

The rollout returns a value $v$ for the player to move at the leaf:

$$
v \;=\; \begin{cases}
+1, & \text{the playout's winner is the leaf's player to move}\\
-1, & \text{the playout's winner is the opponent}\\
+1, & \text{board full and the leaf's player to move is White}\\
-1, & \text{board full and the leaf's player to move is Black}\\
V(s_{\text{cut}}), & \text{after } 32 \text{ plies without a terminal}
\end{cases}
$$

```python
# batched random numbers (much cheaper than one call per ply)
RBN = ROLLOUT_DEPTH + 8
rs = np.random.random(RBN).tolist()
rs_i = [0]

def next_rand():
    k = rs_i[0]
    if k >= RBN:
        rs[:] = np.random.random(RBN).tolist()
        rs_i[0] = k = 0
    rs_i[0] = k + 1
    return rs[k]
```

Randoms come from a **pre-drawn batch**: one `np.random.random()` call per
ply would cost more than the playout step itself; batching cuts that to
~1 call per rollout.

```python
def rollout(node):
    """Depth-capped random playout; returns the value from the
    perspective of the player to move at `node`."""
    bd = node.board[:]            # scratch copy
    empties = [i for i in range(n_cells) if bd[i] == 0]
    to_move = node.player
    draw_value = 1.0 if node.player == WHITE else -1.0
    for _ in range(ROLLOUT_DEPTH):
        if not empties:
            return draw_value     # draw: White wins
        a = empties.pop(int(next_rand() * len(empties)))
        r, c = divmod(a, n_cols)
        bd[a] = to_move
        for dr, dc in DIRS4:
            if line_len(bd, r, c, dr, dc) == 5:
                return 1.0 if to_move == node.player else -1.0
        to_move = -to_move
    return static_eval(bd, node.player)
```

A **scratch copy** of the board is used (discarded afterwards, so the
swap-pop `empties.pop(...)` needs no undo). Each ply places a uniform
random legal stone and checks only the four lines through it — `line_len`
walks both ways **through existing edges** (`conn`), so playouts obey the
game's connectivity.

Why cap the rollout? An uncapped random playout from a mid-game position
runs 100–200 plies — five to ten times the cost — for a mostly noisy
result. Thirty-two plies plus the evaluator is a better accuracy per
millisecond trade.

### Option A: the static evaluator $V(s)$

Let $\Phi_x(s)$ sum, over **all maximal edge-connected runs** of colour $x$
with length $L \geq 2$, an **openness-weighted** contribution. For a run,
let $e \in \{0, 1, 2\}$ be its number of **playable ends**: an empty,
in-bounds cell immediately beyond each extreme of the run, reachable
through an existing edge (diagonals always; in our variant a missing edge
closes an end). Then

$$
g(L, e) \;=\; (L-1)^2 \,\left(1 + \kappa\, e\right),
\qquad \kappa \approx 1.5
\qquad\Longrightarrow\qquad
\Phi_x(s) \;=\; \sum_{\text{runs of } x} g\left(L, e\right)
$$

Two escalation axes, deliberately separated: $(L-1)^2$ is **convex in
length** (a four-line is worth vastly more than a two-line), and
$1 + \kappa e$ is **linear in openness** (the same stones are $\kappa$
times more dangerous when both ends are playable). The resulting hierarchy
at $\kappa = 1.5$ is exactly the threat taxonomy of the glossary:

| run | $L$ | $e$ | $g$ |
| --- | --- | --- | --- |
| closed 2 | 2 | 0 | 1 |
| open 2 / closed 3 | 2 / 3 | 2 / 0 | 4 |
| closed 4 | 4 | 0 | 9 |
| four, one end | 4 | 1 | 22.5 |
| **open 3** | 3 | 2 | **16** |
| **open 4** | 4 | 2 | **36** |

An **open 3** (16) outranks a **fully dead** four (9), but a **one-ended
four** (22.5) outranks the open 3. That ordering is deliberate — a
one-ended four is a winning threat that must be answered *now* — but it is
the reason the evaluator alone cannot be trusted to defend against one:
a heuristic score is not a tactical guarantee. The hard rules and the
search handle forced lines; the evaluator only needs the right *ranking*
at rollout cutoffs. An **open 4** (36) towers over everything, as a forced
win should. Then

$$
\tilde V(s) \;=\; \frac{\Phi_{+1}(s) - \Phi_{-1}(s)}
                        {\left|\Phi_{+1}(s) - \Phi_{-1}(s)\right| + 8}
\;\in\; (-1, 1)
\qquad\quad
V(s) \;=\; \begin{cases}\tilde V(s) & \text{Black to move}\\ -\tilde V(s) & \text{White to move}\end{cases}
$$

```python
def static_eval(bd, for_player):
    """Static evaluation in [-1, 1] from `for_player`'s perspective:
    openness-weighted squared lengths of all maximal connected lines,
    (L-1)^2 * (1 + KAPPA * playable_ends).  An open 3 thus outweighs a
    closed 4, which is the fix for the open-three blindness."""
    sb = sw = 0.0
    for i in range(n_cells):
        v = bd[i]
        if v == 0:
            continue
        r, c = divmod(i, n_cols)
        for dr, dc in DIRS4:
            br, bc = r - dr, c - dc
            if (0 <= br < n_rows and 0 <= bc < n_cols
                    and bd[br * n_cols + bc] == v and conn(br, bc, dr, dc)):
                continue          # not the start of a line
            L = 1
            rr, cc = r, c
            while (0 <= rr + dr < n_rows and 0 <= cc + dc < n_cols
                   and bd[(rr + dr) * n_cols + cc + dc] == v
                   and conn(rr, cc, dr, dc)):
                L += 1
                rr += dr
                cc += dc
            if L >= 2:
                # playable ends: an empty cell just beyond each extreme
                # of the run, reachable through an existing edge
                ends = 0
                er, ec = r - dr, c - dc
                if (0 <= er < n_rows and 0 <= ec < n_cols
                        and bd[er * n_cols + ec] == 0
                        and conn(er, ec, dr, dc)):
                    ends += 1
                er, ec = rr + dr, cc + dc
                if (0 <= er < n_rows and 0 <= ec < n_cols
                        and bd[er * n_cols + ec] == 0
                        and conn(rr, cc, dr, dc)):
                    ends += 1
                w = (L - 1) ** 2 * (1.0 + KAPPA * ends)
                if v == 1:
                    sb += w
                else:
                    sw += w
    score = sb - sw               # black-positive
    score /= (abs(score) + 8.0)   # squash to (-1, 1)
    return score if for_player == 1 else -score
```

The "skip" condition (the cell behind is the same colour **and connected**)
ensures each run is counted exactly once from its start; the
bounds-before-edge-lookup ordering in the `while` condition means `conn`
only ever touches an in-bounds cell's edge. Note the asymmetry in the two
end checks: the edge *before* the run's start belongs to
$(r-dr, c-dc)$ — indexed by the extension cell — while the edge *after*
the run's last cell $(rr, cc)$ belongs to $(rr, cc)$ — indexed by the run.

**How the runs relate to $V(s)$.** The runs are the raw data; $V(s)$ is the
final scalar built from them, in four steps:

$$
\underbrace{\text{maximal edge-connected runs of } x}_{\text{the census: counted once each, } L \geq 2}
\;\longrightarrow\;
\Phi_x(s) = \sum_{\text{runs of } x} g(L, e)
\;\longrightarrow\;
\tilde V(s) = \frac{\Phi_{+1}(s) - \Phi_{-1}(s)}{|\Phi_{+1}(s) - \Phi_{-1}(s)| + 8}
\;\longrightarrow\;
V(s) = \begin{cases}\tilde V(s) & \text{Black to move}\\ -\tilde V(s) & \text{White to move}\end{cases}
$$

Worked example — the only lines on the board are a black **open** 3
($L = 3$, $e = 2$), a black closed 2 ($L = 2$, $e = 0$), and a white closed
2:

$$
\Phi_{+1} = (3-1)^2 (1 + 1.5 \cdot 2) + (2-1)^2 \cdot 1 = 16 + 1 = 17, \qquad
\Phi_{-1} = (2-1)^2 \cdot 1 = 1, \qquad
\tilde V = \frac{16}{16+8} \approx +0.67
$$

Under the old length-only evaluator the same position gave
$\tilde V = 4/13 \approx +0.31$: the open 3 was nearly invisible. Now it
dominates the score — which is precisely what makes "ignore the open three"
branches evaluate badly at every rollout cutoff. In the code, `sb`/`sw` are
literally $\Phi_{+1}$/$\Phi_{-1}$, and the last three lines of
`static_eval` are the squash and perspective steps. This matters because
$V(s_{\text{cut}})$ is the value of every rollout that hits the 32-ply cap
— i.e. the leaf evaluation almost everywhere the game does not end inside
the playout.

### Design rationale for $V(s)$

**Why run length $L$ as the core feature?**

- $L$ is *distance-to-winning in disguise*: the game is won at $L = 5$; a
  run of 4 means win-in-1, a 3 means "threatens a 4". $L$ therefore orders
  every line by urgency *by construction* — no pattern book needed.
- It is the **cheapest sufficient statistic**: computing $L$ needs only
  short local walks (~4–8 cells each). The evaluator runs thousands of
  times per move (once per depth-capped rollout), so it must stay around
  0.1 ms on a 221-cell board — and the import rules forbid anything
  heavier than NumPy anyway.
- It is the correct **first-order approximation** of classical gomoku
  theory, where the patterns that matter (live/dead twos, threes, fours)
  are runs and their openness. Openness is the second coordinate, added by
  the $g(L, e)$ factor below.

**Why the openness factor $1 + \kappa e$?** Openness is a *multiplier* on
threat, not a bonus: the same three stones are $\kappa$ times more
dangerous with both ends playable than with one blocked, because the
dangerous *trajectory* (open 3 → open 4 → win) exists only in the open
case. The form $1 + \kappa e$ has exactly one tuning parameter
($\kappa$ interpolating between the old length-only evaluator at
$\kappa = 0$ and strong threat sensitivity), keeps $g = 1$ for fully dead
runs (today's behaviour as the baseline), and the resulting ranking —
open 3 (16) > closed 4 (9) — matches the glossary's threat taxonomy. A
full $6 \times 3$ lookup table would be more expressive but adds ~18
hand-tuned entries for second-order gains; the search's $Q$ values should
resolve those, not the evaluator. Note the honest ranking: open 3 (16) beats
a *fully dead* four (9) but loses to a *one-ended* four (22.5) — so raising
$\kappa$ cannot, by itself, make the evaluator defend a one-ended four; that
is what the hard rules and the search are for. What $g$ deliberately does
*not* capture — gap patterns like `● ● . ● ●` — is handled by the tree
(which tries the gap cell) and the root win-in-1 rules; the evaluator only
needs to be a good *average* signal at rollout cutoffs.

**Why the $(L-1)^2$ weighting?** It makes urgency **convex** — each extra
stone is worth disproportionately more:

| $L$ | $(L-1)^2$ | meaning |
| --- | --- | --- |
| 2 | 1 | mild potential |
| 3 | 4 | a real threat forming |
| 4 | 9 | win-in-1 — must be answered now |

A single 4-run (9) outweighs *nine* separate 2-runs (9 × 1) — the right
trade, since a near-win dominates any amount of vague potential. A linear
$L$ would allow exactly that absurd trade. The $-1$ keeps lone stones at 0;
the $+8$ in $\tilde V$'s denominator is the matching soft cap — no
*material* lead in gomoku is a certain win until a five actually exists,
so $V$ never quite reaches $\pm 1$.

**Why maximal + edge-connected as the counting unit?** *Maximal* → each
line counted exactly once (no double counting from a run's middle cells).
*Edge-connected* → correctness in this variant: a missing edge genuinely
breaks a line (Figure 5), so the evaluator agrees with `check_result`'s
win rule — an evaluator that believed in phantom lines would misjudge
entire positions.

---

## Phase 4 — Backpropagation

For every edge $(s_t, a_t)$ on the descent path, from the leaf upwards,
with the perspective flipping at each level (zero-sum alternation):

$$
N(s_t, a_t) \leftarrow N(s_t, a_t) + 1, \qquad
W(s_t, a_t) \leftarrow W(s_t, a_t) + v_t, \qquad
Q(s_t, a_t) = \frac{W(s_t, a_t)}{N(s_t, a_t)},
\qquad
v_{t-1} = -v_t
$$

```python
# evaluation phase: a freshly created child may already be terminal
# (create_child sets terminal_value), so consult the node's own state rather
# than a descent-time flag -- otherwise a finished game would be rolled out
# and a forced win evaluated as a random playout.
if node.terminal_value is not None:
    v = node.terminal_value              # for the player to move here
else:
    v = rollout(node)

# backpropagation: exactly one visit per traversed node.  The root is counted
# once here; every traversed child is counted once below.  The sign flips at
# each level.
root.N += 1
v = -v
for pnode, a in reversed(path):
    ch = pnode.children[a]
    ch.N += 1
    ch.W += v
    ch.Q = ch.W / ch.N
    v = -v
```

Sanity check with the two-edge path `root(P1) → n1(P2) → n2(P1)` and a
rollout value $v$ for P1:

- `v = -v` → P2's perspective; the first processed edge `(n1, a2)` — P2's
  move — accumulates $v$. ✓
- flip → P1; the second edge `(root, a1)` — P1's move — accumulates $v$. ✓

A terminal leaf reached by *descent* (not fresh creation) enters the same
loop with $v = z(s)$: identical accounting, because $z$ is already stored
from that node's player-to-move perspective. Both `path` entries were
appended *before* descending, so `pnode.children[a]` is exactly the child
that was visited. The same code path also catches a leaf that became
terminal *during* creation, which is why the descent flag was removed.

---

## Final move selection

After the budget is exhausted the move is sampled from the visit counts,

$$
\pi(a \mid s_0) \;=\; \frac{N(s_0, a)^{1/\tau}}{\sum_b N(s_0, b)^{1/\tau}},
$$

with the temperature $\tau \to 0$, i.e. deterministically the **most
visited** root action (more robust than max-$Q$: the $Q$ of a rarely
visited but once-lucky move can be inflated); $Q$ only breaks ties.

```python
# final move: most-visited root child (tau -> 0); tie-break by Q
best_a, best_key = None, None
for a, ch in root.children.items():
    key = (ch.N, ch.Q)
    if best_key is None or key > best_key:
        best_key, best_a = key, a
if best_a is None:
    # no simulation completed: fall back to the best-prior action
    best_a = root.untried[-1]

return divmod(best_a, n_cols)
```

Defensive fallback: if *zero* simulations completed, return
`root.untried[-1]` — the highest-prior action (`untried` is
prior-ascending and `pop()` drained it from the top). `divmod` converts the
flat index back to $(r, c)$.

---

## Root hard rules

Before any search, two tactical checks can end the call instantly. Let
$\mathcal{W}_p$ be the set of empty cells that complete an exactly-five
line for colour $p$:

$$
\mathcal{W}_p \;=\; \left\{\, a \in \mathcal{A}(s_0) \;:\; \exists\, d \in D,\; L_p(a, d) = 5 \text{ with } a \text{ occupied by } p \,\right\}
$$

```python
# ---- hard tactical rules at the root (cheap, and they save time) ----
empties0 = [i for i in range(n_cells) if board_flat[i] == 0]
if not empties0:
    return (0, 0)                 # game already over (defensive)
if len(empties0) == 1:
    return divmod(empties0[0], n_cols)

def winning_cells(pl):
    out = []
    for a in empties0:
        r, c = divmod(a, n_cols)
        board_flat[a] = pl
        if is_winning_move(board_flat, r, c):
            out.append(a)
        board_flat[a] = 0
    return out

mine = winning_cells(color)
if mine:
    return divmod(mine[0], n_cols)      # win immediately
theirs = winning_cells(-color)
if theirs:
    # if the opponent can win next move, every other move loses: block
    root_actions = theirs
else:
    # ---- open-four threat layer (one ply earlier than win-in-1) ----
    # M: cells where OUR stone creates >= 2 win-in-1 cells -- an open
    #    four.  This is a forced win: the opponent cannot outrace two
    #    win-in-1 cells with a single reply.
    # O: cells where THEIR stone creates >= 2 win-in-1 cells -- if they
    #    get to play one, we lose by force.
    # F: cells where OUR stone creates >= 1 win-in-1 cell -- a forcing
    #    four ("fight threes with fours"): they must answer or lose.
    # Any move outside M | F | O loses by force when O != empty, so the
    # search is restricted to M | F | O; M itself wins outright.
    M = [a for a in empties0 if four_threat_count(board_flat, a, color) >= 2]
    if M:
        return divmod(M[0], n_cols)
    O = [a for a in empties0 if four_threat_count(board_flat, a, -color) >= 2]
    if O:
        F = [a for a in empties0 if four_threat_count(board_flat, a, color) >= 1]
        root_actions = sorted(set(F) | set(O))
    else:
        root_actions = empties0
if len(root_actions) == 1:
    return divmod(root_actions[0], n_cols)

# ---- PUCT-MCTS ----
root = Node(board_flat[:], color)
expand(root, root_actions, rich=True)
```

- **Win-in-1** ($\mathcal{W}_{p} \neq \emptyset$): return such a cell
  immediately — no search can improve on a forced win, and the saved
  seconds are banked.
- **Forced block** ($\mathcal{W}_{-p} \neq \emptyset$, $\mathcal{W}_p = \emptyset$):
  every non-blocking move loses — placing our stone anywhere else leaves
  their winning cell empty and their line intact (our stone elsewhere can
  neither remove nor shorten their line, because *edges are fixed*). The
  root's action set is therefore **restricted to
  $\mathcal{A}(s_0) \leftarrow \mathcal{W}_{-p}$**.
- A single candidate → return without searching.
- **The same priority applies at search nodes.** The root hard rules are the
  cheap pre-search version; `expand` applies the same filter when it builds a
  non-root node's action set: if the mover has an immediate win, only the
  winning moves are kept; otherwise, if the opponent has one, only the
  blocking moves are kept. Candidate cells are restricted to those adjacent to
  a stone (a five must include one), so the filter stays cheap. This makes
  forced reply sequences — "fight threes with fours" — actual search lines
  rather than trusting proximity priors and random rollouts to notice them.
  Note the limitation: admission to $F$ is not proof that the whole
  continuation is safe; the search must still verify the forced sequence
  rather than treating a counterattack as equivalent to a direct defence.

### The open-four threat layer ($M$, $F$, $O$)

Define the **four-threat count** of an empty cell $a$ for colour $p$:

$$
T_p(a) \;=\; \left|\{\, \text{win-in-1 cells for } p \text{ after placing a stone of } p \text{ at } a \,\}\right|
$$

Only the four lines through $a$ can change, so `four_threat_count` places
the stone and validates each candidate within four steps along the four
lines through $a$ with `win_check` — it no longer trusts contiguous run
length alone.  This rejects overlines (a run of six, which the rules do not
count) and catches broken fours (`X . X X X`) whose gap is the winning cell.
`run_extent` is still used to add the classic straight-four extension cells
to the candidate set.  Then:

$$
M = \{a : T_p(a) \geq 2\}, \qquad
O = \{a : T_{-p}(a) \geq 2\}, \qquad
F = \{a : T_p(a) \geq 1\}
$$

with $p$ our colour. The game theory that justifies the cascade:

- **$M \neq \emptyset$ is a forced win** for us: after our move we hold two
  win-in-1 cells; the opponent's single reply can block at most one (and
  they have no win-in-1 of their own to race with, since
  $\mathcal{W}_{-p} = \emptyset$ was handled first). We play our remaining
  win cell and win. Hence $M$ is returned immediately, like `mine`.
- **If $O \neq \emptyset$ and $M = \emptyset$, every move outside
  $M \cup F \cup O$ loses by force**: the opponent answers with some
  $a \in O \setminus \{m\}$ (an open four — two win cells), we hold no
  win-in-1 ($m \notin F$), so we block one cell and they win at the other.
  Restricting the search to $M \cup F \cup O$ is therefore *sound*.
- **Why not just "block"?** Two reasons. First, a naive block-restriction
  would be *unsound*: our own open-four move ($M$) beats their open 3 by
  outracing it, and our four-making moves ($F$) are legal *tempo*
  defences — "fight threes with fours": the opponent must answer our
  win-in-1 or lose, which delays their open four by a ply. Second, the
  choice between blocking ($O$) and counter-attacking ($F$) is a genuine
  value decision — exactly what the search with the openness-aware $V(s)$
  is good at.
- **Variant-awareness**: a "playable" extension cell requires the
  connecting *edge* to exist, so in our variant even the *ends* of a run
  can be closed by a missing edge — $T_p(a)$ counts only real threats.
- **Cost**: three scans over the empty cells, each ~4 short line walks per
  cell → a few milliseconds, once per move. Positions without open-four
  threats pay nothing beyond the $O$-scan.

If $|F \cup O| = 1$ → return it directly (no search); otherwise the root
expands with the **rich** priors over the restricted set and the search
concentrates all ~20k simulations on the cells that matter.

The search runs:

```python
while time.perf_counter() < deadline:
    # one iteration = one full simulation:
    # selection -> expansion -> evaluation -> backpropagation
    ...
```

The deadline is checked once per simulation. A single simulation costs well
under a millisecond (a 32-ply rollout is a few hundred microseconds), so
the overshoot is bounded by milliseconds. With a 4.7 s budget this yields
thousands of simulations; the 0.3 s margin covers the final argmax and any
garbage-collection pause, keeping the move inside the 5-second rule.

---

## Where the game variant enters the algorithm

Every phase touches `edgeH`/`edgeV`:

- `conn()` gates line growth **in the rollouts**;
- the static evaluator counts only **edge-connected** runs;
- the priors reward cells that can actually **participate in lines**.

A cell whose surrounding edges are all missing can never be part of a
winning line — and all four phases consistently learn to ignore it. This is
the structural difference from plain gomoku MCTS, and the reason the
win-check exists in two forms with identical semantics: a full-board scan in
`check_result` (called once per real move) and an incremental
last-move-only version inside the search (called thousands of times per
move).

---

## Tunables

| Knob | Default | Meaning |
| --- | --- | --- |
| `MCTS_TIME_LIMIT` | 4.7 s | Search budget per move (must stay ≤ 5 s) |
| `MCTS_C_PUCT` | 1.4 | Exploration constant $c_{\text{puct}}$ |
| `MCTS_ROLLOUT_DEPTH` | 32 | Plies before the static evaluator takes over |
| `MCTS_KAPPA` | 1.5 | Openness weight $\kappa$ in $g(L, e) = (L-1)^2 (1 + \kappa e)$ |

They can be overridden from outside **without editing the code** (e.g.
`project_skeleton.MCTS_TIME_LIMIT = 0.15` for fast benchmark games); the
defaults apply when the names are not defined.

---

# Performance optimisation (v3)

The search is budget-bound: the strength of the move is (almost) proportional
to how many simulations fit into the 4.7 s. Profiling the v2 engine found the
budget split as follows (mid-game positions):

| Component | Share of the 4.7 s |
| --- | --- |
| Rollouts (total) | ~60–80 % |
| — static evaluator at the 32-ply cutoff | ~27–31 % |
| — win checks inside the playout | ~18–26 % |
| — playout stepping (pick/place) | ~25 % |
| PUCT selection | ~13–26 % |
| Expansion (priors) | ~2–12 % |
| `create_child` | ~2 % |
| Root tactical scans (win-in-1, M/F/O) | ~0.1 % |

So three things were worth attacking: the **win check** (called 4× per
playout ply), the **static evaluator** (once per simulation, a full-board
scan), and small per-simulation overheads (empty-cell lists, random draws,
the full-board draw test). The v2 root tactical layer turned out to be free
and was left alone.

## Measured result

Identical positions (same seed, same stone placements and edges), identical
simulation counter injected into both builds, full 4.7 s budget, medians of
three runs per position:

| Position | v1 (sims/s) | v2 (sims/s) | v3 (sims/s) | v4 (sims/s) | v3 vs v2 | v4 vs v3 |
| --- | --- | --- | --- | --- | --- | --- |
| empty board | 3,198 | 3,203 | 3,577 | 3,514 | **+11.7 %** | −1.8 % |
| early game (12 stones) | 2,168 | 1,784 | 1,877 | 1,591 | **+5.2 %** | −15.2 % |
| early game (12 stones, White) | 2,237 | 1,954 | 2,152 | 2,076 | **+10.1 %** | −3.6 % |
| **mean** | **2,534** | **2,314** | **2,536** | **2,394** | **+9.6 %** | **−5.6 %** |

> **Correction (5 Nov 2026).** An earlier version of this section claimed
> +43 %/+31 %/+14 % for v3 over v2, from single runs per position.  Repeating
> each position three times and taking medians shows the honest figure is
> **+9.6 % in mean sims/s**, not +31–43 %.  The single runs were dominated by
> run-to-run spread; the direction is right, but the magnitude was overstated.
>
> Two further points the repeated measurement makes clear.  First, v3 is **not**
> faster than v1 overall (**+0.1 %** mean): it gains ~12 % on the empty board
> but loses ~13 % on stone-heavy positions.  Second, v2's openness-aware
> evaluator and threat layer cost about 9 % speed, and v3's optimisations
> recovered exactly that — the optimisations did not exceed the pre-v2
> baseline.  "v3" therefore means *faster than v2, equal to v1*, not the
> headline speed-up once quoted.
>
> **v4 (the current build) is the correctness pass on top of v3**: exact
> diagonal threat lengths, exact threat counting, honest visit accounting,
> no rollout of finished games, and the tactical action filter at search
> nodes.  It gives up **5.6 %** mean sims/s against v3 — concentrated on
> stone-heavy positions (−15 %), almost free on the empty board (−1.8 %),
> because the new `expand` filter needs candidate cells adjacent to a stone.
> In exchange it wins **95.8 %** of 120 games against v3.  See
> `benchmarks/results/version_comparison.md` for the full head-to-head data.

(An older measurement quoting ~7,000 sims/s predates v2's open-four threat
layer and openness-aware evaluator; only the table above is apples-to-apples.)

### Why not more?

The remaining cost is irreducible *in pure Python* without changing the
algorithm: every simulation must play out up to 32 plies and evaluate the
leaf. The micro-optimisation wins came from removing per-call indirection
(function calls, list rebuilds) rather than from changing the algorithm —
which is exactly what the rules allow (only the three required functions may
be changed, no extra imports).

## Optimisation 1 — gated win check

`win_check` used to call `line_len` in all four directions for every stone
placed during a playout, even though the vast majority of placements have
**no same-colour neighbour** in most directions (the walk exits immediately,
but the call overhead remains). The fix is a precomputed **gate table** plus
inlined edge lookups:

```python
    # flat edge lists (for inlined lookups in the hot loops):
    #   ehF[r*(n_cols-1)+c] = edge (r,c)-(r,c+1)
    #   evF[r*n_cols+c]     = edge (r,c)-(r+1,c)
    ncm1 = n_cols - 1
    EH_SIZE = n_rows * ncm1
    ehF = np.asarray(state[1]).astype(bool).ravel().tolist()
    evF = np.asarray(state[2]).astype(bool).ravel().tolist()
    edges = ehF + evF                   # gate table indices point into this

    def _edge_idx(r, c, dr, dc):
        """Index into `edges` for the edge (r,c)->(r+dr,c+dc); -1 = diagonal."""
        if dr == 0 and dc == 1:
            return r * ncm1 + c
        if dr == 0 and dc == -1:
            return r * ncm1 + c - 1
        if dr == 1 and dc == 0:
            return EH_SIZE + r * n_cols + c
        if dr == -1 and dc == 0:
            return EH_SIZE + (r - 1) * n_cols + c
        return -1                       # |dr| == |dc| == 1: diagonal, always linked

    # per-cell gate table for the win check: for each of the four directions,
    # (dr, dc, fwd neighbour, fwd edge, bwd neighbour, bwd edge); -1 marks
    # out-of-bounds neighbours / diagonal directions (no edge needed)
    gates4 = []
    for r in range(n_rows):
        for c in range(n_cols):
            entries = []
            for dr, dc in DIRS4:
                fr, fc = r + dr, c + dc
                if 0 <= fr < n_rows and 0 <= fc < n_cols:
                    fnb, fe = fr * n_cols + fc, _edge_idx(r, c, dr, dc)
                else:
                    fnb, fe = -1, -1
                br, bc = r - dr, c - dc
                if 0 <= br < n_rows and 0 <= bc < n_cols:
                    bnb, be = br * n_cols + bc, _edge_idx(br, bc, dr, dc)
                else:
                    bnb, be = -1, -1
                entries.append((dr, dc, fnb, fe, bnb, be))
            gates4.append(tuple(entries))

    def win_check(bd, a):
        """Exact win check for the stone at flat index `a`, with a cheap
        neighbour gate: a direction can only hold a line of >= 2 stones if
        the immediate neighbour through a live connection is ours, so the
        full two-way walk runs only for directions that pass the gate."""
        color = bd[a]
        r, c = divmod(a, n_cols)
        for dr, dc, fnb, fe, bnb, be in gates4[a]:
            fwd = fnb >= 0 and bd[fnb] == color and (fe < 0 or edges[fe])
            bwd = bnb >= 0 and bd[bnb] == color and (be < 0 or edges[be])
            if (fwd or bwd) and line_len(bd, r, c, dr, dc) == 5:
                return True
        return False
```

The gate is **exact**, not an approximation: any five-in-a-row through `a`
must contain a same-colour neighbour of `a` along the line, joined by a live
connection. So a direction whose two adjacent cells fail the gate cannot hold
a win, and skipping it loses nothing.

## Optimisation 2 — inlined edge lookups in the static evaluator

`static_eval` runs once per simulation and scans every stone in four
directions, calling `conn()` at every start check, every walk step and every
playable-end check (hundreds of calls per evaluation). The rewrite inlines
the edge lookups using the flat edge lists:

```python
                # behind cell is the same colour: this is the start of a
                # line only if the connecting edge is missing
                if dr == 0:
                    if ehF[br * ncm1 + (bc if dc == 1 else bc - 1)]:
                        continue
                elif dc == 0:
                    if evF[(br if dr == 1 else br - 1) * n_cols + bc]:
                        continue
                else:
                    continue    # diagonal behind: always connected
```

The formula for $\Phi_x$ (and therefore $V(s)$) is unchanged — only the
plumbing is. Three traps in this transcription, each of which produced a
silently wrong evaluation before being caught by the property test:

1. **The start check must be "skip iff behind is the same colour AND
   connected"**, not "skip iff behind is different" (an inverted condition
   silently skips every run that starts in the open).
2. **Branch on both `dr` and `dc`.** An `elif dr == 1:` branch fires for the
   anti-diagonal `(1, -1)` too, turning "diagonals are always connected" into
   a vertical-edge lookup — the same bug that once broke `conn()`.
3. **An edge belongs to the left/upper cell.** The edge after a run is
   indexed by the run's last cell, not by the extension cell.

## Optimisation 3 — per-simulation bookkeeping

```python
    class Node:
        __slots__ = ('board', 'player', 'children', 'untried', 'prior',
                     'N', 'W', 'Q', 'terminal_value', 'empties')

    def create_child(node, a):
        r, c = divmod(a, n_cols)
        bd = node.board[:]              # copy; the parent's board is reused
        bd[a] = node.player
        child = Node(bd, -node.player)
        child.empties = [i for i in node.empties if i != a]
        if win_check(bd, a):
            child.terminal_value = -1.0                          # mover won
        elif not child.empties:
            child.terminal_value = 1.0 if child.player == WHITE else -1.0
        return child
```

- **Empty-cell lists are carried on the node** (`child.empties`), so the
  rollout no longer rescans all 221 cells, `expand` reuses the list instead
  of rebuilding it, and the **draw test becomes O(1)** (`not child.empties`)
  instead of `0 not in bd` over the whole board.
- **Random numbers are drawn in larger batches** (`RBN = max(4096, ...)`), so
  the generator is consulted roughly once per hundred rollouts instead of
  once per 32 plies.
- The root still stores the **full** empty set (`root.empties`) while its
  `untried` list may be the restricted $F \cup O$ set — the two serve
  different purposes (rollouts vs. action restriction).

## Correctness safeguards

The optimisations touch the two hottest correctness-critical routines, so they
are guarded by:

1. **A property test** comparing the optimised `win_check` and `static_eval`
   against a from-scratch reference implementation over random boards:
   **0 evaluation mismatches and 0 win-check mismatches in 3,676 checks**.
2. **The behavioural suites**: `check_result` cases, the diagonal regression
   (all four directions, `edgeV` all False and all True), the open-three /
   open-four threat suite (including the forced-win $M$ case and the
   double-threat resistance), the MCTS–vs–random games, and the self-play
   smoke test — all green.
3. **Timing compliance**: full-budget moves still land at ~4.72–4.75 s,
   inside the 5 s rule.

> **Status note.** These safeguards describe v3 as originally shipped. The
> review pass below changes the threat layer (`run_extent`,
> `four_threat_count`), the MCTS visit accounting, and the expansion filter,
> so the property test and behavioural suites **must be re-run against the
> updated code** before the "0 mismatches" figures are quoted again.

> **Lesson for the report.** The diagonal bug appeared *three* times in
> different disguises (`conn`, `_edge_idx`, and the inlined evaluator), each
> time because a branch tested `dr` alone instead of `dr` **and** `dc`. In
> this game variant the anti-diagonal must never be treated as a vertical
> edge. A single property test comparing against a reference implementation
> caught all three — much cheaper than debugging losses during play.

---

## Review fixes (post-v3 correctness pass)

A code review of the v3 search found six issues. All are fixed in
`project_skeleton.py`; the sections above have been updated to match. This
list is the authoritative changelog.

### 1. Diagonal run length was doubled (`run_extent`)

The run length was computed as the dot product
`(e_r - s_r) * dr + (e_c - s_c) * dc + 1`. That is exact for horizontal and
vertical steps, but for a diagonal step *both* coordinates advance by one per
step, so the result was `2L - 1` — a run of 3 was reported as 5. Since the
open-four layer only inspects runs with `L == 4`, **both diagonal
orientations were silently excluded from threat detection**: on an otherwise
empty board, an opponent's diagonal open three produced 0 threats instead of 2,
leaving all 218 empty cells as candidate replies.

The fix is the Chebyshev step count, `max(|e_r - s_r|, |e_c - s_c|) + 1`,
which equals the true number of steps in all four unit directions. The
diagonal examples had still "happened to" block in play, but that was MCTS
choosing the move, not the hard rule forcing it — protection that disappears
once competing attractions exist.

### 2. Threat count was not an exact count of winning moves (`four_threat_count`)

Two further defects in the contiguous-run counter:

- **It counted extensions that produce six stones.** After finding a four, it
  counted an empty connected endpoint without checking what lay beyond it. In
  `. X X X X . X`, filling the marked gap produces six, which the rules do not
  count, yet the helper reported a winning cell — and $M$ returns such a move
  as a "forced win".
- **It missed broken fours.** A pattern such as `X . X X X` has an immediate
  winning gap, but the counter only considered contiguous runs of length four
  and reported zero.

The replacement places the stone and **validates each candidate with
`win_check`** — the "count distinct, actual exact-five completion cells"
formulation. Overlines are rejected (a run of six does not satisfy
`line_len == 5`) and broken fours are caught (the gap cell completes five).
`run_extent` is retained only to add the classic straight-four extension cells
to the candidate set. This handles straight fours, broken fours and
intersecting threats consistently.

### 3. "Respond to an open three" is not always "block its endpoint"

The $F \cup O$ restriction deliberately admits our own forcing fours, which can
be correct — but admission to $F$ is **not proof that the continuation is
safe**: the deeper search used proximity priors and random rollouts rather than
consistently playing forced tactical replies. Two mitigations were applied:

- **Tactical priority at search nodes.** `expand` now filters a non-root
  node's action set: if the mover has an immediate win, only winning moves are
  kept; otherwise, if the opponent has one, only the blocking moves are kept.
  Candidates are restricted to cells adjacent to a stone (a five must include
  one), so the filter stays cheap. Forced reply sequences become real search
  lines.
- **Exact threat detection** (fix 2) makes the forcing moves themselves
  trustworthy, which is the prerequisite for the filter to help.

A full "threat-space" reply-sequence search remains the next step; the filter
above is the bounded-cost version.

### 4. MCTS visit counts were doubled

Visits were incremented during descent (`node.N += 1` in both the expansion and
selection branches) **and** at the leaf, **and** again during backup. Non-root
visits were therefore counted twice while values were accumulated once: one
measured search reported `root.N = 2242` but `sum(root child.N) = 4484`. This
inflates the denominator of $Q$ and the PUCT exploration term.

Fix: count the root once per completed simulation, and each traversed child
exactly once during backpropagation. The descent-time increments were removed.
The invariant $N(s) = \sum_b N(s,b)$ now holds.

### 5. Newly created terminal nodes were rolled out

`create_child` stamps `terminal_value`, but the descent broke out without
updating `leaf_is_terminal`, so the evaluation phase could run a random playout
from an already-finished game. Fix: consult the node's own state —

```python
if node.terminal_value is not None:
    v = node.terminal_value
else:
    v = rollout(node)
```

— so a forced win created during the descent is evaluated as the win it is.

### 6. Random-index `pop`, not swap-pop

`empties.pop(int(...))` shifts the remaining list; it is not swap-pop (an
earlier comment said otherwise and was corrected). An isolated microbenchmark on
Python 3.13 — 30 000 repetitions of removing 32 elements from a 221-element
list — measured **0.0874 s** for random-index pop versus **0.0995 s** for
swap-then-pop: swap-pop was about **14 % slower**. The current removal method is
kept; "optimising" it to swap-pop needs a full-engine benchmark first.

### 7. Evaluator ranking claim corrected

The notes previously said an open three "outweighs a closed four". The formula
$g(L,e) = (L-1)^2(1+\kappa e)$ at $\kappa = 1.5$ gives open 3 (16) versus a
**fully dead** four (9) — but a **one-ended** four is 22.5 and outranks the open
three. So the heuristic ranks threats correctly but is *not* a tactical
guarantee against a one-ended four; that is the hard rules' and the search's
job. The text above was corrected accordingly.

### Order of application (and a tuning caveat)

The review's recommended order was: fix diagonal lengths → make threat counts
exact → repair visit/terminal handling → add tactical replies deeper in search
→ optimise expansion and evaluation. **Retune MCTS parameters only afterwards**,
because the previous tuning was compensating for incorrect statistics. In
particular the earlier $C_{PUCT} = 1.4$ settlement was measured on the buggy
visit counts and threat layer, so its verdict should be re-run before being
trusted.
