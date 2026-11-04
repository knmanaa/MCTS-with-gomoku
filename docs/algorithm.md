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
-1, & \text{the move just played won the game (the player to move lost)}\\[2pt]
+1, & \text{draw and the player to move is White (a draw counts as a win for White)}\\[2pt]
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
$L_{+1}\bigl((8,4), (0,1)\bigr)$ is "how long is the horizontal black line
through $(8,4)$?" — the subscript is the colour, the second argument the
orientation. This is exactly how the two prior weight tables use it:
$w_{\text{ext}}\bigl(L_p(a,d)\bigr)$ scores extending **our** line through
$a$, and $w_{\text{blk}}\bigl(L_{-p}(a,d)\bigr)$ scores occupying a cell
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

---

## The node

```python
class Node:
    """Search-tree node.  Edge statistics (N, W, Q) are stored on the
    child and are kept from the perspective of the player who made the
    move leading to it (the parent's player to move)."""
    __slots__ = ('board', 'player', 'children', 'untried', 'prior',
                 'N', 'W', 'Q', 'terminal_value')

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
\underbrace{\text{occ}(a) = \bigl|\{\, j \in \mathcal{N}_8(a) : \text{board}(j) \neq 0 \,\}\bigr|}_{\text{stone proximity}}
$$

$$
\text{score}(a) \;=\; 0.6 \cdot \text{occ}(a)
\;+\; \sum_{d \in D} w_{\text{ext}}\bigl(L_p(a, d)\bigr)
\;+\; \sum_{d \in D} w_{\text{blk}}\bigl(L_{-p}(a, d)\bigr)
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
P(s,a) \;=\; \frac{\text{score}(a) + \varepsilon}{\sum_{a' \in \mathcal{A}(s)} \bigl(\text{score}(a') + \varepsilon\bigr)},
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
a^* \;=\; \arg\max_{a \in \mathcal{A}(s)} \Bigl( Q(s,a) + U(s,a) \Bigr),
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
leaf_is_terminal = False
while True:
    if node.terminal_value is not None:
        leaf_is_terminal = True
        break
    if node.children is None:            # first visit: expand
        expand(node, None, rich=False)
    if node.untried:                     # expansion phase
        node.N += 1
        a = node.untried.pop()
        child = create_child(node, a)
        node.children[a] = child
        path.append((node, a))
        node = child
        break
    node.N += 1                          # selection phase
    a = puct_select(node)
    path.append((node, a))
    node = node.children[a]
```

Four exits, in priority order:

1. **Terminal node** — $z(s)$ was stamped at creation; the exact score is
   known, no evaluation needed.
2. **First visit** — `children is None`: the state has never been seen, so
   build its action set and priors.
3. **Untried actions left** — the node is not fully expanded:
   `untried.pop()` takes the **highest-prior untried action** (the list is
   prior-ascending) — the prior-guided equivalent of
   `select_arbitrary_unvisited`. Create the child, record `(node, action)`
   in `path`, descend, and break: this fresh leaf is what gets evaluated.
   $N(s) \mathrel{+}= 1$ because the parent was visited by this simulation.
4. **Fully tried** — PUCT-select and keep descending. $N(s) \mathrel{+}= 1$
   appears in *both* branches 3 and 4 because $N(s)$ must count every
   simulation passing through — it is the $\sum_b N(s,b)$ under the square
   root.

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
z_{\text{draw}} = \begin{cases}+1 & \text{leaf's player to move is White}\\ -1 & \text{Black}\end{cases} & \text{board full}\\
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
with length $L \geq 2$:

$$
\Phi_x(s) \;=\; \sum_{\text{runs of } x \text{ with length } L \geq 2} (L-1)^2
$$

— quadratic in length, so a four-in-a-line is worth vastly more than a
two-line. Then

$$
\tilde V(s) \;=\; \frac{\Phi_{+1}(s) - \Phi_{-1}(s)}
                        {\bigl|\Phi_{+1}(s) - \Phi_{-1}(s)\bigr| + 8}
\;\in\; (-1, 1)
\qquad\quad
V(s) \;=\; \begin{cases}\tilde V(s) & \text{Black to move}\\ -\tilde V(s) & \text{White to move}\end{cases}
$$

```python
def static_eval(bd, for_player):
    """Static evaluation in [-1, 1] from `for_player`'s perspective:
    compares the squared lengths of all maximal connected lines."""
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
                if v == 1:
                    sb += (L - 1) ** 2
                else:
                    sw += (L - 1) ** 2
    score = sb - sw               # black-positive
    score /= (abs(score) + 8.0)   # squash to (-1, 1)
    return score if for_player == 1 else -score
```

The "skip" condition (the cell behind is the same colour **and connected**)
ensures each run is counted exactly once from its start; the
bounds-before-edge-lookup ordering in the `while` condition means `conn`
only ever touches an in-bounds cell's edge.

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
# evaluation phase
if leaf_is_terminal:
    v = node.terminal_value              # for the player to move here
else:
    v = rollout(node)
node.N += 1

# backpropagation: flip the sign at each level up the tree
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
that was visited.

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
\mathcal{W}_p \;=\; \bigl\{\, a \in \mathcal{A}(s_0) \;:\; \exists\, d \in D,\; L_p(a, d) = 5 \text{ with } a \text{ occupied by } p \,\bigr\}
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
# if the opponent can win next move, every other move loses: block
root_actions = theirs if theirs else empties0
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

The root then expands with the **rich** priors over the (possibly
restricted) action set, and the search runs:

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

They can be overridden from outside **without editing the code** (e.g.
`project_skeleton.MCTS_TIME_LIMIT = 0.15` for fast benchmark games); the
defaults apply when the names are not defined.
