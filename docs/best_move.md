# PUCT-MCTS — Algorithm, Mathematics, and Implementation

This guide develops the ideas behind `best_move(state, color)`: what the
search estimates, how its four phases cooperate, why its values change sign,
and how exact-five tactics and handcrafted heuristics adapt MCTS to this
Gomoku variant. Equations describe the algorithm; implementation notes show
how those equations map to the current `integration` working tree after the
15 November 2026 review. Measurements and proposed extensions are kept
separate from the algorithm itself.

The implementation is inside the function's existing TODO block in
[project_skeleton.py](<../project_skeleton.py>). For result detection and GUI
integration, see [check_result](<check_result.md>) and [main](<main.md>).

The algorithm is **PUCT Monte Carlo tree search without deep learning**:

| Component | Implementation |
| --- | --- |
| Policy prior | Handcrafted line-extension, blocking, proximity, and centre scores |
| Leaf value | Depth-capped uniformly random rollout, then a static line evaluator |
| Tree policy | Prior-guided progressive widening, then PUCT selection |
| Backup | Alternating-sign backup with one visit per traversed edge |
| Final decision | Most-visited root action; mean value breaks ties |
| Tactical exceptions | Exact immediate wins, necessary replies, and root fork handling |

Progressive widening is enabled by default. Tactical rollout policies, tree
reuse, and incremental value updates remain future work.

## Algorithm overview and notation

### Why Monte Carlo tree search?

A move can look strong locally and still lose after a forced reply. Searching
every possible continuation is infeasible: with `m` empty cells, an
unrestricted depth-`h` tree, for `0 <= h <= m`, can have

$$
\prod_{j=0}^{h-1}(m-j)
$$

ordered continuations. MCTS instead spends a finite computation budget on
sampled continuations, progressively concentrating effort where accumulated
evidence and domain knowledge suggest it is useful.

One simulation follows four phases:

1. **Selection:** descend through previously explored actions using PUCT.
2. **Expansion:** if the node has room under its widening schedule, admit the
   highest-prior unvisited action and create its child state.
3. **Evaluation:** use an exact terminal outcome, or a short random playout
   followed by a static estimate.
4. **Backpropagation:** send the result up the path, updating action statistics
   and changing perspective at every alternating turn.

This repeats until the budget is exhausted. The final real move uses the
root visit counts, not the exploratory selection score used inside the tree.

### State, action, and transition

For mathematical purposes, include the player to move in the state:

$$
s=(B,E_H,E_V,p),\qquad
B\in\{-1,0,+1\}^{R\times C},\qquad p\in\{+1,-1\}.
$$

`B` is the stone board, `E_H` and `E_V` are the fixed edge arrays, and `p`
is Black (`+1`) or White (`-1`). The external function passes these as
`state = (board, edgeH, edgeV)` and the separate `color` argument.

The legal action set and deterministic transition are

$$
\mathcal A(s)=\{(r,c):0\leq r<R,\ 0\leq c<C,\ B_{r,c}=0\},
$$

$$
f(s,a)=(B^{p,a},E_H,E_V,-p),\qquad
B^{p,a}_{r,c}=\begin{cases}
p,&(r,c)=a,\\
B_{r,c},&\text{otherwise}.
\end{cases}
$$

Edges influence which runs win, not where stones may be placed. A hypothetical
placement changes one board entry; it does not change connectivity.

| Symbol or term | Meaning |
| --- | --- |
| **ply** | One placement by one player; 32 plies means up to 16 placements per side |
| **run** | Consecutive same-colour stones along one connected straight axis |
| **maximal run** | A run that cannot extend through another connected same-colour stone at either end |
| `d` in `D` | One of the four canonical axes; scanning both signs covers all eight orientations |
| `p(s)` | Player to move in state `s` |
| `N(s)` | Node visit count used by PUCT, including a non-root node's initial evaluation visit |
| `N(s,a)` | Number of simulations traversing action `a` from `s` |
| `W(s,a)` | Sum of backed-up values for that action, from its mover's perspective |
| `Q(s,a)` | Mean backed-up value, `W(s,a) / N(s,a)` for visited actions |
| `P(s,a)` | Normalised heuristic prior assigned before sampling an action |
| `z_p(t)` | Exact terminal outcome of terminal state `t`, viewed by player `p` |
| `V_p(s)` | Nonterminal static value estimate, viewed by player `p` |
| **win-in-1** | An empty cell whose occupation completes a maximal connected run of exactly five |
| **fork** | A move creating at least two distinct win-in-1 cells, not necessarily a straight open four |

### The objective and the role of heuristics

Let `w(t)` be the winner of a terminal state, with `w(t) = -1` for a
full-board draw because White receives that game. Define

$$
z_p(t)=\begin{cases}
+1,&w(t)=p,\\
-1,&w(t)=-p.
\end{cases}
\qquad z_{-p}(t)=-z_p(t).
$$

White's draw advantage breaks colour symmetry but does not break the
zero-sum relationship between the players' utilities.
For an ideal exhaustive solver, value from the current mover's perspective
would obey the negamax recursion

$$
V^*(s)=\begin{cases}
z_{p(s)}(s),&s\text{ terminal},\\
\displaystyle\max_{a\in\mathcal A(s)}[-V^*(f(s,a))],&\text{otherwise}.
\end{cases}
$$

MCTS does not evaluate this entire recursion. For a visited action it builds
an empirical estimate

$$
W(s,a)=\sum_{j=1}^{N(s,a)}G_j(s,a),\qquad
Q(s,a)=\frac{W(s,a)}{N(s,a)},
$$

where each `G_j` is a sampled return converted to the mover's perspective.
As the tree grows, those samples come from changing search continuations;
they are not independent samples from a fixed optimal-play distribution.
A depth-capped heuristic rollout can also be biased. Thus `Q` is a search
estimate, not a certified minimax value or calibrated win probability.

The three sources of guidance have distinct jobs:

- **Exact tactics** prove local wins or necessary responses.
- **The prior `P`** tells the tree which moves to investigate first.
- **The evaluator `V`** estimates how a nonterminal sampled continuation looks.

PUCT can be used without a neural network: replacing a learned policy and
value with handcrafted priors and rollout evaluation preserves its selection
and backup structure, not any claim of AlphaZero-level strength.

## 1. Rules and interface

The input is `state = (board, edgeH, edgeV)` and `color` is `+1` for Black or
`-1` for White. Dimensions come from `board.shape`, not hardcoded board sizes.
An action is any empty point; internal actions use `a = r * n_cols + c`.
The result is a coordinate tuple `(r, c)`. The input arrays are not modified.

Winning means a **maximal connected run of exactly five** same-colour stones:

| Direction `(dr, dc)` | Connectivity |
| --- | --- |
| `(0, 1)` | Horizontal edge at the left endpoint |
| `(1, 0)` | Vertical edge at the upper endpoint |
| `(1, 1)` | Always connected diagonally |
| `(1, -1)` | Always connected diagonally |

### Connectivity and exact-five mathematics

For adjacent points `a = (r,c)` and `b = (r',c')`, define the connection
indicator; out-of-bounds or nonadjacent pairs have value zero:

$$
\chi(a,b)=\begin{cases}
E_H[r,\min(c,c')],&r=r',\ |c-c'|=1,\\
E_V[\min(r,r'),c],&c=c',\ |r-r'|=1,\\
1,&|r-r'|=|c-c'|=1,\\
0,&\text{otherwise}.
\end{cases}
$$

For a cell `a` occupied by `p`, let `m_+(s,a,d)` and `m_-(s,a,d)` count
consecutive same-colour stones reachable in directions `+d` and `-d`, stopping
at an empty/opponent cell, a board boundary, or a missing connection. Then

$$
D=\{(0,1),(1,0),(1,1),(1,-1)\},\qquad
L_p(s,a,d)=1+m_+(s,a,d)+m_-(s,a,d),
$$

$$
\operatorname{win}(s,a,p)
\iff \exists d\in D:\ L_p(s,a,d)=5.
$$

For an empty cell, evaluate this expression on its hypothetical placement
state `s^{p,a} = (B^{p,a}, E_H, E_V, -p)`. In prior formulas below,
`L_p(a,d)` abbreviates this hypothetical-placement run length.

**Maximality is essential:** a connected six contains five-cell subsegments,
but none is an entire maximal run. Missing horizontal/vertical edges can
split six geometrically adjacent stones into separate runs, one of which
may have length five. A gap likewise prevents an unbroken run.

If a unit-step run has extremes `u = (r_s,c_s)` and `v = (r_e,c_e)`, its length is

$$
L=1+\max(|r_e-r_s|,|c_e-c_s|).
$$

This Chebyshev step count works on all four axes. The tempting dot-product
expression `(r_e-r_s)*dr + (c_e-c_s)*dc + 1` instead gives `2L-1` on a diagonal;
it is a historical bug explanation, not a formula to use for run length.

A full board without a winning line is returned as a draw by the detector
but **awarded to White** by the search and GUI.

Callers should stop after a terminal state. If no empty point exists,
`best_move` now raises `ValueError` rather than returning occupied `(0, 0)`.
If exactly one empty point remains, it returns that point directly. The
function does not perform a full-board terminal scan before searching.

### Diagonal interpretation

Both diagonal orientations remain connected even if every horizontal and
vertical edge is disabled. The integration review found no current incorrect
diagonal-win result. Earlier implementations had two separate diagonal bugs:
branching on `dr` alone treated a diagonal as vertical, and a dot-product run
length counted diagonal steps twice. The current step-table/window approach
avoids both calculations in the tactical hot path.

An opponent's open diagonal four with **two actual exact-five completion
cells** is already a forced loss if we have no immediate win: occupying one
completion leaves the other available. That outcome is not itself evidence
of a missed diagonal block. Always distinguish it from a one-ended four.

## 2. Budget and configuration

```python
t0 = time.perf_counter()
TIME_LIMIT = min(4.7, max(0.0, float(globals().get('MCTS_TIME_LIMIT', 4.7))))
C_PUCT = globals().get('MCTS_C_PUCT', 1.4)
ROLLOUT_DEPTH = globals().get('MCTS_ROLLOUT_DEPTH', 32)
KAPPA = globals().get('MCTS_KAPPA', 1.5)
PW_MIN = globals().get('MCTS_PW_MIN', 6)
PW_C = globals().get('MCTS_PW_C', 2.0)
REUSE = bool(globals().get('MCTS_REUSE', False))
TACTIC_EPS = float(globals().get('MCTS_TACTIC_EPS', 1.0))
GAP4_W = float(globals().get('MCTS_GAP4_W', 30.0))
deadline = t0 + TIME_LIMIT
```

| Module-level override | Default | Meaning |
| --- | --- | --- |
| `MCTS_TIME_LIMIT` | `4.7` | Soft total budget, clamped to `[0, 4.7]` seconds |
| `MCTS_C_PUCT` | `1.4` | PUCT exploration coefficient |
| `MCTS_ROLLOUT_DEPTH` | `32` | Maximum rollout plies before static evaluation |
| `MCTS_KAPPA` | `1.5` | Playable-end weight in the static evaluator |
| `MCTS_PW_MIN` | `6` | Minimum admitted children per expanded node |
| `MCTS_PW_C` | `2.0` | Square-root growth coefficient for widening |
| `MCTS_REUSE` | `False` | Enable module-local subtree reuse when the exact reply was explored |
| `MCTS_TACTIC_EPS` | `1.0` | Probability a rollout ignores immediate-win/block tactics (`1.0` = off) |
| `MCTS_GAP4_W` | `30.0` | Static-evaluator weight for an internal-gap broken four such as `XX.XX` |

The widening defaults were selected from paired experiments summarised in
[Section 9](<#9-integration-review-measured-results-and-verification>). They
are implementation defaults, not a claim of global optimality. A larger
full-budget validation is reported later in that section.

Overrides can be assigned by an external caller without editing the function.
Use suitable numeric values and a nonnegative integer rollout depth. The
exploration coefficient and evaluator weights have not been retuned after
this review.

The timer starts before setup, so preprocessing uses part of the budget.
However, preprocessing and individual simulations are not preempted; the
outer loop checks `keep_searching()` between simulations. The 4.7-second
budget leaves nominal margin below the five-second project limit, but is
**not a strict runtime guarantee** for arbitrary board sizes, pauses, or
configurations. A zero budget still performs setup/tactical checks and returns
a tactical move or the highest-prior fallback.

### Cooperative GUI cancellation

`best_move` reads the optional private module hook `_MCTS_EVENT_POLL`.
When present, `keep_searching()` invokes it at simulation boundaries, no more
often than approximately every 30 ms. A truthy return ends the search early.
The ordinary legal-move fallback still applies if no simulation completed.

The search itself does not consume Pygame events or require a display.
[main](<main.md>) temporarily supplies the hook, discards AI-turn clicks,
honours QUIT, and restores the previous hook even on exceptions. Polling is
cooperative: an expensive setup or single simulation can delay it.

## 3. Precomputed geometry and exact win checks

The board and edge arrays are copied into Python-native flat lists for scalar
hot-loop access. Geometry is rebuilt for each call; it is not cached across
real moves.

| Structure | Purpose |
| --- | --- |
| `gates4` | Initial neighbour and edge-index descriptions for the four axes |
| `steps4[a][d]` | `(forward, backward)` connected neighbour indices; `-1` means boundary or missing edge |
| `neighbors8[a]` | In-bounds geometric neighbours for proximity scores and tactical candidate generation |
| `connected_lines` | Maximal geometric lines, split at missing horizontal/vertical edges |
| `windows5` | Every length-five segment within a connected line, plus its connected predecessor/successor |

`neighbors8` is geometric proximity, not proof of a live horizontal/vertical
connection. Conversely, missing all H/V edges does **not** isolate a cell
from diagonal play.

`line_len` walks `steps4` in both directions to measure a maximal same-colour
run through a hypothetical placement. Root priors use its exact length.
`win_check(bd, a)` uses the same step table directly and returns true iff one
axis through the occupied cell `a` has maximal length five. It stops exploring
an axis once the count exceeds five; an overline on that axis cannot win,
but other axes are still checked.

This last-move-only check is sufficient when the incoming search state has
no earlier winner: any newly created winning line must include the newest
stone. The full-board detector remains independent; see
[the interface comparison](<check_result.md#interface-with-the-search-win-check>).

## 4. Exact tactical targets from five-cell windows

`tactical_targets(bd, pl)` returns `(wins, targets)`:

- `wins`: sorted immediate exact-five completion cells for `pl`.
- `targets[a]`: distinct future winning cells obtained by the two-empty
  five-windows containing `a`. Under the root threat layer's precondition of
  **no existing immediate wins for `pl`**, these are exactly the immediate
  winning cells after `pl` hypothetically occupies `a`.

For each connected five-window:

1. Reject it if its connected predecessor or successor already holds `pl`;
   filling the window would produce an overline.
2. Reject it if an opponent stone occurs inside.
3. Four own stones plus one empty cell give an immediate winning cell.
4. Three own stones plus two empty cells `a, b` give the symmetric relation
   `targets[a].add(b)` and `targets[b].add(a)`.

Sets avoid counting the same completion cell twice when multiple windows or
axes describe it. This handles straight, broken, and intersecting threats
without repeatedly placing every candidate and rescanning its neighbours.
It replaces the old `run_extent`/`four_threat_count` helpers.

### Winning cells, forcing moves, and fork proofs

The immediate winning-cell set is

$$
\mathcal W_p(s)=\{a\in\mathcal A(s):
\exists d\in D,\ L_p(s^{p,a},a,d)=5\}.
$$

When `W_p(s)` is empty, the move at `a` creates the completion set and count

$$
T_p(s,a)=\mathcal W_p(s^{p,a}),\qquad
t_p(s,a)=|T_p(s,a)|.
$$

Suppressing the fixed root state in the notation, the useful tactical sets are

$$
M=\{a:t_p(a)\geq2\},\qquad
F=\{a:t_p(a)\geq1\},\qquad
O=\{a:t_{-p}(a)\geq2\}.
$$

`M` contains our fork-launch moves, `F` our forcing-four moves, and `O` the
opponent's fork-launch moves. A straight open four is the familiar example
of two winning cells, but the set-based definition also includes intersecting
and broken threats. A geometrically open run should not be declared a fork
without checking its exact-five completion cells and overline boundaries.

**Why does a fork force a win?** Assume neither player can win immediately
before our move. After `a` in `M`, the opponent cannot win in one placement:
adding our stone cannot create a new line of their colour. Nor can their
single placement remove two distinct empty completion cells of our colour.
At least one winning cell therefore remains for our next turn. The argument
uses distinct completion cells, not the number of windows describing them.

**Why restrict immediate-threat replies?** If we have no immediate win and
the opponent has a winning cell `b`, placing elsewhere cannot remove the
opponent's stones or change fixed edges. The exact-five at `b` remains
available. If there are several distinct such cells, occupying just one
cannot stop all of them; retaining those cells is resistance, not proof of
survival.

### Root tactical cascade

Let `p` be our colour. Write $\mathcal W_p(s)$ as `W_p` in the following
steps, and let `T_p(a)` denote the completion set `targets[a]` under the
precondition above.

1. If `W_p` is nonempty, return an immediate win.
2. Otherwise, if `W_-p` is nonempty, restrict root actions to those cells.
   With several independent opponent winning cells, blocking one need not
   save the position; the search is choosing resistance in an already-lost
   immediate-threat position.
3. With neither side winning immediately, if an own move has
   `|T_p(a)| >= 2`, return that fork. The opponent has no immediate win to
   race with and cannot occupy both distinct completion cells in one reply.
4. Otherwise, handle opponent fork-launch moves as described below.
5. If there are no opponent forks, retain all empty cells.
6. Return a single remaining action directly; otherwise run PUCT-MCTS.

### Correct quiet defences against forks

For each opponent fork-launch cell `a`, define its quiet defence set:

$$
D(a) = \begin{cases}
\{a\} \cup T_{-p}(a), & |T_{-p}(a)| = 2, \\
\{a\}, & |T_{-p}(a)| > 2.
\end{cases}
$$

Occupying the launch cell prevents the fork. Occupying one of exactly two
completion cells also prevents a two-winning-cell fork. When there are more
than two, blocking a single completion still leaves a fork.

The quiet moves that neutralise **every** current opponent fork are

$$
S = \bigcap_{a:\,|T_{-p}(a)|\geq 2} D(a).
$$

Also keep our forcing moves

$$
F = \{a : |T_p(a)| \geq 1\},
$$

because they create an immediate winning threat and gain tempo. The root
searches `sorted(S | F)`. Being in `F` is not proof that the continuation is
safe; search must still verify the forced replies. If `S | F` is empty, the
code retains opponent launch cells as legal resistance in the losing
position, not as a claim of a successful defence.

**Corrected pruning claim:** the old restriction to forcing moves and
opponent launch cells alone was unsound. On a connected one-row board
`[0, -1, -1, 0, -1, 0]`, Black can prevent White's next-move fork by playing
column `0`, `3`, or `5`. The old launch-cell-only rule admitted only `3` and
excluded the two endpoint defences. This demonstrates omitted legal
anti-fork defences, not that either endpoint uniquely wins the game.

The full fork analysis is root-only. At deeper nodes, `expand` uses the
cheaper tactical priority: search our immediate wins if any; otherwise
opponent immediate-winning cells if any; otherwise all legal actions.
Candidates for those immediate checks need only be adjacent to a stone,
but the checks themselves still use exact connectivity and exact-five rules.

## 5. Nodes, priors, and expansion

Each `Node` stores a flat board copy, the player to move, the full empty-cell
list, children, untried actions, priors, terminal value, and visit/value
statistics. `__slots__` avoids a per-node attribute dictionary.

`create_child` copies the board, places the mover's stone, removes its index
from the empty list, and assigns a terminal value when appropriate. Root
`empties` always contains **all** empty cells, even when the root action set
is tactically restricted; rollouts must still be allowed to use them.

### Root prior

The prior is a deterministic, handcrafted policy estimate, not a learned
probability of winning. Let $\mathcal N_8(a)$ be the in-bounds geometric
eight-neighbourhood. Its occupied-neighbour count is

$$
\operatorname{occ}(s,a)=
\left|\{j\in\mathcal N_8(a):B_j\ne0\}\right|.
$$

Proximity encourages local play without permanently forbidding distant
moves. The weak centre preference is

$$
h(r,c) = 1 - \frac{|2r-R+1|}{\max(1,R-1)}
       + 1 - \frac{|2c-C+1|}{\max(1,C-1)}.
$$

Here `R` and `C` are board dimensions and `h` is the centre score. Combining
proximity, centrality, attacking potential, and defensive urgency gives

$$
\operatorname{score}(s,a) = 1 + 0.1h(r,c) + 0.6\operatorname{occ}(s,a)
+ \sum_{d\in D}w_{\rm ext}(L_p(a,d))
+ \sum_{d\in D}w_{\rm block}(L_{-p}(a,d)).
$$

Each length is measured after hypothetically placing the corresponding
colour at `a`, then undoing that scratch-board placement.

The current weight functions are

$$
w_{\rm ext}(L)=\begin{cases}
2,&L=2,\\
8,&L=3,\\
30,&L=4,\\
0,&\text{otherwise},
\end{cases}
\qquad
w_{\rm block}(L)=\begin{cases}
1,&L=2,\\
4,&L=3,\\
20,&L=4,\\
0,&\text{otherwise}.
\end{cases}
$$

The first sum asks, "how much would this move extend our connected runs?"
The second asks, "how attractive would the same point be to our opponent?"
Attacking and blocking scores therefore refer to two different hypothetical
placements, not to both colours occupying the point simultaneously.
Immediate wins are handled before these heuristic weights are needed.

The positive baseline leaves distant legal moves searchable. The small
centre preference removes the old bottom-right empty-board tie bias;
it does not force the final searched move to be central.
Deeper nodes use a cheaper proximity-only score:

$$
\operatorname{score}_{\rm cheap}(s,a)=1+0.5\operatorname{occ}(s,a).
$$

This is an allocation-of-computation choice: computing richer priors at every
new node may consume the budget that could otherwise evaluate continuations.
The root gets a detailed initial ranking because all simulations share it;
deeper nodes start with a cheaper estimate and learn from sampled returns.

For either prior, normalise over the node's retained action set `A`:

$$
P(s,a) = \frac{\operatorname{score}(a)+\varepsilon}
{\sum_{b\in A}\operatorname{score}(b)+|A|\varepsilon},
\qquad \varepsilon=10^{-9}.
$$

```python
total = sum(scores) + len(scores) * 1e-9
node.prior = {a: (s + 1e-9) / total for a, s in zip(actions, scores)}
```

The denominator is the sum of **all** numerators, so

$$
P(s,a)>0,\qquad \sum_{a\in A}P(s,a)=1.
$$

The epsilon is a numerical safeguard; with the present positive score
baseline it is not the main source of exploration. If every raw score were
zero, the same expression would give the uniform prior `1 / |A|`.

Previously the denominator added epsilon only once while every numerator
added it. With all-zero scores on a 221-cell empty board, every prior became
`1`, so their sum was `221` instead of `1`. The corrected denominator
normalises even an all-zero score list.

Actions are sorted ascending by score so `untried.pop()` admits the
highest-prior unvisited action first.

### Prior-guided progressive widening

Expanding every action before revisiting any child spends a large fraction of
the budget broad and shallow: a node with `m` actions receives `m` first visits
before its most promising action can be revisited. Progressive widening instead
allocates visits to breadth only as evidence accumulates.

Let `A(s)` be the node's retained action set, `U(s)` its untried actions, and
`C(s)` its admitted children. Define the integer width

$$
w(s)=\min\left\{|A(s)|,\;
\max\left[\,k_{\min},\;
\left\lfloor k\sqrt{N(s)+1}\right\rfloor\,\right]\right\},
$$

where `k_min = MCTS_PW_MIN` and `k = MCTS_PW_C`. On each descent through a
nonterminal node:

1. If `|C(s)| < w(s)` and `U(s)` is nonempty, remove the highest-prior action
   from `U(s)`, create its child, evaluate that fresh leaf, and stop descending.
2. Otherwise, choose among the admitted children with PUCT.

The square-root schedule has the usual exploration curve: small visit counts
buy depth, while later visits gradually widen to the remaining actions. The
floor at `k_min` guarantees that a newly reached node has several candidate
replies instead of a single deterministic branch. No action is permanently
removed: it waits in `U(s)`.

For PUCT at a widened node, renormalise the priors over the children actually
admitted:

$$
\bar P(s,a)=\frac{P(s,a)}
{\displaystyle\sum_{b\in C(s)}P(s,b)},
\qquad a\in C(s).
$$

Without this renormalisation, the exploration bonus would shrink merely because
many untried actions had not yet been admitted. Once all actions are admitted,
`Pbar = P`. The implementation stores the denominator as `node.prior_mass`
and updates it whenever a child is admitted.

```python
def widening_width(node):
    return min(len(node.prior),
               max(PW_MIN, int(PW_C * (node.N + 1) ** 0.5)))

def can_widen(node):
    return bool(node.untried) and len(node.children) < widening_width(node)

# descent:
if can_widen(node):
    a = node.untried.pop()        # highest prior, prior-ascending list
    child = create_child(node, a)
    node.children[a] = child
    node.prior_mass += node.prior[a]
else:
    a = puct_select(node)
```

The root obeys the same schedule, but its final choice is necessarily among
**admitted** root children. Consequently, a very small time budget trades
broader root coverage for deeper lines. The tactical root rules are evaluated
before widening and are unaffected. At the default budget, the width at the
root is normally much larger than `k_min`.

Progressive widening is an allocation rule, not a pruning proof. It assumes
that low-prior actions are less urgent on average; the exact threat layer
remains responsible for proving immediate wins, necessary replies, and forks.

## 6. Selection and value conventions

Edge statistics are stored on the child:

| Field | Meaning |
| --- | --- |
| `N` | Number of simulations traversing the incoming edge |
| `W` | Sum of backed-up values from the incoming mover's perspective |
| `Q = W / N` | Mean value from that mover's perspective |
| `terminal_value` | Exact outcome from this node's player-to-move perspective |

### Selection: exploitation plus guided exploration

At a fully expanded node, choose

$$
a^*=\underset{a\in A(s)}{\arg\max}\,[Q(s,a)+U(s,a)],
\qquad
U(s,a)=c_{\rm puct}\,P(s,a)\,
\frac{\sqrt{N(s)}}{1+N(s,a)}.
$$

Each term has a separate interpretation:

- **`Q(s,a)`: exploitation.** Prefer actions whose sampled continuations have
  been favourable to the player making the action.
- **`P(s,a)`: prior guidance.** For equal values and visits, investigate a
  more promising handcrafted move more readily.
- **`sqrt(N(s))`: growing exploration pressure.** As more computation is spent
  at the parent, an action whose visit count stays small gets more pressure
  to be revisited.
- **`1 + N(s,a)`: diminishing exploration bonus.** Repeatedly sampling an
  action reduces its bonus relative to less-visited siblings.
- **`c_puct`: the tradeoff.** Larger values favour prior-guided exploration;
  smaller values lean more heavily on current empirical values. Its useful
  scale depends on correctly normalised priors and consistent value ranges.

For example, with `N(s)=100` and `c_puct=1.4`, suppose two visited actions have

| Action | Prior | Visits | Mean value | Exploration bonus | Selection score |
| --- | ---: | ---: | ---: | ---: | ---: |
| `a` | 0.20 | 4 | 0.10 | `1.4 * 0.20 * 10 / 5 = 0.56` | 0.66 |
| `b` | 0.10 | 19 | 0.55 | `1.4 * 0.10 * 10 / 20 = 0.07` | 0.62 |

Even though `b` currently has the better mean, `a` is selected to gather more
evidence. Other children may account for the remaining prior mass and visits.
PUCT is the **search-time allocation rule**; it is not the final move rule.
Its bonus here is not a formally calibrated statistical confidence interval.

The current implementation handles unvisited actions separately, expanding
the highest-prior untried action before PUCT is used at that node. Therefore
its formula need not define `W/N` at zero visits. On each simulation, descent
ends at a terminal node or at the one newly created child.

### Terminal values: state perspective versus edge perspective

A node represents the state **after** its incoming move. Its `player` is the
next player, whereas its incoming `Q` describes the player who just moved.
These are opposite perspectives. In particular,

$$
z(s)=\begin{cases}
-1,&\text{the incoming mover completed five},\\
+1,&\text{board full without five and }p(s)=-1,\\
-1,&\text{board full without five and }p(s)=+1.
\end{cases}
$$

This `z(s)` is stored as `terminal_value` from the **player-to-move**
perspective. An incoming winning move therefore gives a child value of `-1`,
but a parent-edge return of `+1` after the sign flip. A win is checked before
a full-board draw, including on the last cell. Newly created terminal nodes
are recognised immediately; finished games must never be randomly continued.

### Backpropagation: the negamax sign change

Let a sampled path be

$$
s_0\xrightarrow{a_0}s_1\xrightarrow{a_1}\cdots
\xrightarrow{a_{\ell-1}}s_\ell.
$$

If evaluation returns `v_ell` from player `p(s_ell)`'s perspective, then the
return assigned to the edge leaving `s_t` is

$$
G_t=(-1)^{\ell-t}v_\ell,\qquad 0\leq t<\ell.
$$

Every alternating move changes whose interests are being measured. The
updates, performed from the leaf back to the root, are

$$
N(s_t,a_t)\leftarrow N(s_t,a_t)+1,\qquad
W(s_t,a_t)\leftarrow W(s_t,a_t)+G_t,\qquad
Q(s_t,a_t)\leftarrow\frac{W(s_t,a_t)}{N(s_t,a_t)}.
$$

Equivalently, initialise `v = v_ell`, negate it before updating the incoming
leaf edge, and negate it again at each preceding edge. This is exactly the
perspective change in the negamax recursion, not an extra penalty for depth.

The corresponding code is:

```python
root.N += 1
v = -v
for pnode, a in reversed(path):
    ch = pnode.children[a]
    ch.N += 1
    ch.W += v
    ch.Q = ch.W / ch.N
    v = -v
```

**Two-edge sign example.** Suppose the path is Black-to-move root →
White-to-move child → Black-to-move leaf, and the rollout value at the leaf
is `+0.6` for Black. The lower edge was a White move, so it receives `-0.6`;
the root edge was a Black move, so it receives `+0.6`. The same sequence works
for exact terminal values `+1` and `-1`.

No visits are added during descent. The root is counted exactly once per
completed simulation:

$$
N(s_0)=\sum_{a\in A(s_0)}N(s_0,a).
$$

At a non-root node, `node.N` also counts its first incoming visit, when the
node may have been evaluated without traversing any outgoing edge. More
generally, if `E(s)` counts visits evaluated at the node itself,

$$
N_{\rm incoming}(s)=E(s)+\sum_{a\in A(s)}N(s,a).
$$

Consequently, the root equality must not be asserted for every node.
Historical versions incorrectly incremented non-root visits during both
descent and backup while adding values only once, shrinking `Q` and changing
the exploration scale. Counting each traversed edge once retains the
intended empirical mean.

## 7. Rollout and static evaluation

### A truncated Monte Carlo return

At a nonterminal leaf `s_ell`, write `p_ell = p(s_ell)` and let `H` be the
rollout depth cap. The default rollout policy samples uniformly:

$$
\rho(a\mid s)=\frac{1}{|\mathcal A(s)|},\qquad a\in\mathcal A(s).
$$

Starting at the leaf, alternate placements drawn from `rho`. Let `s_cut` be
the state after `H` plies if no terminal occurs first. The returned value is

$$
G_H=\begin{cases}
+1,&\text{a winning run is reached and its colour is }p_\ell,\\
-1,&\text{a winning run is reached and its colour is }-p_\ell,\\
+1,&\text{board fills without five and }p_\ell=-1,\\
-1,&\text{board fills without five and }p_\ell=+1,\\
V_{p_\ell}(s_{\rm cut}),&\text{no terminal after }H\text{ plies}.
\end{cases}
$$

Every branch is expressed from the **original leaf player's** perspective,
regardless of whose turn it is at the cutoff. This is why the evaluator
receives `node.player`, not the rollout's next `to_move`.

One rollout supplies one sample of the truncated-policy value

$$
V_H^\rho(s_\ell)=\mathbb E_\rho[G_H\mid s_\ell].
$$

The full tree does more than estimate a fixed random policy: its upper
levels increasingly choose actions with PUCT and exact tactical filters.
Nevertheless, the leaf estimate is still noisy and heuristic-biased. An
uncapped random game is not automatically a better estimate of optimal play:
it costs more and may average over strategically implausible replies.
The cap trades continuation detail against the number of simulations that
fit the budget; a useful depth should be benchmarked, not assumed.

### Sampling efficiently without changing the uniform policy

Rollouts copy the node board and empty list, alternate colours, and sample
uniformly from remaining empties for at most `ROLLOUT_DEPTH` plies.
Random numbers are batched using `RBN = max(4096, ROLLOUT_DEPTH + 8)`.
Every placement is checked with `win_check` before any draw result.

Removal now uses true swap/pop:

```python
k = int(next_rand() * len(empties))
a = empties[k]
empties[k] = empties[-1]
empties.pop()
```

This avoids the linear tail shift of `pop(k)` while preserving uniform
sampling from the remaining cells. It changes seeded playout sequences.
Constant-time removal does not, by itself, prove a Python wall-clock speedup;
the benchmark below measures the combined engine changes, not this operation
in isolation. Earlier isolated benchmarks favoured random-index pop, so
neither historical nor current results justify a universal microbenchmark
claim about swap/pop.

A rollout returns an exact signed win when terminal. A full board without
five returns `+1` for a White leaf player or `-1` for Black, **including when
the last allowed rollout ply fills the board**. Previously that boundary case
fell through to static evaluation.

### Static value from a census of connected runs

At a nonterminal cutoff, approximate positional advantage by counting
maximal same-colour runs. Let `R_x(s)` denote the runs of colour `x`, and
let `e` in `{0,1,2}` count their **playable ends**: empty in-bounds points
immediately beyond the run, connected by a live edge or a diagonal step.
A missing edge closes an end even if the next geometric point is empty.

Define a run's contribution and the total potential of one colour by

$$
g(L,e)=\begin{cases}
(L-1)^2(1+\kappa e),&2\leq L\leq5,\\
0,&\text{otherwise},
\end{cases}
\qquad
\Phi_x(s)=\sum_{\lambda\in\mathcal R_x(s)}g(L_\lambda,e_\lambda).
$$

The explicit range matters. Lone stones have no run contribution, and a
permanent overline must not be rewarded as more powerful than an exact five.
An overline contributes zero **on its own axis**; its stones may still form
shorter useful runs on other axes. Exact terminal detection takes precedence
over this nonterminal heuristic.

In the implementation, scan `connected_lines` and count consecutive colour
segments. Geometry is already connected, so this census needs no per-step
coordinate arithmetic or edge lookups. Splitting a geometric line at a
missing edge also makes the playable-end rule natural.

### Internal-gap broken fours

The run census above misses a key one-move pattern: `XX.XX` is two short
runs, not one four. Yet placing at the gap completes an exact five. The same
applies to `X.XXX` and `XXX.X`. Define `G4_x(s)` as the number of pairs of
maximal same-colour runs of colour `x` on one connected line that are
separated by **exactly one** empty cell and whose lengths sum to four:

$$
G4_x(s)=\left|\{(\lambda_1,\lambda_2):
\text{same colour }x,\ \text{one empty between them},\
L_{\lambda_1}+L_{\lambda_2}=4\}\right|.
$$

The static evaluator adds

$$
\Phi_x(s)\;+\!=\;\Gamma_4\,G4_x(s),\qquad \Gamma_4=\texttt{MCTS\_GAP4\_W}=30.
$$

This is deliberately computed during the same run scan: only adjacent runs
with one empty between them are paired, so the cost is linear in the line
length and there is no separate five-window pass. An internal-gap broken four
is an immediate win-in-1 and receives a value comparable to an open four
(36), but the exact tactical layer still has priority.

If filling the gap would produce six stones, the two runs contain five
stones in total (or the adjacent run continues through a connected stone), so
`G4_x` does **not** count it. If a missing edge separates the runs, they are
not in the same `connected_line`, so they are not paired. End-gap fours such
as `XXXX.` are already handled by the ordinary four-run term and are not
double-counted here.

The root prior also rewards moves that create a win-in-1 target
(`my_targets`) and moves that deny such a target to the opponent
(`their_targets`) with weights 20 and 15 respectively. This gives
gap-creating setups such as placing at one end of `X.XX` a normal chance in
the search even before the exact threat becomes a terminal position.

### Why squared length and openness?

The two factors express different approximations:

- **`(L-1)^2`: nonlinear urgency.** A two, three, and four have base weights
  `1`, `4`, and `9`, rather than nearly equal weights. Extending a nearly
  complete line should matter more than adding the same stone to a short
  line. Subtracting one keeps isolated stones at zero.
- **`1 + kappa * e`: opportunity to extend.** At fixed length, two playable
  ends normally offer more options than one, which offers more than none.
  This factor is linear in end count and multiplicative with run strength;
  it is not a claim of a measured win probability.
- **Maximal runs:** counting only maximal segments prevents counting the
  same four as several shorter runs along its own axis.
- **Connectivity:** no feature should gain strength from a nonexistent H/V
  edge; diagonal features remain valid independently of those edges.

At `kappa = 1.5`, the hierarchy is

| Run | Length `L` | Playable ends `e` | Contribution `g(L,e)` |
| --- | ---: | ---: | ---: |
| Dead two | 2 | 0 | 1 |
| Open two | 2 | 2 | 4 |
| Dead three | 3 | 0 | 4 |
| Fully dead four | 4 | 0 | 9 |
| Open three | 3 | 2 | 16 |
| One-ended four | 4 | 1 | 22.5 |
| Open four | 4 | 2 | 36 |
| Overline | 6 or more | Any | 0 on that axis |

An open three outweighs a **fully dead** four, not a one-ended four. The
ordering is useful but not a tactical guarantee: an empty reachable endpoint
might merge into a same-colour run beyond the gap and create an overline,
and run features do not fully represent broken patterns. Exact completion
sets, not the static weights, decide immediate wins and forks.

### Squashing and player perspective

First compute Black's raw advantage, then normalise its scale:

$$
\Delta(s)=\Phi_{+1}(s)-\Phi_{-1}(s),\qquad
\widetilde V(s)=\frac{\Delta(s)}{|\Delta(s)|+8}\in(-1,1).
$$

Finally convert to the requested player's perspective:

$$
V_p(s)=p\,\widetilde V(s)
=\begin{cases}
\widetilde V(s),&p=+1,\\
-\widetilde V(s),&p=-1.
\end{cases}
$$

The denominator makes the score bounded and monotone in the raw advantage:
zero advantage maps to zero, and increasing advantage approaches `+1`
without equating a heuristic lead to a proven win. The scale `8` controls
how quickly the score saturates; it is a handcrafted choice. Exact terminal
returns use `-1` and `+1`, while static values stay strictly between them.
Because the transform is odd, `V_-p(s) = -V_p(s)`, as backup requires.

**Worked example.** Suppose the complete run census contains a Black open
three, a Black dead two, and a White dead two, with no other contributing
runs. Then

$$
\Phi_{+1}=4(1+1.5\cdot2)+1=17,\qquad
\Phi_{-1}=1,\qquad
\Delta=16,
$$

$$
\widetilde V=\frac{16}{16+8}=\frac23,\qquad
V_{+1}=+\frac23,\quad V_{-1}=-\frac23.
$$

This is a favourable feature balance for Black, not a two-thirds probability
of winning. The exact threat rules and subsequent sampled replies may still
reverse the search's assessment.

## 8. Final move selection

Exploration is useful while collecting evidence, but the real move should
not retain a large exploration bonus simply because it was rarely tried.
A common way to express a visit-based decision policy is

$$
\pi_\tau(a\mid s_0)=
\frac{N(s_0,a)^{1/\tau}}
{\sum_{b\in A(s_0)}N(s_0,b)^{1/\tau}},\qquad \tau>0,
$$

provided at least one simulation completed. At `tau = 1`, this samples in
proportion to visits; as `tau` tends to zero from above, mass concentrates on
the most-visited actions. If several actions tie, the limit alone does not
specify the implementation's value-based tie-break.

The current algorithm uses a deterministic zero-temperature-style decision,
not an actual sample from a positive-temperature distribution:

$$
a_{\rm final}=\underset{a:\,N(s_0,a)>0}{\arg\max}_{\rm lex}
\bigl(N(s_0,a),Q(s_0,a)\bigr).
$$

That is, maximise visits first, then mean value. A large `Q` based on one
lucky playout is not enough to displace a more thoroughly investigated action
with more visits. Most visits is still a practical decision rule, not a
certificate of optimality.

If no simulation completed, the visit distribution is undefined; the code
instead returns a highest-prior action from `root.untried`. Direct tactical
returns occur before either rule. Finally, `divmod(best_a, n_cols)` converts
the flat action index to `(r, c)`.

## 9. Integration review: measured results and verification

### Runtime and widening tradeoff

The earlier scalar-access and exact-tactics review measured an end-to-end
throughput gain over the pre-review integration code: approximately 22%, 16%,
and 38% more simulations per second on empty, 16-stone, and 40-stone boards.
That measurement remains useful for the shared engine work, but it predates
progressive widening.

Widening deliberately changes tree shape. With the default schedule
`k_min = 6`, `k = 2.0`, a separate local five-seed benchmark at a 0.5-second
budget against the immediately preceding integration HEAD observed:

| Position | Eager-expansion simulations/s | Widened simulations/s | Change |
| --- | ---: | ---: | ---: |
| Empty | 3,398 | 2,474 | −27% |
| 16 stones | 2,448 | 2,079 | −15% |
| 40 stones | 2,402 | 2,230 | −7% |

A simulation that descends deeper is not equivalent to a shallower one, so raw
simulation count is only a diagnostic. The extra depth is exactly the intended
trade. Zero-budget setup remained approximately 5 ms in this measurement.
Single-process full-budget probes stayed around 4.73–4.80 seconds. In the
parallel 24-game run one late-game move reached 5.024 s; see the timing caveat
below. These are observations, not hardware-independent
guarantees; a strict deadline remains cooperative rather than preemptive.

Earlier v1–v4 timing and match results describe historical engines and budgets.
They must not be presented as measurements of this build.

### Correctness checks performed

| Check | Result |
| --- | --- |
| Existing diagonal, result, MCTS gameplay, open-threat, and GUI smoke suites | Passed |
| Independent full-board result reference | 2,000 randomized rectangular/small-board comparisons passed after scalar-access optimisation |
| Independent search win checks | 12,078 occupied-cell probes across 730 random boards passed |
| Exact tactical target maps | 13,810 complete per-action maps matched brute-force second placements under the no-existing-immediate-win precondition |
| Quiet fork defences | 430 candidate defences across 22 qualifying positions matched exact recomputation |
| Long runs | Exact-five/overline probes for lengths 4, 5, 6, 7, 20, and 100 passed |
| Backup and priors | Root visit accounting, `Q = W/N`, and normalised updated priors passed |
| Progressive widening | Audited width bounds, `prior_mass`, untried/child partition, and root visit count on three positions and four seeds; all passed |
| Internal-gap tactical checks | `XX.XX` wins were taken in all four axes for both colours; missing-edge and overline cases were rejected |
| Gap-aware evaluation | `XX.XX` and one-run-one-gap variants score above 0.5; `XX.X` remains low; static-eval throughput cost about 6–8% at 0.5 s |
| GUI integration | Close during actual search, stale-click rejection, one-time result, draw award, and error cleanup passed |

The review-specific differential and GUI checks were executed in memory;
they were not added as new maintained test files. Existing behavioural tests
were run without editing them. All implementation changes were verified to
remain inside the three original TODO bodies.

### Progressive-widening strength experiments

All matches used the same board seeds and **colour-swapped pairings** against
the immediately preceding engine, which expands every action eagerly. Each
game is one independent result; “draws” are awarded to White by the rules.

| Per-move budget | Configuration | Games | Wins | Losses |
| --- | --- | ---: | ---: | ---: |
| 0.08 s | `k_min = 6`, `k = 2.0` (default) | 40 | 34 | 6 |
| 0.08 s | `k_min = 6`, `k = 2.5` | 24 | 21 | 3 |
| 0.08 s | `k_min = 4`, `k = 2.5` | 40 | 30 | 10 |
| 0.08 s | `k_min = 8`, `k = 2.0` | 24 | 18 | 6 |
| 0.08 s | Root eager, deeper widening (`6`, `2.0`) | 16 | 7 | 9 |
| 0.5 s | `k_min = 6`, `k = 2.0` (default) | 8 | 5 | 3 |
| 0.5 s | `k_min = 6`, `k = 2.5` | 8 | 4 | 4 |
| 4.7 s (preliminary) | `k_min = 6`, `k = 2.0` | 8 | 3 | 5 |
| **4.7 s (larger fresh run)** | `k_min = 6`, `k = 2.0` | **24** | **18** | **6** |

The low-budget wins are substantial, and a root-eager variant was rejected
because it spent too much of a small budget broadening the root without
enough depth. At 0.5 seconds the advantage shrinks to a small, statistically
weak sample. The first full-budget sample (8 games, 3–5) was inconclusive.

The larger fresh full-budget run used **12 new board seeds**, both colours,
and 24 games at 4.7 s/move. Widening won **18–6 (75%)**:

- **As Black:** 8–4; **as White:** 10–2.
- Seed pairs: **7 sweeps**, **4 splits**, **1 pair lost 2–0**.
- Wilson 95% confidence interval for the game win rate: **55.1%–88.0%**.
- Exact two-sided binomial test against a 50% null: **p = 0.0227**.
- Two games reached 221 plies and were draws awarded to White. Excluding
  those draw-rule games leaves **17–5**; they contributed one widening win
  and one loss because widening held each colour in one game.

This is moderate but useful statistical evidence that widening is better than
eager expansion at the deployment budget. It is not a claim of a specific Elo
gain: the 24-game interval is wide, and a modest positive effect remains
plausible even below the observed 75%. The default remains `k_min = 6`,
`k = 2.0`; eager expansion can be restored with a very large `MCTS_PW_MIN`.

One timing caveat: the 24-game run executed 12 games in parallel. One
late-game move in a 221-ply game took **5.024 s**, slightly over the five-second
limit under that CPU contention; other moves stayed near 4.74 s. A
single-process full-budget probe is the correct final timing check. The
deadline is cooperative between simulations, so a final simulation or a GC
pause can overshoot.

Wall-clock-limited games can diverge even under identical random seeds because
finite search budgets are not deterministic. The table reports actual games,
not a theoretical Elo estimate. More seeds, balanced openings, and a
single-process timing run would tighten the interval further.

### Tunable re-screen under progressive widening

The old tuning settlement (before progressive widening) recommended
`C_PUCT = 1.4`, `KAPPA = 1.5`, and `ROLLOUT_DEPTH = 32`. Because widening
changes tree shape, these were re-screened against the current default at
0.5 s and then at the full budget.

| Stage | Games/configuration | Main outcomes |
| --- | ---: | --- |
| 0.5 s screen | 32 | Control 12–20; `KAPPA=2.0` 19–13; `C_PUCT=1.0` 18–14; `C_PUCT=1.6` 15–17; depth 24 15–17; depth 40 13–19. The control variance shows the screen was noisy. |
| 4.7 s validation | 16 | Control 8–8; `C_PUCT=1.0` 9–7; `KAPPA=2.0` 9–7; `C_PUCT=1.6, KAPPA=2.0, depth=40` 10–6. |

No candidate separated from the default reliably. The 10–6 full-budget
candidate has a Wilson interval of roughly 39%–82% and its 0.5 s screen was
17–15, so changing the defaults on this evidence would be overfitting to a
small sample. The recommended settings remain:

```python
MCTS_C_PUCT = 1.4
MCTS_KAPPA = 1.5
MCTS_ROLLOUT_DEPTH = 32
MCTS_PW_MIN = 6
MCTS_PW_C = 2.0
MCTS_REUSE = False
MCTS_TACTIC_EPS = 1.0
```

A larger full-budget settlement would be needed before changing any of
`C_PUCT`, `KAPPA`, or rollout depth. In particular, the apparent `KAPPA=2.0`
gain at 0.5 s (19–13) did not reproduce strongly at 4.7 s (9–7).

### Optional experiments evaluated

These features were implemented or prototyped and tested, but are **off by
default** because the evidence did not justify changing production behaviour.

| Feature | Default | Main observation |
| --- | --- | --- |
| Cheap tactical rollouts | `MCTS_TACTIC_EPS = 1.0` (off) | At 0.08 s: mid-tactical `8–8`, always-tactical `7–9`, mostly-tactical `2–14`. Throughput fell from about 2,693 to 254/96 simulations/s for the tested schedules. |
| Tree reuse | `MCTS_REUSE = False` | Reuses only a matching explored opponent reply after validating board, topology, player, and search configuration. At 0.08 s it went `9–7`; at 0.5 s `3–5`; a full-budget opening sample had no reuse hit. No robust gain. |
| Incremental evaluation | Not integrated | A local line-delta prototype reproduced the full evaluator on a random sweep, but 32 per-ply updates cost more than one full evaluator call (~0.16 ms); it would add work, not remove it. |

A combined check under `KAPPA = 2.0` at 0.5 s (16 games per configuration)
was also unfavourable: control 10–6, `KAPPA=2.0 + reuse` 6–10,
`KAPPA=2.0 + tactical` 2–14, and both together 4–12. There is no evidence
that a tunable combination rescues these features.

The tactical shortcut tests the empty cells adjacent to existing stones for
an exact-five win, then for the opponent's exact-five threat. This is exact,
but scanning those candidates on every rollout ply is expensive. The
implementation retains the option; the random rollout remains the default.

Tree reuse stores the node after the chosen move and, on the next call, looks
for the actual opponent reply among that node's explored children. A mismatch
discards the cache. The public `best_move` API has no tree-handle argument, so
this is necessarily a module-local optimisation; it is disabled by default to
keep calls independent.

Incremental evaluation is mathematically promising because placing one stone
only changes the four connected lines through it, and only the runs touching
that cell or its playable ends. However, local run bookkeeping per rollout ply
adds more Python work than the current one-pass cutoff evaluator. It should be
revisited only after profiling on a larger board or with a different runtime.

## 10. Remaining work

1. **Widening follow-up:** the larger 4.7-second run is **18–6** over 24 games
   (Wilson 95% CI 55–88%). Keep `k_min = 6`, `k = 2.0` as the working default;
   collect more balanced-opening games and tune `k_min`/`k` only if a larger
   sample shows a consistent gain. Recheck single-process move time after the
   parallel run's 5.024 s outlier.
2. **Finer deadline checks:** add cooperative checks inside expensive setup,
   descent, or rollout phases if larger boards require them. Never interpret
   an interrupted partial tactical map as a certified complete threat set.
3. **Retuning and maintained tests:** remeasure `C_PUCT`, depth, and weights
   after correctness/expansion changes, using paired matches rather than
   random opponents alone. Preserve regressions for diagonals with no H/V
   edges, broken threes, live-boundary overlines, missing-edge exact fives,
   last-ply draws, normalised priors, widening invariants, tree-reuse metadata,
   and backup perspective/counts.

All search changes remain classical MCTS variants or handcrafted heuristics;
none requires deep learning or additional production imports.
