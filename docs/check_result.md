# Game Result Detection (`check_result`)

`check_result(state)` determines whether the game has ended and, on a win, identifies the five stones that `main` should highlight. It works with any valid rectangular board passed by the caller, using the dimensions of `board`. The implementation is in [project_skeleton.py](<../project_skeleton.py>). These notes include the 15 November 2026 integration review; see also [the MCTS search](<best_move.md>) and [the game loop](<main.md>).

## Rules and interface

The input is `state = (board, edgeH, edgeV)`. For a board with `R` rows and `C` columns, coordinates are zero-based `(row, column)`.

| Array | Shape | Meaning |
| --- | --- | --- |
| `board` | `(R, C)` | `0` for empty, `+1` for black, `-1` for white |
| `edgeH` | `(R, C - 1)` | `edgeH[r, c]` connects `(r, c)` and `(r, c + 1)` when true |
| `edgeV` | `(R - 1, C)` | `edgeV[r, c]` connects `(r, c)` and `(r + 1, c)` when true |

A winning line is a maximal run of **exactly five** adjacent, connected stones of the same color on one straight axis. A gap, a different color, or a missing horizontal or vertical edge ends a run. Diagonally adjacent cells are always connected under the project rules. The function reads the three arrays without modifying them.

| Board result | `gameover` | `winning_line` |
| --- | --- | --- |
| A winning line exists | `True` | A `(5, 2)` integer NumPy array of stone coordinates in scan order |
| No winning line and the board is full | `True` | A `(5, 2)` array filled with `-1` |
| No winning line and at least one empty cell | `False` | A `(5, 2)` array filled with `-1` |

The win scan comes before the full-board check so that a player can win on the last available cell. The no-line sentinel deliberately does not encode a winner: `(True, all -1)` identifies a full-board draw, while `(False, all -1)` means play continues. Under the project rules a drawn game is awarded to **White**. The current `main` displays `White wins!` for the terminal draw sentinel, without inventing five winning coordinates; `best_move` uses the same White-win value for search draws. External scoring should apply that rule as well.

## Scan each connected run once

| Axis | Forward step `(dr, dc)` | Edge rule |
| --- | --- | --- |
| Horizontal | `(0, 1)` | Read `edgeH` at the left cell |
| Vertical | `(1, 0)` | Read `edgeV` at the upper cell |
| Down-right diagonal | `(1, 1)` | Adjacent cells are connected |
| Down-left diagonal | `(1, -1)` | Adjacent cells are connected |

Each axis has two opposite directions. For the horizontal axis, for example, the cell on the left is the predecessor and the scan advances to the right. Four axes therefore cover all eight neighbor directions.

1. For each axis, visit cells in row-major order and skip empty cells.
2. Check the predecessor `(r - dr, c - dc)`. If it contains a connected stone of the same color, skip the current cell because the run starts earlier.
3. From a start cell, advance one step at a time in `(dr, dc)` until the run ends. The local helper `continues_line` checks bounds before indexing, then checks color and the appropriate edge. The same helper works in either direction because the left or upper endpoint determines the index of a shared edge.
4. When the complete run has length five, return the five coordinates, starting at `(r, c)` and advancing by the same step. When there are several winning lines, the function returns the first one in the fixed axis and cell order.
5. If the scan finds no winning line, check whether any empty cell remains.

For a fixed axis, each connected run has exactly one cell with no connected, same-color predecessor. Scanning from that cell reaches the entire run. This is what prevents the last five stones of a six-stone run from being counted as a separate win. A missing edge before or after a five-stone run can make it a valid win even when another stone of the same color is immediately beyond that edge.

With four fixed axes, the worst-case time is `O(R × C)`. Each cell is considered as a start candidate once per axis, and each run is traversed once per axis.

### Scalar-access optimisation and space cost

The current implementation converts the three arrays to Python lists once per call:

```python
cells, h, v = board.tolist(), edgeH.tolist(), edgeV.tolist()
```

`continues_line` and the nested scan then use Python scalar indexing rather than repeated NumPy scalar access. These are independent read copies: neither the board nor its edge arrays are modified. The winning-coordinate construction and final NumPy draw check keep the public interface unchanged.

The scan's counters use constant space, but the list copies require **`O(R × C)` additional space overall**. If no winner is found, `board == 0` also creates a temporary `(R, C)` Boolean array. The previous constant-space description applied only to the scan before these copies were introduced.

## Integration review verification — 15 November 2026

No correctness defect was found in the staged integration detector. In particular, both diagonal orientations are always connected, predecessor checks do not wrap at array boundaries, an overline cannot be accepted as its last five stones, and a win on the final empty point takes priority over a draw.

After the scalar-access optimisation, **2,000 randomized comparisons** against an independent five-window/end-boundary reference passed, including varying rectangular and small dimensions, exact winning coordinates, and unchanged input arrays. The existing result and diagonal regression suites also passed, including five versus six diagonal stones when every horizontal and vertical edge is disabled.

The search's independently optimised last-move check also passed **12,078 occupied-cell probes** across 730 randomized boards, plus long-run cases; see [the search verification summary](<best_move.md#correctness-checks-performed>). These are separate checks, not additional full-board detector comparisons.

Local timing observations for the detector:

| Workload | Before Python-list scalar access | After | Interpretation |
| --- | ---: | ---: | --- |
| Empty 17-by-13 board, 1,000 calls | About 244 µs/call | About 101 µs/call | About 2.4× faster on this workload |
| 100 populated random boards, repeated 10 times | About 561 µs/call | About 309 µs/call | Separate short benchmark; about 1.8× faster |

These timings are workload- and machine-dependent, not guarantees for every board. The in-memory differential checks were not added as maintained test files.

## Historical integration verification

The following records describe the earlier integration checks against Member B's branch, not newly rerun counts for the optimised implementation. The fixed cases below are representative of the **15 cases** checked directly against the required result and coordinate format. The checks also verified that the input arrays were unchanged.

| Position or condition | Expected result |
| --- | --- |
| Five connected black or white stones in any of the four axes | Win with exactly those five coordinates |
| Five diagonal stones with every horizontal and vertical edge disabled | Win |
| Six connected stones with an empty cell elsewhere | Game continues if no other line wins |
| Five connected stones followed by a sixth same-color stone across a missing edge | Win for the connected group of five |
| A gap or a missing edge inside a candidate line | That line cannot win |
| A winning run touching an array boundary | Win without reading across the boundary |
| Empty board, full board with no winner, or a win on a full board | Continue, draw, or win, respectively |

Three comparisons checked the implementation at different levels:

| Comparison | Result |
| --- | --- |
| 1,000 randomized boards with varying dimensions and edge probabilities, starting with seed `5002` | The result and coordinates matched an independent reference that enumerates five-cell windows and checks both ends. They also matched Member B's standalone `check_result`. |
| 3,307 legal placements across 60 randomized games, stopping each game at the first win or draw | After every placement, the presence of a winning line matched Member B's internal Fast Check for that placement. |
| One fixed position with an immediate win, run through Pygame using a dummy display | The original B version and the version with this `check_result` chose the same winning move, reported the same winner, and highlighted the same five cells. |

The comparisons used Member B's branch at [commit 749898b](https://github.com/knmanaa/MCTS-with-gomoku/tree/749898b816be5b0606e499df926ac9ee77ea7aff). To reproduce the six-stone boundary case, start Python from the repository root with the project environment active and run the following.

```python
import numpy as np

from project_skeleton import check_result

board = np.array([[1, 1, 1, 1, 1, 1, 0]])
edge_h = np.ones((1, 6), dtype=bool)
edge_v = np.zeros((0, 7), dtype=bool)
state = (board, edge_h, edge_v)

gameover, line = check_result(state)
assert not gameover and np.all(line == -1)

edge_h[0, 4] = False  # Break the edge between the fifth and sixth stones.
gameover, line = check_result(state)
assert gameover and line.tolist() == [[0, c] for c in range(5)]
```

## Interface with the search win check

The two checks share the same winning rule and answer different calls. `main` calls `check_result(state)` with the complete board after a real move and uses returned coordinates to highlight a winning line. Inside `best_move`, `win_check(bd, a)` receives a flat search-board copy and the index of the stone just placed, then returns a Boolean for whether that placement wins. It walks the precomputed `steps4` connectivity table rather than rechecking edge arrays on each step. If there was no winner before the placement, any newly created win must pass through that stone, so the search can restrict its check to the four axes through that cell.

The checks stay independent in the program. Compare the search Boolean with the **presence of a winning line**, not with `gameover` alone, since `gameover` is also true for draws. The full-board sentinel remains `(True, all -1)` for a draw; the search separately assigns White's signed terminal value. Its rollout now performs the draw check even when the final allowed rollout ply fills the board, instead of evaluating that position heuristically.

The fixed positions and placement-by-placement comparison provide regressions whenever either implementation changes. Do not compare a last-move-only check against arbitrary boards with pre-existing unrelated winners: its equivalence relies on the no-earlier-winner precondition.
