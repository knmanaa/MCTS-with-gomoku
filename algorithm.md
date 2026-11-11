# Game Result Detection (`check_result`)

`check_result(state)` determines whether the game has ended and, on a win, identifies the five stones that `main` should highlight. It works with any valid rectangular board passed by the caller, using the dimensions of `board`. The implementation is in [project_skeleton.py](project_skeleton.py).

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

The win scan comes before the full-board check so that a player can win on the last available cell. The draw result lets `main` display a draw. Competition scoring handles the rule that awards a drawn game to White.

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

With four fixed axes, the worst-case time is `O(R × C)`. Each cell is considered as a start candidate once per axis, and each run is traversed once per axis. The line scan uses constant additional space. If no winner is found, the NumPy expression `board == 0` for the draw check creates a temporary `(R, C)` Boolean array.

## Verification

The fixed cases below are representative of the **15 cases** checked directly against the required result and coordinate format. The checks also verified that the input arrays were unchanged.

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

## Interface with Member B's Fast Check

The two checks share the same winning rule and answer different calls. `main` calls `check_result(state)` with a full board after a real move and uses the returned coordinates to highlight the winning line. The Fast Check inside `best_move` receives the location of a stone just placed on a search board and returns a Boolean for whether that placement wins. If there was no winner before the placement, any newly created win must pass through that stone, so the search can restrict its check to the four axes through that cell.

The checks stay independent in the program. In the comparison, the Fast Check's Boolean was compared with the presence of a winning line from `check_result`, since `gameover` is also true for draws. If the group changes either implementation, the fixed positions and placement-by-placement comparison give reviewers a way to check that the two still apply the same rule.
