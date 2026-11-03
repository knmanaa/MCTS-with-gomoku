# MCTS with Gomoku

An interactive implementation of a variant of the classic game **five-in-a-row** (also known as *gomoku* or *wuziqi*), in which a human player competes against a computer opponent that selects its moves using a strategy based on **Monte Carlo tree search (MCTS)**.

## Rules of the game

The game is played on a rectangular grid board with `N_R = 17` rows and `N_C = 13` columns. The gridlines of the board are **randomly initialized** according to two parameters:

- **Horizontal connectivity** `C_H = 0.85`: the probability that two horizontally adjacent grid points are connected by a line.
- **Vertical connectivity** `C_V = 0.7`: the probability that two vertically adjacent grid points are connected by a line.

A line is drawn between connected grid points. Although no diagonal lines are drawn, grid points are always considered connected diagonally (in both `/` and `\` directions).

Two players, **Black** and **White**, alternately place their stones at any empty grid point, with **Black moving first**. The rules for winning are:

- When an unbroken line of **exactly five** stones of the same colour is formed, that player wins the game.
- A line connecting **more than five** stones of the same colour does **not** count as a win, and the game continues.
- Stones separated by gaps do not form a winning line.
- When there are no more empty points, the game ends in a draw — but White is considered to have won, because Black enjoys the advantage of moving first.

## Representation of the game

The top left corner of the board is the origin `(0, 0)`. The row number `r` is in `[0, N_R)` and counts from top to bottom; the column index `c` is in `[0, N_C)` and counts from left to right. A game state is represented by the tuple:

```
state = (board, edgeH, edgeV)
```

consisting of:

- `board`: an `N_R x N_C` integer array
  - `board[r, c] = +1` — the point `(r, c)` is occupied by a **black** stone
  - `board[r, c] = -1` — the point `(r, c)` is occupied by a **white** stone
  - `board[r, c] =  0` — the point `(r, c)` is empty
- `edgeH`: an `N_R x (N_C - 1)` Boolean array — `edgeH[r, c] = True` means the points `(r, c)` and `(r, c+1)` are horizontally connected
- `edgeV`: an `(N_R - 1) x N_C` Boolean array — `edgeV[r, c] = True` means the points `(r, c)` and `(r+1, c)` are vertically connected

## How the program works

The program is built upon `project_skeleton.py`, which provides the constants, drawing utilities, and the structure of the game loop. Three functions need to be completed:

### `main(human_color)`

Running `main(human_color)` pops up a game window showing a randomly initialized board, after which the human player and the computer take turns placing stones (Black moves first).

- `human_color = +1` — the human plays Black, the computer plays White
- `human_color = -1` — the human plays White, the computer plays Black

When the human clicks a valid, empty position during their turn, a stone of their colour is placed at that grid point. During the computer's turn, it places a stone at the point returned by `best_move(state, color)`. Each newly placed stone is drawn with a highlighted border that remains until the next stone is placed. After every move, `check_result(state)` is called; if the game has ended, the result ("Black wins!", "White wins!", or "Draw!") is printed in the window and the five stones in the winning line are highlighted. The window is locked once the game ends, but can be closed at any time.

As an optional extra, when `abs(human_color) != 1` the program shows a game between the computer and itself (like an animation), which is useful for verifying that the entire program runs correctly.

### `best_move(state, color)`

Given the current state, this function uses a strategy **based on Monte Carlo tree search (MCTS)** to determine and return the best point `(r, c)` at which to place a stone. MCTS may be combined with other tactics, such as occupying a specific point due to the configuration of the board. The function must spend **at most five seconds per move** — a program is considered frozen if a move takes too much time. It does not modify `board[r, c]` itself.

### `check_result(state)`

Given the current state, this function checks whether the game has ended. It returns `(gameover, winning_line)`:

- `gameover` — `True` if a player has won or the game ends in a draw; `False` otherwise
- `winning_line` — a `5 x 2` NumPy array with the positions `(r, c)` of the five stones forming the winning line, or ten `-1` elements if no one has won yet

## Requirements

- Python 3 with **Pygame** and **NumPy**
- Only Pygame, NumPy, and `time` may be imported
- Constants defined in the skeleton (board size, connectivities, colours, window dimensions, etc.) must not be hardcoded, since they may be modified; `SEED` may be changed to initialize a different board

## Setting up the environment

It is recommended to use [uv](https://docs.astral.sh/uv/) to manage the Python environment and dependencies. After installing uv, create a virtual environment and install the dependencies:

```bash
uv venv
uv pip install pygame numpy
```

Alternatively, you can run the program directly in a managed environment:

```bash
uv run --with pygame --with numpy python project_skeleton.py
```

## Running

```bash
python project_skeleton.py
```

The seed used to initialize the board is set at the bottom of the script via `SEED`. Since `best_move(state, color)` depends only on what the skeleton provides, it can also be imported and used by an external master program — for example, to pit two different strategies against each other in a head-to-head match, where each match consists of two rounds with the groups alternating as Black and White, and the player spending less total thinking time winning in the event of a tied match.
