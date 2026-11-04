import pygame as pg
import numpy as np
import time

#%% Constants that may be modified during tests ###############################

COLOR_EDGE  = [209, 139,  71] # dark  brown
COLOR_BOARD = [255, 206, 158] # light brown
COLOR_STONE  = {+1: [  0,   0,   0], # black
                -1: [255, 255, 255]} # white
COLOR_BORDER = {+1: [ 75,  75,  75], # dark  grey
                -1: [235, 235, 235]} # light grey
COLOR_HIGHLIGHT = [255,   0,   0] # red

WINDOW_WIDTH  = 640
WINDOW_HEIGHT = 720
WINDOW_CAPTION = 'Five-in-a-row'

NUM_OF_ROWS = 17 # N_R
NUM_OF_COLS = 13 # N_C

PAD_x = WINDOW_WIDTH  / NUM_OF_COLS
PAD_y = WINDOW_HEIGHT / NUM_OF_ROWS

SEP_x = (WINDOW_WIDTH -2*PAD_x) / (NUM_OF_COLS-1) # separation between columns
SEP_y = (WINDOW_HEIGHT-2*PAD_y) / (NUM_OF_ROWS-1) # separation between rows

STONE_RADIUS = min(SEP_x, SEP_y) * 0.3

CONNECTIVITY_HORIZONTAL = .85 # C_H = P(a horizontal edge exists)
CONNECTIVITY_VERTICAL   = .7  # C_V = P(a vertical   edge exists)

#%% Utility functions that you can use ########################################

def draw_board(state):
    """ This function draws an empty board in the game window.
    """
    board, edgeH, edgeV = state
    
    surface = pg.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    surface.fill(COLOR_BOARD)

    def draw_line(xy1, xy2, thickness):
        pg.draw.line(surface, COLOR_EDGE, xy1, xy2, thickness)
    
    for r in range(NUM_OF_ROWS):
        for c in range(NUM_OF_COLS):
            x = PAD_x + c * SEP_x
            y = PAD_y + r * SEP_y
            pg.draw.circle(surface, COLOR_EDGE, (x, y), STONE_RADIUS/2)
            if c > 0: draw_line((x-SEP_x, y), (x, y), 3*edgeH[r, c-1])
            if r > 0: draw_line((x, y-SEP_y), (x, y), 3*edgeV[r-1, c])
                        
    pg.display.set_caption(WINDOW_CAPTION)
    pg.display.update()
    return surface

def draw_stone(surface, r, c, color, highlight=False):
    """ This function draws a stone in the rth row and cth column of
        the game board (0 <= r < NUM_OF_ROWS and 0 <= c < NUM_OF_COLS).
        
        If color = +1, the stone is black.
        If color = -1, the stone is white.
        If highlight = True, the stone has a highlighted border.
    """
    x = PAD_x + c * SEP_x
    y = PAD_y + r * SEP_y

    def draw_circle(color, width=0):
        pg.draw.circle(surface, color, [x, y], STONE_RADIUS, width)
    
    draw_circle(COLOR_STONE[color])
    draw_circle(COLOR_BORDER[color], 2)
    if highlight: draw_circle(COLOR_HIGHLIGHT, 3)
    pg.display.update()

def print_result(surface, winner):
    """ The function prints the game result
        in the upper left corner of the game window.
    """
    if   winner ==  0: msg, c = 'Draw!'      , COLOR_EDGE
    elif winner == +1: msg, c = 'Black wins!', COLOR_STONE[+1]
    elif winner == -1: msg, c = 'White wins!', COLOR_STONE[-1]
    
    font = pg.font.Font('freesansbold.ttf', 32)
    text = font.render(msg, True, c)
    textRect = text.get_rect()
    textRect.topleft = (0, 0)
    surface.blit(text, textRect)
    pg.display.update()

#%% Functions that you need to complete #######################################

### TODO
def best_move(state, color):
    """ Given the current state=(board, edgeH, edgeV), 
        this function uses your designed strategy,
        which must be related to Monte Carlo tree search (MCTS),
        to select and return the best point (r, c) to place a stone at.
        
        If color = +1, the stone is black.
        If color = -1, the stone is white.
        
        Keep in mind that this function does not change board[r, c].
        
        You can write between the two rows of #.
    """
    ###########################################################################
    # PUCT-MCTS (AlphaZero-style search without deep learning).
    #
    # Replacements w.r.t. AlphaZero:
    #   * policy prior P(s, a) -> heuristic prior (line potential, urgency,
    #     proximity), computed from the board alone
    #   * value network V(s)   -> depth-capped random rollouts, with a static
    #     evaluation function at the cutoff
    # The PUCT selection, expansion, and sign-alternating backup follow the
    # standard scheme; the final move is the most-visited root action
    # (temperature tau -> 0).
    #
    # Game-specific points:
    #   * a line of EXACTLY five stones wins, and consecutive stones must be
    #     joined by an existing edge (horizontal/vertical); diagonals are
    #     always connected (same rule as check_result, cf. Figures 2-5)
    #   * a draw counts as a win for White (rules, section 1)
    #   * the input state is never modified
    #   * the 5-second move limit is respected (4.7 s deadline)
    #
    # Tunables can be overridden from outside for benchmarking WITHOUT
    # editing this file (they default to the values below):
    #   MCTS_TIME_LIMIT, MCTS_C_PUCT, MCTS_ROLLOUT_DEPTH, MCTS_KAPPA

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

    def conn(r, c, dr, dc):
        """Whether (r, c) and (r+dr, c+dc) are joined by an existing edge.
        Diagonals are always connected."""
        if dr == 0 and dc == 1:  return edgeH_l[r][c]
        if dr == 0 and dc == -1: return edgeH_l[r][c - 1]
        if dr == 1 and dc == 0:  return edgeV_l[r][c]
        if dr == -1 and dc == 0: return edgeV_l[r - 1][c]
        return True   # |dr| == 1 and |dc| == 1: diagonals

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

    def is_winning_move(bd, r, c):
        """Whether the stone just placed at (r, c) completes an unbroken
        line of exactly five stones (the game would be over)."""
        for dr, dc in DIRS4:
            if line_len(bd, r, c, dr, dc) == 5:
                return True
        return False

    def run_extent(bd, r, c, dr, dc):
        """For the maximal unbroken line of bd[r, c] through (r, c) along
        (dr, dc), return (L, start, end): the length and the two extreme
        cells of the run (start is the cell furthest along -d)."""
        color = bd[r * n_cols + c]
        s_r, s_c = r, c
        br, bc = r - dr, c - dc
        while (0 <= br < n_rows and 0 <= bc < n_cols
               and bd[br * n_cols + bc] == color
               and conn(br, bc, dr, dc)):
            s_r, s_c = br, bc
            br -= dr
            bc -= dc
        e_r, e_c = r, c
        fr, fc = r + dr, c + dc
        while (0 <= fr < n_rows and 0 <= fc < n_cols
               and bd[fr * n_cols + fc] == color
               and conn(e_r, e_c, dr, dc)):
            e_r, e_c = fr, fc
            fr += dr
            fc += dc
        L = (e_r - s_r) * dr + (e_c - s_c) * dc + 1   # dot with the unit step
        return L, (s_r, s_c), (e_r, e_c)

    def four_threat_count(bd, a, pl):
        """Number of win-in-1 cells colour `pl` would gain by placing a
        stone at flat index a (only the four lines through a can change).
        Requires: a is empty and no win-in-1 cells for pl exist on bd yet
        (both hold at the point of use: mine/theirs were handled first)."""
        r, c = divmod(a, n_cols)
        bd[a] = pl
        total = 0
        for dr, dc in DIRS4:
            L, s, e = run_extent(bd, r, c, dr, dc)
            if L != 4:
                continue
            pr, pc = s[0] - dr, s[1] - dc     # extension cell before the run
            if (0 <= pr < n_rows and 0 <= pc < n_cols
                    and bd[pr * n_cols + pc] == 0 and conn(pr, pc, dr, dc)):
                total += 1
            qr, qc = e[0] + dr, e[1] + dc     # extension cell after the run
            if (0 <= qr < n_rows and 0 <= qc < n_cols
                    and bd[qr * n_cols + qc] == 0 and conn(e[0], e[1], dr, dc)):
                total += 1
        bd[a] = 0
        return total

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

    def puct_select(node):
        """PUCT: argmax_a Q(s, a) + c_puct * P(s, a) * sqrt(N(s)) / (1 + N(s, a))."""
        sq_n = node.N ** 0.5
        best_a, best_s = None, -1e18
        for a, ch in node.children.items():
            s = ch.Q + C_PUCT * node.prior[a] * sq_n / (1 + ch.N)
            if s > best_s:
                best_s, best_a = s, a
        return best_a

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

    while time.perf_counter() < deadline:
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

    # final move: most-visited root child (tau -> 0); tie-break by Q
    best_a, best_key = None, None
    for a, ch in root.children.items():
        key = (ch.N, ch.Q)
        if best_key is None or key > best_key:
            best_key, best_a = key, a
    if best_a is None:
        # no simulation completed: fall back to the best-prior action
        best_a = root.untried[-1]

    ###########################################################################
    return divmod(best_a, n_cols)

### TODO
def check_result(state):
    """ Given the current state=(board, edgeH, edgeV),
        this function checks if a player has already won the game.
        
        This function returns (gameover, winning_line).
        
        gameover: bool
            * True if a player has won or the game ends in a draw.
            * False if the game has not yet ended.
        
        winning_line: a 5*2 NumPy array
            * The positions (r, c) of the five stones that make a player win.
            * If no one wins (yet), its 10 elements are all -1.
            
        You can write between the two rows of #.
    """
    ###########################################################################

    board, edgeH, edgeV = state
    n_rows, n_cols = board.shape

    def connected(r, c, dr, dc):
        """Whether (r, c) and (r+dr, c+dc) are connected.

        Horizontal/vertical connections exist only where the corresponding
        edge was initialized; a missing edge breaks the line.  Diagonal
        connections always exist (even though no diagonal lines are drawn).
        """
        if dr == 0 and dc == 1:
            return edgeH[r, c]
        if dr == 1 and dc == 0:
            return edgeV[r, c]
        return True

    # A win is an UNBROKEN line of EXACTLY five stones of the same colour:
    #   * stones separated by empty points do not count (Figure 3)
    #   * a line of six or more connected stones does not count (Figure 4)
    #   * horizontally/vertically, consecutive stones must be joined by an
    #     existing edge, otherwise the line is broken (Figure 5)
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        for r in range(n_rows):
            for c in range(n_cols):
                color = board[r, c]
                if color == 0:
                    continue

                # only start counting at the beginning of a line
                pr, pc = r - dr, c - dc
                if (0 <= pr < n_rows and 0 <= pc < n_cols
                        and board[pr, pc] == color
                        and connected(pr, pc, dr, dc)):
                    continue

                # measure the length of the line starting at (r, c)
                length = 1
                rr, cc = r, c
                while True:
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < n_rows and 0 <= nc < n_cols):
                        break
                    if board[nr, nc] != color:
                        break
                    if not connected(rr, cc, dr, dc):
                        break
                    length += 1
                    rr, cc = nr, nc

                if length == 5:
                    winning_line = np.array([[r + i * dr, c + i * dc]
                                             for i in range(5)], dtype=int)
                    return True, winning_line

    # no more empty points: the game ends in a draw (no winning line)
    if np.all(board != 0):
        return True, np.ones((5, 2)) * -1

    ###########################################################################
    return False, np.ones((5, 2)) * -1

### TODO
def main(human_color):
    """ This function is the cornerstone of the program.
        In short, it creates a game window, initializes the game,
        and updates the game according to two players' movements.
        
        If human_color = +1, you are Black, and the computer is White.
        If human_color = -1, you are White, and the computer is Black.
        Otherwise, the window is closed.
       (Optionally, it shows a game between the computer and itself.)
        
        The computer places its stones according to best_move(state, color).
        
        Several variables (running, gameover, congratulated, previous_move)
        have been defined for you.
        You must update them according to the progress of the game.
        
        You can write between the two rows of #.
    """
    pg.init()
    board = np.zeros((NUM_OF_ROWS, NUM_OF_COLS)).astype(int)

    random = np.random.random
    edgeH = random((NUM_OF_ROWS, NUM_OF_COLS-1)) < CONNECTIVITY_HORIZONTAL
    edgeV = random((NUM_OF_ROWS-1, NUM_OF_COLS)) < CONNECTIVITY_VERTICAL

    state = (board, edgeH, edgeV)
    surface = draw_board(state)
    
    running = True # turns False once you try to close the game window
    gameover = False # turns True once the game has ended
    congratulated = False # turns True once the game result is printed
    
    previous_move = None # after the first move, it stores the position
                         # (r, c) at which a stone has just been placed
    
    ###########################################################################

    # Any human_color other than +1/-1 means there is no human player:
    # the computer plays against itself (optional task).
    human = human_color if human_color in (+1, -1) else 0

    current = +1  # Black always moves first

    def end_turn(r, c):
        """Place the move at (r, c), redraw the previous stone without its
        highlighted border, then check the result and update the flags."""
        nonlocal current, gameover, congratulated, previous_move

        color = board[r, c]

        # remove the highlight border from the previous move (if any)
        if previous_move is not None:
            pr, pc = previous_move
            draw_stone(surface, pr, pc, board[pr, pc], highlight=False)

        # draw the new stone with a highlighted border
        draw_stone(surface, r, c, color, highlight=True)
        previous_move = (r, c)

        # check whether the game has ended after this move
        over, winning_line = check_result(state)
        if over:
            gameover = True
            congratulated = True

            if np.all(winning_line == -1):
                print_result(surface, 0)  # draw
            else:
                winner = board[winning_line[0, 0], winning_line[0, 1]]
                print_result(surface, winner)
                for rr, cc in winning_line:  # highlight the winning five stones
                    draw_stone(surface, rr, cc, winner, highlight=True)
        else:
            current = -current  # switch players

    while running:                   # this double loop handles
        for event in pg.event.get(): # all events you initiate

            if event.type == pg.QUIT: # detects whether you
                running = False       # try to close the window

            if (event.type == pg.MOUSEBUTTONDOWN   # a click during the human's
                and not gameover                   # turn is only accepted while
                and human != 0                     # the game is not over and
                and current == human):             # the window is unlocked

                x, y = event.pos # the position at which you click
                                 # (0, 0) is the top left corner of the window

                r = round( (y-PAD_y) / SEP_y )
                c = round( (x-PAD_x) / SEP_x )

                # the click must land on a valid, empty grid point
                if (0 <= r < NUM_OF_ROWS and 0 <= c < NUM_OF_COLS
                        and board[r, c] == 0):
                    board[r, c] = current
                    end_turn(r, c)

        # the computer's turn (also covers the computer-vs-itself mode,
        # where human = 0 so every move is made by the computer)
        if running and not gameover and current != human:
            if human == 0:
                pg.time.delay(300) # slow down the self-play animation

            r, c = best_move(state, current)

            # only accept a valid, empty point
            if board[r, c] == 0:
                board[r, c] = current
                end_turn(r, c)

    ###########################################################################
    
    pg.quit()
    return state

if __name__ == '__main__':
    
    SEED = 5002 # used to initalize board; you can change it
    WINDOW_CAPTION += f' (seed: {SEED})'
    
    np.random.seed(SEED)
    state = main(human_color=+1)