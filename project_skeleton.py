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
    # PUCT selection, sign-alternating backup, and the final most-visited
    # root action follow the standard scheme.  Expansion uses prior-guided
    # progressive widening: low-prior branches are deferred, never deleted.
    #
    # Game-specific points:
    #   * a line of EXACTLY five stones wins, and consecutive stones must be
    #     joined by an existing edge (horizontal/vertical); diagonals are
    #     always connected (same rule as check_result, cf. Figures 2-5)
    #   * a draw counts as a win for White (rules, section 1)
    #   * the input state is never modified
    #   * a soft 4.7-second deadline leaves margin below the 5-second limit;
    #     setup and the final simulation are not preempted
    #
    # Tunables can be overridden from outside for benchmarking WITHOUT
    # editing this file (they default to the values below):
    #   MCTS_TIME_LIMIT, MCTS_C_PUCT, MCTS_ROLLOUT_DEPTH, MCTS_KAPPA,
    #   MCTS_PW_MIN, MCTS_PW_C

    t0 = time.perf_counter()
    TIME_LIMIT    = min(4.7, max(0.0, float(globals().get('MCTS_TIME_LIMIT', 4.7))))
    C_PUCT        = globals().get('MCTS_C_PUCT', 1.4)
    ROLLOUT_DEPTH = globals().get('MCTS_ROLLOUT_DEPTH', 32)
    KAPPA         = globals().get('MCTS_KAPPA', 1.5)
    # Progressive widening: a node starts with a small, prior-ordered action
    # set and admits another action once it has enough visits.  All actions
    # remain eligible; this delays low-prior branches instead of deleting them.
    PW_MIN        = max(1, int(globals().get('MCTS_PW_MIN', 6)))
    PW_C          = max(0.0, float(globals().get('MCTS_PW_C', 2.0)))
    deadline = t0 + TIME_LIMIT
    event_poll = globals().get('_MCTS_EVENT_POLL')
    next_poll = t0

    def keep_searching():
        # Optional cooperative GUI cancellation; external callers need no Pygame
        # display and never have their event queue consumed by best_move itself.
        nonlocal next_poll
        now = time.perf_counter()
        if event_poll is not None and now >= next_poll:
            next_poll = now + 0.03
            if event_poll():
                return False
        return now < deadline

    n_rows, n_cols = state[0].shape
    n_cells = n_rows * n_cols
    WHITE = -1
    DIRS4 = ((0, 1), (1, 0), (1, 1), (1, -1))

    # python-native flat copies: fast scalar access in the hot loops
    board_flat = np.asarray(state[0]).ravel().astype(int).tolist()

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

    # Resolve connectivity once, not during every rollout/line walk.
    # Each pair is (forward, backward), with -1 for a boundary or missing edge.
    steps4 = [tuple((f if f >= 0 and (fe < 0 or edges[fe]) else -1,
                     b if b >= 0 and (be < 0 or edges[be]) else -1)
                    for _, _, f, fe, b, be in entries)
              for entries in gates4]
    connected_lines = []
    windows5 = []
    for d in range(4):
        for a in range(n_cells):
            if steps4[a][d][1] >= 0:
                continue
            line = []
            i = a
            while i >= 0:
                line.append(i)
                i = steps4[i][d][0]
            connected_lines.append(tuple(line))
            for k in range(len(line) - 4):
                windows5.append((tuple(line[k:k + 5]),
                                 line[k - 1] if k else -1,
                                 line[k + 5] if k + 5 < len(line) else -1))

    def line_len(bd, r, c, dr, dc):
        a = r * n_cols + c
        pl = bd[a]
        d = DIRS4.index((dr, dc))
        length = 1
        for side in (0, 1):
            i = steps4[a][d][side]
            while i >= 0 and bd[i] == pl:
                length += 1
                i = steps4[i][d][side]
        return length

    def win_check(bd, a):
        pl = bd[a]
        if pl == 0:
            return False
        for d, (f, b) in enumerate(steps4[a]):
            length = 1
            while f >= 0 and bd[f] == pl:
                length += 1
                if length > 5:
                    break
                f = steps4[f][d][0]
            if length > 5:
                continue
            while b >= 0 and bd[b] == pl:
                length += 1
                if length > 5:
                    break
                b = steps4[b][d][1]
            if length == 5:
                return True
        return False

    def tactical_targets(bd, pl):
        """Exact wins and a -> set of winning replies after playing a.

        A connected five-window with three stones and two empties describes
        both forcing moves at once, including broken fours. Boundary stones
        of our colour invalidate a window: filling it would make an overline.
        Sets deduplicate the same winning cell shared by different windows.
        """
        wins, targets = set(), {}
        for cells, before, after in windows5:
            if (before >= 0 and bd[before] == pl
                    or after >= 0 and bd[after] == pl):
                continue
            empty = []
            for a in cells:
                v = bd[a]
                if v == -pl:
                    break
                if v == 0:
                    empty.append(a)
                    if len(empty) > 2:
                        break
            else:
                if len(empty) == 1:
                    wins.add(empty[0])
                elif len(empty) == 2:
                    a, b = empty
                    targets.setdefault(a, set()).add(b)
                    targets.setdefault(b, set()).add(a)
        return sorted(wins), targets

    class Node:
        """Search-tree node.  Edge statistics (N, W, Q) are stored on the
        child and are kept from the perspective of the player who made the
        move leading to it (the parent's player to move)."""
        __slots__ = ('board', 'player', 'children', 'untried', 'prior',
                     'prior_mass', 'N', 'W', 'Q', 'terminal_value', 'empties')

        def __init__(self, board, player):
            self.board = board          # flat list, player to move = `player`
            self.player = player
            self.children = None        # dict {action: Node}, set on expansion
            self.untried = None         # actions not yet tried, prior-ascending
            self.prior = None           # dict {action: P(s, a)}
            self.prior_mass = 0.0       # sum of priors of admitted children
            self.N = 0                  # visit count N(s)
            self.W = 0.0                # total action value
            self.Q = 0.0                # mean action value W / N
            self.terminal_value = None  # value for the player to move if over
            self.empties = None         # flat indices of this board's empties

    def create_child(node, a):
        """Apply action `a` for node's player to move; return the child."""
        bd = node.board[:]              # copy; the parent's board is reused
        bd[a] = node.player
        child = Node(bd, -node.player)
        child.empties = [i for i in node.empties if i != a]
        if win_check(bd, a):
            child.terminal_value = -1.0                          # mover won
        elif not child.empties:
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
            # Positive baseline keeps distant legal moves searchable; a small
            # centre preference avoids the old bottom-right opening tie bias.
            centre = (1.0 - abs(2 * r - n_rows + 1) / max(1, n_rows - 1)
                      + 1.0 - abs(2 * c - n_cols + 1) / max(1, n_cols - 1))
            s = 1.0 + 0.1 * centre + 0.6 * sum(1 for j in neighbors8[a] if bd[j] != 0)
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
            actions = node.empties   # set at creation; no rescan needed
            # ---- tactical priority at search nodes (not just the root) ----
            # If the player to move has an immediate win, explore only winning
            # moves.  Otherwise, if the opponent has an immediate win, explore
            # only the blocking moves.  This makes forced reply sequences
            # (fight threes with fours) actual search lines instead of relying
            # on proximity priors and random rollouts to notice them.  A five
            # must include a cell adjacent to an existing stone, so only those
            # candidate cells need testing.
            board = node.board
            me = node.player
            cand = [a for a in actions
                    if any(board[j] != 0 for j in neighbors8[a])]
            if cand:
                wins = []
                for a in cand:
                    board[a] = me
                    if win_check(board, a):
                        wins.append(a)
                    board[a] = 0
                if wins:
                    actions = wins
                else:
                    blocks = []
                    for a in cand:
                        board[a] = -me
                        if win_check(board, a):
                            blocks.append(a)
                        board[a] = 0
                    if blocks:
                        actions = blocks
        if rich:
            scores = rich_scores(node.board, node.player, actions)
        else:
            scores = [1.0 + 0.5 * sum(1 for j in neighbors8[a] if node.board[j] != 0)
                      for a in actions]
        total = sum(scores) + len(scores) * 1e-9
        node.prior = {a: (s + 1e-9) / total for a, s in zip(actions, scores)}
        # ascending by prior, so that list.pop() yields the best-prior action
        node.untried = [a for _, a in sorted(zip(scores, actions))]
        node.children = {}
        node.prior_mass = 0.0

    def widening_width(node):
        """Maximum admitted children at the node's current visit count.

        width(N) = min(|A|, max(PW_MIN, floor(PW_C * sqrt(N + 1)))).  The
        square-root growth lets visits deepen promising lines first, while
        every legal action remains eligible once the budget is large enough.
        """
        return min(len(node.prior), max(PW_MIN, int(PW_C * (node.N + 1) ** 0.5)))

    def can_widen(node):
        return bool(node.untried) and len(node.children) < widening_width(node)

    def puct_select(node):
        """PUCT among admitted children, with priors renormalised to their
        currently admitted mass so widening does not shrink exploration."""
        sq_n = node.N ** 0.5
        inv_mass = 1.0 / node.prior_mass if node.prior_mass > 0.0 else 0.0
        best_a, best_s = None, -1e18
        for a, ch in node.children.items():
            s = ch.Q + C_PUCT * (node.prior[a] * inv_mass) * sq_n / (1 + ch.N)
            if s > best_s:
                best_s, best_a = s, a
        return best_a

    # batched random numbers (much cheaper than one call per ply); the large
    # batch means a refill is needed only once per ~100 rollouts
    RBN = max(4096, ROLLOUT_DEPTH + 8)
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
        (L-1)^2 * (1 + KAPPA * playable_ends).  An open 3 (16) outweighs a
        fully dead four (9) but NOT a one-ended four (22.5): the score ranks
        threats, it is not a tactical guarantee -- forced lines are handled by
        the hard rules and by the search.

        Connected geometry is precomputed: this loop uses only flat Python
        list accesses, without coordinate arithmetic or edge lookups."""
        sb = sw = 0.0
        for line in connected_lines:
            k, size = 0, len(line)
            while k < size:
                v = bd[line[k]]
                if v == 0:
                    k += 1
                    continue
                start = k
                k += 1
                while k < size and bd[line[k]] == v:
                    k += 1
                length = k - start
                # Overlines are NOT strong threats in this exact-five game.
                if length < 2 or length > 5:
                    continue
                ends = int(start > 0 and bd[line[start - 1]] == 0)
                ends += int(k < size and bd[line[k]] == 0)
                w = (length - 1) ** 2 * (1.0 + KAPPA * ends)
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
        empties = list(node.empties)  # scratch copy for swap/pop removal
        to_move = node.player
        draw_value = 1.0 if node.player == WHITE else -1.0
        for _ in range(ROLLOUT_DEPTH):
            if not empties:
                return draw_value     # draw: White wins
            k = int(next_rand() * len(empties))
            a = empties[k]
            empties[k] = empties[-1]  # O(1) removal instead of shifting the tail
            empties.pop()
            bd[a] = to_move
            if win_check(bd, a):
                return 1.0 if to_move == node.player else -1.0
            to_move = -to_move
        # The last allowed ply can fill the board: do not statically evaluate
        # that draw (which is a White win) as a nonterminal position.
        if not empties:
            return draw_value
        return static_eval(bd, node.player)

    # ---- hard tactical rules at the root (cheap, and they save time) ----
    empties0 = [i for i in range(n_cells) if board_flat[i] == 0]
    if not empties0:
        raise ValueError('best_move requires at least one empty point')
    if len(empties0) == 1:
        return divmod(empties0[0], n_cols)

    mine, my_targets = tactical_targets(board_flat, color)
    if mine:
        return divmod(mine[0], n_cols)
    theirs, their_targets = tactical_targets(board_flat, -color)
    if theirs:
        root_actions = theirs
    else:
        forks = [a for a, targets in my_targets.items() if len(targets) >= 2]
        if forks:
            return divmod(min(forks), n_cols)
        opponent_forks = {a: targets for a, targets in their_targets.items()
                          if len(targets) >= 2}
        if opponent_forks:
            # A quiet defence may occupy the launch cell OR one of its two
            # winning endpoints. Restricting to launch cells alone discards
            # legal defences, especially against broken/diagonal threats.
            safe = set(empties0)
            for a, targets in opponent_forks.items():
                safe.intersection_update({a} | (targets if len(targets) == 2 else set()))
            # A forcing four gains tempo and must also remain searchable.
            root_actions = sorted(safe | set(my_targets))
            if not root_actions:  # already lost: still choose a legal resistance
                root_actions = sorted(opponent_forks)
        else:
            root_actions = empties0
    if len(root_actions) == 1:
        return divmod(root_actions[0], n_cols)

    # ---- PUCT-MCTS ----
    root = Node(board_flat[:], color)
    root.empties = empties0         # full empty set (root_actions may be a subset)
    expand(root, root_actions, rich=True)

    while keep_searching():
        node, path = root, []
        while True:
            if node.terminal_value is not None:
                break                        # game already decided at `node`
            if node.children is None:            # first visit: expand
                expand(node, None, rich=False)
            if can_widen(node):                   # progressive widening
                a = node.untried.pop()
                child = create_child(node, a)
                node.children[a] = child
                node.prior_mass += node.prior[a]
                path.append((node, a))
                node = child
                break
            a = puct_select(node)                # selection phase
            path.append((node, a))
            node = node.children[a]

        # evaluation phase.  A freshly created child may already be terminal
        # (create_child sets terminal_value), so consult the node's own state
        # rather than a descent-time flag: otherwise a finished game would be
        # rolled out and a forced win evaluated as a random playout.
        if node.terminal_value is not None:
            v = node.terminal_value              # for the player to move here
        else:
            v = rollout(node)

        # backpropagation: exactly one visit per traversed node.  The root is
        # counted once here; every traversed child is counted once below (the
        # descent used to increment each node as well, so non-root visits were
        # counted twice while values were summed once -- which distorted Q and
        # the PUCT exploration term).  The sign flips at every level.
        root.N += 1
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
    rows, cols = board.shape
    # Python scalar indexing avoids NumPy overhead in the small nested loops.
    cells, h, v = board.tolist(), edgeH.tolist(), edgeV.tolist()

    def continues_line(r, c, nr, nc, color):
        # Test bounds before indexing; negative NumPy indices would wrap.
        if not (0 <= nr < rows and 0 <= nc < cols):
            return False
        if cells[nr][nc] != color:
            return False

        # Horizontal edges belong to the left cell, vertical edges to the upper cell.
        if r == nr:
            return h[r][min(c, nc)]
        if c == nc:
            return v[min(r, nr)][c]
        return True  # Adjacent diagonal points are always connected.

    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        for r in range(rows):
            for c in range(cols):
                color = cells[r][c]
                # Skip empty cells and any cell that is not the start of its run.
                if color == 0 or continues_line(r, c, r - dr, c - dc, color):
                    continue

                length = 1
                end_r, end_c = r, c
                while continues_line(end_r, end_c, end_r + dr, end_c + dc, color):
                    length += 1
                    end_r += dr
                    end_c += dc

                # Count the entire connected run so six in a row cannot win.
                if length == 5:
                    winning_line = np.array(
                        [(r + i * dr, c + i * dc) for i in range(5)], dtype=int
                    )
                    return True, winning_line

    # Check for a draw only after ruling out a win on the last empty point.
    if not np.any(board == 0):
        return True, np.full((5, 2), -1.0)

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
    self_play = abs(human_color) != 1
    current_color = +1
    clock = pg.time.Clock()
    missing_poll = object()

    def commit_move(r, c):
        nonlocal current_color, gameover, congratulated, previous_move
        board[r, c] = current_color
        if previous_move is not None:
            pr, pc = previous_move
            draw_stone(surface, pr, pc, board[pr, pc])
        draw_stone(surface, r, c, current_color, highlight=True)
        previous_move = (r, c)
        gameover, winning_line = check_result(state)
        if gameover:
            if np.all(winning_line == -1):
                # Keep the detector's draw sentinel, but award the game to
                # White as required by the rules (Black moved first).
                print_result(surface, -1)
            else:
                first_r, first_c = winning_line[0]
                winner = board[first_r, first_c]
                for win_r, win_c in winning_line:
                    draw_stone(surface, win_r, win_c, winner, highlight=True)
                print_result(surface, winner)
            congratulated = True
        else:
            current_color = -current_color

    def poll_during_search():
        nonlocal running
        # All clicks during an AI turn are stale. Consume them, but honour
        # QUIT promptly instead of waiting for the full search budget.
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
        return not running

    try:
        while running:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    running = False
                    break
                if (event.type == pg.MOUSEBUTTONDOWN and event.button == 1
                        and not gameover and not self_play
                        and current_color == human_color):
                    x, y = event.pos
                    r = round((y - PAD_y) / SEP_y)
                    c = round((x - PAD_x) / SEP_x)
                    if (0 <= r < NUM_OF_ROWS and 0 <= c < NUM_OF_COLS
                            and board[r, c] == 0):
                        commit_move(r, c)

            if (running and not gameover
                    and (self_play or current_color != human_color)):
                # The optional hook keeps best_move usable by an external master
                # program; restore any pre-existing hook even on an exception.
                old_poll = globals().get('_MCTS_EVENT_POLL', missing_poll)
                globals()['_MCTS_EVENT_POLL'] = poll_during_search
                try:
                    r, c = best_move(state, current_color)
                finally:
                    if old_poll is missing_poll:
                        globals().pop('_MCTS_EVENT_POLL', None)
                    else:
                        globals()['_MCTS_EVENT_POLL'] = old_poll
                poll_during_search()
                if not running:
                    break
                if not (isinstance(r, (int, np.integer))
                        and isinstance(c, (int, np.integer))
                        and 0 <= r < NUM_OF_ROWS and 0 <= c < NUM_OF_COLS
                        and board[r, c] == 0):
                    # Do not silently search forever if an engine violates its API.
                    raise ValueError('best_move returned an illegal move')
                commit_move(r, c)

            clock.tick(60)
    except BaseException:
        pg.quit()
        raise

    ###########################################################################
    
    pg.quit()
    return state

if __name__ == '__main__':
    
    SEED = 5002 # used to initalize board; you can change it
    WINDOW_CAPTION += f' (seed: {SEED})'
    
    np.random.seed(SEED)
    state = main(human_color=+1)