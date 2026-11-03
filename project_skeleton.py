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

    r = np.random.randint(NUM_OF_ROWS) # a sample which
    c = np.random.randint(NUM_OF_COLS) # you can remove

    ###########################################################################
    return r, c

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
    
    # Using the sample below, you will see a game window pop out,
    # then you can click in it to place black stones and see
    # the computer place white (and highlighted) stones at random positions.
    
    while running:                   # this double loop handles
        for event in pg.event.get(): # all events you initiate
            
            if event.type == pg.QUIT: # detects whether you
                running = False       # try to close the window
            
            if (event.type == pg.MOUSEBUTTONDOWN # detects whether you
                                                 # click in the window                   
                and not gameover): # and allows the board to change
                                   # only before the game ends, i.e.
                                   # the window is locked after it

                x, y = event.pos # the position at which you click
                                 # (0, 0) is the top left corner of the window
                
                r = round( (y-PAD_y) / SEP_y )
                c = round( (x-PAD_x) / SEP_x )
                draw_stone(surface, r, c, +1, highlight=False)
                
                r, c = best_move(state, -1)
                draw_stone(surface, r, c, -1, highlight=True)
                
    # You can remove this sample.
                
    ###########################################################################
    
    pg.quit()
    return state

if __name__ == '__main__':
    
    SEED = 5002 # used to initalize board; you can change it
    WINDOW_CAPTION += f' (seed: {SEED})'
    
    np.random.seed(SEED)
    state = main(human_color=+1)