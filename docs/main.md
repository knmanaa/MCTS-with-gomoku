# Game Loop (`main`) — Integration Notes

This document describes the current `main(human_color)` implementation in
[project_skeleton.py](<../project_skeleton.py>) after the 15 November 2026
integration review. See also [the search](<best_move.md>) and
[result detection](<check_result.md>).

## Modes and lifecycle

| `human_color` | Mode |
| --- | --- |
| `+1` | Human Black, computer White |
| `-1` | Human White, computer Black |
| Other numeric values with `abs(human_color) != 1` | Computer versus itself |

Black always starts. The function initialises Pygame, an empty board, and
random H/V edges using the skeleton's constants. After the window closes,
it shuts down Pygame and returns `state = (board, edgeH, edgeV)`.

The loop maintains the supplied `running`, `gameover`, `congratulated`, and
`previous_move` variables. A Pygame clock caps idle/event-loop iterations at
60 Hz; a long search already consumes its frame time rather than receiving
an unconditional extra animation delay.

All game-loop changes remain inside the existing `main` TODO body. The
skeleton's drawing functions, initialisation, imports, and constants were
not changed by the review.

## One move-commit path

Both human and computer moves use the nested `commit_move(r, c)` helper:

1. Place `current_color` at the selected point.
2. Redraw the previous stone without its latest-move highlight.
3. Draw the new stone highlighted and update `previous_move`.
4. Call `check_result(state)` once for the real placement.
5. If terminal, highlight the returned five stones when present, announce
   the result once, and set `congratulated = True`.
6. Otherwise, switch `current_color` to its opposite.

This removes the former duplicated human/AI placement and result logic.
Once terminal, both human placement and AI search are disabled; the window
continues processing close events without repeating the announcement.

### Full-board draws

The detector still returns `(True, all -1)` when the board is full with no
winning line. The GUI now calls `print_result(surface, -1)`, displaying
**White wins!**, because the project awards drawn games to White.
It does not fabricate or highlight a White five-stone line.

This differs from the earlier GUI, which displayed `Draw!` and delegated
White's award solely to external scoring. Search terminal values, GUI
awards, and competition scoring should now use the same rule, while the
result detector retains its original sentinel API.

## Human input

A placement is accepted only for a left mouse-button press when:

- the game has not ended;
- the mode has a human player;
- the current colour is the human's colour;
- the nearest grid point, obtained by rounding the skeleton's pixel-to-grid
  transform, is inside the board; and
- that point is empty.

Clicks on occupied or out-of-bounds points and non-left buttons do not
place stones. The implementation retains nearest-grid-point rounding; it
does not add a separate pixel-distance tolerance around intersections.

## Responsive AI turns without new imports

`best_move` still runs synchronously on the main thread. Responsiveness is
provided by **cooperative polling**, not a worker thread or a second event
loop.

Immediately around each AI call, `main` temporarily installs the private
module hook `_MCTS_EVENT_POLL = poll_during_search`. The search's
`keep_searching()` helper calls this hook at simulation boundaries, no more
often than approximately once per 30 ms.

`poll_during_search` drains pending Pygame events:

- QUIT sets `running = False` and requests early search termination.
- Mouse clicks received during the AI turn are discarded rather than
  replayed on the human's next turn.
- Other queued events are consumed without additional handlers.

After the search returns, the previous hook is restored in `finally`,
including the distinction between an absent hook and an explicitly stored
`None`. The event queue is drained again before any AI placement. If QUIT
was received, the returned move is **not** committed.

Standalone calls to `best_move` do not touch the Pygame event queue unless
a caller explicitly supplies the hook. No display is required for ordinary
search use. The module-global hook is intended for this single-threaded
GUI; it is not a thread-local or concurrent-search cancellation API.

### Timing limitation

The polling interval is a target between cooperative checkpoints, not a
hard event-latency guarantee. Geometry setup, root tactics, tree descent,
and an individual rollout are not preempted. Larger boards or unusually
expensive phases can therefore delay cancellation. The default-board test
confirmed early cancellation during an actual 4.7-second-budget search,
without placing the cancelled move; it did not establish a universal
30 ms window-close bound.

## AI output validation and cleanup

Before indexing or committing an AI move, the loop requires:

- both coordinates to be Python or NumPy integers;
- in-bounds row and column indices; and
- an empty destination.

An invalid coordinate pair raises `ValueError` rather than silently
retrying forever. Malformed return values can raise during tuple unpacking;
engine exceptions propagate as well. In each case, the event hook is
restored and the loop's exception handler calls `pg.quit()` before
re-raising. Normal window closure also calls `pg.quit()`.

The exception-safe loop starts after the skeleton's initial Pygame/window
setup; this is not a claim that errors in the unchanged initialisation
code are also protected by that handler.

## Verification performed

The review used a dummy SDL display and in-memory instrumentation, without
adding production imports or changing maintained test files.

| Scenario | Observed result |
| --- | --- |
| QUIT scheduled during actual MCTS with a 4.7-second budget | Search cancelled early; no move committed; Pygame closed |
| Mouse click arriving during an AI search | Not placed on the following human turn |
| Full 2-by-2 board with no five | Four AI placements; White awarded the game exactly once |
| Winning terminal state | One announcement; no subsequent AI search |
| Post-terminal event processing | Window remains closable; results are not repeated |
| Invalid bounds, fractional coordinate, malformed return, or engine exception | Error propagates; Pygame closes; previous hook is restored |
| Explicitly pre-existing hook value `None` | Restored exactly, not removed |
| Existing headless self-play smoke test | Passed; both colours placed stones |

The current loop uses `pg.event.get()` without positional event filters, so
the existing no-argument event-queue mock remains compatible. Because search
polls and post-search drains also call `event.get()`, a mock's event-get call
count is **not** a count of rendered frames or AI moves. The smoke test's
historical comment equating these quantities should not be used to infer
how many turns were played.

## Remaining recommendations

- Maintain explicit GUI regressions for both human colours, stale clicks,
  close-before-placement, terminal lockout, and hook/error cleanup.
- Add finer cooperative checkpoints if larger boards need tighter close
  latency, without treating partially computed tactical sets as complete.
- Keep draw detection separate from the rule awarding White the game.
- Progressive widening is now enabled in the search; tactical rollouts and
  tree reuse remain future work in [the search recommendations](<best_move.md#10-recommended-next-work--not-implemented>).
