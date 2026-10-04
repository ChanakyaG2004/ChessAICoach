# Physical Chess Coach

This app watches a physical chessboard through a webcam. When you press
**Check my move**, it detects settled square changes, accepts one move that fits the current legal position, and uses a
local Stockfish engine to coach a game. The human plays from the **top of the
rotated board image**; the coach plays from the bottom. On the coach's turn,
the app shows a move to make on the physical board. On the human's turn, it
reviews the accepted move and shows feedback. It never moves a piece itself.

Small dark marks on pale pieces can improve camera contrast. The move detector
compares changes in local contours, including those marks, against the confirmed
position. Piece names come from that recorded position and legal moves; identical
marks do not visually identify a pawn, knight, or another piece type.

**Save now** saves the game and local camera diagnostics, including the accepted
reference image, the current view, board corners, and square scores. These help
reproduce a missed move without replacing the confirmed reference.

The game is saved as PGN after every accepted move. Resync, undo, and new-game
controls provide a way to recover when the camera and physical board diverge.

**Text is the primary coaching interface.** Open `http://127.0.0.1:8765` in
Codex's built-in browser after starting the app. Ask about a better move, why
it works, your last move, a candidate move, or chess concepts. Follow-up
questions use the current position and the preceding recommendation.
Optional voice uses the same chat and keeps every explanation visible.

## Set up and run on macOS

From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
brew install stockfish
python main.py --top-color black --board-rotation cw
```

For this already configured Mac, run `./Start\ Chess\ Coach.command` from
Codex. It uses the camera orientation tested here. On a fresh checkout,
`bash scripts/setup.sh` creates the Python environment and installs Stockfish
through Homebrew when available.

## Text conversation and optional voice

The local chat page shows the expected position, the camera board, the current
physical action, and the conversation. It also offers setup confirmation,
resync, undo, new game, save, and manual move controls. Leave it open beside
the physical board while playing.

Try:

- “What is a better move here?”
- “Why?”
- “How was my last move?”
- “What if I move my knight to f3?”
- “What can my opponent do next?”

Stockfish supplies the suggested moves and evaluations. The local Ollama model
writes conversational explanations using the recorded position, verified move
effects, checked continuations, and recent conversation. It knows the move shown
on screen, so “why should I move that pawn?” works without first asking for a
recommendation in chat. Follow-ups such as “explain that more simply” or “why not
move a knight first?” can explore the idea and compare legal alternatives.

The conversational model defaults to `llama3.1:latest`, already installed on this Mac.
Set `CHESS_COACH_MODEL` to another installed Ollama model to change it. If the model
is unavailable or its response fails the evidence checks, the coach uses factual
explanations, including move purpose and tradeoffs. `--no-language-model` uses
that fallback directly. No cloud API key is required.

Voice is **off by default**. Enable it in the page, click the microphone to
record a short question, then stop recording. Check or edit the transcript
before sending it. Spoken replies are optional, and you can stop playback.
Recording begins only when you click the microphone. Allow microphone access
when Codex/macOS asks. Text chat stays usable if microphone access is denied.

Local speech recognition requires whisper.cpp and an English model:

```bash
bash scripts/setup_voice.sh
```

Speech is transcribed on this Mac; replies use the installed macOS voice.
The page reports unavailable speech components. Recordings are temporary and
are removed after processing. Chat history stays in memory for the current
app session; only the chess game is saved to PGN.

Use `--chat-port 8766` if the default port is busy, or `--chat-port 0` for an
available port (printed at startup). `--no-chat` keeps only the camera windows.
The server binds to this computer's loopback address and is not published.

This Mac has Stockfish 19 installed through Homebrew at
`/opt/homebrew/bin/stockfish`. If it is installed elsewhere, pass
`--engine-path /path/to/stockfish` or set `STOCKFISH_EXECUTABLE`. The app
searches standard Homebrew paths automatically. If Stockfish is unavailable,
move tracking continues without coaching. Use `--no-engine` to run that way
intentionally. `--analysis-time 0.25` controls the seconds spent on each
engine search; the default is 0.25.

On macOS, allow **ChatGPT** under System Settings → Privacy & Security → Camera
when running through Codex. Camera access from a restricted Codex command can
still fail with that switch enabled; run the command from Codex with camera
access outside its restricted command sandbox. The app uses OpenCV camera 0
by default; use `--camera-index N` for another camera.

The example above matches a camera view with Black on the **left** and White on
the right. Clockwise rotation puts Black at the top and White at the bottom.
For a different setup, choose `--board-rotation none|cw|ccw|half` and set
`--top-color` to the color at the top **after** rotation. The top side is the
human side. If omitted, `--top-color` is requested in the terminal.

The physical pieces must match the position the program expects before it
records its first stable image. The app pauses at startup while you check the
physical setup against **Expected pieces** in the browser (or the FEN in the
terminal). Clear your hand and click **Confirm physical position**, or press
`y` in a camera window, when the board outline is reliable. `--assume-setup` skips
this confirmation only when you have already checked the board. It starts
from the standard position unless you supply a complete FEN:

```bash
python main.py --top-color black --board-rotation cw \
  --fen "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"
```

The FEN includes the side to move, castling rights, en passant square, and move
numbers. The camera does **not** identify every piece in the starting layout.
Check the physical setup and top color yourself. A legal move is checked
against that supplied position, not a fully recognized piece map.

## Play in Codex

Keep the local browser chat open as the main interface. **Physical camera**
shows the real board, and **Expected pieces** shows the position the app knows.
During undo, new game, or manual confirmation, Expected pieces shows the target
position to arrange before confirming. The recorded game remains unchanged
until confirmation succeeds.

The two supporting windows are **Camera - board outline** and **Physical Chess
Coach**. The camera view shows the detected board outline. The coach view shows
the rectified board with square labels, whose turn it is, the next physical
action, engine feedback, detection status, recent moves, and the PGN path.
Keep a window focused when using keys.

Make **one complete physical move at a time**, remove your hand, then press
**Check my move** in the browser. In a camera window, you can also press `Space`.
The app collects fresh stable frames and compares the resulting square changes with the legal
moves for the recorded position and turn. If it accepts a human move, it saves
the game and reviews the move with Stockfish. On the coach's turn, play the
displayed suggestion on the physical board, clear your hand, and press **Check my move** again.
If a different legal move is played for the coach, the app follows the actual
physical board and notes the difference. A clear out-of-turn move is reported
and rejected. Rejected changes leave the internal position untouched.

Moves are recorded only after a check request. Each press checks at most one
move; a failed check or no detected change ends that request. Finish or correct
the move, then press the button again. If tracking is lost or the board does
not settle within a few seconds, the request ends without changing the game.
For castling, move both the king and rook before checking. After a promotion
check, choose the promoted piece in the displayed controls.

To reset the board, press **New game**, arrange the standard starting position,
then press **Confirm physical position**. This records a fresh reference and
starts a new PGN. Checking moves is disabled until setup is confirmed.

Pass `--debug-windows` at launch to show the edge, Hough-line, candidate,
lighting-normalized, warped-board, and square-change diagnostic windows.
The square-change scores and visual fit are image heuristics, not calibrated
probabilities.

## Keys and recovery

| Key | Action |
| --- | --- |
| `Space` | Check one completed physical move using the camera. |
| `q` | Quit; close camera and engine. |
| `y` at startup | Confirm the physical setup after checking it against the expected FEN in the terminal. |
| `c` | Click four outer board corners in the **Camera** window, in any order. After the new outline is accepted, confirm a resync with `y`. |
| `r` | Resync the image baseline to the current recorded position. Arrange every physical piece to match the displayed FEN, clear your hand, then press `y`. |
| `u` | Undo the last accepted move. Restore the previous physical position, clear your hand, then press `y`. |
| `n` | Start a new standard game and a new PGN. Set up all physical pieces, clear your hand, then press `y`. |
| `m` | Record a missed move manually: type UCI such as `e2e4` (or `a7a8q` for promotion), press Enter, check the physical position, then `y`. The move must be legal. |
| `y` / `Esc` | Confirm / cancel a pending resync, undo, or new game. |
| `s` | Save PGN now and save camera, edge, candidate, coach, and board screenshots in `debug_captures/`. |
| `1`–`4` | Confirm a detected promotion as queen, rook, bishop, or knight. |
| `Enter` | Confirm a displayed rook move that may be an unfinished castle. |

Resync updates the **image baseline only**; it does not infer missing pieces or
change the recorded FEN. Use it only after checking that the physical board
matches the shown position. Undo and new game also depend on you arranging the
physical pieces first. The `y` confirmation checks for a reliable board outline
and several stable frames, but does not verify each piece's identity. If the camera moves, recalibrate the
corners before confirming a resync.

If a completed legal move is missed because an earlier piece adjustment changed
the image, use **Manual move** to record that one move and refresh the baseline.
This is an explicit confirmation by you. Resync alone never advances the game.
While typing a manual move in the camera window, press Esc before other hotkeys.

If automatic board detection misses the playing surface, press `c` and click
the four outer corners in the Camera window. For a fixed camera, you can also
provide four pixel coordinates at launch:

```bash
python main.py --top-color black --board-rotation cw \
  --corners 560,322 1063,384 1009,883 458,808
```

Those coordinates came from one 1920×1080 capture and must be recalibrated if
the camera or board moves. Camera resolution and measured brightness/FPS appear
in the terminal. The app requests 1920×1080 at 60 FPS and requests 30 FPS if
the initial high-rate frames are severely underexposed.

## Save and resume

New games are saved automatically to `games/game_<timestamp>.pgn` beside
`main.py`. Set a specific output path with `--save-pgn /path/to/game.pgn`.
The PGN contains the initial FEN, human color, full accepted move history, and
current FEN. Writes replace the file atomically, and a damaged or inconsistent
game is rejected on resume.

To continue an earlier game, arrange the physical board to match the PGN's
**final position**, then run the command below. Check the printed expected FEN
and press `y` once the physical setup and board outline are correct:

```bash
python main.py --resume games/game_YYYYMMDD_HHMMSS_microseconds.pgn \
  --board-rotation cw
```

The saved human color is restored automatically. You may pass the same
`--top-color` explicitly, but a conflicting value is rejected. `--resume` and
`--fen` cannot be combined. The camera records a fresh image baseline for the
resumed position. The existing PGN is updated after accepted moves unless
`--save-pgn` points to another file.

## Detection limits

The image matcher knows the expected chess position and tests the visual
footprint of each legal move. It handles ordinary moves, captures, castling,
en passant, and promotion candidates. Promotions require a key choice because
different promoted pieces change the same squares. A rook moved first during
castling may need the king move completed or an explicit `Enter` confirmation.

Tall pieces, shadows, glare, low light, similar-looking capture replacements,
camera movement, or a stationary obstruction can confuse a visual comparison.
The app waits for a stable board and rejects weak or ambiguous changes instead
of guessing. It pauses when board tracking is unreliable. If multiple moves
occur before one is accepted, restore the last accepted physical position or
use the recovery controls with the correct setup. Increase
`--stable-seconds 1.5` for slower hand movements; adjust
`--change-threshold` only after examining the debug square scores.

Coaching is based on a short local Stockfish search. Its suggested move and
grade are guidance for the current recorded position, not a guarantee of
perfect play. The engine never sees the camera image and cannot correct a
wrong physical setup.

## Tests

```bash
python -m unittest discover -v
```

Unit tests cover board detection, legal move matching, recovery, PGN storage,
engine coaching, and the dashboard. Move-check tests verify that a click is
required, each request records at most one move, failed checks preserve the
reference, and resetting cancels pending checks. Live camera checks have verified ordinary
moves, out-of-turn rejection, and a capture on one board and lighting setup;
other physical boards and lighting may need threshold calibration.

Chat tests cover move recommendations, follow-up explanations, legal hypothetical
moves, current threats, and stale responses. Local voice tests cover audio
validation, timeouts, and temporary-file cleanup. HTTP integration tests need
loopback networking permission when run from Codex's command sandbox.
