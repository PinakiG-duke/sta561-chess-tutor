import chess
import chess.engine
import os
import asyncio
import numpy as np
from collections import Counter
from dotenv import load_dotenv

load_dotenv()

# Windows asyncio fix
if os.name == "nt":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

STOCKFISH_PATH = os.getenv("STOCKFISH_PATH")

# ELO → Stockfish Skill Level mapping
ELO_TO_SKILL = {
    800: 1,
    1000: 3,
    1200: 5,
    1400: 8,
    1600: 11,
    1800: 14,
    2000: 17,
    2200: 19,
    2400: 20,
}


def elo_to_skill_level(elo: int) -> int:
    closest = min(ELO_TO_SKILL.keys(), key=lambda k: abs(k - elo))
    return ELO_TO_SKILL[closest]


def get_board_from_fen(fen: str) -> chess.Board:
    try:
        return chess.Board(fen)
    except ValueError as e:
        raise ValueError(f"Invalid FEN: {e}")


def get_legal_moves(board: chess.Board) -> list:
    return [move.uci() for move in board.legal_moves]


def get_best_move(board: chess.Board, elo: int, time_limit: float = 1.0) -> dict:
    skill_level = elo_to_skill_level(elo)
    with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:
        engine.configure({"Skill Level": skill_level})
        result = engine.play(
            board, chess.engine.Limit(time=time_limit), info=chess.engine.INFO_ALL
        )
        move = result.move
        move_san = board.san(move)
        info = engine.analyse(board, chess.engine.Limit(time=0.5))
        score = info.get("score")
        if score:
            pov = score.white()
            if pov.is_mate():
                eval_str = f"Mate in {pov.mate()}"
            else:
                cp = pov.score()
                eval_str = f"{cp/100:+.2f}" if cp is not None else "N/A"
        else:
            eval_str = "N/A"
    return {
        "move_uci": move.uci(),
        "move_san": move_san,
        "skill_level": skill_level,
        "elo": elo,
        "evaluation": eval_str,
        "fen_before": board.fen(),
    }


def get_position_evaluation(board: chess.Board) -> str:
    with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:
        info = engine.analyse(board, chess.engine.Limit(time=0.5))
        score = info.get("score")
        if score:
            pov = score.white()
            if pov.is_mate():
                return f"Mate in {pov.mate()}"
            cp = pov.score()
            if cp is not None:
                adv = "White" if cp > 0 else "Black"
                return f"{adv} +{abs(cp/100):.2f}"
    return "Equal"


def get_position_evaluation_numeric(board: chess.Board) -> float:
    with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:
        info = engine.analyse(board, chess.engine.Limit(time=0.5))
        score = info.get("score")
        if score:
            pov = score.white()
            if pov.is_mate():
                m = pov.mate()
                return 10.0 if m and m > 0 else -10.0
            cp = pov.score()
            if cp is not None:
                return round(cp / 100, 2)
    return 0.0


def apply_move(board: chess.Board, move_uci: str) -> chess.Board:
    move = chess.Move.from_uci(move_uci)
    if move not in board.legal_moves:
        raise ValueError(f"Illegal move: {move_uci}")
    board.push(move)
    return board


def get_game_status(board: chess.Board) -> dict:
    return {
        "is_game_over": board.is_game_over(),
        "is_checkmate": board.is_checkmate(),
        "is_stalemate": board.is_stalemate(),
        "is_check": board.is_check(),
        "turn": "White" if board.turn == chess.WHITE else "Black",
        "fullmove_number": board.fullmove_number,
    }


# ── Position Complexity Meter ─────────────────────────────────────────────────
def get_position_complexity(
    board: chess.Board, n_samples: int = 15, skill_level: int = 10
) -> dict:
    """
    Measure position complexity via Shannon entropy of move distribution.
    Samples Stockfish n_samples times at a mid-range skill level.
    High entropy = many plausible moves = complex position.
    Low entropy  = one dominant move = clear position.

    Returns:
        entropy     : Shannon entropy in bits
        label       : 'Clear' / 'Moderate' / 'Complex'
        top_move    : most frequently chosen move (SAN)
        top_pct     : selection rate of top move
        unique_moves: number of distinct moves sampled
    """
    move_counts = Counter()
    with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:
        engine.configure({"Skill Level": skill_level})
        for _ in range(n_samples):
            result = engine.play(board, chess.engine.Limit(time=0.05))
            try:
                san = board.san(result.move)
            except Exception:
                san = result.move.uci()
            move_counts[san] += 1

    total = sum(move_counts.values())
    probs = np.array([c / total for c in move_counts.values()])
    probs = probs[probs > 0]
    entropy = float(-np.sum(probs * np.log2(probs))) if len(probs) > 0 else 0.0

    top_move, top_count = move_counts.most_common(1)[0]
    top_pct = round(top_count / total * 100, 1)

    if entropy < 0.8:
        label = "Clear"
    elif entropy < 1.8:
        label = "Moderate"
    else:
        label = "Complex"

    return {
        "entropy": round(entropy, 3),
        "label": label,
        "top_move": top_move,
        "top_pct": top_pct,
        "unique_moves": len(move_counts),
        "n_samples": n_samples,
    }


# ── Evaluation Confidence Indicator ───────────────────────────────────────────
def get_evaluation_confidence(board: chess.Board) -> dict:
    """
    Compare Stockfish evaluation at shallow (depth 5) vs deep (depth 15).
    Large gap = evaluation is volatile / position is tactically sharp.
    Small gap = evaluation is stable / engine agrees with itself.

    Returns:
        shallow_eval : centipawn score at depth 5
        deep_eval    : centipawn score at depth 15
        gap          : absolute difference in pawn units
        confidence   : 'High' / 'Medium' / 'Low'
        label        : human-readable string
    """

    def eval_at_depth(engine, board, depth):
        info = engine.analyse(board, chess.engine.Limit(depth=depth))
        score = info.get("score")
        if score:
            pov = score.white()
            if pov.is_mate():
                m = pov.mate()
                return 10.0 if m and m > 0 else -10.0
            cp = pov.score()
            if cp is not None:
                return round(cp / 100, 2)
        return 0.0

    with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:
        shallow = eval_at_depth(engine, board, depth=5)
        deep = eval_at_depth(engine, board, depth=15)

    gap = abs(deep - shallow)

    if gap < 0.3:
        confidence = "High"
        label = "Evaluation stable — position is clear"
    elif gap < 0.8:
        confidence = "Medium"
        label = "Some depth-sensitivity — moderate complexity"
    else:
        confidence = "Low"
        label = "Evaluation volatile — position is tactically sharp"

    return {
        "shallow_eval": shallow,
        "deep_eval": deep,
        "gap": round(gap, 3),
        "confidence": confidence,
        "label": label,
    }
