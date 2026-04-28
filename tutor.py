import anthropic
import chess
import os
import numpy as np
from scipy.stats import norm
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# ── Strategy profiles ──────────────────────────────────────────────────────────
STRATEGY_PROFILES = {
    "Aggressive": {
        "description": "Attack the opponent's king, create threats, sacrifice material for initiative.",
        "move_bias": "Prefer moves that create immediate threats, open lines toward the opponent's king, or gain attacking initiative — even at the cost of some material.",
        "plan_bias": "Focus the plan on building an attack, opening files, or creating mating threats.",
    },
    "Solid": {
        "description": "Keep your position safe, avoid risks, defend well before attacking.",
        "move_bias": "Prefer moves that consolidate the position, eliminate opponent threats, and avoid unnecessary complications.",
        "plan_bias": "Focus the plan on neutralising opponent threats and maintaining a sound structure.",
    },
    "Positional": {
        "description": "Improve your pieces, control key squares, build long-term advantages.",
        "move_bias": "Prefer moves that improve piece placement, control important squares, or create long-term structural advantages.",
        "plan_bias": "Focus the plan on piece improvement, outpost creation, and pawn structure advantages.",
    },
    "Balanced": {
        "description": "Play objectively — the best move regardless of style.",
        "move_bias": "Prefer the objectively strongest move without stylistic bias.",
        "plan_bias": "Describe the most accurate plan for the position regardless of style.",
    },
}

# ── ELO profiles ───────────────────────────────────────────────────────────────
ELO_PROFILES = {
    (0, 999): {
        "label": "complete beginner",
        "vocab_rule": (
            "Use only everyday English. Never use: 'tempo', 'initiative', 'outpost', "
            "'file', 'rank', 'battery', 'zwischenzug', 'prophylaxis', 'compensation'. "
            "Define any tactic name immediately after using it."
        ),
        "depth": "Focus only on: is my piece safe? am I leaving something free to capture?",
        "strategy_frame": "Think one move ahead only.",
        "analogy_style": "Use simple everyday analogies.",
        "comparison_depth": "One sentence on what goes wrong with the alternative.",
    },
    (1000, 1199): {
        "label": "casual player",
        "vocab_rule": "Basic chess words only: check, capture, pawn, piece, square. Define any tactic name you use.",
        "depth": "Focus on piece safety, basic tactics (forks, pins), not walking into threats.",
        "strategy_frame": "Think 2 moves ahead.",
        "analogy_style": "Simple analogies welcome but not required.",
        "comparison_depth": "Explain what the alternative allows the opponent to do.",
    },
    (1200, 1399): {
        "label": "intermediate beginner",
        "vocab_rule": "Standard vocabulary fine: fork, pin, skewer, open file, development. Avoid: prophylaxis, zugzwang, compensation, dynamic imbalance.",
        "depth": "Tactics and basic positional ideas: piece activity, king safety, pawn structure.",
        "strategy_frame": "Short-term plan over next 2-3 moves.",
        "analogy_style": "Analogies optional.",
        "comparison_depth": "Compare top 2 moves: what does each give up and gain?",
    },
    (1400, 1599): {
        "label": "intermediate player",
        "vocab_rule": "Full chess vocabulary. Reference pawn structure, piece coordination, open files, weak squares.",
        "depth": "Balance tactics and strategy. Explain the positional justification.",
        "strategy_frame": "Medium-term plan 3-5 moves.",
        "analogy_style": "No analogies needed.",
        "comparison_depth": "Compare top 2-3 candidate moves with concrete lines.",
    },
    (1600, 1799): {
        "label": "strong club player",
        "vocab_rule": "Full technical vocabulary: prophylaxis, initiative, compensation, imbalance, outpost, battery.",
        "depth": "Deep positional and tactical explanation including opponent's best response.",
        "strategy_frame": "Strategic plan in terms of the position type.",
        "analogy_style": "No analogies.",
        "comparison_depth": "Concrete variation comparison with evaluation reasoning.",
    },
    (1800, 9999): {
        "label": "advanced player",
        "vocab_rule": "Fully technical. Assume strong pattern recognition and calculation ability.",
        "depth": "Engine-aware analysis. Include nuance, subtle ideas, opponent counterplay.",
        "strategy_frame": "Long-term structural implications.",
        "analogy_style": "No analogies.",
        "comparison_depth": "Deep comparison with concrete lines and evaluation differences.",
    },
}


def get_elo_profile(elo: int) -> dict:
    for (low, high), profile in ELO_PROFILES.items():
        if low <= elo <= high:
            return profile
    return ELO_PROFILES[(1400, 1599)]


def get_alternative_moves(board: chess.Board, recommended_uci: str, n: int = 2) -> list:
    alts = []
    for move in board.legal_moves:
        if move.uci() == recommended_uci:
            continue
        try:
            alts.append(board.san(move))
        except Exception:
            continue
        if len(alts) >= n:
            break
    return alts


def get_move_explanation(
    board: chess.Board,
    move_san: str,
    move_uci: str,
    elo: int,
    evaluation: str,
    strategy: str = "Balanced",
    move_history: list = None,
) -> str:
    profile = get_elo_profile(elo)
    strat = STRATEGY_PROFILES.get(strategy, STRATEGY_PROFILES["Balanced"])
    turn = "White" if board.turn == chess.WHITE else "Black"
    alts = get_alternative_moves(board, move_uci, n=2)
    alt_str = ", ".join(alts) if alts else "none available"
    history_str = ""
    if move_history and len(move_history) > 0:
        history_str = f"Recent moves: {', '.join(move_history[-6:])}\n"

    prompt = f"""You are a chess tutor explaining a move to a {profile['label']} (ELO ~{elo}).
The player has chosen a {strategy} style: {strat['description']}

POSITION (FEN): {board.fen()}
{history_str}TURN: {turn}
RECOMMENDED MOVE: {move_san}
EVALUATION: {evaluation}
ALTERNATIVES: {alt_str}

YOUR EXPLANATION MUST HAVE THREE LABELLED SECTIONS:

**Why this move?**
Explain why {move_san} fits the {strategy} style.
{profile['depth']}
{strat['move_bias']}
Vocabulary rule (strictly enforced): {profile['vocab_rule']}

**Your strategy going forward**
{strat['plan_bias']}
{profile['strategy_frame']}
What should the player do over the next few moves?

**How does this compare to other moves?**
{profile['comparison_depth']}
Reference specifically: {alt_str}

HARD RULES: 150-250 words max. Each section present and labelled."""

    try:
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text.strip()
    except anthropic.APIError as e:
        return f"Explanation unavailable: {e}"


def get_position_commentary(
    board: chess.Board, elo: int, strategy: str = "Balanced"
) -> str:
    profile = get_elo_profile(elo)
    strat = STRATEGY_PROFILES.get(strategy, STRATEGY_PROFILES["Balanced"])
    turn = "White" if board.turn == chess.WHITE else "Black"
    statuses = []
    if board.is_check():
        statuses.append("The king is in check.")
    if board.is_checkmate():
        statuses.append("Checkmate.")
    if board.is_stalemate():
        statuses.append("Stalemate.")

    prompt = f"""You are a chess tutor giving brief commentary to a {profile['label']} (ELO ~{elo}).
The player uses a {strategy} style: {strat['description']}

Position (FEN): {board.fen()}
{' '.join(statuses)}
Turn: {turn}

In 2-3 sentences:
1. Most important feature of this position to notice
2. One concrete thing to look for next, consistent with {strategy} style

Vocabulary rule: {profile['vocab_rule']}
Be direct and specific."""

    try:
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text.strip()
    except anthropic.APIError as e:
        return f"Commentary unavailable: {e}"


# ── Bayesian ELO Estimator ─────────────────────────────────────────────────────
ELO_BANDS = np.array([800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400])
FEEDBACK_OPTIONS = ["Too Simple", "Right Level", "Too Complex"]


def _likelihood(
    feedback: str, true_elo: int, displayed_elo: int, sigma: float = 300
) -> float:
    """
    P(feedback | true_elo, displayed_elo)
    Models feedback as a Gaussian observation of ELO gap.
    """
    delta = true_elo - displayed_elo
    threshold = sigma / 2
    z_upper = (threshold - delta) / sigma
    z_lower = (-threshold - delta) / sigma
    p_too_complex = norm.cdf(z_lower)
    p_right_level = norm.cdf(z_upper) - norm.cdf(z_lower)
    p_too_simple = 1 - norm.cdf(z_upper)
    liks = {
        "Too Simple": max(p_too_simple, 1e-10),
        "Right Level": max(p_right_level, 1e-10),
        "Too Complex": max(p_too_complex, 1e-10),
    }
    return liks[feedback]


class BayesianELOEstimator:
    """
    Online Bayesian estimator of player true ELO.

    Prior:    Uniform over ELO_BANDS
    Likelihood: Gaussian model of feedback-ELO-gap relationship
    Update:   Bayes rule after each explanation feedback
    Output:   MAP estimate drives displayed ELO
    """

    def __init__(self, sigma: float = 300):
        self.sigma = sigma
        self.elo_bands = ELO_BANDS
        self.posterior = np.ones(len(ELO_BANDS)) / len(ELO_BANDS)
        self.history = []

    @property
    def map_estimate(self) -> int:
        return int(self.elo_bands[np.argmax(self.posterior)])

    @property
    def posterior_mean(self) -> float:
        return float(np.dot(self.posterior, self.elo_bands))

    @property
    def posterior_std(self) -> float:
        mean = self.posterior_mean
        variance = float(np.dot(self.posterior, (self.elo_bands - mean) ** 2))
        return float(np.sqrt(variance))

    def update(self, feedback: str, displayed_elo: int):
        """Bayesian update: posterior ∝ likelihood × prior."""
        liks = np.array(
            [
                _likelihood(feedback, int(e), displayed_elo, self.sigma)
                for e in self.elo_bands
            ]
        )
        unnorm = liks * self.posterior
        self.posterior = unnorm / unnorm.sum()
        self.history.append(
            {
                "displayed_elo": displayed_elo,
                "feedback": feedback,
                "map_estimate": self.map_estimate,
                "posterior_mean": round(self.posterior_mean, 1),
                "posterior_std": round(self.posterior_std, 1),
                "posterior_snapshot": self.posterior.copy(),
            }
        )
        return self.map_estimate

    def reset(self):
        self.posterior = np.ones(len(ELO_BANDS)) / len(ELO_BANDS)
        self.history = []

    def confidence_label(self) -> str:
        std = self.posterior_std
        if std < 150:
            return "High confidence"
        elif std < 300:
            return "Moderate confidence"
        else:
            return "Still learning your level"
