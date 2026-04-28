import streamlit as st
import chess
import chess.svg
import base64
import pickle
import os
import random
import numpy as np
import pandas as pd
import altair as alt
from engine import (
    get_board_from_fen,
    get_best_move,
    get_legal_moves,
    get_position_evaluation,
    get_position_evaluation_numeric,
    get_position_complexity,
    get_evaluation_confidence,
    apply_move,
    get_game_status,
    elo_to_skill_level,
)
from tutor import (
    get_move_explanation,
    get_position_commentary,
    STRATEGY_PROFILES,
    BayesianELOEstimator,
    ELO_BANDS,
    FEEDBACK_OPTIONS,
)
from endgame_env import KingPawnEnv, BOARD_SIZE

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Chess Tutor", page_icon="♟️", layout="wide")
st.title("♟️ Arete")
st.caption(
    "The Chess Tutor That Does Not Try to Win · ELO-calibrated recommendations · Bayesian ELO inference · Endgame Trainer"
)


# ── Load endgame agent ─────────────────────────────────────────────────────────
@st.cache_resource
def load_endgame_agent():
    path = os.path.join(os.path.dirname(__file__), "endgame_agent.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


ENDGAME_AGENT = load_endgame_agent()

# ── Session state ──────────────────────────────────────────────────────────────
DEFAULTS = {
    "board": chess.Board(),
    "move_history": [],
    "eval_history": [],
    "last_explanation": "",
    "last_move": "",
    "mode": "tutor",
    "bot_commentary": "",
    "pending_move": "",
    "pending_san": "",
    "strategy": "Balanced",
    "bayesian": BayesianELOEstimator(),
    "use_bayesian": False,
    "complexity": None,
    "confidence": None,
    "feedback_given": False,
    # Endgame trainer state
    "eg_env": KingPawnEnv(),
    "eg_started": False,
    "eg_history": [],
    "eg_commentary": "",
    "eg_result": None,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Helpers ────────────────────────────────────────────────────────────────────
def render_board(board: chess.Board, last_move_uci: str = None) -> str:
    last_move = None
    if last_move_uci:
        try:
            last_move = chess.Move.from_uci(last_move_uci)
        except Exception:
            pass
    svg = chess.svg.board(
        board,
        lastmove=last_move,
        size=400,
        colors={
            "square light": "#f0d9b5",
            "square dark": "#b58863",
            "square light lastmove": "#cdd16e",
            "square dark lastmove": "#aaa23a",
        },
    )
    b64 = base64.b64encode(svg.encode()).decode()
    return f'<img src="data:image/svg+xml;base64,{b64}" width="400"/>'


def record_eval(move_san, board_after, mover):
    try:
        cp = get_position_evaluation_numeric(board_after)
        cp = max(-10.0, min(10.0, cp))
        st.session_state.eval_history.append(
            {
                "move_number": len(st.session_state.eval_history) + 1,
                "move": move_san,
                "eval": cp,
                "side": mover,
            }
        )
    except Exception:
        pass


def render_eval_chart():
    if len(st.session_state.eval_history) < 1:
        st.caption("Play moves to see your position trend.")
        return
    df = pd.DataFrame(st.session_state.eval_history)
    df["color"] = df["eval"].apply(lambda x: "Improved" if x > 0 else "Worsened")
    zero = (
        alt.Chart(pd.DataFrame({"y": [0]}))
        .mark_rule(color="gray", strokeDash=[4, 4])
        .encode(y="y:Q")
    )
    area = (
        alt.Chart(df)
        .mark_area(
            line={"color": "#4a90d9"},
            color=alt.Gradient(
                gradient="linear",
                stops=[
                    alt.GradientStop(color="#d63031", offset=0),
                    alt.GradientStop(color="#00b894", offset=1),
                ],
                x1=1,
                x2=1,
                y1=1,
                y2=0,
            ),
            opacity=0.4,
        )
        .encode(
            x=alt.X("move_number:O", title="Move", axis=alt.Axis(labelAngle=0)),
            y=alt.Y(
                "eval:Q", title="Evaluation (pawns)", scale=alt.Scale(domain=[-10, 10])
            ),
            tooltip=["move_number", "move", "eval", "side"],
        )
    )
    points = (
        alt.Chart(df)
        .mark_point(size=80, filled=True)
        .encode(
            x=alt.X("move_number:O"),
            y=alt.Y("eval:Q"),
            color=alt.Color(
                "color:N",
                scale=alt.Scale(
                    domain=["Improved", "Worsened"], range=["#00b894", "#d63031"]
                ),
                legend=alt.Legend(title="Position"),
            ),
            tooltip=["move_number", "move", "eval", "side"],
        )
    )
    st.altair_chart(
        (zero + area + points)
        .properties(height=200, title="Position Evaluation Over Time")
        .interactive(),
        use_container_width=True,
    )


def render_posterior_chart(estimator):
    df = pd.DataFrame(
        {
            "ELO": [str(e) for e in ELO_BANDS],
            "Probability": estimator.posterior,
        }
    )
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X(
                "Probability:Q", scale=alt.Scale(domain=[0, 1]), title="P(true ELO)"
            ),
            y=alt.Y("ELO:O", sort=None, title="ELO Band"),
            color=alt.condition(
                alt.datum.Probability == float(estimator.posterior.max()),
                alt.value("#2ecc71"),
                alt.value("#3498db"),
            ),
            tooltip=["ELO", alt.Tooltip("Probability:Q", format=".3f")],
        )
        .properties(height=220, title="Posterior Belief — Your True ELO")
    )
    st.altair_chart(chart, use_container_width=True)


def get_active_elo(manual_elo):
    if st.session_state.use_bayesian:
        return st.session_state.bayesian.map_estimate
    return manual_elo


def agent_act(state, legal_actions, q_table):
    """Greedy action selection from the loaded Q-table."""
    if not legal_actions:
        return None
    if state in q_table and q_table[state]:
        return max(legal_actions, key=lambda a: q_table[state].get(a, 0.0))
    return random.choice(legal_actions)


def eg_move_label(action):
    atype, nr, nc = action
    piece = "King" if atype == "K" else "Pawn"
    return f"{piece} to {chr(96+nc)}{nr}"


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")

    mode = st.radio("Mode", ["Tutor Mode", "Play vs Bot", "Endgame Trainer"], index=0)
    st.session_state.mode = {
        "Tutor Mode": "tutor",
        "Play vs Bot": "play",
        "Endgame Trainer": "endgame",
    }[mode]

    st.divider()

    if st.session_state.mode != "endgame":
        st.subheader("ELO Configuration")
        use_bayesian = st.toggle(
            "Auto-detect my ELO (Bayesian)",
            value=st.session_state.use_bayesian,
            help="System infers your ELO from explanation feedback.",
        )
        st.session_state.use_bayesian = use_bayesian

        if use_bayesian:
            bayes = st.session_state.bayesian
            st.info(
                f"**Estimated ELO: {bayes.map_estimate}**  \n"
                f"Uncertainty: ±{bayes.posterior_std:.0f} pts  \n"
                f"{bayes.confidence_label()}"
            )
            render_posterior_chart(bayes)
            if st.button("Reset ELO estimation", use_container_width=True):
                st.session_state.bayesian.reset()
                st.rerun()
            manual_elo = bayes.map_estimate
        else:
            manual_elo = st.select_slider(
                "Your ELO Rating",
                options=[800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400],
                value=1200,
            )
            st.caption(f"Stockfish Skill Level: {elo_to_skill_level(manual_elo)} / 20")

        st.divider()
        st.subheader("Playing Style")
        strategy_options = list(STRATEGY_PROFILES.keys())
        selected_strategy = st.radio(
            "Choose your style",
            strategy_options,
            index=strategy_options.index(st.session_state.strategy),
        )
        st.session_state.strategy = selected_strategy
        st.info(
            f"**{selected_strategy}:** "
            f"{STRATEGY_PROFILES[selected_strategy]['description']}"
        )

        st.divider()
        st.subheader("Position Setup")
        fen_input = st.text_input(
            "Custom FEN (optional)", placeholder="Leave blank for starting position"
        )
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Load Position", use_container_width=True):
                try:
                    st.session_state.board = (
                        get_board_from_fen(fen_input.strip())
                        if fen_input.strip()
                        else chess.Board()
                    )
                    st.session_state.move_history = []
                    st.session_state.eval_history = []
                    st.session_state.last_explanation = ""
                    st.session_state.last_move = ""
                    st.session_state.complexity = None
                    st.session_state.confidence = None
                    st.session_state.feedback_given = False
                    st.success("Loaded")
                except ValueError as e:
                    st.error(str(e))
        with c2:
            if st.button("Reset Board", use_container_width=True):
                st.session_state.board = chess.Board()
                st.session_state.move_history = []
                st.session_state.eval_history = []
                st.session_state.last_explanation = ""
                st.session_state.last_move = ""
                st.session_state.complexity = None
                st.session_state.confidence = None
                st.session_state.feedback_given = False
                st.success("Reset")

    else:  # Endgame mode sidebar
        st.subheader("Endgame Trainer")
        st.info(
            "Play as **White** (King + Pawn) against a trained RL agent playing Black (King only). "
            "Promote your pawn to row 5 to win. The agent will try to capture it."
        )

        if ENDGAME_AGENT is None:
            st.warning(
                "Agent not loaded. Run the RL notebook and save the agent first."
            )
        else:
            st.success(f"RL agent loaded ({len(ENDGAME_AGENT):,} states)")

        eg_elo = st.select_slider(
            "Your ELO (for commentary)",
            options=[800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400],
            value=1200,
        )

        if st.button("New Game", use_container_width=True, type="primary"):
            env = st.session_state.eg_env
            env.reset()
            st.session_state.eg_started = True
            st.session_state.eg_history = [
                {"wk": env.wk, "wp": env.wp, "bk": env.bk, "move": "Start"}
            ]
            st.session_state.eg_commentary = ""
            st.session_state.eg_result = None
            st.rerun()


# ── Main layout ────────────────────────────────────────────────────────────────
board = st.session_state.board
status = get_game_status(board)
strategy = st.session_state.strategy
elo = get_active_elo(manual_elo) if st.session_state.mode != "endgame" else 1200

col_board, col_panel = st.columns([1, 1], gap="large")

# ════════════════════════════════════════════════════════════════════════════
# ENDGAME TRAINER MODE
# ════════════════════════════════════════════════════════════════════════════
if st.session_state.mode == "endgame":
    env = st.session_state.eg_env

    with col_board:
        st.markdown("### Endgame Trainer — 5×5 Board")
        st.caption("White: ♔ King + ♙ Pawn  |  Black: ♚ King (RL agent)")

        if st.session_state.eg_started:
            st.markdown(env.render_svg(size=320), unsafe_allow_html=True)

            if env.done:
                if env.winner == "White":
                    st.success("🎉 You promoted the pawn — White wins!")
                elif env.winner == "Black":
                    st.error("The agent captured your pawn — Black wins.")
            else:
                st.info(
                    f"Your turn (White) | Pawn at row {env.wp[0]} — "
                    f"needs {BOARD_SIZE - env.wp[0]} more advance(s) to promote"
                )
        else:
            st.info("Click **New Game** in the sidebar to start.")

        # Move history
        if st.session_state.eg_history:
            st.markdown("#### Move History")
            moves = [h["move"] for h in st.session_state.eg_history[1:]]
            pairs = []
            for i in range(0, len(moves), 2):
                w = moves[i]
                b = moves[i + 1] if i + 1 < len(moves) else "..."
                pairs.append(f"{i//2+1}. {w}  |  {b}")
            st.text("\n".join(pairs) if pairs else "No moves yet.")

    with col_panel:
        st.markdown("### Make Your Move")

        if not st.session_state.eg_started:
            st.info("Start a new game from the sidebar.")

        elif env.done:
            st.markdown(f"**Game over — {env.winner} wins.**")
            st.caption("Start a new game from the sidebar.")

        elif env.turn == 0:  # Player's turn
            legal = env.legal_white()
            if not legal:
                st.warning("No legal moves available.")
            else:
                # From square selector
                from_options = sorted(set(f"{chr(96+a[2])}{a[1]}" for a in legal))
                # Separate king and pawn actions
                king_actions = [a for a in legal if a[0] == "K"]
                pawn_actions = [a for a in legal if a[0] == "P"]

                st.markdown("**King moves:**")
                king_labels = {f"King to {chr(96+a[2])}{a[1]}": a for a in king_actions}
                pawn_labels = {f"Pawn to {chr(96+a[2])}{a[1]}": a for a in pawn_actions}

                all_labels = {**king_labels, **pawn_labels}
                chosen_label = st.selectbox(
                    "Select your move",
                    [""] + list(all_labels.keys()),
                    key="eg_move_select",
                )

                if st.button(
                    "Make Move",
                    use_container_width=True,
                    type="primary",
                    disabled=not chosen_label,
                ):
                    if chosen_label:
                        action = all_labels[chosen_label]
                        state = env._state()
                        _, reward, done = env.step(action)

                        st.session_state.eg_history.append(
                            {
                                "wk": env.wk,
                                "wp": env.wp,
                                "bk": env.bk,
                                "move": chosen_label,
                            }
                        )

                        if not done and env.turn == 1:
                            # Agent responds
                            agent_state = env._state()
                            agent_actions = env.legal_black()
                            if ENDGAME_AGENT and agent_actions:
                                agent_action = agent_act(
                                    agent_state, agent_actions, ENDGAME_AGENT
                                )
                            else:
                                agent_action = env.black_heuristic()

                            if agent_action:
                                _, _, done = env.step(agent_action)
                                agent_label = (
                                    f"Black King to "
                                    f"{chr(96+agent_action[2])}"
                                    f"{agent_action[1]}"
                                )
                                st.session_state.eg_history.append(
                                    {
                                        "wk": env.wk,
                                        "wp": env.wp,
                                        "bk": env.bk,
                                        "move": agent_label,
                                    }
                                )

                        st.session_state.eg_result = env.winner
                        st.rerun()

        st.divider()
        st.markdown("### Position Commentary")

        if st.session_state.eg_started and not env.done:
            if st.button("Get coaching tip", use_container_width=True):
                with st.spinner("Analysing..."):
                    # Create a proxy chess board for commentary
                    # Map 5x5 coordinates to commentary context via prompt
                    commentary_prompt = (
                        f"You are coaching a chess player on a small 5x5 endgame board. "
                        f"White King is at {chr(96+env.wk[1])}{env.wk[0]}, "
                        f"White Pawn is at {chr(96+env.wp[1])}{env.wp[0]}, "
                        f"Black King is at {chr(96+env.bk[1])}{env.bk[0]}. "
                        f"The pawn needs to reach row 5 to promote. "
                        f"In 2 sentences, give one concrete coaching tip for White's next move. "
                        f"Speak to a player rated around {eg_elo} ELO. "
                        f"Use simple language for lower ratings, technical language for higher."
                    )
                    try:
                        import anthropic
                        from dotenv import load_dotenv

                        load_dotenv()
                        client = anthropic.Anthropic(
                            api_key=os.getenv("ANTHROPIC_API_KEY")
                        )
                        msg = client.messages.create(
                            model="claude-haiku-4-5-20251001",
                            max_tokens=150,
                            messages=[{"role": "user", "content": commentary_prompt}],
                        )
                        st.session_state.eg_commentary = msg.content[0].text.strip()
                    except Exception as e:
                        st.session_state.eg_commentary = f"Commentary unavailable: {e}"
                st.rerun()

            if st.session_state.eg_commentary:
                st.info(st.session_state.eg_commentary)

        st.divider()
        st.markdown("### How This Works")
        st.caption(
            "The Black king is controlled by a reinforcement learning agent trained "
            "through 500,000 games of trial and error — with no prior knowledge of chess. "
            "It learned entirely from experience which positions are winning and which are losing. "
            "The same learning algorithm underpins full-scale chess AI. "
            "What changes at larger scale is the representation, not the logic."
        )

# ════════════════════════════════════════════════════════════════════════════
# BOARD COLUMN — Tutor and Play modes
# ════════════════════════════════════════════════════════════════════════════
elif st.session_state.mode in ("tutor", "play"):
    with col_board:
        st.markdown("### Board")
        st.markdown(
            render_board(board, st.session_state.last_move), unsafe_allow_html=True
        )

        if status["is_checkmate"]:
            winner = "Black" if status["turn"] == "White" else "White"
            st.error(f"Checkmate — {winner} wins!")
        elif status["is_stalemate"]:
            st.warning("Stalemate — draw!")
        elif status["is_check"]:
            st.warning(f"⚠️ {status['turn']} is in check!")
        else:
            st.info(
                f"Turn: {status['turn']} | Move {status['fullmove_number']} "
                f"| ELO: {elo} | Style: {strategy}"
            )

        if not status["is_game_over"]:
            eval_str = get_position_evaluation(board)
            conf = st.session_state.confidence
            col_e, col_c = st.columns(2)
            with col_e:
                st.metric("Position Evaluation", eval_str)
            with col_c:
                if conf:
                    conf_color = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}
                    st.metric(
                        "Eval Confidence",
                        f"{conf_color.get(conf['confidence'], '')} "
                        f"{conf['confidence']}",
                        help=conf["label"],
                    )

        if st.session_state.complexity:
            comp = st.session_state.complexity
            color_map = {"Clear": "🟢", "Moderate": "🟡", "Complex": "🔴"}
            st.markdown(
                f"**Position Complexity:** "
                f"{color_map.get(comp['label'], '')} {comp['label']}  \n"
                f"Entropy: {comp['entropy']:.2f} bits · "
                f"Dominant move: {comp['top_move']} ({comp['top_pct']}%) · "
                f"{comp['unique_moves']} distinct moves sampled"
            )

        st.markdown("### Position Trend")
        render_eval_chart()

    # ── PANEL COLUMN ────────────────────────────────────────────────────────
    with col_panel:

        # ── TUTOR MODE ───────────────────────────────────────────────────────
        if st.session_state.mode == "tutor":
            st.markdown("### Tutor Mode")

            if not status["is_game_over"]:
                if st.button(
                    "🔍 Analyse Position + Get Move",
                    use_container_width=True,
                    type="primary",
                ):
                    with st.spinner("Analysing..."):
                        try:
                            skill = elo_to_skill_level(elo)
                            comp = get_position_complexity(
                                board, n_samples=15, skill_level=skill
                            )
                            conf = get_evaluation_confidence(board)
                            st.session_state.complexity = comp
                            st.session_state.confidence = conf

                            result = get_best_move(board, elo)
                            explanation = get_move_explanation(
                                board=board,
                                move_san=result["move_san"],
                                move_uci=result["move_uci"],
                                elo=elo,
                                evaluation=result["evaluation"],
                                strategy=strategy,
                                move_history=st.session_state.move_history,
                            )
                            st.session_state.last_move = result["move_uci"]
                            st.session_state.last_explanation = explanation
                            st.session_state.pending_move = result["move_uci"]
                            st.session_state.pending_san = result["move_san"]
                            st.session_state.feedback_given = False
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")

                if st.session_state.last_explanation:
                    st.markdown("#### Recommended Move")
                    st.code(st.session_state.pending_san, language=None)
                    st.markdown("#### Explanation")
                    st.markdown(st.session_state.last_explanation)

                    if (
                        st.session_state.use_bayesian
                        and not st.session_state.feedback_given
                    ):
                        st.markdown("#### How was that explanation?")
                        st.caption("Your feedback updates the ELO estimate.")
                        fb_cols = st.columns(3)
                        feedback_given = None
                        with fb_cols[0]:
                            if st.button("⬇ Too Simple", use_container_width=True):
                                feedback_given = "Too Simple"
                        with fb_cols[1]:
                            if st.button(
                                "✓ Right Level",
                                use_container_width=True,
                                type="primary",
                            ):
                                feedback_given = "Right Level"
                        with fb_cols[2]:
                            if st.button("⬆ Too Complex", use_container_width=True):
                                feedback_given = "Too Complex"

                        if feedback_given:
                            new_elo = st.session_state.bayesian.update(
                                feedback_given, elo
                            )
                            st.session_state.feedback_given = True
                            st.success(
                                f"Feedback: **{feedback_given}**  \n"
                                f"Updated ELO: **{new_elo}** "
                                f"(±{st.session_state.bayesian.posterior_std:.0f} pts)"
                            )
                            st.rerun()

                    elif (
                        st.session_state.use_bayesian
                        and st.session_state.feedback_given
                    ):
                        st.caption("✓ Feedback recorded.")

                    if st.button("▶ Apply This Move", use_container_width=True):
                        try:
                            mover = "White" if board.turn == chess.WHITE else "Black"
                            new_board = apply_move(board, st.session_state.pending_move)
                            st.session_state.board = new_board
                            st.session_state.move_history.append(
                                st.session_state.pending_san
                            )
                            record_eval(st.session_state.pending_san, new_board, mover)
                            st.session_state.last_explanation = ""
                            st.session_state.complexity = None
                            st.session_state.confidence = None
                            st.session_state.feedback_given = False
                            st.rerun()
                        except ValueError as e:
                            st.error(str(e))
            else:
                st.info("Game over. Reset from the sidebar.")

            st.divider()
            st.markdown("#### Make Your Own Move")
            legal = get_legal_moves(board)
            from_squares = sorted(set(m[:2] for m in legal))
            from_sq = st.selectbox("From square", [""] + from_squares, key="tutor_from")
            to_options = (
                sorted(set(m[2:4] for m in legal if m.startswith(from_sq)))
                if from_sq
                else []
            )
            to_sq = st.selectbox("To square", [""] + to_options, key="tutor_to")

            if st.button("Apply Move", use_container_width=True):
                if from_sq and to_sq:
                    move_uci = from_sq + to_sq
                    if any(m.startswith(move_uci) and len(m) == 5 for m in legal):
                        move_uci += "q"
                    try:
                        board_copy = chess.Board(board.fen())
                        san = board_copy.san(chess.Move.from_uci(move_uci))
                        mover = "White" if board.turn == chess.WHITE else "Black"
                        new_board = apply_move(board, move_uci)
                        st.session_state.board = new_board
                        st.session_state.move_history.append(san)
                        st.session_state.last_move = move_uci
                        st.session_state.last_explanation = ""
                        st.session_state.complexity = None
                        st.session_state.confidence = None
                        st.session_state.feedback_given = False
                        record_eval(san, new_board, mover)
                        st.rerun()
                    except ValueError as e:
                        st.error(str(e))
                else:
                    st.warning("Select both From and To squares.")

        # ── PLAY VS BOT ──────────────────────────────────────────────────────
        elif st.session_state.mode == "play":
            st.markdown("### Play vs Bot")
            st.caption(f"Bot ELO: {elo} | Style: {strategy}")

            if not status["is_game_over"]:
                legal = get_legal_moves(board)
                from_squares = sorted(set(m[:2] for m in legal))
                from_sq = st.selectbox(
                    "From square", [""] + from_squares, key="play_from"
                )
                to_options = (
                    sorted(set(m[2:4] for m in legal if m.startswith(from_sq)))
                    if from_sq
                    else []
                )
                to_sq = st.selectbox("To square", [""] + to_options, key="play_to")

                if st.button("Make Move", use_container_width=True, type="primary"):
                    if from_sq and to_sq:
                        move_uci = from_sq + to_sq
                        if any(m.startswith(move_uci) and len(m) == 5 for m in legal):
                            move_uci += "q"
                        try:
                            board_copy = chess.Board(board.fen())
                            san = board_copy.san(chess.Move.from_uci(move_uci))
                            mover = "White" if board.turn == chess.WHITE else "Black"
                            new_board = apply_move(board, move_uci)
                            st.session_state.board = new_board
                            st.session_state.move_history.append(san)
                            st.session_state.last_move = move_uci
                            record_eval(san, new_board, mover)

                            bot_status = get_game_status(new_board)
                            if not bot_status["is_game_over"]:
                                with st.spinner("Bot thinking..."):
                                    comp = get_position_complexity(
                                        new_board,
                                        n_samples=15,
                                        skill_level=elo_to_skill_level(elo),
                                    )
                                    conf = get_evaluation_confidence(new_board)
                                    st.session_state.complexity = comp
                                    st.session_state.confidence = conf

                                    bot_result = get_best_move(new_board, elo)
                                    commentary = get_position_commentary(
                                        new_board, elo, strategy
                                    )
                                    st.session_state.bot_commentary = (
                                        f"Bot played **{bot_result['move_san']}**. "
                                        f"{commentary}"
                                    )
                                    bot_board = apply_move(
                                        new_board, bot_result["move_uci"]
                                    )
                                    st.session_state.board = bot_board
                                    st.session_state.move_history.append(
                                        bot_result["move_san"]
                                    )
                                    st.session_state.last_move = bot_result["move_uci"]
                                    record_eval(
                                        bot_result["move_san"], bot_board, "Bot"
                                    )

                            st.session_state.feedback_given = False
                            st.rerun()
                        except ValueError as e:
                            st.error(str(e))
                    else:
                        st.warning("Select both From and To squares.")

                if st.session_state.bot_commentary:
                    st.markdown("#### Bot Commentary")
                    st.info(st.session_state.bot_commentary)

                    if (
                        st.session_state.use_bayesian
                        and not st.session_state.feedback_given
                    ):
                        st.markdown("#### How was the commentary?")
                        fb_cols = st.columns(3)
                        feedback_given = None
                        with fb_cols[0]:
                            if st.button(
                                "⬇ Too Simple",
                                use_container_width=True,
                                key="play_fb_simple",
                            ):
                                feedback_given = "Too Simple"
                        with fb_cols[1]:
                            if st.button(
                                "✓ Right Level",
                                use_container_width=True,
                                type="primary",
                                key="play_fb_right",
                            ):
                                feedback_given = "Right Level"
                        with fb_cols[2]:
                            if st.button(
                                "⬆ Too Complex",
                                use_container_width=True,
                                key="play_fb_complex",
                            ):
                                feedback_given = "Too Complex"
                        if feedback_given:
                            new_elo = st.session_state.bayesian.update(
                                feedback_given, elo
                            )
                            st.session_state.feedback_given = True
                            st.success(
                                f"**{feedback_given}** → ELO: **{new_elo}** "
                                f"(±{st.session_state.bayesian.posterior_std:.0f} pts)"
                            )
                            st.rerun()
            else:
                st.info("Game over. Reset from sidebar.")

        # ── Move history ─────────────────────────────────────────────────────
        st.divider()
        st.markdown("#### Move History")
        if st.session_state.move_history:
            history = st.session_state.move_history
            pairs = []
            for i in range(0, len(history), 2):
                w = history[i]
                b = history[i + 1] if i + 1 < len(history) else "..."
                pairs.append(f"{i//2+1}. {w}  {b}")
            st.text("\n".join(pairs))
        else:
            st.caption("No moves yet.")
