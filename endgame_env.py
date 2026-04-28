"""
endgame_env.py
King+Pawn (White) vs King (Black) on a 5x5 board.
Shared between the Jupyter notebook and the Streamlit app.
"""

import random
import base64

BOARD_SIZE = 5
PROMOTE_ROW = BOARD_SIZE
STEP_PENALTY = -0.02
ROW_REWARD = 0.3


def valid(r, c):
    return 1 <= r <= BOARD_SIZE and 1 <= c <= BOARD_SIZE


def king_moves(r, c):
    return [
        (r + dr, c + dc)
        for dr in [-1, 0, 1]
        for dc in [-1, 0, 1]
        if (dr, dc) != (0, 0) and valid(r + dr, c + dc)
    ]


def kings_adjacent(r1, c1, r2, c2):
    return abs(r1 - r2) <= 1 and abs(c1 - c2) <= 1


class KingPawnEnv:
    """
    5x5 King+Pawn vs King endgame environment.
    White wins by promoting the pawn to row 5.
    Black wins by capturing the pawn.
    Supports randomised or fixed starting positions.
    """

    def reset(self, fixed=None):
        if fixed:
            self.wk = fixed["wk"]
            self.wp = fixed["wp"]
            self.bk = fixed["bk"]
        else:
            self.wk, self.wp, self.bk = self._random_position()
        self.turn = 0  # 0 = White, 1 = Black
        self.done = False
        self.winner = None
        return self._state()

    def _random_position(self):
        for _ in range(1000):
            wp = (random.randint(2, 3), random.randint(1, BOARD_SIZE))
            wk_cands = [(r, c) for r, c in king_moves(*wp) if (r, c) != wp]
            if not wk_cands:
                continue
            wk = random.choice(wk_cands)
            bk_cands = [
                (r, c)
                for r in range(4, BOARD_SIZE + 1)
                for c in range(1, BOARD_SIZE + 1)
                if (r, c) != wk
                and (r, c) != wp
                and not kings_adjacent(r, c, wk[0], wk[1])
            ]
            if not bk_cands:
                continue
            bk = random.choice(bk_cands)
            return wk, wp, bk
        return (3, 2), (3, 3), (5, 1)  # safe fallback

    def _state(self):
        return (
            self.wk[0],
            self.wk[1],
            self.wp[0],
            self.wp[1],
            self.bk[0],
            self.bk[1],
            self.turn,
        )

    def legal_white(self):
        actions = []
        for nr, nc in king_moves(*self.wk):
            if (nr, nc) == self.wp or (nr, nc) == self.bk:
                continue
            actions.append(("K", nr, nc))
        pr, pc = self.wp
        nr = pr + 1
        if valid(nr, pc) and (nr, pc) != self.bk and (nr, pc) != self.wk:
            actions.append(("P", nr, pc))
        return actions

    def legal_black(self):
        return [
            ("K", nr, nc)
            for nr, nc in king_moves(*self.bk)
            if (nr, nc) != self.wk
            and not kings_adjacent(nr, nc, self.wk[0], self.wk[1])
        ]

    def step(self, action):
        if self.done:
            return self._state(), 0, True
        atype, nr, nc = action
        if self.turn == 0:  # White
            if atype == "K":
                self.wk = (nr, nc)
                reward = STEP_PENALTY
            else:  # Pawn advance
                self.wp = (nr, nc)
                reward = ROW_REWARD + STEP_PENALTY
                if nr == PROMOTE_ROW:
                    self.done = True
                    self.winner = "White"
                    return self._state(), +1.0, True
            self.turn = 1
            return self._state(), reward, False
        else:  # Black
            self.bk = (nr, nc)
            if self.bk == self.wp:
                self.done = True
                self.winner = "Black"
                return self._state(), -1.0, True
            self.turn = 0
            return self._state(), 0, False

    def black_heuristic(self):
        """Capture pawn if possible, otherwise approach it."""
        actions = self.legal_black()
        if not actions:
            return None
        pr, pc = self.wp
        for a in actions:
            if (a[1], a[2]) == (pr, pc):
                return a
        return min(actions, key=lambda a: abs(a[1] - pr) + abs(a[2] - pc))

    def render_svg(self, size=320) -> str:
        """Render the 5x5 board as an HTML img tag with embedded SVG."""
        sq = size // BOARD_SIZE
        pad = 28
        w = size + pad
        h = size + pad

        lines = [
            f'<svg width="{w}" height="{h}" '
            f'xmlns="http://www.w3.org/2000/svg" '
            f'style="background:#1a1a2e;border-radius:6px;">'
        ]

        # Board squares
        for r in range(1, BOARD_SIZE + 1):
            for c in range(1, BOARD_SIZE + 1):
                x = pad + (c - 1) * sq
                y = size - r * sq  # row 1 at bottom
                fill = "#f0d9b5" if (r + c) % 2 == 0 else "#b58863"
                lines.append(
                    f'<rect x="{x}" y="{y}" width="{sq}" '
                    f'height="{sq}" fill="{fill}"/>'
                )

        # Coordinate labels
        for i in range(1, BOARD_SIZE + 1):
            y_label = size - (i - 1) * sq - sq // 2 + 5
            lines.append(
                f'<text x="13" y="{y_label}" font-size="12" '
                f'fill="#ccc" text-anchor="middle">{i}</text>'
            )
            x_label = pad + (i - 1) * sq + sq // 2
            lines.append(
                f'<text x="{x_label}" y="{size + 20}" font-size="12" '
                f'fill="#ccc" text-anchor="middle">{chr(96 + i)}</text>'
            )

        # Pieces
        piece_map = {
            self.wk: ("♔", "#ffffff", "#333333"),
            self.wp: ("♙", "#ffffff", "#333333"),
            self.bk: ("♚", "#111111", "#eeeeee"),
        }
        for (r, c), (sym, fill, stroke) in piece_map.items():
            x = pad + (c - 1) * sq + sq // 2
            y = size - (r - 1) * sq - sq // 2 + int(sq * 0.28)
            lines.append(
                f'<text x="{x}" y="{y}" '
                f'font-size="{int(sq * 0.68)}" '
                f'fill="{fill}" text-anchor="middle" '
                f'stroke="{stroke}" stroke-width="0.6">'
                f"{sym}</text>"
            )

        lines.append("</svg>")
        svg_str = "\n".join(lines)
        b64 = base64.b64encode(svg_str.encode()).decode()
        return f'<img src="data:image/svg+xml;base64,{b64}" width="{w}"/>'
