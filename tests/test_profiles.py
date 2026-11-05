import numpy as np
import pandas as pd

from similar_players.evaluate import ranks
from similar_players.events import COUNT_FIELDS
from similar_players.find import similar
from similar_players.profiles import FEATURES, build_profile, build_profiles, split_rows


def row(player, match, minutes=90, passes=40, position="Midfielder"):
    base = {field: 1 for field in COUNT_FIELDS}
    base.update(match_id=match, player_id=player, player=f"P{player}", team="T", position=position, minutes=minutes,
                passes=passes, passes_completed=passes // 2, actions=10, x_total=500, width_total=100)
    return base


def test_rates_are_per_90():
    profile = build_profile(pd.DataFrame([row(1, 1, minutes=45, passes=20), row(1, 2, minutes=45, passes=20)]))
    assert profile["passes_p90"] == 40
    assert profile["pass_completion"] == 0.5


def test_short_minutes_and_goalkeepers_are_left_out():
    rows = pd.DataFrame([row(1, m) for m in range(3)] + [row(2, 1)] + [row(3, m, position="Goalkeeper") for m in range(3)])
    assert list(build_profiles(rows)["player_id"]) == [1]


def test_split_halves_share_no_match():
    a, b = split_rows(pd.DataFrame([row(1, m) for m in range(6)]))
    assert not set(a["match_id"]) & set(b["match_id"])


def test_ranks_find_the_matching_row_and_can_drop_teammates():
    query = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert list(ranks(query, query)) == [1, 1]
    swapped = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert list(ranks(query, swapped)) == [2, 2]
    assert list(ranks(query, swapped, np.array(["T", "T"]))) == [1, 1]


def test_similar_stays_in_position_and_skips_the_player():
    rng = np.random.default_rng(1)
    profiles = pd.DataFrame(rng.normal(size=(8, len(FEATURES))), columns=FEATURES)
    profiles["player_id"] = range(8)
    profiles["player"] = [f"Player {i}" for i in range(8)]
    profiles["team"] = "T"
    profiles["minutes"] = 300
    profiles["position"] = ["Winger"] * 5 + ["Forward"] * 3
    target, table = similar(profiles, "Player 1", 3)
    assert target["player_id"] == 1
    assert "Player 1" not in set(table["player"]) and "Player 6" not in set(table["player"])
