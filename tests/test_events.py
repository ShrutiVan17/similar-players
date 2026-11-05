from similar_players.events import in_box, is_progressive, match_rows, minutes_played, position_group


def event(kind, minute=1, player=(1, "A"), location=(60.0, 40.0), team="X", **extra):
    base = {
        "type": {"name": kind}, "minute": minute, "second": 0, "team": {"name": team},
        "player": {"id": player[0], "name": player[1]}, "position": {"name": "Left Wing"}, "location": list(location),
    }
    base.update(extra)
    return base


def lineup(*ids):
    return {
        "type": {"name": "Starting XI"}, "minute": 0, "second": 0, "team": {"name": "X"},
        "tactics": {"lineup": [{"player": {"id": player}} for player in ids]},
    }


def test_progressive_means_a_quarter_closer_to_goal():
    assert is_progressive([60, 40], [100, 40])
    assert not is_progressive([60, 40], [70, 40])


def test_box_limits():
    assert in_box([110, 40]) and not in_box([100, 40]) and not in_box([110, 10])


def test_position_groups():
    assert position_group("Left Center Back") == "Centre back"
    assert position_group("Right Wing Back") == "Full back"
    assert position_group("Left Wing") == "Winger"
    assert position_group("Center Forward") == "Forward"
    assert position_group("Center Defensive Midfield") == "Midfielder"


def test_substitution_minutes():
    sub = event("Substitution", minute=60, substitution={"replacement": {"id": 2, "name": "B"}})
    minutes = minutes_played([lineup(1), sub, event("Pass", minute=90, player=(2, "B"), **{"pass": {"end_location": [61, 40]}})])
    assert minutes == {1: 60, 2: 30}


def test_counts_skip_set_pieces_and_penalties():
    events = [
        lineup(1),
        event("Pass", **{"pass": {"end_location": [100, 40], "length": 40, "shot_assist": True}}),
        event("Pass", **{"pass": {"end_location": [61, 40], "type": {"name": "Corner"}}}),
        event("Pass", **{"pass": {"end_location": [100, 40], "outcome": {"name": "Incomplete"}}}),
        event("Shot", shot={"type": {"name": "Open Play"}, "statsbomb_xg": 0.25}),
        event("Shot", minute=90, shot={"type": {"name": "Penalty"}, "statsbomb_xg": 0.78}),
    ]
    row = match_rows(events, 9)[0]
    assert row["passes"] == 2 and row["passes_completed"] == 1
    assert row["progressive_passes"] == 1 and row["key_passes"] == 1 and row["long_passes"] == 1
    assert row["shots"] == 1 and row["xg"] == 0.25
    assert row["position"] == "Winger"
