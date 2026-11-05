"""Turn StatsBomb event files into per-player, per-match counts."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

GOAL = (120.0, 40.0)
BOX_X, BOX_Y_LOW, BOX_Y_HIGH = 102.0, 18.0, 62.0
COUNT_FIELDS = [
    "actions", "passes", "passes_completed", "progressive_passes", "long_passes", "key_passes", "crosses",
    "progressive_carries", "dribbles", "dribbles_completed", "shots", "xg", "box_touches",
    "pressures", "tackles", "interceptions", "recoveries", "clearances", "blocks", "fouls_committed",
    "x_total", "width_total",
]


def goal_distance(point: list[float]) -> float:
    return math.hypot(GOAL[0] - point[0], GOAL[1] - point[1])


def is_progressive(start: list[float], end: list[float]) -> bool:
    """A move counts as progressive if it cuts the distance to goal by at least 25%."""
    return goal_distance(end) <= 0.75 * goal_distance(start)


def in_box(point: list[float]) -> bool:
    return point[0] >= BOX_X and BOX_Y_LOW <= point[1] <= BOX_Y_HIGH


def position_group(name: str | None) -> str:
    if not name:
        return "Unknown"
    if "Goalkeeper" in name:
        return "Goalkeeper"
    if "Back" in name:
        return "Centre back" if "Center" in name else "Full back"
    if "Wing" in name:
        return "Winger"
    if "Forward" in name or "Striker" in name:
        return "Forward"
    if "Midfield" in name:
        return "Midfielder"
    return "Unknown"


def minutes_played(events: list[dict]) -> dict[int, float]:
    """Work out minutes from starting lineups, substitutions and the last whistle."""
    match_end = max(event["minute"] + event["second"] / 60 for event in events)
    on, off = {}, {}
    for event in events:
        kind = event["type"]["name"]
        if kind == "Starting XI":
            for entry in event["tactics"]["lineup"]:
                on[entry["player"]["id"]] = 0.0
        elif kind == "Substitution":
            time = event["minute"] + event["second"] / 60
            off[event["player"]["id"]] = time
            on[event["substitution"]["replacement"]["id"]] = time
        elif kind == "Player Off" or (kind == "Bad Behaviour" and "card" in event.get("bad_behaviour", {})
                                      and "Red" in event["bad_behaviour"]["card"]["name"]):
            if "player" in event:
                off[event["player"]["id"]] = event["minute"] + event["second"] / 60
    return {player: off.get(player, match_end) - start for player, start in on.items()}


def _count(row: Counter, event: dict) -> None:
    kind = event["type"]["name"]
    location = event.get("location")
    if location:
        row["actions"] += 1
        row["x_total"] += location[0]
        row["width_total"] += abs(location[1] - 40.0)
        if in_box(location) and kind in {"Pass", "Carry", "Shot", "Ball Receipt*", "Dribble"}:
            row["box_touches"] += 1

    if kind == "Pass":
        detail = event["pass"]
        if detail.get("type", {}).get("name") in {"Corner", "Free Kick", "Goal Kick", "Kick Off", "Throw-in"}:
            return
        row["passes"] += 1
        if "outcome" not in detail:
            row["passes_completed"] += 1
            if is_progressive(location, detail["end_location"]):
                row["progressive_passes"] += 1
        if detail.get("length", 0) >= 32:
            row["long_passes"] += 1
        if detail.get("shot_assist") or detail.get("goal_assist"):
            row["key_passes"] += 1
        if detail.get("cross"):
            row["crosses"] += 1
    elif kind == "Carry":
        if is_progressive(location, event["carry"]["end_location"]):
            row["progressive_carries"] += 1
    elif kind == "Dribble":
        row["dribbles"] += 1
        if event["dribble"].get("outcome", {}).get("name") == "Complete":
            row["dribbles_completed"] += 1
    elif kind == "Shot":
        if event["shot"]["type"]["name"] != "Penalty":
            row["shots"] += 1
            row["xg"] += event["shot"].get("statsbomb_xg", 0.0)
    elif kind == "Pressure":
        row["pressures"] += 1
    elif kind == "Duel" and event["duel"].get("type", {}).get("name") == "Tackle":
        row["tackles"] += 1
    elif kind == "Interception":
        row["interceptions"] += 1
    elif kind == "Ball Recovery":
        row["recoveries"] += 1
    elif kind == "Clearance":
        row["clearances"] += 1
    elif kind == "Block":
        row["blocks"] += 1
    elif kind == "Foul Committed":
        row["fouls_committed"] += 1


def match_rows(events: list[dict], match_id: int) -> list[dict]:
    """Return one row per player for a single match."""
    minutes = minutes_played(events)
    counts: dict[int, Counter] = defaultdict(Counter)
    names: dict[int, str] = {}
    teams: dict[int, str] = {}
    positions: dict[int, Counter] = defaultdict(Counter)
    for event in events:
        player = event.get("player")
        if not player:
            continue
        names[player["id"]] = player["name"]
        teams[player["id"]] = event["team"]["name"]
        if "position" in event:
            positions[player["id"]][event["position"]["name"]] += 1
        _count(counts[player["id"]], event)

    rows = []
    for player_id, played in minutes.items():
        if player_id not in names or played <= 0:
            continue
        common = positions[player_id].most_common(1)
        row = {
            "match_id": match_id,
            "player_id": player_id,
            "player": names[player_id],
            "team": teams[player_id],
            "position": position_group(common[0][0] if common else None),
            "minutes": round(played, 2),
        }
        row.update({field: counts[player_id][field] for field in COUNT_FIELDS})
        rows.append(row)
    return rows


def load_match(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))
