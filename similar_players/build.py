"""Read raw StatsBomb files and write data/match_rows.csv and data/profiles.csv."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from similar_players.events import load_match, match_rows
from similar_players.profiles import build_profiles

COMPETITIONS = {(43, 106): "World Cup 2022", (55, 282): "Euro 2024", (223, 282): "Copa America 2024", (44, 107): "MLS 2023"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--open-data", required=True, help="Path to a copy of statsbomb/open-data")
    parser.add_argument("--out", default="data")
    args = parser.parse_args()
    root = Path(args.open_data) / "data"

    rows = []
    for (competition, season), name in COMPETITIONS.items():
        matches = json.loads((root / "matches" / str(competition) / f"{season}.json").read_text(encoding="utf-8"))
        for match in matches:
            for row in match_rows(load_match(root / "events" / f"{match['match_id']}.json"), match["match_id"]):
                row["competition"] = name
                rows.append(row)
    frame = pd.DataFrame(rows)
    out = Path(args.out)
    frame.to_csv(out / "match_rows.csv", index=False)
    profiles = build_profiles(frame.drop(columns=["competition"]))
    profiles.to_csv(out / "profiles.csv", index=False)
    print(f"{len(frame)} player matches, {frame['match_id'].nunique()} matches, {len(profiles)} player profiles")
    print(profiles["position"].value_counts().to_string())


if __name__ == "__main__":
    main()
