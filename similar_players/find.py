"""Show the players most similar to one player, within the same position."""

from __future__ import annotations

import argparse

import pandas as pd
from sklearn.preprocessing import StandardScaler

from similar_players.profiles import FEATURES, cosine_matrix


def similar(profiles: pd.DataFrame, name: str, count: int = 5) -> tuple[pd.Series, pd.DataFrame]:
    matches = profiles[profiles["player"].str.contains(name, case=False, regex=False)]
    if matches.empty:
        raise SystemExit(f"No player matching {name!r}. They need at least 270 minutes in the data.")
    target = matches.sort_values("minutes", ascending=False).iloc[0]
    group = profiles[profiles["position"] == target["position"]].reset_index(drop=True)
    scaled = StandardScaler().fit(profiles[FEATURES]).transform(group[FEATURES])
    index = group.index[group["player_id"] == target["player_id"]][0]
    group = group.assign(similarity=cosine_matrix(scaled[[index]], scaled)[0].round(3))
    others = group[group["player_id"] != target["player_id"]]
    return target, others.nlargest(count, "similarity")[["player", "team", "similarity", "minutes"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("player")
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--profiles", default="data/profiles.csv")
    args = parser.parse_args()
    target, table = similar(pd.read_csv(args.profiles), args.player, args.top)
    print(f"{target['player']} ({target['team']}, {target['position']}, {target['minutes']:.0f} minutes)\n")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
