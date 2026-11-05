"""Test the similarity search on a question with a known answer.

Each player's matches are split into two halves that share no match. Given one
half, the search should find the same player's other half among all candidates.
The method is picked on one group of players and scored on a different group,
so the reported number is not tuned on the players it scores.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from similar_players.profiles import FEATURES, build_profile, cosine_matrix, split_rows  # noqa: E402


def half_profiles(rows: pd.DataFrame) -> pd.DataFrame:
    records = {player_id: build_profile(group) for player_id, group in rows.groupby("player_id")}
    return pd.DataFrame(records).T[FEATURES].sort_index().astype(float)


def ranks(query: np.ndarray, candidates: np.ndarray, teams: np.ndarray | None = None) -> np.ndarray:
    """Return the rank of the correct match for each query, where 1 is best.

    With teams given, teammates are removed from the candidates, so a player
    cannot be found just because his team plays a recognizable style.
    """
    scores = cosine_matrix(query, candidates)
    correct = np.diag(scores).copy()
    if teams is not None:
        same_team = teams[:, None] == teams[None, :]
        np.fill_diagonal(same_team, False)
        scores = np.where(same_team, -np.inf, scores)
    return (scores > correct[:, None]).sum(axis=1) + 1


def summarize(values: np.ndarray) -> dict:
    return {
        "top_1": round(float((values <= 1).mean()), 3),
        "top_5": round(float((values <= 5).mean()), 3),
        "top_10": round(float((values <= 10).mean()), 3),
        "median_rank": float(np.median(values)),
    }


def run(rows: pd.DataFrame, seed: int = 7) -> dict:
    first, second = split_rows(rows[~rows["position"].isin(["Goalkeeper", "Unknown"])])
    a, b = half_profiles(first), half_profiles(second)
    teams = rows.groupby("player_id")["team"].agg(lambda values: values.mode().iloc[0]).reindex(a.index).to_numpy()
    order = np.random.default_rng(seed).permutation(len(a))
    tune_ids, test_ids = order[: len(order) // 2], order[len(order) // 2 :]

    scaler = StandardScaler().fit(pd.concat([a.iloc[tune_ids], b.iloc[tune_ids]]))
    a_scaled, b_scaled = scaler.transform(a), scaler.transform(b)
    pooled = np.vstack([a_scaled[tune_ids], b_scaled[tune_ids]])

    tuning = {"scaled features": float(np.median(ranks(a_scaled[tune_ids], b_scaled[tune_ids])))}
    for k in range(3, len(FEATURES)):
        pca = PCA(n_components=k, whiten=True, random_state=seed).fit(pooled)
        tuning[str(k)] = float(np.median(ranks(pca.transform(a_scaled[tune_ids]), pca.transform(b_scaled[tune_ids]))))
    best_pca = min((key for key in tuning if key != "scaled features"), key=lambda key: (tuning[key], int(key)))
    chosen = "scaled features" if tuning["scaled features"] <= tuning[best_pca] else f"pca {best_pca}"

    pca = PCA(n_components=int(best_pca), whiten=True, random_state=seed).fit(pooled)
    test_teams = teams[test_ids]
    count = len(test_ids)
    return {
        "players_tested": int(count),
        "players_used_for_tuning": int(len(tune_ids)),
        "tuning_median_rank": {"scaled features": tuning["scaled features"], f"pca {best_pca}": tuning[best_pca]},
        "chosen_on_tuning_players": chosen,
        "random_guess": {
            "top_1": round(1 / count, 3),
            "top_5": round(5 / count, 3),
            "top_10": round(10 / count, 3),
            "median_rank": (count + 1) / 2,
        },
        "scaled_features": summarize(ranks(a_scaled[test_ids], b_scaled[test_ids])),
        "scaled_features_no_teammates": summarize(ranks(a_scaled[test_ids], b_scaled[test_ids], test_teams)),
        "pca": summarize(ranks(pca.transform(a_scaled[test_ids]), pca.transform(b_scaled[test_ids]))),
    }


def chart(result: dict, path: Path) -> None:
    labels = ["Top 1", "Top 5", "Top 10"]
    keys = ["top_1", "top_5", "top_10"]
    x = np.arange(3)
    fig, axis = plt.subplots(figsize=(6.4, 3.6), dpi=150)
    axis.bar(x - 0.2, [result["random_guess"][k] * 100 for k in keys], 0.4, label="Random guess", color="#b8bcc2")
    axis.bar(x + 0.2, [result["scaled_features"][k] * 100 for k in keys], 0.4, label="This model", color="#1f6f54")
    axis.set_xticks(x, labels)
    axis.set_ylabel("Same player found (%)")
    axis.set_title(f"Finding a player's other matches among {result['players_tested']} players")
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", default="data/match_rows.csv")
    args = parser.parse_args()
    result = run(pd.read_csv(args.rows).drop(columns=["competition"], errors="ignore"))
    Path("results/evaluation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    chart(result, Path("results/retrieval.png"))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
