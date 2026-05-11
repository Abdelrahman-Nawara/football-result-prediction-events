#!/usr/bin/env python3
"""
Football match-result prediction from in-game event data.

This script is designed for the AUC Fundamentals of Machine Learning project:
"Predicting Football Games Result using In-game Events".

It loads the Wyscout soccer event dataset, performs data cleaning and EDA,
builds match-level features from events observed up to a configurable match
minute, trains multiple classifiers, evaluates them, and exports artifacts that
can be used directly in the project report and presentation.

Example:
    python football_result_prediction.py --cutoff-minute 75

For a faster smoke run on one competition:
    python football_result_prediction.py --competitions England --cv-folds 2
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

try:
    import joblib
except Exception:  # pragma: no cover - fallback for minimal environments
    joblib = None

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    from sklearn.compose import ColumnTransformer
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.inspection import permutation_importance
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        log_loss,
    )
    from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, cross_validate
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    SKLEARN_AVAILABLE = True
    SKLEARN_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - depends on the user's local environment
    SKLEARN_AVAILABLE = False
    SKLEARN_IMPORT_ERROR = exc


RANDOM_STATE = 42
RESULT_ORDER = ["away_win", "draw", "home_win"]
PERIOD_OFFSETS = {
    "1H": 0.0,
    "2H": 45.0,
    "E1": 90.0,
    "E2": 105.0,
    "P": 120.0,
}

SELECTED_TAGS = {
    101: "goal",
    102: "own_goal",
    201: "opportunity",
    301: "assist",
    302: "key_pass",
    401: "left_foot",
    402: "right_foot",
    403: "head_or_body",
    701: "duel_lost",
    702: "duel_neutral",
    703: "duel_won",
    801: "high",
    802: "low",
    901: "through",
    1001: "fair_play",
    1101: "direct",
    1102: "indirect",
    1301: "feint",
    1302: "missed_ball",
    1401: "interception",
    1501: "clearance_tag",
    1601: "sliding_tackle",
    1701: "red_card",
    1702: "yellow_card",
    1703: "second_yellow_card",
    1801: "accurate",
    1802: "not_accurate",
    1901: "counter_attack",
    2001: "dangerous_ball_lost",
    2101: "blocked",
}

GOAL_LOCATION_TAGS = set(range(1201, 1210))
POST_LOCATION_TAGS = set(range(1217, 1224))
OUT_LOCATION_TAGS = set(range(1210, 1217))
ON_TARGET_TAGS = GOAL_LOCATION_TAGS | POST_LOCATION_TAGS | {101}


@dataclass
class CleaningReport:
    """Counts cleaning decisions made while processing the raw event stream."""

    raw_events: int = 0
    duplicate_events: int = 0
    events_after_cutoff: int = 0
    events_for_unknown_match: int = 0
    events_for_unknown_team: int = 0
    events_without_valid_time: int = 0
    events_with_missing_coordinates: int = 0
    kept_events: int = 0
    event_name_counts: Counter[str] = field(default_factory=Counter)
    subevent_name_counts: Counter[str] = field(default_factory=Counter)
    tag_counts: Counter[int] = field(default_factory=Counter)

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "raw_events": self.raw_events,
            "duplicate_events": self.duplicate_events,
            "events_after_cutoff": self.events_after_cutoff,
            "events_for_unknown_match": self.events_for_unknown_match,
            "events_for_unknown_team": self.events_for_unknown_team,
            "events_without_valid_time": self.events_without_valid_time,
            "events_with_missing_coordinates": self.events_with_missing_coordinates,
            "kept_events": self.kept_events,
            "top_event_names": self.event_name_counts.most_common(25),
            "top_subevent_names": self.subevent_name_counts.most_common(35),
            "top_tag_ids": self.tag_counts.most_common(35),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an ML model that predicts football match results from in-game events."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("Data set"),
        help="Path to the dataset folder containing matches/, events/, teams.json, and mapping CSVs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory where EDA plots, metrics, feature tables, and model files will be saved.",
    )
    parser.add_argument(
        "--cutoff-minute",
        type=float,
        default=75.0,
        help=(
            "Only events up to this in-game minute are used as features. "
            "Use 45 for half-time, 60 or 75 for live prediction, or 90 for full-match event summaries."
        ),
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
        help="Fraction of matches reserved for the chronological holdout test set.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of stratified CV folds on the training split. Set to 0 to skip CV.",
    )
    parser.add_argument(
        "--permutation-repeats",
        type=int,
        default=8,
        help="Number of permutation-importance repeats for the best model.",
    )
    parser.add_argument(
        "--competitions",
        nargs="*",
        default=None,
        help=(
            "Optional competition subset, e.g. England Spain World_Cup. "
            "By default all available competitions are used."
        ),
    )
    parser.add_argument(
        "--split",
        choices=["chronological", "stratified"],
        default="chronological",
        help="Holdout split strategy. Chronological is more realistic for future-match prediction.",
    )
    parser.add_argument(
        "--include-team-ids",
        action="store_true",
        help=(
            "Include home/away team IDs as categorical features. This can improve accuracy by capturing team "
            "strength, but the default excludes them to keep the model focused on in-game events."
        ),
    )
    return parser.parse_args()


def safe_int(value: Any, default: int | None = None) -> int | None:
    try:
        if value is None or value == "":
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def slug(value: Any, fallback: str = "unknown") -> str:
    text = str(value or "").strip().lower()
    if not text:
        return fallback
    clean = []
    for char in text:
        if char.isalnum():
            clean.append(char)
        else:
            clean.append("_")
    return "_".join(part for part in "".join(clean).split("_") if part) or fallback


def ensure_output_dirs(output_dir: Path) -> dict[str, Path]:
    paths = {
        "root": output_dir,
        "figures": output_dir / "figures",
        "tables": output_dir / "tables",
        "models": output_dir / "models",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def competition_from_file(path: Path, prefix: str) -> str:
    return path.stem.replace(prefix, "", 1)


def load_team_names(data_dir: Path) -> dict[int, str]:
    teams_path = data_dir / "teams.json"
    if not teams_path.exists():
        return {}
    with teams_path.open("r", encoding="utf-8") as fh:
        teams = json.load(fh)
    return {
        safe_int(team.get("wyId")): team.get("officialName") or team.get("name") or str(team.get("wyId"))
        for team in teams
        if safe_int(team.get("wyId")) is not None
    }


def extract_team_by_side(match: dict[str, Any], side: str) -> dict[str, Any] | None:
    for value in (match.get("teamsData") or {}).values():
        if value.get("side") == side:
            return value
    return None


def parse_match_rows(data_dir: Path, competitions: set[str] | None, team_names: dict[int, str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    matches_dir = data_dir / "matches"
    match_paths = sorted(matches_dir.glob("matches_*.json"))

    if not match_paths:
        raise FileNotFoundError(f"No match files found in {matches_dir}")

    for path in match_paths:
        competition = competition_from_file(path, "matches_")
        if competitions and competition not in competitions:
            continue

        with path.open("r", encoding="utf-8") as fh:
            matches = json.load(fh)

        for match in matches:
            if match.get("status") and match.get("status") != "Played":
                continue

            home = extract_team_by_side(match, "home")
            away = extract_team_by_side(match, "away")
            if home is None or away is None:
                continue

            home_team_id = safe_int(home.get("teamId"))
            away_team_id = safe_int(away.get("teamId"))
            if home_team_id is None or away_team_id is None:
                continue

            home_score = safe_int(home.get("score"), 0)
            away_score = safe_int(away.get("score"), 0)
            if home_score > away_score:
                result = "home_win"
            elif home_score < away_score:
                result = "away_win"
            else:
                result = "draw"

            rows.append(
                {
                    "match_id": safe_int(match.get("wyId")),
                    "competition": competition,
                    "season_id": safe_int(match.get("seasonId")),
                    "round_id": safe_int(match.get("roundId")),
                    "gameweek": safe_int(match.get("gameweek")),
                    "dateutc": pd.to_datetime(match.get("dateutc"), errors="coerce", utc=True),
                    "venue": match.get("venue"),
                    "duration": match.get("duration"),
                    "winner": safe_int(match.get("winner"), 0),
                    "home_team_id": home_team_id,
                    "away_team_id": away_team_id,
                    "home_team_name": team_names.get(home_team_id, str(home_team_id)),
                    "away_team_name": team_names.get(away_team_id, str(away_team_id)),
                    "home_score": home_score,
                    "away_score": away_score,
                    "goal_diff_final": home_score - away_score,
                    "total_goals": home_score + away_score,
                    "target": result,
                }
            )

    matches_df = pd.DataFrame(rows)
    if matches_df.empty:
        raise ValueError("No playable matches were loaded. Check --data-dir and --competitions.")

    matches_df = matches_df.drop_duplicates(subset=["match_id"]).sort_values("dateutc").reset_index(drop=True)
    matches_df["home_team_id"] = matches_df["home_team_id"].astype(str)
    matches_df["away_team_id"] = matches_df["away_team_id"].astype(str)
    matches_df["gameweek"] = matches_df["gameweek"].fillna(0).astype(int)
    return matches_df


def event_minute(event: dict[str, Any]) -> float | None:
    period = event.get("matchPeriod")
    if period not in PERIOD_OFFSETS:
        return None
    seconds = safe_float(event.get("eventSec"), default=np.nan)
    if not np.isfinite(seconds) or seconds < 0:
        return None
    return PERIOD_OFFSETS[period] + seconds / 60.0


def position_value(event: dict[str, Any], index: int, key: str) -> float | None:
    positions = event.get("positions") or []
    if index >= len(positions) or not isinstance(positions[index], dict):
        return None
    value = safe_float(positions[index].get(key), default=np.nan)
    if not np.isfinite(value) or value < 0 or value > 100:
        return None
    return value


def shot_quality_proxy(start_x: float | None, start_y: float | None, tags: set[int], subevent_name: str) -> float:
    """Estimate chance quality from location and context when true xG labels are unavailable."""
    if 101 in tags:
        return 0.85
    if subevent_name == "penalty":
        return 0.76
    if start_x is None or start_y is None:
        base = 0.06
    else:
        distance = math.sqrt(((100.0 - start_x) * 1.05) ** 2 + ((50.0 - start_y) * 0.68) ** 2)
        centrality = max(0.0, 1.0 - abs(start_y - 50.0) / 50.0)
        raw = 1.8 - 0.085 * distance + 0.65 * centrality
        base = 1.0 / (1.0 + math.exp(-raw))

    if 403 in tags:
        base *= 0.75
    if 2101 in tags:
        base *= 0.55
    if subevent_name == "free_kick_shot":
        base *= 0.65
    if tags & POST_LOCATION_TAGS:
        base *= 0.90
    if tags & OUT_LOCATION_TAGS:
        base *= 0.45
    return float(np.clip(base, 0.01, 0.85))


def increment(stats: defaultdict[str, float], key: str, value: float = 1.0) -> None:
    stats[key] += value


def update_event_features(stats: defaultdict[str, float], event: dict[str, Any]) -> bool:
    event_name = slug(event.get("eventName"), "event_unknown")
    subevent_name = slug(event.get("subEventName"), "subevent_unknown")
    raw_subevent_name = str(event.get("subEventName") or "").strip().lower().replace(" ", "_")
    period = slug(event.get("matchPeriod"), "period_unknown")
    tags = {safe_int(tag.get("id")) for tag in event.get("tags", []) if isinstance(tag, dict)}
    tags.discard(None)

    start_x = position_value(event, 0, "x")
    start_y = position_value(event, 0, "y")
    end_x = position_value(event, 1, "x")
    end_y = position_value(event, 1, "y")
    has_missing_coordinates = any(value is None for value in (start_x, start_y, end_x, end_y))

    increment(stats, "events")
    increment(stats, f"event_{event_name}")
    increment(stats, f"subevent_{subevent_name}")
    increment(stats, f"period_{period}")

    for tag_id, tag_name in SELECTED_TAGS.items():
        if tag_id in tags:
            increment(stats, f"tag_{tag_name}")

    if start_x is not None and start_y is not None:
        increment(stats, "positioned_events")
        increment(stats, "start_x_sum", start_x)
        increment(stats, "start_y_sum", start_y)
        if start_x >= 66.0:
            increment(stats, "final_third_events")
        if start_x >= 84.0 and 19.0 <= start_y <= 81.0:
            increment(stats, "penalty_area_events")

    if end_x is not None and end_y is not None:
        increment(stats, "end_positioned_events")
        increment(stats, "end_x_sum", end_x)
        increment(stats, "end_y_sum", end_y)
        if end_x >= 66.0:
            increment(stats, "final_third_entries")
        if end_x >= 84.0 and 19.0 <= end_y <= 81.0:
            increment(stats, "box_entries")

    if start_x is not None and end_x is not None:
        progression = end_x - start_x
        increment(stats, "progression_sum", progression)
        increment(stats, "progression_events")
        if progression >= 10.0:
            increment(stats, "progressive_events")

    if event_name == "pass":
        increment(stats, "passes")
        if 1801 in tags:
            increment(stats, "accurate_passes")
        if raw_subevent_name == "cross":
            increment(stats, "crosses")
            if 1801 in tags:
                increment(stats, "accurate_crosses")
        if raw_subevent_name == "smart_pass":
            increment(stats, "smart_passes")
        if raw_subevent_name == "high_pass":
            increment(stats, "high_passes")
        if raw_subevent_name == "simple_pass":
            increment(stats, "simple_passes")
        if 901 in tags:
            increment(stats, "through_passes")
        if start_x is not None and end_x is not None and end_x - start_x >= 10.0:
            increment(stats, "progressive_passes")

    elif event_name == "shot":
        increment(stats, "shots")
        if 101 in tags:
            increment(stats, "goals_scored_by_event_team")
        if 2101 in tags:
            increment(stats, "blocked_shots")
        if tags & ON_TARGET_TAGS:
            increment(stats, "shots_on_target_or_post")
        increment(stats, "xg_proxy", shot_quality_proxy(start_x, start_y, tags, raw_subevent_name))

    elif event_name == "duel":
        increment(stats, "duels")
        if 703 in tags:
            increment(stats, "duels_won")
        if 701 in tags:
            increment(stats, "duels_lost")
        if 702 in tags:
            increment(stats, "duels_neutral")

    elif event_name == "foul":
        increment(stats, "fouls")

    elif event_name == "free_kick":
        increment(stats, "free_kicks")
        if raw_subevent_name == "corner":
            increment(stats, "corners")
        elif raw_subevent_name == "free_kick_shot":
            increment(stats, "free_kick_shots")
        elif raw_subevent_name == "penalty":
            increment(stats, "penalties")
        elif raw_subevent_name == "throw_in":
            increment(stats, "throw_ins")

    elif event_name == "save_attempt":
        increment(stats, "save_attempts")

    elif event_name == "offside":
        increment(stats, "offsides")

    elif event_name == "others_on_the_ball":
        if raw_subevent_name == "clearance":
            increment(stats, "clearances")
        elif raw_subevent_name == "acceleration":
            increment(stats, "accelerations")
        elif raw_subevent_name == "touch":
            increment(stats, "touches")

    if 101 in tags:
        increment(stats, "goal_events_for")
    if 102 in tags:
        increment(stats, "own_goals_committed")
    if 1701 in tags or 1703 in tags:
        increment(stats, "red_cards_total")
    if 1702 in tags:
        increment(stats, "yellow_cards_total")

    return has_missing_coordinates


def load_event_features(
    data_dir: Path,
    matches_df: pd.DataFrame,
    competitions: set[str] | None,
    cutoff_minute: float,
) -> tuple[dict[tuple[int, str], defaultdict[str, float]], CleaningReport]:
    events_dir = data_dir / "events"
    event_paths = sorted(events_dir.glob("events_*.json"))
    if not event_paths:
        raise FileNotFoundError(f"No event files found in {events_dir}")

    if competitions:
        event_paths = [path for path in event_paths if competition_from_file(path, "events_") in competitions]

    match_team_lookup: dict[tuple[int, str], str] = {}
    match_ids = set(matches_df["match_id"].astype(int).tolist())
    for row in matches_df.itertuples(index=False):
        match_team_lookup[(int(row.match_id), str(row.home_team_id))] = "home"
        match_team_lookup[(int(row.match_id), str(row.away_team_id))] = "away"

    team_stats: dict[tuple[int, str], defaultdict[str, float]] = defaultdict(lambda: defaultdict(float))
    cleaning = CleaningReport()
    seen_event_ids: set[int] = set()

    for path in event_paths:
        competition = competition_from_file(path, "events_")
        print(f"Loading events from {competition}...")
        with path.open("r", encoding="utf-8") as fh:
            events = json.load(fh)

        for event in events:
            cleaning.raw_events += 1
            event_id = safe_int(event.get("id"))
            if event_id is not None:
                if event_id in seen_event_ids:
                    cleaning.duplicate_events += 1
                    continue
                seen_event_ids.add(event_id)

            match_id = safe_int(event.get("matchId"))
            if match_id not in match_ids:
                cleaning.events_for_unknown_match += 1
                continue

            minute = event_minute(event)
            if minute is None:
                cleaning.events_without_valid_time += 1
                continue
            if minute > cutoff_minute:
                cleaning.events_after_cutoff += 1
                continue

            team_id = safe_int(event.get("teamId"))
            key = (match_id, str(team_id))
            if key not in match_team_lookup:
                cleaning.events_for_unknown_team += 1
                continue

            cleaning.kept_events += 1
            event_name = str(event.get("eventName") or "Unknown")
            subevent_name = str(event.get("subEventName") or "Unknown")
            cleaning.event_name_counts[event_name] += 1
            cleaning.subevent_name_counts[subevent_name] += 1
            for tag in event.get("tags", []):
                if isinstance(tag, dict):
                    tag_id = safe_int(tag.get("id"))
                    if tag_id is not None:
                        cleaning.tag_counts[tag_id] += 1

            has_missing_coordinates = update_event_features(team_stats[key], event)
            if has_missing_coordinates:
                cleaning.events_with_missing_coordinates += 1

    return team_stats, cleaning


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def finalize_team_features(raw_stats: defaultdict[str, float], cutoff_minute: float) -> dict[str, float]:
    stats = dict(raw_stats)

    events = stats.get("events", 0.0)
    passes = stats.get("passes", 0.0)
    shots = stats.get("shots", 0.0)
    crosses = stats.get("crosses", 0.0)
    duels = stats.get("duels", 0.0)

    stats["events_per_minute"] = safe_ratio(events, max(cutoff_minute, 1.0))
    stats["passes_per_minute"] = safe_ratio(passes, max(cutoff_minute, 1.0))
    stats["shots_per_minute"] = safe_ratio(shots, max(cutoff_minute, 1.0))
    stats["pass_accuracy"] = safe_ratio(stats.get("accurate_passes", 0.0), passes)
    stats["cross_accuracy"] = safe_ratio(stats.get("accurate_crosses", 0.0), crosses)
    stats["shot_accuracy_proxy"] = safe_ratio(stats.get("shots_on_target_or_post", 0.0), shots)
    stats["goal_conversion_event_team"] = safe_ratio(stats.get("goals_scored_by_event_team", 0.0), shots)
    stats["xg_per_shot"] = safe_ratio(stats.get("xg_proxy", 0.0), shots)
    stats["duel_win_rate"] = safe_ratio(stats.get("duels_won", 0.0), duels)
    stats["duel_loss_rate"] = safe_ratio(stats.get("duels_lost", 0.0), duels)
    stats["final_third_event_rate"] = safe_ratio(stats.get("final_third_events", 0.0), events)
    stats["box_entry_rate"] = safe_ratio(stats.get("box_entries", 0.0), events)
    stats["progressive_event_rate"] = safe_ratio(stats.get("progressive_events", 0.0), events)
    stats["progressive_pass_rate"] = safe_ratio(stats.get("progressive_passes", 0.0), passes)
    stats["dangerous_loss_rate"] = safe_ratio(stats.get("tag_dangerous_ball_lost", 0.0), events)
    stats["counter_attack_rate"] = safe_ratio(stats.get("tag_counter_attack", 0.0), events)

    positioned_events = stats.get("positioned_events", 0.0)
    end_positioned_events = stats.get("end_positioned_events", 0.0)
    progression_events = stats.get("progression_events", 0.0)
    stats["mean_start_x"] = safe_ratio(stats.get("start_x_sum", 0.0), positioned_events)
    stats["mean_start_y"] = safe_ratio(stats.get("start_y_sum", 0.0), positioned_events)
    stats["mean_end_x"] = safe_ratio(stats.get("end_x_sum", 0.0), end_positioned_events)
    stats["mean_end_y"] = safe_ratio(stats.get("end_y_sum", 0.0), end_positioned_events)
    stats["mean_progression"] = safe_ratio(stats.get("progression_sum", 0.0), progression_events)

    for internal_key in [
        "start_x_sum",
        "start_y_sum",
        "end_x_sum",
        "end_y_sum",
        "progression_sum",
        "positioned_events",
        "end_positioned_events",
        "progression_events",
    ]:
        stats.pop(internal_key, None)

    return stats


def build_feature_matrix(
    matches_df: pd.DataFrame,
    team_stats: dict[tuple[int, str], defaultdict[str, float]],
    cutoff_minute: float,
) -> pd.DataFrame:
    finalized: dict[tuple[int, str], dict[str, float]] = {
        key: finalize_team_features(value, cutoff_minute) for key, value in team_stats.items()
    }
    all_team_feature_names = sorted({feature for values in finalized.values() for feature in values})

    rows: list[dict[str, Any]] = []
    for match in matches_df.itertuples(index=False):
        match_id = int(match.match_id)
        home_team_id = str(match.home_team_id)
        away_team_id = str(match.away_team_id)
        home_stats = finalized.get((match_id, home_team_id), {})
        away_stats = finalized.get((match_id, away_team_id), {})

        home_current_goals = home_stats.get("goal_events_for", 0.0) + away_stats.get("own_goals_committed", 0.0)
        away_current_goals = away_stats.get("goal_events_for", 0.0) + home_stats.get("own_goals_committed", 0.0)
        home_events = home_stats.get("events", 0.0)
        away_events = away_stats.get("events", 0.0)
        total_events = home_events + away_events
        home_passes = home_stats.get("passes", 0.0)
        away_passes = away_stats.get("passes", 0.0)
        total_passes = home_passes + away_passes

        row = {
            "match_id": match_id,
            "dateutc": match.dateutc,
            "competition": match.competition,
            "season_id": match.season_id,
            "round_id": match.round_id,
            "gameweek": match.gameweek,
            "duration": match.duration,
            "home_team_id": home_team_id,
            "away_team_id": away_team_id,
            "home_team_name": match.home_team_name,
            "away_team_name": match.away_team_name,
            "home_score": match.home_score,
            "away_score": match.away_score,
            "goal_diff_final": match.goal_diff_final,
            "total_goals": match.total_goals,
            "target": match.target,
            "cutoff_minute": cutoff_minute,
            "home_current_goals": home_current_goals,
            "away_current_goals": away_current_goals,
            "current_goal_diff": home_current_goals - away_current_goals,
            "abs_current_goal_diff": abs(home_current_goals - away_current_goals),
            "is_draw_at_cutoff": float(home_current_goals == away_current_goals),
            "home_leading_at_cutoff": float(home_current_goals > away_current_goals),
            "away_leading_at_cutoff": float(home_current_goals < away_current_goals),
            "home_event_share": safe_ratio(home_events, total_events),
            "away_event_share": safe_ratio(away_events, total_events),
            "home_pass_share": safe_ratio(home_passes, total_passes),
            "away_pass_share": safe_ratio(away_passes, total_passes),
            "event_share_diff": safe_ratio(home_events - away_events, total_events),
            "pass_share_diff": safe_ratio(home_passes - away_passes, total_passes),
        }

        for feature in all_team_feature_names:
            home_value = home_stats.get(feature, 0.0)
            away_value = away_stats.get(feature, 0.0)
            row[f"home_{feature}"] = home_value
            row[f"away_{feature}"] = away_value
            row[f"diff_{feature}"] = home_value - away_value

        rows.append(row)

    feature_df = pd.DataFrame(rows).sort_values("dateutc").reset_index(drop=True)
    return feature_df


def save_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)


def plot_target_distribution(feature_df: pd.DataFrame, figures_dir: Path) -> None:
    plt.figure(figsize=(8, 5))
    order = [label for label in RESULT_ORDER if label in set(feature_df["target"])]
    ax = sns.countplot(data=feature_df, x="target", order=order, hue="target", palette="Set2", legend=False)
    ax.set_title("Final Match Result Distribution")
    ax.set_xlabel("Result from home-team perspective")
    ax.set_ylabel("Matches")
    plt.tight_layout()
    plt.savefig(figures_dir / "target_distribution.png", dpi=180)
    plt.close()


def plot_competition_distribution(feature_df: pd.DataFrame, figures_dir: Path) -> None:
    crosstab = pd.crosstab(feature_df["competition"], feature_df["target"], normalize="index")
    crosstab = crosstab[[col for col in RESULT_ORDER if col in crosstab.columns]]
    ax = crosstab.plot(kind="bar", stacked=True, figsize=(10, 5), colormap="Set2")
    ax.set_title("Result Mix by Competition")
    ax.set_xlabel("Competition")
    ax.set_ylabel("Share of matches")
    ax.legend(title="Result", loc="upper right")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(figures_dir / "result_mix_by_competition.png", dpi=180)
    plt.close()


def plot_event_summary(cleaning: CleaningReport, figures_dir: Path) -> None:
    event_counts = pd.DataFrame(cleaning.event_name_counts.most_common(), columns=["event_name", "count"])
    if event_counts.empty:
        return
    plt.figure(figsize=(10, 5))
    ax = sns.barplot(
        data=event_counts.head(15),
        x="count",
        y="event_name",
        hue="event_name",
        palette="viridis",
        legend=False,
    )
    ax.set_title("Most Common In-game Event Types Kept for Modeling")
    ax.set_xlabel("Events")
    ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig(figures_dir / "event_type_counts.png", dpi=180)
    plt.close()


def plot_correlation_heatmap(feature_df: pd.DataFrame, feature_cols: list[str], figures_dir: Path) -> None:
    numeric_cols = [col for col in feature_cols if pd.api.types.is_numeric_dtype(feature_df[col])]
    if len(numeric_cols) < 3:
        return

    corr_frame = feature_df[numeric_cols].copy()
    corr_frame["target_code"] = feature_df["target"].map({"away_win": -1, "draw": 0, "home_win": 1})
    top_features = (
        corr_frame.corr(numeric_only=True)["target_code"]
        .drop("target_code")
        .abs()
        .sort_values(ascending=False)
        .head(18)
        .index.tolist()
    )
    if not top_features:
        return

    plt.figure(figsize=(12, 9))
    corr = feature_df[top_features].corr(numeric_only=True)
    ax = sns.heatmap(corr, cmap="coolwarm", center=0.0, square=False, linewidths=0.3)
    ax.set_title("Correlation Among Most Target-related Engineered Features")
    plt.tight_layout()
    plt.savefig(figures_dir / "top_feature_correlation_heatmap.png", dpi=180)
    plt.close()


def make_confusion_matrix_df(y_true: Iterable[str], y_pred: Iterable[str], labels: list[str]) -> pd.DataFrame:
    label_to_index = {label: index for index, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for actual, predicted in zip(y_true, y_pred):
        if actual in label_to_index and predicted in label_to_index:
            matrix[label_to_index[actual], label_to_index[predicted]] += 1
    return pd.DataFrame(matrix, index=labels, columns=labels)


def plot_confusion_matrix(y_true: pd.Series, y_pred: np.ndarray, figures_dir: Path) -> pd.DataFrame:
    labels = [label for label in RESULT_ORDER if label in set(y_true) | set(y_pred)]
    if SKLEARN_AVAILABLE:
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    else:
        cm_df = make_confusion_matrix_df(y_true, y_pred, labels)
    plt.figure(figsize=(7, 6))
    ax = sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", cbar=False)
    ax.set_title("Best Model Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(figures_dir / "best_model_confusion_matrix.png", dpi=180)
    plt.close()
    return cm_df


def plot_feature_importance(importance_df: pd.DataFrame, figures_dir: Path) -> None:
    if importance_df.empty:
        return
    plot_df = importance_df.head(25).iloc[::-1]
    plt.figure(figsize=(10, 8))
    ax = sns.barplot(
        data=plot_df,
        x="importance_mean",
        y="feature",
        hue="feature",
        palette="mako",
        legend=False,
    )
    ax.set_title("Permutation Importance of Best Model")
    ax.set_xlabel("Mean macro-F1 decrease after permutation")
    ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig(figures_dir / "permutation_feature_importance.png", dpi=180)
    plt.close()


def make_eda_artifacts(
    feature_df: pd.DataFrame,
    feature_cols: list[str],
    cleaning: CleaningReport,
    output_paths: dict[str, Path],
    cutoff_minute: float,
) -> None:
    figures_dir = output_paths["figures"]
    tables_dir = output_paths["tables"]

    target_summary = feature_df["target"].value_counts().rename_axis("target").reset_index(name="matches")
    competition_summary = (
        feature_df.groupby("competition", as_index=False)
        .agg(matches=("match_id", "count"), avg_total_goals=("total_goals", "mean"))
        .sort_values("matches", ascending=False)
    )
    high_signal_features = [
        "current_goal_diff",
        "home_event_share",
        "diff_xg_proxy",
        "diff_shots",
        "diff_shots_on_target_or_post",
        "diff_pass_accuracy",
        "diff_final_third_entries",
        "diff_box_entries",
        "diff_duel_win_rate",
        "diff_red_cards_total",
    ]
    available_high_signal = [col for col in high_signal_features if col in feature_df.columns]
    signal_summary = feature_df.groupby("target")[available_high_signal].mean().reset_index()

    target_summary.to_csv(tables_dir / "eda_target_distribution.csv", index=False)
    competition_summary.to_csv(tables_dir / "eda_competition_summary.csv", index=False)
    signal_summary.to_csv(tables_dir / "eda_high_signal_feature_means.csv", index=False)

    plot_target_distribution(feature_df, figures_dir)
    plot_competition_distribution(feature_df, figures_dir)
    plot_event_summary(cleaning, figures_dir)
    plot_correlation_heatmap(feature_df, feature_cols, figures_dir)

    eda_summary = {
        "cutoff_minute": cutoff_minute,
        "matches": int(feature_df.shape[0]),
        "features": int(len(feature_cols)),
        "date_min": feature_df["dateutc"].min(),
        "date_max": feature_df["dateutc"].max(),
        "competitions": feature_df["competition"].value_counts().to_dict(),
        "target_distribution": feature_df["target"].value_counts().to_dict(),
        "average_total_goals": float(feature_df["total_goals"].mean()),
        "median_events_per_match_before_cutoff": float(
            (feature_df["home_events"] + feature_df["away_events"]).median()
            if {"home_events", "away_events"}.issubset(feature_df.columns)
            else 0.0
        ),
        "cleaning_report": cleaning.to_jsonable(),
    }
    save_json(tables_dir / "eda_summary.json", eda_summary)


def modeling_columns(feature_df: pd.DataFrame, include_team_ids: bool = False) -> tuple[list[str], list[str], list[str]]:
    metadata_cols = {
        "match_id",
        "dateutc",
        "target",
        "home_score",
        "away_score",
        "goal_diff_final",
        "total_goals",
        "home_team_name",
        "away_team_name",
        "winner",
        "venue",
        "duration",
    }
    categorical_candidates = ["competition"]
    if include_team_ids:
        categorical_candidates.extend(["home_team_id", "away_team_id"])
    else:
        metadata_cols.update({"home_team_id", "away_team_id"})
    categorical_cols = [col for col in categorical_candidates if col in feature_df.columns]
    feature_cols = [col for col in feature_df.columns if col not in metadata_cols]
    numeric_cols = [
        col
        for col in feature_cols
        if col not in categorical_cols and pd.api.types.is_numeric_dtype(feature_df[col])
    ]
    return feature_cols, numeric_cols, categorical_cols


def manual_stratified_indices(y: pd.Series, test_size: float) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(RANDOM_STATE)
    train_indices: list[int] = []
    test_indices: list[int] = []
    for _, class_indices in y.groupby(y).groups.items():
        indices = np.array(list(class_indices), dtype=int)
        rng.shuffle(indices)
        n_test = max(1, int(round(len(indices) * test_size)))
        n_test = min(n_test, len(indices) - 1) if len(indices) > 1 else len(indices)
        test_indices.extend(indices[:n_test].tolist())
        train_indices.extend(indices[n_test:].tolist())
    rng.shuffle(train_indices)
    rng.shuffle(test_indices)
    return np.array(train_indices, dtype=int), np.array(test_indices, dtype=int)


def split_dataset(
    feature_df: pd.DataFrame,
    feature_cols: list[str],
    test_size: float,
    split: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = feature_df[feature_cols].copy()
    y = feature_df["target"].copy()

    if split == "stratified":
        if SKLEARN_AVAILABLE:
            splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=RANDOM_STATE)
            train_idx, test_idx = next(splitter.split(X, y))
        else:
            train_idx, test_idx = manual_stratified_indices(y, test_size)
    else:
        ordered = feature_df.sort_values("dateutc").reset_index()
        split_index = int(round((1.0 - test_size) * len(ordered)))
        split_index = min(max(split_index, 1), len(ordered) - 1)
        train_idx = ordered.loc[: split_index - 1, "index"].to_numpy()
        test_idx = ordered.loc[split_index:, "index"].to_numpy()

        if y.iloc[test_idx].nunique() < min(3, y.nunique()):
            warnings.warn(
                "Chronological test split lost at least one class. Falling back to stratified split.",
                RuntimeWarning,
            )
            if SKLEARN_AVAILABLE:
                splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=RANDOM_STATE)
                train_idx, test_idx = next(splitter.split(X, y))
            else:
                train_idx, test_idx = manual_stratified_indices(y, test_size)

    return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]


def build_preprocessor(numeric_cols: list[str], categorical_cols: list[str], scale_numeric: bool) -> ColumnTransformer:
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_transformer = Pipeline(steps=numeric_steps)
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_candidate_models(numeric_cols: list[str], categorical_cols: list[str]) -> dict[str, Pipeline]:
    linear_preprocessor = build_preprocessor(numeric_cols, categorical_cols, scale_numeric=True)
    tree_preprocessor = build_preprocessor(numeric_cols, categorical_cols, scale_numeric=False)

    return {
        "dummy_most_frequent": Pipeline(
            steps=[
                ("preprocess", tree_preprocessor),
                ("model", DummyClassifier(strategy="most_frequent")),
            ]
        ),
        "multinomial_logistic_regression": Pipeline(
            steps=[
                ("preprocess", linear_preprocessor),
                (
                    "model",
                    LogisticRegression(
                        max_iter=4000,
                        class_weight="balanced",
                        multi_class="auto",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocess", tree_preprocessor),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=500,
                        min_samples_leaf=3,
                        max_features="sqrt",
                        class_weight="balanced_subsample",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "extra_trees": Pipeline(
            steps=[
                ("preprocess", tree_preprocessor),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=600,
                        min_samples_leaf=2,
                        max_features="sqrt",
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            steps=[
                ("preprocess", tree_preprocessor),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_iter=350,
                        learning_rate=0.035,
                        l2_regularization=0.08,
                        early_stopping=True,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def evaluate_holdout(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    predictions = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "balanced_accuracy": balanced_accuracy_score(y_test, predictions),
        "macro_f1": f1_score(y_test, predictions, average="macro"),
        "weighted_f1": f1_score(y_test, predictions, average="weighted"),
    }
    if hasattr(model, "predict_proba"):
        try:
            probabilities = model.predict_proba(X_test)
            labels = list(model.classes_)
            metrics["log_loss"] = log_loss(y_test, probabilities, labels=labels)
        except Exception:
            metrics["log_loss"] = np.nan
    return metrics


def cross_validation_metrics(model: Pipeline, X_train: pd.DataFrame, y_train: pd.Series, cv_folds: int) -> dict[str, float]:
    if cv_folds <= 1:
        return {}
    min_class_count = int(y_train.value_counts().min())
    folds = min(cv_folds, min_class_count)
    if folds <= 1:
        return {}

    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring={"accuracy": "accuracy", "macro_f1": "f1_macro", "balanced_accuracy": "balanced_accuracy"},
        n_jobs=-1,
        error_score="raise",
    )
    return {
        "cv_accuracy_mean": float(scores["test_accuracy"].mean()),
        "cv_accuracy_std": float(scores["test_accuracy"].std()),
        "cv_macro_f1_mean": float(scores["test_macro_f1"].mean()),
        "cv_macro_f1_std": float(scores["test_macro_f1"].std()),
        "cv_balanced_accuracy_mean": float(scores["test_balanced_accuracy"].mean()),
        "cv_balanced_accuracy_std": float(scores["test_balanced_accuracy"].std()),
    }


class NumpyMajorityClassifier:
    """Tiny fallback baseline used only when scikit-learn cannot be imported."""

    def fit(self, _x: np.ndarray, y: Iterable[str]) -> "NumpyMajorityClassifier":
        counts = Counter(y)
        self.classes_ = np.array([label for label in RESULT_ORDER if label in counts])
        self.majority_class_ = counts.most_common(1)[0][0]
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.repeat(self.majority_class_, x.shape[0])


class NumpySoftmaxClassifier:
    """Multiclass logistic regression trained with full-batch gradient descent."""

    def __init__(self, learning_rate: float = 0.08, l2: float = 0.01, epochs: int = 900) -> None:
        self.learning_rate = learning_rate
        self.l2 = l2
        self.epochs = epochs

    def fit(self, x: np.ndarray, y: Iterable[str]) -> "NumpySoftmaxClassifier":
        y_array = np.array(list(y))
        self.classes_ = np.array([label for label in RESULT_ORDER if label in set(y_array)])
        class_to_index = {label: index for index, label in enumerate(self.classes_)}
        y_index = np.array([class_to_index[label] for label in y_array], dtype=int)
        y_one_hot = np.eye(len(self.classes_))[y_index]

        rng = np.random.default_rng(RANDOM_STATE)
        x_augmented = self._augment(x)
        self.weights_ = rng.normal(0.0, 0.01, size=(x_augmented.shape[1], len(self.classes_)))

        counts = np.bincount(y_index, minlength=len(self.classes_)).astype(float)
        class_weights = len(y_index) / (len(self.classes_) * np.maximum(counts, 1.0))
        sample_weights = class_weights[y_index]
        sample_weights = sample_weights / sample_weights.mean()

        for _ in range(self.epochs):
            probabilities = self._softmax(x_augmented @ self.weights_)
            residual = (probabilities - y_one_hot) * sample_weights[:, None]
            gradient = (x_augmented.T @ residual) / sample_weights.sum()
            gradient[:-1] += self.l2 * self.weights_[:-1]
            self.weights_ -= self.learning_rate * gradient

        return self

    @staticmethod
    def _augment(x: np.ndarray) -> np.ndarray:
        return np.column_stack([x, np.ones(x.shape[0])])

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp_values = np.exp(shifted)
        return exp_values / exp_values.sum(axis=1, keepdims=True)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self._softmax(self._augment(x) @ self.weights_)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(x), axis=1)]

    def coefficient_importance(self) -> np.ndarray:
        return np.mean(np.abs(self.weights_[:-1]), axis=1)


def fallback_encode_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    categorical_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str], pd.Series, pd.Series]:
    combined = pd.concat([X_train, X_test], axis=0)
    combined = pd.get_dummies(combined, columns=[col for col in categorical_cols if col in combined.columns], dummy_na=True)
    combined = combined.apply(pd.to_numeric, errors="coerce")
    combined = combined.replace([np.inf, -np.inf], np.nan)

    train_encoded = combined.iloc[: len(X_train)].copy()
    test_encoded = combined.iloc[len(X_train) :].copy()
    medians = train_encoded.median(axis=0).fillna(0.0)
    train_encoded = train_encoded.fillna(medians)
    test_encoded = test_encoded.fillna(medians)

    means = train_encoded.mean(axis=0)
    stds = train_encoded.std(axis=0).replace(0.0, 1.0).fillna(1.0)
    train_scaled = ((train_encoded - means) / stds).to_numpy(dtype=float)
    test_scaled = ((test_encoded - means) / stds).to_numpy(dtype=float)
    return train_scaled, test_scaled, train_encoded.columns.tolist(), means, stds


def simple_classification_report_df(y_true: Iterable[str], y_pred: Iterable[str]) -> pd.DataFrame:
    y_true_array = np.array(list(y_true))
    y_pred_array = np.array(list(y_pred))
    labels = [label for label in RESULT_ORDER if label in set(y_true_array) | set(y_pred_array)]
    rows: dict[str, dict[str, float]] = {}

    for label in labels:
        tp = float(np.sum((y_true_array == label) & (y_pred_array == label)))
        fp = float(np.sum((y_true_array != label) & (y_pred_array == label)))
        fn = float(np.sum((y_true_array == label) & (y_pred_array != label)))
        support = float(np.sum(y_true_array == label))
        precision = safe_ratio(tp, tp + fp)
        recall = safe_ratio(tp, tp + fn)
        f1 = safe_ratio(2.0 * precision * recall, precision + recall)
        rows[label] = {"precision": precision, "recall": recall, "f1-score": f1, "support": support}

    accuracy = float(np.mean(y_true_array == y_pred_array))
    macro = {
        "precision": float(np.mean([rows[label]["precision"] for label in labels])),
        "recall": float(np.mean([rows[label]["recall"] for label in labels])),
        "f1-score": float(np.mean([rows[label]["f1-score"] for label in labels])),
        "support": float(len(y_true_array)),
    }
    weights = np.array([rows[label]["support"] for label in labels], dtype=float)
    weight_total = weights.sum() if weights.sum() else 1.0
    weighted = {
        metric: float(
            np.sum([rows[label][metric] * rows[label]["support"] for label in labels]) / weight_total
        )
        for metric in ["precision", "recall", "f1-score"]
    }
    weighted["support"] = float(len(y_true_array))

    rows["accuracy"] = {"precision": accuracy, "recall": accuracy, "f1-score": accuracy, "support": len(y_true_array)}
    rows["macro avg"] = macro
    rows["weighted avg"] = weighted
    return pd.DataFrame(rows).T


def simple_metrics(y_true: Iterable[str], y_pred: Iterable[str]) -> dict[str, float]:
    report = simple_classification_report_df(y_true, y_pred)
    return {
        "accuracy": float(report.loc["accuracy", "f1-score"]),
        "balanced_accuracy": float(report.loc["macro avg", "recall"]),
        "macro_f1": float(report.loc["macro avg", "f1-score"]),
        "weighted_f1": float(report.loc["weighted avg", "f1-score"]),
    }


def save_fallback_model(path: Path, payload: dict[str, Any]) -> None:
    if joblib is not None:
        joblib.dump(payload, path)
    else:
        with path.open("wb") as fh:
            pickle.dump(payload, fh)


def train_and_evaluate_numpy_models(
    feature_df: pd.DataFrame,
    feature_cols: list[str],
    categorical_cols: list[str],
    output_paths: dict[str, Path],
    test_size: float,
    split: str,
    permutation_repeats: int,
) -> tuple[str, Any, pd.DataFrame]:
    tables_dir = output_paths["tables"]
    models_dir = output_paths["models"]
    figures_dir = output_paths["figures"]

    print(f"scikit-learn is unavailable, using NumPy fallback models. Import error: {SKLEARN_IMPORT_ERROR}")
    X_train, X_test, y_train, y_test = split_dataset(feature_df, feature_cols, test_size, split)
    x_train, x_test, encoded_feature_names, means, stds = fallback_encode_features(X_train, X_test, categorical_cols)

    models: dict[str, Any] = {
        "dummy_most_frequent_numpy": NumpyMajorityClassifier(),
        "softmax_logistic_regression_numpy": NumpySoftmaxClassifier(learning_rate=0.08, l2=0.01, epochs=900),
        "softmax_logistic_regression_low_l2_numpy": NumpySoftmaxClassifier(learning_rate=0.06, l2=0.002, epochs=1100),
    }

    rows: list[dict[str, Any]] = []
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        rows.append({"model": name, **simple_metrics(y_test, predictions)})

    comparison_df = pd.DataFrame(rows).sort_values(["macro_f1", "accuracy"], ascending=False)
    comparison_df.to_csv(tables_dir / "model_comparison.csv", index=False)
    best_name = str(comparison_df.iloc[0]["model"])
    best_model = models[best_name]
    best_predictions = best_model.predict(x_test)

    report_df = simple_classification_report_df(y_test, best_predictions)
    report_df.to_csv(tables_dir / "classification_report_best_model.csv")
    cm_df = plot_confusion_matrix(y_test, best_predictions, figures_dir)
    cm_df.to_csv(tables_dir / "confusion_matrix_best_model.csv")

    if hasattr(best_model, "coefficient_importance"):
        importance_values = best_model.coefficient_importance()
    else:
        importance_values = np.zeros(len(encoded_feature_names))
    importance_df = (
        pd.DataFrame(
            {
                "feature": encoded_feature_names,
                "importance_mean": importance_values,
                "importance_std": np.zeros(len(encoded_feature_names)),
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )
    if permutation_repeats <= 0:
        importance_df["importance_type"] = "coefficient_magnitude"
    else:
        importance_df["importance_type"] = "coefficient_magnitude_fallback"
    importance_df.to_csv(tables_dir / "permutation_importance.csv", index=False)
    plot_feature_importance(importance_df, figures_dir)

    holdout_manifest = {
        "split": split,
        "test_size": test_size,
        "train_matches": int(len(X_train)),
        "test_matches": int(len(X_test)),
        "train_target_distribution": y_train.value_counts().to_dict(),
        "test_target_distribution": y_test.value_counts().to_dict(),
        "best_model": best_name,
        "best_model_metrics": comparison_df.iloc[0].to_dict(),
        "sklearn_available": False,
        "sklearn_import_error": str(SKLEARN_IMPORT_ERROR),
        "fallback_note": "NumPy softmax regression was used because scikit-learn could not be imported.",
    }
    save_json(tables_dir / "modeling_summary.json", holdout_manifest)
    save_fallback_model(
        models_dir / "best_football_result_model.joblib",
        {
            "model": best_model,
            "encoded_feature_names": encoded_feature_names,
            "means": means,
            "stds": stds,
            "categorical_cols": categorical_cols,
        },
    )

    return best_name, best_model, comparison_df


def train_and_evaluate_models(
    feature_df: pd.DataFrame,
    feature_cols: list[str],
    numeric_cols: list[str],
    categorical_cols: list[str],
    output_paths: dict[str, Path],
    test_size: float,
    split: str,
    cv_folds: int,
    permutation_repeats: int,
) -> tuple[str, Pipeline, pd.DataFrame]:
    if not SKLEARN_AVAILABLE:
        return train_and_evaluate_numpy_models(
            feature_df=feature_df,
            feature_cols=feature_cols,
            categorical_cols=categorical_cols,
            output_paths=output_paths,
            test_size=test_size,
            split=split,
            permutation_repeats=permutation_repeats,
        )

    tables_dir = output_paths["tables"]
    models_dir = output_paths["models"]
    figures_dir = output_paths["figures"]

    X_train, X_test, y_train, y_test = split_dataset(feature_df, feature_cols, test_size, split)
    models = build_candidate_models(numeric_cols, categorical_cols)

    rows: list[dict[str, Any]] = []
    fitted_models: dict[str, Pipeline] = {}

    for name, model in models.items():
        print(f"Training {name}...")
        cv_metrics = cross_validation_metrics(model, X_train, y_train, cv_folds)
        model.fit(X_train, y_train)
        fitted_models[name] = model
        holdout_metrics = evaluate_holdout(model, X_test, y_test)
        rows.append({"model": name, **cv_metrics, **holdout_metrics})

    comparison_df = pd.DataFrame(rows).sort_values(["macro_f1", "accuracy"], ascending=False)
    comparison_df.to_csv(tables_dir / "model_comparison.csv", index=False)

    best_name = str(comparison_df.iloc[0]["model"])
    best_model = fitted_models[best_name]
    best_predictions = best_model.predict(X_test)
    report_df = pd.DataFrame(classification_report(y_test, best_predictions, output_dict=True)).T
    report_df.to_csv(tables_dir / "classification_report_best_model.csv")
    cm_df = plot_confusion_matrix(y_test, best_predictions, figures_dir)
    cm_df.to_csv(tables_dir / "confusion_matrix_best_model.csv")

    importance_df = pd.DataFrame()
    if permutation_repeats > 0:
        print("Computing permutation importance for the best model...")
        importance = permutation_importance(
            best_model,
            X_test,
            y_test,
            scoring="f1_macro",
            n_repeats=permutation_repeats,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        importance_df = (
            pd.DataFrame(
                {
                    "feature": feature_cols,
                    "importance_mean": importance.importances_mean,
                    "importance_std": importance.importances_std,
                }
            )
            .sort_values("importance_mean", ascending=False)
            .reset_index(drop=True)
        )
        importance_df.to_csv(tables_dir / "permutation_importance.csv", index=False)
        plot_feature_importance(importance_df, figures_dir)

    holdout_manifest = {
        "split": split,
        "test_size": test_size,
        "train_matches": int(len(X_train)),
        "test_matches": int(len(X_test)),
        "train_target_distribution": y_train.value_counts().to_dict(),
        "test_target_distribution": y_test.value_counts().to_dict(),
        "best_model": best_name,
        "best_model_metrics": comparison_df.iloc[0].to_dict(),
    }
    save_json(tables_dir / "modeling_summary.json", holdout_manifest)
    if joblib is not None:
        joblib.dump(best_model, models_dir / "best_football_result_model.joblib")
    else:
        with (models_dir / "best_football_result_model.joblib").open("wb") as fh:
            pickle.dump(best_model, fh)

    return best_name, best_model, comparison_df


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    args = parse_args()
    output_paths = ensure_output_dirs(args.output_dir)

    competitions = set(args.competitions) if args.competitions else None
    print("Loading matches and team metadata...")
    team_names = load_team_names(args.data_dir)
    matches_df = parse_match_rows(args.data_dir, competitions, team_names)
    print(f"Loaded {len(matches_df):,} played matches from {matches_df['competition'].nunique()} competitions.")

    print(f"Engineering event features up to minute {args.cutoff_minute:g}...")
    team_stats, cleaning = load_event_features(args.data_dir, matches_df, competitions, args.cutoff_minute)
    feature_df = build_feature_matrix(matches_df, team_stats, args.cutoff_minute)

    feature_cols, numeric_cols, categorical_cols = modeling_columns(feature_df, include_team_ids=args.include_team_ids)
    feature_df.to_csv(output_paths["tables"] / "engineered_match_features.csv", index=False)

    print("Creating EDA artifacts...")
    make_eda_artifacts(feature_df, feature_cols, cleaning, output_paths, args.cutoff_minute)

    print("Training and evaluating models...")
    best_name, _, comparison_df = train_and_evaluate_models(
        feature_df=feature_df,
        feature_cols=feature_cols,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        output_paths=output_paths,
        test_size=args.test_size,
        split=args.split,
        cv_folds=args.cv_folds,
        permutation_repeats=args.permutation_repeats,
    )

    print("\nModel comparison:")
    print(comparison_df.to_string(index=False))
    print(f"\nBest model: {best_name}")
    print(f"Artifacts saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
