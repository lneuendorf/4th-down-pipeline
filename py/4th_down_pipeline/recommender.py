from os.path import join
import pickle
import argparse
import logging

import numpy as np
import pandas as pd

from data_loader import data_loader as dl
from feature_engineering import feature_engineering as fe
from feature_engineering.team_strength import impute_missing_team_strengths
from inference import(
    field_goal,
    fourth_down_attempt,
    punt
)

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
    level=logging.INFO
)
LOG = logging.getLogger(__name__)


def get_recommendations(
    year: int, 
    week: int, 
    season_type: str, 
    force_data_update: bool = False
) -> pd.DataFrame:
    """ Get 4th down recommendations for all games in a specific year, week, and season type """
    # Load all necessary data
    games = dl.load_games(year, week, season_type, force_data_update)
    plays = dl.load_plays(year, week, season_type, force_data_update)
    weather = dl.load_weather(year, week, season_type, force_data_update)
    venues = dl.load_venues(force_data_update)
    lines = dl.load_lines(year, week, season_type, force_data_update)
    elo = dl.load_elo(year, week, season_type, force_data_update)
    team_strengths = dl.load_team_strengths(year, week, season_type, force_data_update)

    data = fe.engineer_features(games, plays, weather, venues, lines, elo, team_strengths)
    data = fe.add_decision(data)

    # Other filtering
    data = (
        data
        .query('~((action == "field_goal") & (yards_to_goal > 60))')
        .query('offense_division == "fbs" or defense_division == "fbs"')
        .reset_index(drop=True)
    )

    data = impute_missing_team_strengths(data)

    int_cols = ['offense_score', 'defense_score',
       'offense_timeouts', 'defense_timeouts', 'yards_to_goal', 'down',
       'distance', 'score_diff',
       'season', 'week', 'neutral_site', 
       'completed', 'is_home_team', 'grass',
       'game_indoors', 'pressure_rating', 
       'game_seconds_remaining',
       'can_kneel_out', 'can_kneel_out_30', 'can_kneel_out_60', 'can_kneel_out_90',
    ]
    for col in int_cols:
        if col in data.columns:
            data[col] = data[col].astype(int)

    float_cols = [
       'pct_game_played', 'elevation', 'seconds_left_in_half',
       'home_pregame_elo', 'away_pregame_elo', 'pregame_elo_diff', 'pregame_offense_elo',
       'pregame_defense_elo', 'precipitation', 'wind_speed', 'temperature',
       'home_spread', 'pregame_spread',
       'diff_time_ratio', 'spread_time_ratio', 'offense_strength',
       'defense_strength']
    for col in float_cols:
        if col in data.columns:
            data[col] = data[col].astype(float)

    # Compute the expected win probabilities after each possible decision outcome
    data = field_goal.compute_field_goal_eWP(data)
    data = punt.compute_punt_eWP(data)
    data = fourth_down_attempt.compute_fourth_down_attempt_eWP(data)

    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 4th down recommender pipeline")
    parser.add_argument("--year", type=int, required=True, help="Year of the season (e.g., 2025)")
    parser.add_argument("--week", type=int, required=True, help="Week of the season")
    parser.add_argument("--season_type", type=str, required=True, choices=["regular", "postseason"], help="Season type")
    parser.add_argument("--force_data_update", type=lambda x: str(x).lower() == "true", default=False,
                        help="Force refresh of data (default: False)")
    
    args = parser.parse_args()

    data = get_recommendations(
        year=args.year,
        week=args.week,
        season_type=args.season_type,
        force_data_update=args.force_data_update,
    )