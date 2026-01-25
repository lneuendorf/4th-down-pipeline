from os.path import join
import logging
import os

from data_loader import data_loader as dl
from feature_engineering import feature_engineering as fe
from feature_engineering.team_strength import impute_missing_team_strengths
from inference import(
    field_goal,
    fourth_down_attempt,
    punt
)

OUTPUT_DIR = 'results'

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
    level=logging.INFO
)
LOG = logging.getLogger(__name__)


def generate_recommendations(
    year: int, 
    week: int, 
    season_type: str, 
    force_data_update: bool = False
) -> None:
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
        .query('offense_division == "fbs"') # Only FBS teams
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

    # Add start_date to data
    data = data.merge(
        games[['id', 'start_date']].rename(columns={'id':'game_id'}), 
        on='game_id', 
        how='left'
    )

    # Store results
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    cols = [
        'season', 'week', 'season_type', 'start_date', 'game_id', 'play_id', 
        'offense', 'offense_division', 'offense_timeouts', 'offense_score', 'offense_strength', 'pregame_offense_elo',
        'defense', 'defense_division', 'defense_timeouts', 'defense_score', 'defense_strength', 'pregame_defense_elo',
        'period', 'clock_minutes', 'clock_seconds', 'pct_game_played', 'pct_half_played', 
        'game_seconds_remaining', 'seconds_left_in_half',
        'seconds_after_kneelout', 'seconds_after_punt_and_opponent_kneelout', 'can_kneel_out',
        'can_kneel_out_30', 'can_kneel_out_60', 'can_kneel_out_90', 
        'play_type', 'play_text', 
        'score_diff', 'pregame_elo_diff', 'pregame_spread', 
        'diff_time_ratio', 'spread_time_ratio', 
        'yards_to_goal', 'down', 'distance', 
        'home_id', 'home_team', 'home_conference', 
        'away_id', 'away_team', 'away_conference', 
        'is_home_team', 
        'home_division', 'home_pregame_elo',
        'away_division', 'away_pregame_elo',
        'precipitation', 'elevation', 'grass', 'wind_speed', 'temperature', 'game_indoors',
        'pressure_rating', 
        'fg_make_proba', 'wp_make_proba',
        'wp_miss_proba', 'receiving_team_yards_to_goal',
        'fourth_down_conversion_proba', 'wp_convert_proba',
        'wp_fail_proba', 
        'action','decision','exp_wp_fg','exp_wp_go', 'exp_wp_punt']
    data[cols].to_parquet(join(OUTPUT_DIR, f'{year}_{week}_{season_type}.parquet'))

    return data[cols]