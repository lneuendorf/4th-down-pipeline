from os.path import join
import pickle
import argparse
import logging

import numpy as np
import pandas as pd

from data_loader import data_loader as dl
from feature_engineering import feature_engineering as fe
from inference import(
    win_probability,
    fourth_down_attempt,
    punt,
    field_goal
)

MODEL_DIR = 'models'

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

    data = _impute_missing_team_strengths(data)

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
    data = _compute_field_goal(data)
    data = _compute_punt(data)
    data = _compute_fourth_down_attempt(data)
    breakpoint()
    return data

def _compute_field_goal(data: pd.DataFrame) -> pd.DataFrame:
    """ Compute field goal probability and expected win probabilities """
    LOG.info('Predicing field goal make probability.')
    data['fg_make_proba'] = field_goal.predict_field_goal_make_probability(data)
    # Hardcoded probabilities for long FG attempts
    rules = [
        ('50 <= yards_to_goal <= 52', 0.15),
        ('53 <= yards_to_goal <= 55', 0.11),
        ('56 <= yards_to_goal <= 58', 0.07),
        ('59 <= yards_to_goal <= 61', 0.03),
        ('62 <= yards_to_goal <= 64', 0.01),
        ('65 <= yards_to_goal <= 70', 0.0001),
        ('yards_to_goal > 70', 0.0)
    ]
    for rule, base_prob in rules:
        data.loc[data.query(rule).index, 'fg_make_proba'] = base_prob

    LOG.info('Predicting win probabilities after FG make.')
    wp_features = [
        'score_diff',
        'diff_time_ratio',
        'spread_time_ratio',
        'pregame_offense_elo',
        'pregame_defense_elo',
        'pct_game_played',
        'seconds_left_in_half',
        'is_home_team',
        'offense_timeouts',
        'defense_timeouts',
        'yards_to_goal',
        'down',
        'distance',
        'can_kneel_out',
        'can_kneel_out_30',
        'can_kneel_out_60',
        'can_kneel_out_90',
    ]

    five_seconds_pct = 5 / (15 * 60 * 4)

    # Assumptions: 
    # 1. Kicking a FG takes 5 seconds off the clock
    # 2. The receiving team yards to goal post FG attempt is 80 yards
    #NOTE: this is WP team of the team NOT kicking the FG
    wp_make_data = (
        data.assign(
            score_diff=lambda x: -x['score_diff'] - 3,
            diff_time_ratio=lambda x: (
                -x['score_diff'] - 3) * np.exp(4 * (3600 - np.maximum(x['game_seconds_remaining'] - 5, 0)) / 3600
            ),
            spread_time_ratio=lambda x: (
                -x['pregame_spread']) * np.exp(4 * (3600 - np.maximum(x['game_seconds_remaining'] - 5, 0)) / 3600
            ),
            pregame_offense_elo_new=lambda x: x.pregame_defense_elo,
            pregame_defense_elo_new=lambda x: x.pregame_offense_elo,
            pct_game_played=lambda x: np.minimum(x['pct_game_played'] + five_seconds_pct, 1.0),
            seconds_left_in_half=lambda x: np.maximum(x['seconds_left_in_half'] - 5, 0),
            is_home_team=lambda x: np.select([x['is_home_team'] == 1, x['is_home_team'] == -1], [-1, 1], default=0),
            offense_timeouts_new=lambda x: x.defense_timeouts,
            defense_timeouts_new=lambda x: x.offense_timeouts,
            yards_to_goal=80,
            down=1,
            distance=10,
            can_kneel_out=lambda x: np.where(
                x.seconds_after_kneelout - 5 <= 0, 1, 0
            ),
            can_kneel_out_30=lambda x: np.where(
                x.seconds_after_kneelout - 5 <= 30, 1, 0
            ),
            can_kneel_out_60=lambda x: np.where(
                x.seconds_after_kneelout - 5 <= 60, 1, 0
            ),
            can_kneel_out_90=lambda x: np.where(
                x.seconds_after_kneelout - 5 <= 90, 1, 0
            )

        )
        .drop(columns=['offense_timeouts','defense_timeouts',
                      'pregame_offense_elo', 'pregame_defense_elo'])
        .rename(columns={
            'offense_timeouts_new':'offense_timeouts',
            'defense_timeouts_new':'defense_timeouts',
            'pregame_offense_elo_new':'pregame_offense_elo',
            'pregame_defense_elo_new':'pregame_defense_elo'
        })
        [wp_features]
    )

    # the "1 -" here is to flip the WP back to the team that is kicking the FG
    probas = 1 - win_probability.predict_win_probability(wp_make_data)

    # Set WP to 1 or 0 if the game is over after the FG
    probas[(wp_make_data['pct_game_played'] == 1.0) & ((-1 * wp_make_data['score_diff']) > 0)] = 1.0
    probas[(wp_make_data['pct_game_played'] == 1.0) & ((-1 * wp_make_data['score_diff']) < 0)] = 0.0

    # round probas to 3 decimal places
    data['wp_make_proba'] = np.round(probas, 3)

    LOG.info('Predicting win probabilities after FG miss.')
    #NOTE: this is WP team of the team NOT kicking the FG
    wp_miss_data = (
        data.assign(
            score_diff=lambda x: (-1 * x['score_diff']),
            diff_time_ratio=lambda x: (
                -x['score_diff']) * np.exp(4 * (3600 - np.maximum(x['game_seconds_remaining'] - 5, 0)) / 3600
            ),
            spread_time_ratio=lambda x: (
                -x['pregame_spread']) * np.exp(4 * (3600 - np.maximum(x['game_seconds_remaining'] - 5, 0)) / 3600
            ),
            pregame_offense_elo_new=lambda x: x.pregame_defense_elo,
            pregame_defense_elo_new=lambda x: x.pregame_offense_elo,
            pct_game_played=lambda x: np.minimum(x['pct_game_played'] + five_seconds_pct, 1.0),
            seconds_left_in_half=lambda x: np.maximum(x['seconds_left_in_half'] - 5, 0),
            is_home_team=lambda x: np.select([x['is_home_team'] == 1, x['is_home_team'] == -1], [-1, 1], default=0),
            offense_timeouts_new=lambda x: x.defense_timeouts,
            defense_timeouts_new=lambda x: x.offense_timeouts,
            yards_to_goal=lambda x: 100 - x['yards_to_goal'],
            down=1,
            distance=10,
            can_kneel_out=lambda x: np.where(
                x.seconds_after_kneelout - 5 <= 0, 1, 0
            ),
            can_kneel_out_30=lambda x: np.where(
                x.seconds_after_kneelout - 5 <= 30, 1, 0
            ),
            can_kneel_out_60=lambda x: np.where(
                x.seconds_after_kneelout - 5 <= 60, 1, 0
            ),
            can_kneel_out_90=lambda x: np.where(
                x.seconds_after_kneelout - 5 <= 90, 1, 0
            )
        )
        .drop(columns=['offense_timeouts','defense_timeouts',
                       'pregame_offense_elo', 'pregame_defense_elo'])
        .rename(columns={
            'offense_timeouts_new':'offense_timeouts',
            'defense_timeouts_new':'defense_timeouts',
            'pregame_offense_elo_new':'pregame_offense_elo',
            'pregame_defense_elo_new':'pregame_defense_elo'
        })
        [wp_features]
    )

    # the "1 -" here is to flip the WP back to the team that is kicking the FG
    probas = 1 - win_probability.predict_win_probability(wp_miss_data)

    # Set WP to 1 or 0 if the game is over after the FG
    probas[(wp_miss_data['pct_game_played'] == 1.0) & ((-1 * wp_miss_data['score_diff']) > 0)] = 1.0
    probas[(wp_miss_data['pct_game_played'] == 1.0) & ((-1 * wp_miss_data['score_diff']) < 0)] = 0.0

    # round probas to 3 decimal places
    data['wp_miss_proba'] = np.round(probas, 3)

    data['exp_wp_fg'] = (data['fg_make_proba'] * data['wp_make_proba']) + ((1 - data['fg_make_proba']) * data['wp_miss_proba']).round(4)

    return data

def _compute_punt(data: pd.DataFrame) -> pd.DataFrame:
    """ Compute punt expected win probabilities """
    LOG.info('Predicting punt yards to goal.')
    data['receiving_team_yards_to_goal'] = punt.predict_receiving_team_yards_to_goal(
        data.rename(columns={
            'yards_to_goal':'punt_team_end_yards_to_goal',
            'pregame_offense_elo':'punting_team_pregame_elo',
            'pregame_defense_elo':'receiving_team_pregame_elo',
        })
    )

    LOG.info('Predicting win probabilities after punt.')
    wp_features = [
        'score_diff',
        'diff_time_ratio',
        'spread_time_ratio',
        'pregame_offense_elo',
        'pregame_defense_elo',
        'pct_game_played',
        'seconds_left_in_half',
        'is_home_team',
        'offense_timeouts',
        'defense_timeouts',
        'yards_to_goal',
        'down',
        'distance',
        'can_kneel_out',
        'can_kneel_out_30',
        'can_kneel_out_60',
        'can_kneel_out_90',
    ]

    five_seconds_pct = 5 / (15 * 60 * 4)

    # Assumptions: 
    # 1. Punt takes 5 seconds off the clock
    #NOTE: this is WP of the team receiving the punt
    wp_data = (
        data.assign(
            score_diff=lambda x: -x['score_diff'],
            diff_time_ratio=lambda x: (
                -x['score_diff']) * np.exp(4 * (3600 - np.maximum(x['game_seconds_remaining'] - 5, 0)) / 3600
            ),
            spread_time_ratio=lambda x: (
                -x['pregame_spread']) * np.exp(4 * (3600 - np.maximum(x['game_seconds_remaining'] - 5, 0)) / 3600
            ),
            pregame_offense_elo_new=lambda x: x.pregame_defense_elo,
            pregame_defense_elo_new=lambda x: x.pregame_offense_elo,
            pct_game_played=lambda x: np.minimum(x['pct_game_played'] + five_seconds_pct, 1.0),
            seconds_left_in_half=lambda x: np.maximum(x['seconds_left_in_half'] - 5, 0),
            pregame_spread=lambda x: -x['pregame_spread'],
            is_home_team=lambda x: np.select(
                condlist=[x['is_home_team'] == 1, x['is_home_team'] == -1], 
                choicelist=[-1, 1], 
                default=0
            ),
            offense_timeouts_new=lambda x: x.defense_timeouts,
            defense_timeouts_new=lambda x: x.offense_timeouts,
            yards_to_goal=None,
            down=1,
            distance=10,
            can_kneel_out=lambda x: np.where(
                x.seconds_after_kneelout - 5 <= 0, 1, 0
            ),
            can_kneel_out_30=lambda x: np.where(
                x.seconds_after_kneelout - 5 <= 30, 1, 0
            ),
            can_kneel_out_60=lambda x: np.where(
                x.seconds_after_kneelout - 5 <= 60, 1, 0
            ),
            can_kneel_out_90=lambda x: np.where(
                x.seconds_after_kneelout - 5 <= 90, 1, 0
            )   
        )
        .drop(columns=['offense_timeouts','defense_timeouts'
                       ,'pregame_offense_elo', 'pregame_defense_elo'])
        .rename(columns={
            'offense_timeouts_new':'offense_timeouts',
            'defense_timeouts_new':'defense_timeouts',
            'pregame_offense_elo_new':'pregame_offense_elo',
            'pregame_defense_elo_new':'pregame_defense_elo'
        })
        [wp_features]
    )

    # Any potential punt inside the 40 yard line is assumed to be downed at the 89 yard line
    data.loc[data.yards_to_goal < 40, 'punt_yards_to_goal'] = 89

    wp_data['yards_to_goal'] = data['punt_yards_to_goal']

    # the "1 -" here is to flip the WP back to the team that is punting
    probas = (1 - win_probability.predict_win_probability(wp_data))

    # Set WP to 1 or 0 if the game is over after the punt
    probas[(wp_data['pct_game_played'] == 1.0) & ((-1 * wp_data['score_diff']) > 0)] = 1.0
    probas[(wp_data['pct_game_played'] == 1.0) & ((-1 * wp_data['score_diff']) < 0)] = 0.0

    data['exp_wp_punt'] = np.round(probas, 4)

    return data

def _compute_fourth_down_attempt(data: pd.DataFrame) -> pd.DataFrame:
    """ Compute fourth down attempt expected win probabilities """
    LOG.info('Predicting fourth down conversion probability.')
    data['fourth_down_conversion_proba'] = fourth_down_attempt.predict_conversion_probability(data)

    LOG.info('Predicting win probabilities after successful fourth down conversion.')
    wp_features = [
        'score_diff',
        'diff_time_ratio',
        'spread_time_ratio',
        'pregame_offense_elo',
        'pregame_defense_elo',
        'pct_game_played',
        'seconds_left_in_half',
        'is_home_team',
        'offense_timeouts',
        'defense_timeouts',
        'yards_to_goal',
        'down',
        'distance',
        'can_kneel_out',
        'can_kneel_out_30',
        'can_kneel_out_60',
        'can_kneel_out_90',
    ]

    five_seconds_pct = 5 / (15 * 60 * 4)

    # WP Assumptions: 
    # 1. Fourth down attempt takes 5 seconds off the clock
    # 2. Only assuming they pick up exactly the yards needed for the first down (conservative estimate)
    # 3. On a touchdown, the receiving team gets the ball at the 80 yard line
    # 4. If the yards to goal is LEQ 1, then the offense team scores a touchdown if they convert
    #NOTE: this is the WP of the offense team if distance != yards_to_goal, else the WP of the defense team
    wp_convert_data = (
        data
        .assign(
            score_diff=lambda x: np.where(
                x.yards_to_goal <= 1,  # if scored touchdown
                (-1 * x['score_diff']) - 7, # flip defense to offense team
                x['score_diff']
            ),
            diff_time_ratio=lambda x: np.where(
                x.yards_to_goal <= 1,  # if scored touchdown
                (-x['score_diff'] - 7) * np.exp(4 * (3600 - np.maximum(x['game_seconds_remaining'] - 5, 0)) / 3600),
                (x['score_diff']) * np.exp(4 * (3600 - np.maximum(x['game_seconds_remaining'] - 5, 0)) / 3600)
            ),
            spread_time_ratio=lambda x: np.where(
                x.yards_to_goal <= 1,  # if scored touchdown
                (-x['pregame_spread']) * np.exp(4 * (3600 - np.maximum(x['game_seconds_remaining'] - 5, 0)) / 3600),
                (x['pregame_spread']) * np.exp(4 * (3600 - np.maximum(x['game_seconds_remaining'] - 5, 0)) / 3600)
            ),
            pregame_offense_elo_new=lambda x: x.pregame_offense_elo,
            pregame_defense_elo_new=lambda x: x.pregame_defense_elo,
            pct_game_played=lambda x: np.minimum(x['pct_game_played'] + five_seconds_pct, 1.0),
            seconds_left_in_half=lambda x: np.maximum(x['seconds_left_in_half'] - 5, 0),
            is_home_team=lambda x: np.where(
                x.yards_to_goal <= 1, # if scored touchdown
                np.select([x['is_home_team'] == 1, x['is_home_team'] == -1], [-1, 1], default=0), 
                x['is_home_team']
            ),
            offense_timeouts_new=lambda x: np.where(
                x.yards_to_goal <= 1, # if scored touchdown
                x['defense_timeouts'],
                x['offense_timeouts']
            ),
            defense_timeouts_new=lambda x: np.where(
                x.yards_to_goal <= 1, # if scored touchdown
                x['offense_timeouts'],
                x['defense_timeouts']
            ),
            yards_to_goal=lambda x: np.where(
                x.yards_to_goal <= 1, # if scored touchdown
                80,
                x['yards_to_goal'] - x['distance']
            ),
            down=1,
            distance=10,
            can_kneel_out=lambda x: np.where(
                x.seconds_after_kneelout - 5 <= 0, 1, 0
            ),
            can_kneel_out_30=lambda x: np.where(
                x.seconds_after_kneelout - 5 <= 30, 1, 0
            ),
            can_kneel_out_60=lambda x: np.where(
                x.seconds_after_kneelout - 5 <= 60, 1, 0
            ),
            can_kneel_out_90=lambda x: np.where(
                x.seconds_after_kneelout - 5 <= 90, 1, 0
            )
        )
        .drop(columns=['offense_timeouts', 'defense_timeouts'
                       ,'pregame_offense_elo', 'pregame_defense_elo'])
        .rename(columns={
            'offense_timeouts_new': 'offense_timeouts',
            'defense_timeouts_new': 'defense_timeouts',
            'pregame_offense_elo_new': 'pregame_offense_elo',
            'pregame_defense_elo_new': 'pregame_defense_elo'
        })
        [wp_features]
    )

    # If the conversion leads to a TD, then flip the WP back from defense to offense
    probas = win_probability.predict_win_probability(wp_convert_data)
    probas = np.where(
        data.yards_to_goal.values <= 1,
        1 - probas,
        probas
    )
    # Set WP to 1 or 0 if the game is over after the FG
    pct_game_played = wp_convert_data['pct_game_played'].values
    yards_to_goal = data.yards_to_goal.values
    score_diff = wp_convert_data['score_diff'].values
    game_over_win = (pct_game_played == 1.0) & (
        ((yards_to_goal <= 1) & ((-1 * score_diff) > 0)) |
        ((yards_to_goal > 1) & (score_diff > 0))
    )
    game_over_loss = (pct_game_played == 1.0) & (
        ((yards_to_goal <= 1) & ((-1 * score_diff) < 0)) |
        ((yards_to_goal > 1) & (score_diff < 0))
    )
    probas[game_over_win] = 1.0
    probas[game_over_loss] = 0.0
    data['wp_convert_proba'] = probas.round(4)

    LOG.info('Predicting win probabilities after failed fourth down conversion.')
    # Assumptions:
    # 1. Assumes 4th down attempt gains zero yards on failure
    #NOTE: this is WP of the team defending the fourth down attempt after the attempt
    wp_fail_data = (
        data
        .assign(
            score_diff=lambda x: x.score_diff * -1,
            diff_time_ratio=lambda x: (
                -x['score_diff'] * np.exp(4 * (3600 - np.maximum(x['game_seconds_remaining'] - 5, 0)) / 3600)
            ),
            spread_time_ratio=lambda x: (
                -x['pregame_spread']) * np.exp(4 * (3600 - np.maximum(x['game_seconds_remaining'] - 5, 0)) / 3600
            ),
            pregame_offense_elo_new=lambda x: x.pregame_defense_elo,
            pregame_defense_elo_new=lambda x: x.pregame_offense_elo,
            pct_game_played=lambda x: np.minimum(x['pct_game_played'] + five_seconds_pct, 1.0),
            seconds_left_in_half=lambda x: np.maximum(x['seconds_left_in_half'] - 5, 0),
            is_home_team=lambda x: np.select(
                condlist=[x['is_home_team'] == 1, x['is_home_team'] == -1], 
                choicelist=[-1, 1], 
                default=0
            ),
            offense_timeouts_new=lambda x: x.defense_timeouts,
            defense_timeouts_new=lambda x: x.offense_timeouts,
            yards_to_goal=lambda x: 100 - x['yards_to_goal'],
            down=1,
            distance=10,
            can_kneel_out=lambda x: np.where(
                x.seconds_after_kneelout - 5 <= 0, 1, 0
            ),
            can_kneel_out_30=lambda x: np.where(
                x.seconds_after_kneelout - 5 <= 30, 1, 0
            ),
            can_kneel_out_60=lambda x: np.where(
                x.seconds_after_kneelout - 5 <= 60, 1, 0
            ),
            can_kneel_out_90=lambda x: np.where(
                x.seconds_after_kneelout - 5 <= 90, 1, 0
            )
        )
        .drop(columns=['offense_timeouts','defense_timeouts'
                       ,'pregame_offense_elo', 'pregame_defense_elo'])
        .rename(columns={
            'offense_timeouts_new':'offense_timeouts',
            'defense_timeouts_new':'defense_timeouts',
            'pregame_offense_elo_new':'pregame_offense_elo',
            'pregame_defense_elo_new':'pregame_defense_elo'
        })
        [wp_features]
    )
    # the "1 -" here is to flip the WP back to the team that is attempting the 4th down
    probas = (1 - win_probability.predict_win_probability(wp_fail_data))
    # Set WP to 1 or 0 if the game is over after the FG
    probas[(wp_fail_data['pct_game_played'] == 1.0) & ((-1 * wp_fail_data['score_diff']) > 0)] = 1.0
    probas[(wp_fail_data['pct_game_played'] == 1.0) & ((-1 * wp_fail_data['score_diff']) < 0)] = 0.0
    data['wp_fail_proba'] = probas.round(4)

    data['exp_wp_go'] = (
        (data['fourth_down_conversion_proba'] * data['wp_convert_proba']) + 
        ((1 - data['fourth_down_conversion_proba']) * data['wp_fail_proba'])
    ).round(4)
    
    return data

def _impute_missing_team_strengths(data: pd.DataFrame) -> pd.DataFrame:
    """ Impute missing team strengths using linear regression based on ELO ratings. """
    offense_model = pickle.load(open(join(MODEL_DIR, 'team_strength', 'offense.pkl'), 'rb'))
    defense_model = pickle.load(open(join(MODEL_DIR, 'team_strength', 'defense.pkl'), 'rb'))

    # Offense Strength
    mask = data['offense_strength'].isna()
    if mask.sum() != 0:
        data.loc[mask, 'offense_strength'] = offense_model.predict(
            data.loc[mask, ['pregame_offense_elo']]
        )

    # Defense Strength
    mask = data['defense_strength'].isna()
    if mask.sum() != 0:
        data.loc[mask, 'defense_strength'] = defense_model.predict(
            data.loc[mask, ['pregame_defense_elo']]
        )

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