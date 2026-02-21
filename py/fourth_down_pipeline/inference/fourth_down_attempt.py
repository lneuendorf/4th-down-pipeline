import logging
import xgboost as xgb
import pandas as pd
import numpy as np
from inference import win_probability
from feature_engineering.feature_engineering import add_kneel_features

MODEL_PATH = 'models/fourth_down/xgb_classifier.json'

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
    level=logging.INFO
)
LOG = logging.getLogger(__name__)

def compute_fourth_down_attempt_eWP(data: pd.DataFrame) -> pd.DataFrame:
    """ Compute fourth down attempt expected win probabilities """
    LOG.info('Predicting fourth down conversion probability.')
    data['fourth_down_conversion_proba'] = predict_conversion_probability(data)

    LOG.info('Predicting win probabilities after successful fourth down conversion.')
    wp_features = [
        'score_diff',
        'offense_score',
        'defense_score',
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
        'seconds_after_kneelout',
        'seconds_after_punt_and_opponent_kneelout',
        # 'can_kneel_out',
        # 'can_kneel_out_30',
        # 'can_kneel_out_60',
        # 'can_kneel_out_90',
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
            diff_time_ratio=lambda x: np.where(
                x.yards_to_goal <= x.distance, # if scored touchdown
                (-x['score_diff'] - 7) * np.exp(4 * (3600 - np.maximum(x['game_seconds_remaining'] - 5, 0)) / 3600),
                (x['score_diff']) * np.exp(4 * (3600 - np.maximum(x['game_seconds_remaining'] - 5, 0)) / 3600)
            ),
            score_diff=lambda x: np.where(
                x.yards_to_goal <= x.distance,  # if scored touchdown
                (-1 * x['score_diff']) - 7, # flip defense to offense team
                x['score_diff']
            ),
            spread_time_ratio=lambda x: np.where(
                x.yards_to_goal <= x.distance,  # if scored touchdown
                (-x['pregame_spread']) * np.exp(-4 * (3600 - np.maximum(x['game_seconds_remaining'] - 5, 0)) / 3600),
                (x['pregame_spread']) * np.exp(-4 * (3600 - np.maximum(x['game_seconds_remaining'] - 5, 0)) / 3600)
            ),
            pregame_offense_elo_new=lambda x: x.pregame_offense_elo,
            pregame_defense_elo_new=lambda x: x.pregame_defense_elo,
            pct_game_played=lambda x: np.minimum(x['pct_game_played'] + five_seconds_pct, 1.0),
            seconds_left_in_half=lambda x: np.maximum(x['seconds_left_in_half'] - 5, 0),
            game_seconds_remaining=lambda x: np.maximum(x['game_seconds_remaining'] - 5, 0),
            is_home_team=lambda x: np.where(
                x.yards_to_goal <= x.distance, # if scored touchdown
                np.select([x['is_home_team'] == 1, x['is_home_team'] == -1], [-1, 1], default=0), 
                x['is_home_team']
            ),
            offense_timeouts_new=lambda x: np.where(
                x.yards_to_goal <= x.distance, # if scored touchdown
                x['defense_timeouts'],
                x['offense_timeouts']
            ),
            defense_timeouts_new=lambda x: np.where(
                x.yards_to_goal <= x.distance, # if scored touchdown
                x['offense_timeouts'],
                x['defense_timeouts']
            ),
            yards_to_goal=lambda x: np.where(
                x.yards_to_goal <= x.distance, # if scored touchdown
                80,
                x['yards_to_goal'] - x['distance']
            ),
            down=1,
            distance=10
        )
        .drop(columns=['offense_timeouts', 'defense_timeouts'
                       ,'pregame_offense_elo', 'pregame_defense_elo'])
        .rename(columns={
            'offense_timeouts_new': 'offense_timeouts',
            'defense_timeouts_new': 'defense_timeouts',
            'pregame_offense_elo_new': 'pregame_offense_elo',
            'pregame_defense_elo_new': 'pregame_defense_elo'
        })
    )
    
    wp_convert_data = add_kneel_features(wp_convert_data)
    wp_convert_data = wp_convert_data[wp_features]

    # If the conversion leads to a TD, then flip the WP back from defense to offense
    probas = win_probability.predict_win_probability(wp_convert_data)
    probas = np.where(
        data.yards_to_goal.values <= data.distance.values,
        1 - probas,
        probas
    )
    # Set WP to 1 or 0 if the game is over after the FG
    pct_game_played = wp_convert_data['pct_game_played'].values
    yards_to_goal = data.yards_to_goal.values
    distance = data.distance.values
    score_diff = wp_convert_data['score_diff'].values
    game_over_win = (pct_game_played == 1.0) & (
        ((yards_to_goal <= distance) & ((-1 * score_diff) > 0)) |
        ((yards_to_goal > distance) & (score_diff > 0))
    )
    game_over_loss = (pct_game_played == 1.0) & (
        ((yards_to_goal <= distance) & ((-1 * score_diff) < 0)) |
        ((yards_to_goal > distance) & (score_diff < 0))
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
            diff_time_ratio=lambda x: (
                (-x['score_diff'] * np.exp(4 * (3600 - np.maximum(x['game_seconds_remaining'] - 5, 0)) / 3600))
            ),
            score_diff=lambda x: x.score_diff * -1,
            spread_time_ratio=lambda x: (
                (-x['pregame_spread']) * np.exp(-4 * (3600 - np.maximum(x['game_seconds_remaining'] - 5, 0)) / 3600)
            ),
            pregame_offense_elo_new=lambda x: x.pregame_defense_elo,
            pregame_defense_elo_new=lambda x: x.pregame_offense_elo,
            pct_game_played=lambda x: np.minimum(x['pct_game_played'] + five_seconds_pct, 1.0),
            seconds_left_in_half=lambda x: np.maximum(x['seconds_left_in_half'] - 5, 0),
            game_seconds_remaining=lambda x: np.maximum(x['game_seconds_remaining'] - 5, 0),
            is_home_team=lambda x: np.select(
                condlist=[x['is_home_team'] == 1, x['is_home_team'] == -1], 
                choicelist=[-1, 1], 
                default=0
            ),
            offense_timeouts_new=lambda x: x.defense_timeouts,
            defense_timeouts_new=lambda x: x.offense_timeouts,
            yards_to_goal=lambda x: 100 - x['yards_to_goal'],
            down=1,
            distance=10
        )
        .drop(columns=['offense_timeouts','defense_timeouts'
                       ,'pregame_offense_elo', 'pregame_defense_elo'])
        .rename(columns={
            'offense_timeouts_new':'offense_timeouts',
            'defense_timeouts_new':'defense_timeouts',
            'pregame_offense_elo_new':'pregame_offense_elo',
            'pregame_defense_elo_new':'pregame_defense_elo'
        })
    )
    wp_fail_data = add_kneel_features(wp_fail_data)
    wp_fail_data = wp_fail_data[wp_features]
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

def predict_conversion_probability(
    df: pd.DataFrame
) -> pd.Series:
    """ 
    Given a dataframe of features, return conversion probability predictions.

    Args:
        df (pd.DataFrame): Input dataframe containing feature columns.

    Returns:
        pd.Series: Conversion probability predictions (values between 0 and 1).
    """
    
    feature_names = [
        'distance',
        'diff_time_ratio',
        'is_home_team',
        'precipitation', 
        'wind_speed', 
        'temperature', 
        'yards_to_goal', 
        'offense_strength', 
        'defense_strength'
    ]
    dmatrix = xgb.DMatrix(df[feature_names])

    model = _load_model(MODEL_PATH)
    preds = model.predict(dmatrix)
    
    return pd.Series(preds, index=df.index, name="conversion_probability")



def _load_model(model_path: str) -> xgb.Booster:
    model = xgb.Booster()
    model.load_model(model_path)
    return model