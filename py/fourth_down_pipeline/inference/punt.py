import logging
import xgboost as xgb
import pandas as pd
import numpy as np
import statsmodels.api as sm
from inference import win_probability
from feature_engineering.feature_engineering import add_kneel_features

XGB_MODEL_PATH = 'models/punt_yards_to_goal/xgb_classifier.json'
LR_MODEL_PATH = 'models/punt_yards_to_goal/linear_regression_model.pkl'

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
    level=logging.INFO
)
LOG = logging.getLogger(__name__)

def compute_punt_eWP(data: pd.DataFrame) -> pd.DataFrame:
    """ Compute punt expected win probabilities """
    LOG.info('Predicting punt yards to goal.')
    data['receiving_team_yards_to_goal'] = np.where(
        data.yards_to_goal < 40,
        88,
        predict_receiving_team_yards_to_goal(
            data.rename(columns={
                'yards_to_goal':'punt_team_end_yards_to_goal',
                'pregame_offense_elo':'punting_team_pregame_elo',
                'pregame_defense_elo':'receiving_team_pregame_elo',
            })
        )
    )

    LOG.info('Predicting win probabilities after punt.')
    wp_features = [
        'play_id',
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
        'seconds_after_kneelout',
        'seconds_after_punt_and_opponent_kneelout',
        # 'can_kneel_out',
        # 'can_kneel_out_30',
        # 'can_kneel_out_60',
        # 'can_kneel_out_90',
    ]

    five_seconds_pct = 5 / (15 * 60 * 4)

    # Assumptions: 
    # 1. Punt takes 5 seconds off the clock
    #NOTE: this is WP of the team receiving the punt
    wp_data = (
        data.assign(
            diff_time_ratio=lambda x: (
                (-x['score_diff']) * np.exp(4 * (3600 - np.maximum(x['game_seconds_remaining'] - 5, 0)) / 3600)
            ),
            score_diff=lambda x: -x['score_diff'],
            spread_time_ratio=lambda x: (
                (-x['pregame_spread']) * np.exp(-4 * (3600 - np.maximum(x['game_seconds_remaining'] - 5, 0)) / 3600)
            ),
            pregame_offense_elo_new=lambda x: x.pregame_defense_elo,
            pregame_defense_elo_new=lambda x: x.pregame_offense_elo,
            pct_game_played=lambda x: np.minimum(x['pct_game_played'] + five_seconds_pct, 1.0),
            seconds_left_in_half=lambda x: np.maximum(x['seconds_left_in_half'] - 5, 0),
            game_seconds_remaining=lambda x: np.maximum(x['game_seconds_remaining'] - 5, 0),
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

    wp_data = add_kneel_features(wp_data)

    wp_data['yards_to_goal'] = data['receiving_team_yards_to_goal']
    wp_data = wp_data[wp_features]

    # the "1 -" here is to flip the WP back to the team that is punting
    probas = (1 - win_probability.predict_win_probability(wp_data))

    # Set WP to 1 or 0 if the game is over after the punt
    probas[(wp_data['pct_game_played'] == 1.0) & ((-1 * wp_data['score_diff']) > 0)] = 1.0
    probas[(wp_data['pct_game_played'] == 1.0) & ((-1 * wp_data['score_diff']) < 0)] = 0.0

    data['exp_wp_punt'] = np.round(probas, 4)

    return data

def predict_receiving_team_yards_to_goal(
    df: pd.DataFrame
) -> pd.Series:
    """ 
    Given a dataframe of features, return receiving team yards to goal predictions.

    Args:
        df (pd.DataFrame): Input dataframe containing feature columns.

    Returns:
        pd.Series: Receiving team yards to goal predictions (values between 0 and 100).
    """

    feature_names = [  
        'punt_team_end_yards_to_goal', 
        'elevation',  
        'temperature', 
        'punting_team_pregame_elo', 
        'receiving_team_pregame_elo',
    ]
    data = (
        df[feature_names].copy()
        .assign(const=1)
        .set_index('const', append=True)
        .reset_index()
        .drop(columns=['level_0'])
    )
    lr_model = sm.load(LR_MODEL_PATH)
    lr_preds = lr_model.predict(data)


    feature_names = [  
        'punt_team_end_yards_to_goal',
        'elevation', 
        'wind_speed', 
        'precipitation', 
        'temperature', 
        'punting_team_pregame_elo', 
        'receiving_team_pregame_elo',
    ]
    dmatrix = xgb.DMatrix(df[feature_names])

    xgb_model = _load_model(XGB_MODEL_PATH)
    xgb_preds = xgb_model.predict(dmatrix)

    preds = 0.5 * (lr_preds + xgb_preds)
    
    return pd.Series(preds, index=df.index, name="receiving_team_yards_to_goal")

def _load_model(model_path: str) -> xgb.Booster:
    model = xgb.Booster()
    model.load_model(model_path)
    return model