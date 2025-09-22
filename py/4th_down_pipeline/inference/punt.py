import logging
import xgboost as xgb
import pandas as pd
import numpy as np
from inference import win_probability

MODEL_PATH = 'models/punt_yards_to_goal/xgb_classifier.json'

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
    level=logging.INFO
)
LOG = logging.getLogger(__name__)

def compute_punt_eWP(data: pd.DataFrame) -> pd.DataFrame:
    """ Compute punt expected win probabilities """
    LOG.info('Predicting punt yards to goal.')
    data['receiving_team_yards_to_goal'] = predict_receiving_team_yards_to_goal(
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
                (-x['score_diff']) * np.exp(4 * (3600 - np.maximum(x['game_seconds_remaining'] - 5, 0)) / 3600)
            ),
            spread_time_ratio=lambda x: (
                (-x['pregame_spread']) * np.exp(-4 * (3600 - np.maximum(x['game_seconds_remaining'] - 5, 0)) / 3600)
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
        'wind_speed', 
        'precipitation', 
        'temperature', 
        'punting_team_pregame_elo', 
        'receiving_team_pregame_elo',
    ]
    dmatrix = xgb.DMatrix(df[feature_names])

    model = _load_model(MODEL_PATH)
    preds = model.predict(dmatrix)
    
    return pd.Series(preds, index=df.index, name="receiving_team_yards_to_goal")



def _load_model(model_path: str) -> xgb.Booster:
    model = xgb.Booster()
    model.load_model(model_path)
    return model