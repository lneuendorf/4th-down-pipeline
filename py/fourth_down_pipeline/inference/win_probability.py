import xgboost as xgb
import pandas as pd

MODEL_PATH = 'models/win_probability/xgb_classifier.json'

def predict_win_probability(
    df: pd.DataFrame, 
) -> pd.Series:
    """
    Given a dataframe of features, return win probability predictions.

    Args:
        df (pd.DataFrame): Input dataframe containing feature columns.
    
    Returns:
        pd.Series: Win probability predictions (values between 0 and 1).
    """

    feature_names = [
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
    dmatrix = xgb.DMatrix(df[feature_names])

    model = _load_model(MODEL_PATH)
    preds = model.predict(dmatrix)
    
    return pd.Series(preds, index=df.index, name="win_probability")

def _load_model(model_path: str) -> xgb.Booster:
    model = xgb.Booster()
    model.load_model(model_path)
    return model