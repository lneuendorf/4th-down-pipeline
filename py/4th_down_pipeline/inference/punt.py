import xgboost as xgb
import pandas as pd

MODEL_PATH = 'models/punt_yards_to_goal/xgb_classifier.json'

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