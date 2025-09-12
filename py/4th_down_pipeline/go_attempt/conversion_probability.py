import xgboost as xgb
import pandas as pd

MODEL_PATH = '../../../models/fourth_down/xgb_classifier.json'

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