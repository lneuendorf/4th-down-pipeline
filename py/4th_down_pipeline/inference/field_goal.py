import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

SELECTION_MODEL_PATH = 'models/fg_probability/selection_model.pkl'
OUTCOME_MODEL_PATH = 'models/fg_probability/outcome_model.pkl'

def predict_field_goal_probability(
    df: pd.DataFrame
) -> pd.Series:
    """ 
    Given a dataframe of features, return field goal probability predictions.

    Args:
        df (pd.DataFrame): Input dataframe containing feature columns.

    Returns:
        pd.Series: Field goal probability predictions (values between 0 and 1).
    """
    
    selection_features = [
        'yards_to_goal', 
        'score_diff',
        'pct_game_played', 
        'pregame_offense_elo', 
        'pregame_defense_elo', 
        'distance', 
        'pressure_rating', 
        'is_home_team',
        'wind_speed', 
        'temperature',
        'elevation', 
        'grass', 
        'game_indoors'
    ]

    
    selection_df = df[selection_features]
    selection_model = _load_model(SELECTION_MODEL_PATH)
    df['probit_score'] = selection_model.predict(sm.add_constant(selection_df))
    
    # Inverse Mills Ratio λ = φ / Φ
    W_gamma = df.prodit_score
    phi = norm.pdf(W_gamma)
    Phi = norm.cdf(W_gamma)
    df['lambda'] = phi / Phi

    outcome_features = [
        'yards_to_goal', 
        'score_diff', 
        'pregame_offense_elo', 
        'pregame_defense_elo', 
        'pressure_rating',
        'wind_speed', 
        'elevation', 
        'lambda'
    ]
    outcome_df = df[outcome_features]
    outcome_model = _load_model(OUTCOME_MODEL_PATH)
    preds = outcome_model.predict(sm.add_constant(outcome_df))

    return pd.Series(preds, index=df.index, name="field_goal_probability")

def _load_model(model_path: str) -> sm.Probit:
    model = sm.load(model_path)
    return model