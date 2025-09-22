import logging
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import norm
from inference import win_probability

SELECTION_MODEL_PATH = 'models/fg_probability/selection_model.pkl'
OUTCOME_MODEL_PATH = 'models/fg_probability/outcome_model.pkl'

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
    level=logging.INFO
)
LOG = logging.getLogger(__name__)

def compute_field_goal_eWP(data: pd.DataFrame) -> pd.DataFrame:
    """ Compute field goal probability and expected win probabilities """
    LOG.info('Predicing field goal make probability.')
    data['fg_make_proba'] = predict_field_goal_make_probability(data)
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
                (-x['score_diff'] - 3) * np.exp(4 * (3600 - np.maximum(x['game_seconds_remaining'] - 5, 0)) / 3600)
            ),
            spread_time_ratio=lambda x: (
                (-x['pregame_spread']) * np.exp(-4 * (3600 - np.maximum(x['game_seconds_remaining'] - 5, 0)) / 3600)
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
                (-x['score_diff']) * np.exp(4 * (3600 - np.maximum(x['game_seconds_remaining'] - 5, 0)) / 3600)
            ),
            spread_time_ratio=lambda x: (
                (-x['pregame_spread']) * np.exp(-4 * (3600 - np.maximum(x['game_seconds_remaining'] - 5, 0)) / 3600)
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

    data['exp_wp_fg'] = (
        (data['fg_make_proba'] * data['wp_make_proba']) + 
        ((1 - data['fg_make_proba']) * data['wp_miss_proba'])
    ).round(4)
    
    return data

def predict_field_goal_make_probability(
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
    W_gamma = df.probit_score
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