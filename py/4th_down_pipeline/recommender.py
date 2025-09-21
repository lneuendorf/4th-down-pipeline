from os.path import join
import pickle
import pandas as pd
from sklearn.linear_model import LinearRegression
import data_loader as dl
import feature_engineering as fe

MODEL_DIR = '../../models'


def get_recommendations(
    year: int, 
    week: int, 
    season_type: str, 
    force_data_update: bool = False
) -> pd.DataFrame:
    """ Get 4th down recommendations for all games in a specific year, week, and season type 
    
    Args:
        year (int): Year of the season (e.g., 2023)
        week (int): Week number of the season (e.g., 1-15 for regular season)
        season_type (str): Type of the season ('regular' or 'postseason')
        force_data_update (bool): If True, forces data to be fetched from API even 
            if cached data exists.

    Returns:
        pd.DataFrame: DataFrame containing 4th down recommendations for each game
    """

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

    # TODO: generate predictions and expected win probabilities

    return data

def _impute_missing_team_strengths(df: pd.DataFrame) -> pd.DataFrame:
    """ Impute missing team strenghts using linear regression based on ELO ratings.
    
    Args:
        df (pd.DataFrame): DataFrame containing team strengths with possible 
        missing values along with pregame ELO ratings.

    Returns:
        pd.DataFrame: DataFrame with missing team strengths imputed.
    """
    offense_model = pickle.load(open(join(MODEL_DIR, 'team_strength', 'offense.pkl'), 'rb'))
    defense_model = pickle.load(open(join(MODEL_DIR, 'team_strength', 'defense.pkl'), 'rb'))

    # Offense Strength
    mask = df['offense_strength'].isna()
    if mask.sum() != 0:
        df.loc[mask, 'offense_strength'] = offense_model.predict(
            df.loc[mask, ['pregame_offense_elo']]
        )

    # Defense Strength
    mask = df['defense_strength'].isna()
    if mask.sum() != 0:
        df.loc[mask, 'defense_strength'] = defense_model.predict(
            df.loc[mask, ['pregame_defense_elo']]
        )

    return df