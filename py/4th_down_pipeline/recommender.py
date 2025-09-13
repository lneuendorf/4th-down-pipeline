import pandas as pd
import data_loader as dl


def get_recommendations(year: int, week: int, season_type: str) -> pd.DataFrame:
    """ Get 4th down recommendations for all games in a specific year, week, and season type 
    
    Args:
        year (int): Year of the season (e.g., 2023)
        week (int): Week number of the season (e.g., 1-15 for regular season)
        season_type (str): Type of the season ('regular' or 'postseason')

    Returns:
        pd.DataFrame: DataFrame containing 4th down recommendations for each game
    """
    games = dl.load_games(year, week, season_type)

    plays = dl.load_plays(year, week, season_type)

    breakpoint()
    return None
    