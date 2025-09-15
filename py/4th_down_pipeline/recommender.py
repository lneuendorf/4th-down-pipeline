import pandas as pd
import data_loader as dl


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
    games = dl.load_games(year, week, season_type, force_data_update)
    plays = dl.load_plays(year, week, season_type, force_data_update)
    weather = dl.load_weather(year, week, season_type, force_data_update)
    venues = dl.load_venues(force_data_update)
    lines = dl.load_lines(year, week, season_type, force_data_update)
    ppa = dl.load_ppa(year, week, season_type, force_data_update)
    breakpoint()
    return None
    