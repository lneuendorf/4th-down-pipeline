import re
import os
import json
import logging
from os.path import join
import pandas as pd
import cfbd

CONFIG_PATH = '../../config.json'
DATA_PATH = '../../data'

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
    level=logging.INFO
)
LOG = logging.getLogger(__name__)

with open(CONFIG_PATH) as f:
    configuration = cfbd.Configuration(
        access_token = json.load(f)['CFBD_API_KEY']
    )

def load_games(year: int, week: int, season_type: str) -> pd.DataFrame:
    """Load games data for a specific year, week, and season type from CFBD API."""
    id_cols = [
        'id', 'season', 'week', 'season_type', 'completed', 'neutral_site',
        'venue_id', 'start_date'
    ]
    home_cols = [
        'home_id', 'home_team', 'home_conference', 'home_points', 
        'home_pregame_elo'
    ]
    away_cols = [
        'away_id', 'away_team', 'away_conference', 'away_points', 
        'away_pregame_elo'
    ]
    cols = id_cols + home_cols + away_cols

    games_dir = join(DATA_PATH, 'games')
    os.makedirs(games_dir, exist_ok=True)
    file_path = join(games_dir, f'{year}.parquet')

    if os.path.exists(file_path):
        LOG.info(f'Reading {year} games data from cached data')
        games = pd.read_parquet(file_path)
        if not games.query('season_type == @season_type and week == @week').empty:
            return games[
                (games['week'] == week) &
                (games['season_type'] == season_type)
            ].reset_index(drop=True)
        LOG.info(f'Missing week {week}, fetching from CFBD API')
    else:
        LOG.info(f'Fetching {year} games data from CFBD API')

    # Fetch from CFBD API
    with cfbd.ApiClient(configuration) as api_client:
        api_instance = cfbd.GamesApi(api_client)
        data = api_instance.get_games(year=year)
    games = pd.DataFrame([val.to_dict() for val in data])
    games.columns = _convert_to_snake_case(games.columns)

    # Keep only required columns
    games = games[cols + ['home_classification', 'away_classification']]
    games['season_type'] = games['season_type'].apply(lambda x: x.value)

    games = games.assign(
        home_division=games['home_classification'].apply(
            lambda x: x.value if x is not None else None
        ),
        away_division=games['away_classification'].apply(
            lambda x: x.value if x is not None else None
        )
    ).drop(columns=['home_classification', 'away_classification'])
    games = games[cols + ['home_division', 'away_division']]

    games.to_parquet(file_path)

    return games[
        (games['week'] == week) &
        (games['season_type'] == season_type)
    ].reset_index(drop=True)

def load_plays(year: int, week: int, season_type: str) -> pd.DataFrame:
    # Load plays data
    cols = [
        'season', 'week', 'season_type',
        'id', 'drive_id', 'game_id', 'drive_number', 'play_number', 'offense',
        'offense_conference', 'offense_score', 'defense', 'home', 'away',
        'defense_conference', 'defense_score', 'period', 'offense_timeouts',
        'defense_timeouts', 'yardline', 'yards_to_goal', 'down', 'distance',
        'yards_gained', 'scoring', 'play_type', 'play_text', 'ppa',
        'clock_minutes', 'clock_seconds'
    ]
    plays_dir = join(DATA_PATH, 'plays')
    os.makedirs(plays_dir, exist_ok=True)
    file_path = join(plays_dir, f'{year}.parquet')

    plays_cached = pd.DataFrame()
    if os.path.exists(file_path):
        LOG.info(f'Reading {year} plays from cached data')
        plays_cached = pd.read_parquet(file_path)

        if not plays_cached.query('season_type == @season_type and week == @week').empty:
            return plays_cached[
                (plays_cached['week'] == week) &
                (plays_cached['season_type'] == season_type)
            ].reset_index(drop=True)
        LOG.info(f'Missing {year} week {week}, fetching from CFBD API')
    else:
        LOG.info(f'Fetching {year} week {week} plays from CFBD API')

    # Fetch from CFBD API
    with cfbd.ApiClient(configuration) as api_client:
        api_instance = cfbd.PlaysApi(api_client)
        data = api_instance.get_plays(
            year=year, 
            week=week, 
            season_type=cfbd.SeasonType(season_type)
        )
    plays = pd.DataFrame([val.to_dict() for val in data])
    plays.columns = _convert_to_snake_case(plays.columns)
    plays = (
        plays.assign(
            season=year,
            week=week,
            season_type=season_type,
            clock_minutes=lambda x: x['clock'].apply(lambda y: y['minutes']),
            clock_seconds=lambda x: x['clock'].apply(lambda y: y['seconds'])
        )
        .drop(columns=['clock'])
    )

    # Combine with cached data and save
    plays_all = pd.concat([plays_cached, plays[cols]], ignore_index=True)
    plays_all.drop_duplicates(inplace=True)
    plays_all.to_parquet(file_path)

    return plays[
        (plays['week'] == week) &
        (plays['season_type'] == season_type)
    ].reset_index(drop=True)
    
# TODO: UPDATE THE PLAYS FUNCTION TO CALL BY YEAR -> NEED TO MINIMIZE API CALLS

def _convert_to_snake_case(cols):
    cols_new = []
    for c in cols:
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', c)
        cols_new.append(re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower())
    return cols_new