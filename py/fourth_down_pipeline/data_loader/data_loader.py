import re, os, json, logging
from os.path import join
from typing import Optional
import pandas as pd
import cfbd
from feature_engineering.elo_updater import update_elo
from feature_engineering.team_strength import calculate_team_strengths

CONFIG_PATH = 'config.json'
DATA_PATH = 'data'

pd.set_option('future.no_silent_downcasting', True)

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
    level=logging.INFO
)
LOG = logging.getLogger(__name__)

with open(CONFIG_PATH) as f:
    configuration = cfbd.Configuration(
        access_token = json.load(f)['CFBD_API_KEY']
    )

def load_games(
    year: int, 
    week: Optional[int] = None,
    season_type: Optional[str] = None,
    force_data_update: Optional[bool] = False
) -> pd.DataFrame:
    """Load games data for a specific year, week, and season type from CFBD API.
    Must pass a value for year, but if week and season_type are not provided,
    all games for the year will be returned.
    
    Args:
        year (int): Year of the season (e.g., 2023)
        week (int, optional): Week number of the season (e.g., 1-15 for regular season)
        season_type (str, optional): Type of the season ('regular' or 'postseason')
        force_data_update (bool, optional): If True, forces data to be fetched 
            from API even if cached data exists.
    """
    if season_type is None and week is not None:
        raise ValueError('If week is provided, season_type must also be provided')
    if season_type is not None and week is None:
        raise ValueError('If season_type is provided, week must also be provided')

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

    if os.path.exists(file_path) and not force_data_update:
        LOG.info(f'Reading {year} games data from cached data')
        games = pd.read_parquet(file_path)
        if season_type is None and week is None:
            return games.reset_index(drop=True)
        if not games.query('season_type == @season_type and week == @week').empty:
            return games[
                (games['week'] == week) &
                (games['season_type'] == season_type)
            ].reset_index(drop=True)
        else:
            LOG.info(f'Missing week {week}, fetching from CFBD API')
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

    if season_type is None and week is None:
        return games.reset_index(drop=True)

    return games[
        (games['week'] == week) &
        (games['season_type'] == season_type)
    ].reset_index(drop=True)

def load_plays(
    year: int, 
    week: Optional[int] = None,
    season_type: Optional[str] = None,
    force_data_update: bool = False
) -> pd.DataFrame:
    """Load plays data for a specific year, week, and season type from CFBD API.
    Must pass a value for year, but if week and season_type are not provided,
    all plays for the year will be returned.
    
    Args:
        year (int): Year of the season (e.g., 2023)
        week (int, optional): Week number of the season (e.g., 1-15 for regular season)
        season_type (str, optional): Type of the season ('regular' or 'postseason')
        force_data_update (bool): If True, forces data to be fetched from API even 
            if cached data exists.
    """
    if season_type is None and week is not None:
        raise ValueError('If week is provided, season_type must also be provided')
    if season_type is not None and week is None:
        raise ValueError('If season_type is provided, week must also be provided')

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

    # Check if cached data exists
    if os.path.exists(file_path) and not force_data_update:
        LOG.info(f'Reading {year} plays from cached data')
        plays_cached = pd.read_parquet(file_path)
        
        # Return all plays for the year if no week/season_type specified
        if season_type is None and week is None:
            return plays_cached.reset_index(drop=True)
        
        # Return specific week if it exists in cache
        if not plays_cached.query('season_type == @season_type and week == @week').empty:
            return plays_cached[
                (plays_cached['week'] == week) &
                (plays_cached['season_type'] == season_type)
            ].reset_index(drop=True)
        else:
            LOG.info(f'Missing {year} week {week}, fetching from CFBD API')
    else:
        LOG.info(f'Fetching {year} plays data from CFBD API')
        plays_cached = pd.DataFrame()

    # If we need to fetch data for a specific week
    if week is not None and season_type is not None:
        # Fetch from CFBD API for specific week
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
        
        return plays_all[
            (plays_all['week'] == week) &
            (plays_all['season_type'] == season_type)
        ].reset_index(drop=True)
    
    # If we need to fetch all plays for the year
    else:
        # Fetch all weeks and season types for the year
        all_plays = []
        
        # Fetch regular season weeks
        LOG.info(f'Fetching {year} regular season plays from CFBD API')
        with cfbd.ApiClient(configuration) as api_client:
            api_instance = cfbd.PlaysApi(api_client)
            
            # Get regular season weeks (typically 1-15)
            for week_num in range(1, 16):  # Adjust range as needed
                try:
                    data = api_instance.get_plays(
                        year=year, 
                        week=week_num, 
                        season_type=cfbd.SeasonType('regular')
                    )
                    week_plays = pd.DataFrame([val.to_dict() for val in data])
                    week_plays.columns = _convert_to_snake_case(week_plays.columns)
                    week_plays = (
                        week_plays.assign(
                            season=year,
                            week=week_num,
                            season_type='regular',
                            clock_minutes=lambda x: x['clock'].apply(lambda y: y['minutes']),
                            clock_seconds=lambda x: x['clock'].apply(lambda y: y['seconds'])
                        )
                        .drop(columns=['clock'])
                    )
                    all_plays.append(week_plays[cols])
                    LOG.info(f'Fetched week {week_num} regular season plays')
                except Exception as e:
                    LOG.warning(f'Failed to fetch week {week_num} regular season: {e}')
            
            # Get postseason weeks
            LOG.info(f'Fetching {year} postseason plays from CFBD API')
            try:
                for week in range(1, 5): 
                    data = api_instance.get_plays(
                        year=year, 
                        week=week,
                        season_type=cfbd.SeasonType('postseason')
                    )
                    postseason_plays = pd.DataFrame([val.to_dict() for val in data])
                    postseason_plays.columns = _convert_to_snake_case(postseason_plays.columns)
                    postseason_plays = (
                        postseason_plays.assign(
                            season=year,
                            week=week,
                            season_type='postseason',
                            clock_minutes=lambda x: x['clock'].apply(lambda y: y['minutes']),
                            clock_seconds=lambda x: x['clock'].apply(lambda y: y['seconds'])
                        )
                        .drop(columns=['clock'])
                    )
                    all_plays.append(postseason_plays[cols])
                LOG.info('Fetched postseason plays')
            except Exception as e:
                LOG.warning(f'Failed to fetch postseason plays: {e}')
        
        # Combine all plays
        if all_plays:
            plays_all = pd.concat(all_plays, ignore_index=True)
            plays_all.drop_duplicates(inplace=True)
            plays_all.to_parquet(file_path)
            return plays_all.reset_index(drop=True)
        else:
            return pd.DataFrame(columns=cols)

def load_teams(
    year: int,
    force_data_update: bool = False
) -> pd.DataFrame:
    """Load teams data from CFBD API.

    Args:
        year (int): Year of the season (e.g., 2023)
        force_data_update (bool): If True, forces data to be fetched from API even
            if cached data exists.
    Returns:
        pd.DataFrame: DataFrame containing team information
    """
    teams_dir = join(DATA_PATH, 'teams')
    os.makedirs(teams_dir, exist_ok=True)
    file_path = join(teams_dir, f'{year}.parquet')

    if os.path.exists(file_path) and not force_data_update:
        LOG.info(f'Reading teams data from cached data')
        teams = pd.read_parquet(file_path)
        if not teams.query('season == @year').empty:
            return teams[teams['season'] == year].reset_index(drop=True)
        LOG.info(f'Missing {year} teams')
    LOG.info(f'Fetching {year} teams data from CFBD API')

    with cfbd.ApiClient(configuration) as api_client:
        api_instance = cfbd.TeamsApi(api_client)
        data = api_instance.get_teams(year=year)
    teams = pd.DataFrame([val.to_dict() for val in data])
    teams.columns = _convert_to_snake_case(teams.columns)
    teams.insert(0, 'season', year)
    teams.to_parquet(file_path)
    return teams

def load_weather(
    year: int, 
    week: int, 
    season_type: str, 
    force_data_update: bool = False
) -> pd.DataFrame:
    """Load weather data for a specific year, week, and season type from CFBD API.

    Args:
        year (int): Year of the season (e.g., 2023)
        week (int): Week number of the season (e.g., 1-15 for regular season)
        season_type (str): Type of the season ('regular' or 'postseason')
        force_data_update (bool): If True, forces data to be fetched from API even 
            if cached data exists.
    """
    weather_dir = join(DATA_PATH, 'weather')
    os.makedirs(weather_dir, exist_ok=True)
    file_path = join(weather_dir, f'{year}.parquet')

    if os.path.exists(file_path) and not force_data_update:
        LOG.info(f'Reading {year} weather data from cached data')
        weather = pd.read_parquet(file_path)
        if not weather.query('season_type == @season_type and week == @week').empty:
            return weather[
                (weather['week'] == week) &
                (weather['season_type'] == season_type)
            ].reset_index(drop=True)
        LOG.info(f'Missing week {week}, fetching from CFBD API')
    LOG.info(f'Fetching {year} weather data from CFBD API')

    # Fetch from CFBD API
    with cfbd.ApiClient(configuration) as api_client:
        api_instance = cfbd.GamesApi(api_client)
        data = api_instance.get_weather(year=year)
    weather = pd.DataFrame([val.to_dict() for val in data])
    weather.columns = _convert_to_snake_case(weather.columns)
    weather['season_type'] = weather['season_type'].apply(lambda x: x.value)

    weather.to_parquet(file_path)

    return weather[
        (weather['week'] == week) &
        (weather['season_type'] == season_type)
    ].reset_index(drop=True)

def load_venues(force_data_update: bool = False) -> pd.DataFrame:
    """Load venue data from CFBD API.

    Args:
        force_data_update (bool): If True, forces data to be fetched from API even
            if cached data exists.
    Returns:
        pd.DataFrame: DataFrame containing venue information
    """
    venue_dir = join(DATA_PATH, 'venues')
    os.makedirs(venue_dir, exist_ok=True)
    file_path = join(venue_dir, f'venues.parquet')

    if os.path.exists(file_path) and not force_data_update:
        LOG.info(f'Reading venues data from cached data')
        venues = pd.read_parquet(file_path)
        return venues
    LOG.info(f'Fetching venue data from CFBD API')

    with cfbd.ApiClient(configuration) as api_client:
        api_instance = cfbd.VenuesApi(api_client)
        data = api_instance.get_venues()
    venues = pd.DataFrame([val.to_dict() for val in data])
    venues.columns = _convert_to_snake_case(venues.columns)
    venues.to_parquet(file_path)
    return venues

def load_lines(
    year: int, 
    week: int, 
    season_type: str, 
    force_data_update: bool = False
) -> pd.DataFrame:
    """Get betting lines for a specific year, week, and season type.

    Args:
        year (int): Year of the season (e.g., 2023)
        week (int): Week number of the season (e.g., 1-15 for regular season)
        season_type (str): Type of the season ('regular' or 'postseason')
        force_data_update (bool): If True, forces data to be fetched from API even 
            if cached data exists.
    Returns:
        pd.DataFrame: DataFrame containing betting lines for each game
    """
    lines_dir = join(DATA_PATH, 'lines')
    os.makedirs(lines_dir, exist_ok=True)
    file_path = join(lines_dir, f'{year}.parquet')

    if os.path.exists(file_path) and not force_data_update:
        LOG.info(f'Reading {year} lines data from cached data')
        lines = pd.read_parquet(file_path)
        if not lines.query('season_type == @season_type and week == @week').empty:
            season_week = lines.query('season_type == @season_type and week == @week')
            if season_week.home_spread.isnull().sum() == 0: # if no missing spreads, return
                return season_week.reset_index(drop=True)
        LOG.info(f'Missing {year} week {week} lines')
    LOG.info(f'Fetching {year} lines data from CFBD API')

    # Fetch from CFBD API
    with cfbd.ApiClient(configuration) as api_client:
        api_instance = cfbd.BettingApi(api_client)
        data = api_instance.get_lines(year=year)
    lines = pd.DataFrame([val.to_dict() for val in data])
    lines.columns = _convert_to_snake_case(lines.columns)
    lines['season_type'] = lines['season_type'].apply(lambda x: x.value)
    lines = lines.assign(
        home_division=lines['home_classification'].apply(
            lambda x: x.value if x is not None else None
        ),
        away_division=lines['away_classification'].apply(
            lambda x: x.value if x is not None else None
        )
    ).drop(columns=['home_classification', 'away_classification'])

    lines_exploded = lines.explode('lines')

    lines_exploded['provider'] = lines_exploded['lines'].apply(
        lambda x: x['provider'] if pd.notnull(x) else None
    )
    lines_exploded['spread'] = lines_exploded['lines'].apply(
        lambda x: x['spread'] if pd.notnull(x) and 'spread' in x else None
    )

    unique_providers = lines_exploded['provider'].dropna().unique()

    spreads = lines_exploded.pivot_table(
        index=[col for col in lines_exploded.columns if col not in ['lines', 'provider', 'spread']],
        columns='provider',
        values='spread',
        aggfunc='first'
    ).reset_index()

    spreads.columns = [f'{col}_spread' if col in unique_providers else col for col in spreads.columns]
    lines = lines.drop(columns=['lines']).drop_duplicates().merge(
        spreads,
        on=[col for col in lines.columns if col != 'lines'],
        how='left'
    )

    potenial_spread_cols = ['consensus_spread', 'teamrankings_spread', 
        'numberfire_spread', 'Bovada_spread', 'ESPN Bet_spread', 'DraftKings_spread', 
        'Caesars_spread', 'SugarHouse_spread', 'William Hill (New Jersey)_spread',
        'Caesars Sportsbook (Colorado)_spread', 'Caesars (Pennsylvania)_spread'
    ]

    lines['home_spread'] = None

    for col in potenial_spread_cols:
        if col in lines.columns:
            lines['home_spread'] = lines['home_spread'].fillna(lines[col])

    cols = ['id', 'season', 'season_type', 'week', 'start_date', 'home_team',
       'home_conference', 'away_team', 'away_conference', 'home_score',
       'away_score', 'home_division', 'away_division', 'home_spread']
    lines = lines[cols]

    lines.to_parquet(file_path)

    return lines[
        (lines['week'] == week) &
        (lines['season_type'] == season_type)
    ].reset_index(drop=True)

def load_ppa(
    year: int, 
    week: Optional[int] = None,
    season_type: Optional[str] = None,
    current_week: Optional[int] = None,
    current_season_type: Optional[str] = None,
    force_data_update: bool = False
) -> pd.DataFrame:
    """Load PPA data for a specific year, week, and season type from CFBD API.

    Args:
        year (int): Year of the season (e.g., 2023)
        week Optional[int]: Week number of the season (e.g., 1-15 for regular season)
        season_type Optional[str]: Type of the season ('regular' or 'postseason')
        current_week (int, optional): Current week of the season
        force_data_update (bool): If True, forces data to be fetched from API even 
            if cached data exists.
    Returns:
        pd.DataFrame: DataFrame containing PPA data for each play
    """
    ppa_dir = join(DATA_PATH, 'ppa')
    os.makedirs(ppa_dir, exist_ok=True)
    file_path = join(ppa_dir, f'{year}.parquet')

    if (season_type is None and week is not None) or (season_type is not None and week is None):
        raise ValueError('If week is provided, season_type must also be provided, and vice versa')
    if season_type is None and week is None and year is not None:
        pull_all = True
    else:
        pull_all = False

    if os.path.exists(file_path) and not force_data_update:
        LOG.info(f'Reading {year} PPA data from cached data')
        ppa = pd.read_parquet(file_path)
        if pull_all:
            if current_week is not None and current_season_type is not None: 
                if not ppa.query('season_type == @current_season_type and week == @current_week').empty:
                    if current_season_type == 'regular':
                        return (
                            ppa
                            .query('season_type == "regular" and week <= @current_week')
                            .reset_index(drop=True)
                        )
                    else:
                        return pd.concat([
                            ppa[ppa['season_type'] == 'regular'],
                            ppa[
                                (ppa['season_type'] == current_season_type) & 
                                (ppa['week'] <= current_week)
                            ]
                        ], ignore_index=True).reset_index(drop=True)
                LOG.info(f'Missing up to current week {current_week}, fetching from CFBD API')
            else:
                return ppa.reset_index(drop=True)
        if not ppa.query('season_type == @season_type and week == @week').empty:
            return ppa[
                (ppa['week'] == week) &
                (ppa['season_type'] == season_type)
            ].reset_index(drop=True)
        LOG.info(f'Missing week {week}, fetching from CFBD API')
    LOG.info(f'Fetching {year} PPA data from CFBD API')

    # Fetch from CFBD API
    with cfbd.ApiClient(configuration) as api_client:
        api_instance = cfbd.MetricsApi(api_client)
        data = api_instance.get_predicted_points_added_by_game(
            year=year,
            exclude_garbage_time=True
        )
    ppa = pd.DataFrame([val.to_dict() for val in data])

    ppa.columns = _convert_to_snake_case(ppa.columns)
    ppa['season_type'] = ppa['season_type'].apply(lambda x: x.value)
    ppa = (
        ppa.assign(
            offense_ppa = ppa['offense'].apply(lambda x: x['overall']),
            defense_ppa = ppa['defense'].apply(lambda x: x['overall'])
        )
        .drop(columns=['offense', 'defense'])
    )

    ppa.to_parquet(file_path)

    if pull_all:
        if current_week is not None and current_season_type is not None:
            if current_season_type == 'regular':
                return (
                    ppa
                    .query('season_type == "regular" and week <= @current_week')
                    .reset_index(drop=True)
                )
            else:
                return pd.concat([
                    ppa[ppa['season_type'] == 'regular'],
                    ppa[
                        (ppa['season_type'] == current_season_type) & 
                        (ppa['week'] <= current_week)
                    ]
                ], ignore_index=True).reset_index(drop=True)
        else:
            return ppa.reset_index(drop=True)
    return ppa[
        (ppa['week'] == week) &
        (ppa['season_type'] == season_type)
    ].reset_index(drop=True)

def load_elo(
    year: int, 
    week: int, 
    season_type: str, 
    force_data_update: bool = False
) -> pd.DataFrame:
    """Load ELO data for a specific year, week, and season type from CFBD API.

    Args:
        year (int): Year of the season (e.g., 2023)
        week (int): Week number of the season (e.g., 1-15 for regular season)
        season_type (str): Type of the season ('regular' or 'postseason')
        force_data_update (bool): If True, forces data to be fetched from API even 
            if cached data exists.
    Returns:
        pd.DataFrame: DataFrame containing ELO ratings for each team
    """
    elo_dir = join(DATA_PATH, 'elo')
    os.makedirs(elo_dir, exist_ok=True)
    file_path = join(elo_dir, f'{year}.parquet')

    if os.path.exists(file_path) and not force_data_update:
        LOG.info(f'Reading {year} ELO data from cached data')
        elo = pd.read_parquet(file_path)
        if not elo.query('season_type == @season_type and week == @week').empty:
            return elo[
                (elo['week'] == week) &
                (elo['season_type'] == season_type)
            ].reset_index(drop=True)
        LOG.info(f'Missing {year} week {week} ELO data')
    LOG.info(f'Generating {year} week {week} ELO data')

    years = range(1930, year + 1)
    games, teams = [], []
    for year in years:
        games.append(load_games(year, force_data_update=force_data_update))
        teams.append(load_teams(year, force_data_update=force_data_update))
    games = pd.concat(games, ignore_index=True)
    teams = pd.concat(teams, ignore_index=True)

    elo = update_elo(games, teams, year, week, season_type)

    return elo[
        (elo['week'] == week) &
        (elo['season_type'] == season_type)
    ].reset_index(drop=True)

def load_team_strengths(
    year: int,
    week: int,
    season_type: str,
    force_data_update: bool = False
) -> pd.DataFrame:
    """Load team strength data for a specific year, week, and season type.

    Args:
        year (int): Year of the season (e.g., 2023)
        week (int): Week number of the season (e.g., 1-15 for regular season)
        season_type (str): Type of the season ('regular' or 'postseason')
        force_data_update (bool): If True, forces data to be fetched from API even 
            if cached data exists.
    Returns:
        pd.DataFrame: DataFrame containing team strength metrics for each team
    """
    team_strengths_dir = join(DATA_PATH, 'team_strengths')
    os.makedirs(team_strengths_dir, exist_ok=True)
    file_path = join(team_strengths_dir, f'{year}_{week}_{season_type}.parquet')

    ppa = pd.concat([
        load_ppa(year - 1),
        load_ppa(year, current_week=week, current_season_type=season_type)
    ])

    if os.path.exists(file_path) and not force_data_update:
        LOG.info(f'Reading {year} team strengths from cached data')
        team_strengths = pd.read_parquet(file_path)
        if not team_strengths.query('season_type == @season_type and week == @week').empty:
            return team_strengths[
                (team_strengths['season'] == year) &
                (team_strengths['week'] == week) &
                (team_strengths['season_type'] == season_type)
            ].reset_index(drop=True)
        LOG.info(f'Missing {year} week {week} team strengths')
    LOG.info(f'Generating {year} week {week} team strengths')

    team_strengths = calculate_team_strengths(ppa, year, week, season_type)

    team_strengths = (
        team_strengths
        .query('season == @year and week == @week and season_type == @season_type')
        .reset_index(drop=True)
    )

    team_strengths.to_parquet(file_path)

    return team_strengths


def _convert_to_snake_case(cols):
    cols_new = []
    for c in cols:
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', c)
        cols_new.append(re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower())
    return cols_new