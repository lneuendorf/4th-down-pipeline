import json
import re
import os
from os.path import join
from datetime import datetime

import pandas as pd
import numpy as np
from tqdm import tqdm

import cfbd
from cfbd.rest import ApiException

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

with open("../config.json") as f:
    configuration = cfbd.Configuration(
        access_token = json.load(f)["CFBD_API_KEY"]
    )

# Current year for weekly updates
CURRENT_YEAR = 2025
DATA_DIR = "../data"

def convert_to_snake_case(cols):
    cols_new = []
    for c in cols:
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', c)
        cols_new.append(re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower())
    return cols_new

def get_current_week():
    """Get the current week of the CFB season"""
    with cfbd.ApiClient(configuration) as api_client:
        api_instance = cfbd.GamesApi(api_client)
        try:
            # Get current week information
            current_week = api_instance.get_calendar(year=CURRENT_YEAR)
            # Find the current week based on today's date
            today = datetime.now().date()
            for week in current_week:
                if week.first_game_start <= today.isoformat() and week.last_game_start >= today.isoformat():
                    return week.week
            # If no current week found, return the latest completed week
            return max([w.week for w in current_week if w.last_game_start <= today.isoformat()])
        except ApiException as e:
            print(f"Exception when calling GamesApi->get_calendar: {e}")
            return None

def load_historical_elo_data():
    """Load historical ELO data from parquet files"""
    elo_dir = join(DATA_DIR, "elo")
    all_elo_dfs = []
    
    if not os.path.exists(elo_dir):
        return pd.DataFrame()
    
    # Get all available historical ELO files
    elo_files = [f for f in os.listdir(elo_dir) if f.endswith('.parquet')]
    
    for file in elo_files:
        year = int(file.split('.')[0])
        if year < CURRENT_YEAR:  # Only load previous years
            df = pd.read_parquet(join(elo_dir, file))
            all_elo_dfs.append(df)
    
    if all_elo_dfs:
        return pd.concat(all_elo_dfs, ignore_index=True)
    return pd.DataFrame()

def get_team_division_mapping():
    """Get a mapping of team IDs to their divisions"""
    teams_dir = join(DATA_DIR, "teams")
    team_division_map = {}
    
    # Try to get the most recent team data
    for year in range(CURRENT_YEAR-1, CURRENT_YEAR+1):
        file_path = join(teams_dir, f"{year}.parquet")
        if os.path.exists(file_path):
            df_teams = pd.read_parquet(file_path)
            for _, row in df_teams.iterrows():
                team_division_map[row['id']] = row.get('classification', 'fbs')
    
    return team_division_map

def update_games_data():
    """Update games data for the current year and week"""
    games_dir = join(DATA_DIR, "games")
    if not os.path.exists(games_dir):
        os.makedirs(games_dir)
    
    current_week = get_current_week()
    if current_week is None:
        print("Could not determine current week")
        return None
    
    # Load existing games data for current year if it exists
    current_year_file = join(games_dir, f"{CURRENT_YEAR}.parquet")
    if os.path.exists(current_year_file):
        df_current = pd.read_parquet(current_year_file)
        max_week = df_current['week'].max() if not df_current.empty else 0
    else:
        df_current = pd.DataFrame()
        max_week = 0
    
    # If we already have data for the current week, no need to update
    if max_week >= current_week:
        print(f"Games data for {CURRENT_YEAR} week {current_week} already exists")
        return df_current
    
    # Fetch new games data for the current week
    print(f"Fetching {CURRENT_YEAR} games data for week {current_week} from CFBD API")
    with cfbd.ApiClient(configuration) as api_client:
        api_instance = cfbd.GamesApi(api_client)
        try:
            # Get games for the current week
            games = api_instance.get_games(year=CURRENT_YEAR, week=current_week)
            df_new = pd.DataFrame([val.to_dict() for val in games])
            
            if df_new.empty:
                print(f"No games found for {CURRENT_YEAR} week {current_week}")
                return df_current
            
            df_new.columns = convert_to_snake_case(df_new.columns)
            
            # Select and rename columns
            id_cols = ['id', 'season', 'week', 'season_type', 'completed', 'neutral_site', 'venue_id']
            home_cols = ['home_id', 'home_team', 'home_conference', 'home_points']
            away_cols = ['away_id', 'away_team', 'away_conference', 'away_points']
            
            df_new = df_new[id_cols + home_cols + away_cols]
            df_new['season_type'] = df_new['season_type'].apply(lambda x: x.value if hasattr(x, 'value') else x)
            
            # Merge with existing data or create new
            if df_current.empty:
                df_updated = df_new
            else:
                df_updated = pd.concat([df_current, df_new], ignore_index=True)
            
            # Save updated data
            df_updated.to_parquet(current_year_file)
            print(f"Updated games data for {CURRENT_YEAR} week {current_week}")
            
            return df_updated
            
        except ApiException as e:
            print(f"Exception when calling GamesApi->get_games: {e}")
            return df_current

def calculate_weekly_elo():
    """Calculate ELO ratings for the current week"""
    # Load historical ELO data
    df_historical_elo = load_historical_elo_data()
    
    # Get the most recent ELO ratings for each team
    if not df_historical_elo.empty:
        latest_elo = df_historical_elo.sort_values(['season', 'week']).groupby('team_id').last().reset_index()
        elo_cache = dict(zip(latest_elo['team_id'], latest_elo['elo']))
    else:
        elo_cache = {}
    
    # Get team division mapping for initial ELO values
    team_division_map = get_team_division_mapping()
    
    # ELO parameters (these should be tuned periodically)
    HFA = 100  # Home Field Advantage
    K = 100    # K-factor
    DIVISOR = 800  # Probability scaling
    
    # Initial ELO values by division
    elo_initial = {
        'fbs': 1500.0,
        'fcs': 1300.0,
        'ii': 1000.0,
        'iii': 800.0
    }
    
    # Update games data
    df_games = update_games_data()
    if df_games is None or df_games.empty:
        print("No games data available")
        return
    
    # Filter for current year and week
    current_week = get_current_week()
    df_current = df_games[(df_games['season'] == CURRENT_YEAR) & (df_games['week'] == current_week)].copy()
    
    if df_current.empty:
        print(f"No games found for {CURRENT_YEAR} week {current_week}")
        return
    
    # Add result columns
    df_current.loc[:, 'home_result'] = np.select(
        [df_current['home_points'] > df_current['away_points'], 
         df_current['home_points'] < df_current['away_points']],
        [1, 0],
        default=0.5
    )
    df_current.loc[:, 'away_result'] = 1 - df_current.loc[:, 'home_result']
    
    # Add division information
    df_current.loc[:, 'home_classification'] = df_current['home_id'].map(team_division_map).fillna('fbs')
    df_current.loc[:, 'away_classification'] = df_current['away_id'].map(team_division_map).fillna('fbs')
    
    # Set initial ELO values
    df_current.loc[:, 'home_elo'] = df_current['home_id'].map(elo_cache).fillna(
        df_current['home_classification'].map(elo_initial))
    df_current.loc[:, 'away_elo'] = df_current['away_id'].map(elo_cache).fillna(
        df_current['away_classification'].map(elo_initial))
    
    # Calculate new ELO ratings
    df_current.loc[:, 'home_elo_post'] = (
        df_current['home_elo'] +
        K * np.log(np.abs(df_current['home_points'] - df_current['away_points']) + 1) / 2
        * (df_current['home_result'] - 1 / (1 + 10 ** ((df_current['away_elo'] - HFA - df_current['home_elo']) / DIVISOR)))
    )
    df_current.loc[:, 'away_elo_post'] = (
        df_current['away_elo'] +
        K * np.log(np.abs(df_current['away_points'] - df_current['home_points']) + 1) / 2
        * (df_current['away_result'] - 1 / (1 + 10 ** ((df_current['home_elo'] + HFA - df_current['away_elo']) / DIVISOR)))
    )
    
    # Prepare ELO data for saving
    elo_data = pd.concat([
        df_current[['season', 'week', 'season_type', 'home_id', 'home_team', 'home_classification', 'home_elo_post']]
        .rename(columns={'home_id': 'team_id', 'home_team': 'team', 'home_classification': 'division', 'home_elo_post': 'elo'}),
        df_current[['season', 'week', 'season_type', 'away_id', 'away_team', 'away_classification', 'away_elo_post']]
        .rename(columns={'away_id': 'team_id', 'away_team': 'team', 'away_classification': 'division', 'away_elo_post': 'elo'})
    ])
    
    # Append to historical ELO data
    elo_dir = join(DATA_DIR, "elo")
    if not os.path.exists(elo_dir):
        os.makedirs(elo_dir)
    
    # Load existing ELO data for current year if it exists
    current_elo_file = join(elo_dir, f"{CURRENT_YEAR}.parquet")
    if os.path.exists(current_elo_file):
        df_existing_elo = pd.read_parquet(current_elo_file)
        # Remove any existing data for this week (in case we're re-running)
        df_existing_elo = df_existing_elo[df_existing_elo['week'] != current_week]
        # Combine with new data
        df_updated_elo = pd.concat([df_existing_elo, elo_data], ignore_index=True)
    else:
        df_updated_elo = elo_data
    
    # Save updated ELO data
    df_updated_elo.to_parquet(current_elo_file)
    print(f"Updated ELO ratings for {CURRENT_YEAR} week {current_week}")
    
    return df_updated_elo

if __name__ == "__main__":
    print(f"Running ELO update for {CURRENT_YEAR}")
    updated_elo = calculate_weekly_elo()
    if updated_elo is not None:
        print(f"ELO update completed successfully")
        print(f"Updated {len(updated_elo)} team-week records")
    else:
        print("ELO update failed")