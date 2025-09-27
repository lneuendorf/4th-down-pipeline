import pickle
from os.path import join
import numpy as np
import pandas as pd
from scipy.optimize import minimize

MODEL_DIR = 'models'

def calculate_team_strengths(
    ppa: pd.DataFrame,
    year: int,
    week: int,
    season_type: str,
    window_size: int = 10,

):
    # Convert season/week to a numerical value for sorting
    ppa = ppa.copy()
    if set(ppa['season_type'].unique().tolist()) != set(['regular', 'postseason']):
        raise ValueError("PPA data must contain both 'regular' and 'postseason' season types")
    ppa['season_type_indicator'] = ppa['season_type'].apply(lambda x: 1 if x == 'regular' else 2)
    ppa['season_week'] = ppa['season'] * 1000 + ppa['season_type_indicator'] * 100 + ppa['week']
    season_type_indicator = 1 if season_type == 'regular' else 2
    target_sw = year * 1000 + season_type_indicator * 100 + week
    
    # Get all games before the target week
    df_prior = ppa[ppa['season_week'] < target_sw].copy()
    
    if len(df_prior) == 0:
        raise ValueError(f"No historical data available before season {year} week {week}")
    
    # Sort by season and week (most recent first)
    df_prior = df_prior.sort_values(
        ['season', 'season_type_indicator', 'week'], 
        ascending=[False, False, False]
    )
    
    # Get the most recent window_size weeks of games
    unique_weeks = df_prior[['season','season_type_indicator', 'week']].drop_duplicates()
    recent_weeks = unique_weeks.head(window_size)
    
    # Filter to just these weeks
    df_window = pd.merge(
        left=df_prior, 
        right=recent_weeks, 
        on=['season', 'season_type_indicator', 'week'], 
        how='inner'
    )
    
    if len(df_window) == 0:
        raise ValueError(f"Insufficient data available - found {len(df_prior)} "
                         f"prior games but none in last {window_size} weeks")
    
    # Get all unique teams in the data window
    all_teams = pd.concat([df_window['team'], df_window['opponent']]).unique()
    num_teams = len(all_teams)

    # Create mappings between team names and indices
    team_to_idx = {team: idx for idx, team in enumerate(all_teams)}
    
    # Variables: offensive and defensive strengths for each team
    num_vars = 2 * num_teams
    
    df_window['offense_idx'] = df_window['team'].map(team_to_idx)
    df_window['defense_idx'] = df_window['opponent'].map(team_to_idx)
    offense_indices = df_window['offense_idx'].values
    defense_indices = df_window['defense_idx'].values
    actual_ppas = df_window['offense_ppa'].values - df_window['offense_ppa'].mean()

    # Objective function: minimize the squared error between predicted and actual offensive PPA
    def objective(x):
        predicted_ppas = x[offense_indices] - x[num_teams + defense_indices]
        return np.sum((predicted_ppas - actual_ppas) ** 2)
    
    # Initial guess (zeros for all variables)
    x0 = np.zeros(num_vars)
    
    # Bounds: allow strengths to be between -1 and 1 (adjust as needed)
    bounds = [(-1, 1) for _ in range(num_vars)]

    constraints = [
        {"type": "eq", "fun": lambda x: np.mean(x[:num_teams])},  # mean offense = 0
        {"type": "eq", "fun": lambda x: np.mean(x[num_teams:])}   # mean defense = 0
    ]
        
    # Solve the optimization problem
    result = minimize(
        objective, 
        x0, 
        bounds=bounds, 
        method='SLSQP', 
        options={'maxiter': 10000},
        constraints=constraints
    )
    
    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")
    
    # Extract the offensive and defensive strengths
    offensive_strengths = result.x[:num_teams]
    defensive_strengths = result.x[num_teams:]
    
    # Create the output DataFrame
    df_strengths = pd.DataFrame({
        'team': all_teams,
        'offensive_strength': offensive_strengths,
        'defensive_strength': defensive_strengths,
        'season': year,
        'week': week,
        'season_type': season_type,
        'games_used': len(df_window),
        'min_season': df_window['season'].min(),
        'min_week': df_window['week'].min(),
        'max_season': df_window['season'].max(),
        'max_week': df_window['week'].max()
    })
    
    return df_strengths

def impute_missing_team_strengths(data: pd.DataFrame) -> pd.DataFrame:
    """ Impute missing team strengths using linear regression based on ELO ratings. """
    offense_model = pickle.load(open(join(MODEL_DIR, 'team_strength', 'offense.pkl'), 'rb'))
    defense_model = pickle.load(open(join(MODEL_DIR, 'team_strength', 'defense.pkl'), 'rb'))

    # Offense Strength
    mask = data['offense_strength'].isna()
    if mask.sum() != 0:
        data.loc[mask, 'offense_strength'] = offense_model.predict(
            data.loc[mask, ['pregame_offense_elo']]
        )

    # Defense Strength
    mask = data['defense_strength'].isna()
    if mask.sum() != 0:
        data.loc[mask, 'defense_strength'] = defense_model.predict(
            data.loc[mask, ['pregame_defense_elo']]
        )

    return data