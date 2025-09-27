import os
from os.path import join
import pandas as pd
import numpy as np

DATA_PATH = 'data'

def update_elo(
    games: pd.DataFrame, 
    teams: pd.DataFrame, 
    year: int, 
    week: int, 
    season_type: str, 

) -> pd.DataFrame:
    """ Update and save ELO ratings based on the provided games and teams data.
    
    Args:
        games (pd.DataFrame): DataFrame containing game data
        teams (pd.DataFrame): DataFrame containing team data
        year (int): Year of the season (e.g., 2023)
        week (int): Week number of the season (e.g., 1-15 for regular season)
        season_type (str): Type of the season ('regular' or 'postseason')
    Returns:
        pd.DataFrame: DataFrame containing updated ELO ratings for each team
    """

    # Merge in team division
    team_cols = ['season','id','classification']
    df = (
        games[['season','week','season_type','home_id','home_team','home_points',
               'away_id','away_team','away_points']]
        .merge(
            teams[team_cols].rename(columns={
                'id': 'home_id', 
                'classification': 'home_division'
            }),
            left_on=['season','home_id'],
            right_on=['season','home_id'], 
            how='left'
        ).merge(
            teams[team_cols].rename(columns={
                'id': 'away_id', 
                'classification': 'away_division'
            }),
            left_on=['season','away_id'],
            right_on=['season','away_id'],
            how='left'
        )
        [['season', 'week', 'season_type', 'home_id', 'home_team', 'home_division',
          'home_points', 'away_id', 'away_team', 'away_division', 'away_points']]
    )

    # Determine game results
    df.loc[:,'home_result'] = np.select(
        [df['home_points'] > df['away_points'], df['home_points'] < df['away_points']],
        [1, 0],
        default=0.5
    )
    df.loc[:,'away_result'] = 1 - df.loc[:,'home_result']

    df = _fill_missing_divisions(df)

    df = _generate_elo(df)
    
    df_final = (
        pd.concat([
            df[['season','week','season_type','home_id','home_team',
                'home_division','home_elo']]
            .rename(columns={
                'home_id': 'team_id', 
                'home_team': 'team', 
                'home_division': 'division', 
                'home_elo': 'elo'
            }),
            df[['season','week','season_type','away_id','away_team',
                'away_division','away_elo']]
            .rename(columns={
                'away_id': 'team_id', 
                'away_team': 'team', 
                'away_division': 'division', 
                'away_elo': 'elo'
            })
        ])
        .sort_values(
            ['team_id','season','season_type','week'], 
            ascending=[True, True, False, True]
        )
        .reset_index(drop=True)
    )

    # Directory for caching
    elo_dir = join(DATA_PATH, "elo")
    if not os.path.exists(elo_dir):
        os.makedirs(elo_dir)

    for season in df_final.season.unique():
        df_final \
            .query("season == @season") \
            .reset_index(drop=True) \
            .to_parquet(join(elo_dir, f"{season}.parquet"))
        
    return (
        df_final
        .query("season == @year and week == @week and season_type == @season_type")
        .reset_index(drop=True)
    )


def _fill_missing_divisions(df: pd.DataFrame) -> pd.DataFrame:
    division_heirarchy = {
        'iii': 0,
        'ii': 1,
        'fcs': 2,
        'fbs': 3
    }

    # Loop twice to handle for cases where both teams are missing division values
    for i in range(2):
        # assume missing division values are most commonly occuring opponent division
        confs = (
            pd.concat([
                df[['home_id','away_division']]
                .rename(columns={
                    'home_id': 'team_id', 
                    'away_division': 'opp_division'
                })
                .dropna(subset=['opp_division']),
                df[['away_id','home_division']]
                .rename(columns={
                    'away_id': 'team_id', 
                    'home_division': 'opp_division'
                })
                .dropna(subset=['opp_division'])
            ])
            .groupby(['team_id','opp_division'])
            .agg({'opp_division': 'count'})
            .rename(columns={'opp_division': 'count'})
            .reset_index()
            .assign(
                division_heirarchy=lambda x: x['opp_division'].map(division_heirarchy)
            )
            .sort_values(
                ['team_id','count','division_heirarchy'], 
                ascending=[True,False,True]
            )
            .dropna(subset=['opp_division'])
            .drop_duplicates('team_id')
            .drop(columns=['count','division_heirarchy'])
            .rename(columns={'opp_division': 'division'})
        )
        df = (
            df.merge(
                confs.rename(columns={
                    'team_id': 'home_id', 
                    'division': 'home_division'
                }),
                on='home_id',
                how='left',
                suffixes=('','_y')
            ).merge(
                confs.rename(columns={
                    'team_id': 'away_id', 
                    'division': 'away_division'
                }),
                on='away_id',
                how='left',
                suffixes=('','_y')
            )
            .assign(
                home_division=lambda x: x['home_division'].fillna(x['home_division_y']),
                away_division=lambda x: x['away_division'].fillna(x['away_division_y'])
            )
            .drop(columns=['home_division_y','away_division_y'])
        )

        df.dropna(subset=['home_points','away_points'], inplace=True)
    return df

def _generate_elo(df: pd.DataFrame) -> pd.DataFrame:
    HFA = 100
    K = 100
    DIVISOR = 800

    elo_cache = {}

    elo_initial = {
        'fbs': 1500.0,
        'fcs': 1300.0,
        'ii': 1000.0,
        'iii': 800.0
    }

    df['home_elo'] = df['home_division'].apply(lambda x: elo_initial[x])
    df['away_elo'] = df['away_division'].apply(lambda x: elo_initial[x])

    df['home_elo_post'] = df['home_elo']
    df['away_elo_post'] = df['away_elo']

    for season in df.season.unique():
        df_season = df.query("season == @season")
        
        for week in sorted(df_season.week.unique()):
            df_week = df_season.query("week == @week").copy()
            
            df_week['home_elo'] = df_week['home_id'].map(elo_cache).fillna(df_week['home_elo'])
            df_week['away_elo'] = df_week['away_id'].map(elo_cache).fillna(df_week['away_elo'])

            df_week['home_elo_post'] = (
                df_week['home_elo'] +
                K * np.log(np.abs(df_week['home_points'] - df_week['away_points']) + 1) / 2  # Margin of victory adjustment
                * (df_week['home_result'] - 1 / (1 + 10 ** ((df_week['away_elo'] - HFA - df_week['home_elo']) / DIVISOR)))
            )
            df_week['away_elo_post'] = (
                df_week['away_elo'] +
                K * np.log(np.abs(df_week['away_points'] - df_week['home_points']) + 1) / 2  # Margin of victory adjustment
                * (df_week['away_result'] - 1 / (1 + 10 ** ((df_week['home_elo'] + HFA - df_week['away_elo']) / DIVISOR)))
            )

            df.loc[df_week.index, 'home_elo'] = df_week['home_elo'].astype(float)
            df.loc[df_week.index, 'away_elo'] = df_week['away_elo'].astype(float)
            df.loc[df_week.index, 'home_elo_post'] = df_week['home_elo_post'].astype(float)
            df.loc[df_week.index, 'away_elo_post'] = df_week['away_elo_post'].astype(float)

            elo_cache.update(df_week.set_index('home_id')['home_elo_post'].to_dict())
            elo_cache.update(df_week.set_index('away_id')['away_elo_post'].to_dict())
    return df