from os.path import join
import pickle
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np

MODEL_DIR = 'models'

def engineer_features(
    games: pd.DataFrame,
    plays: pd.DataFrame,
    weather: pd.DataFrame,
    venues: pd.DataFrame,
    lines: pd.DataFrame,
    elo: pd.DataFrame,
    team_strengths: pd.DataFrame
) -> pd.DataFrame:
    """ Engineer features for 4th down decision making model.

    Args:
        games (pd.DataFrame): DataFrame containing game information
        plays (pd.DataFrame): DataFrame containing play-by-play data
        weather (pd.DataFrame): DataFrame containing weather information
        venues (pd.DataFrame): DataFrame containing venue information
        lines (pd.DataFrame): DataFrame containing betting lines
        elo (pd.DataFrame): DataFrame containing ELO ratings
        team_strengths (pd.DataFrame): DataFrame containing team strength metrics

    Returns:
        pd.DataFrame: DataFrame with engineered features for 4th down decision making
    """

    play_cols = [
        'game_id','drive_id','id','offense','defense','period','clock_minutes',
        'clock_seconds','offense_score','defense_score', 'offense_timeouts',
        'defense_timeouts','yards_to_goal','down','distance', 'play_type','play_text'
    ]
    game_cols = [
        'season','week','id','season_type','neutral_site','venue_id', 'completed',
        'home_id','home_team','home_conference',
        'away_id','away_team','away_conference',
    ]
    venue_cols = [
        'id', 'elevation', 'grass'
    ]
    elo_cols = ['season', 'week', 'division', 'team_id', 'elo']
    weather_cols = [
        'id','precipitation','wind_speed','temperature','game_indoors'
    ]
    spread_cols = ['id','home_spread']

    data = (
        plays[play_cols].rename(columns={'id':'play_id'})
        .query('down==4')
        .assign(
            clock_minutes=lambda x: np.maximum(np.minimum(x['clock_minutes'], 15), 0),
            clock_seconds=lambda x: np.maximum(np.minimum(x['clock_seconds'], 59), 0),
            offense_score=lambda x: np.maximum(x['offense_score'], 0),
            defense_score=lambda x: np.maximum(x['defense_score'], 0),
        )
        .assign(
            pct_game_played=lambda x: (((x['period'] - 1) * 15 * 60) + (15* 60) - 
                                    (x['clock_minutes'] * 60 + x['clock_seconds'])
                                    ) / (15 * 60 * 4),
            score_diff=lambda x: x['offense_score'] - x['defense_score'],
        )
        .query('0 < period <= 4')
        .query('0 <= yards_to_goal <= 100')
        .query('0 <= distance <= 100')
        # .drop(columns=['period','clock_minutes','clock_seconds'])
        .merge(
            games[game_cols].rename(columns={'id':'game_id'}),
            how='left',
            on=['game_id']
        )
        .query('completed == True')
        .assign(
            is_home_team=lambda x: np.select(
                condlist=[x.neutral_site, x.offense == x.home_team],
                choicelist=[0, 1], 
                default=-1
            ),
        )
        .merge(
            venues[venue_cols].rename(columns={'id':'venue_id'}),
            how='left',
            on=['venue_id']
        )
        .merge(
            elo[elo_cols].rename(columns={'team_id': 'home_id', 
                                            'division': 'home_division', 
                                            'elo': 'home_pregame_elo'}),
            on=['season', 'week', 'home_id'],
            how='left'
        )
        .merge(
            elo[elo_cols].rename(columns={'team_id': 'away_id', 
                                            'division': 'away_division', 
                                            'elo': 'away_pregame_elo'}),
            on=['season', 'week', 'away_id'],
            how='left'
        )
        .assign(
            pregame_elo_diff=lambda x: np.where(
                x.offense == x.home_team,
                x.home_pregame_elo - x.away_pregame_elo,
                x.away_pregame_elo - x.home_pregame_elo
            ),
            pregame_offense_elo=lambda x: np.where(
                x.offense == x.home_team,
                x.home_pregame_elo,
                x.away_pregame_elo
            ),
            pregame_defense_elo=lambda x: np.where(
                x.offense == x.home_team,
                x.away_pregame_elo,
                x.home_pregame_elo
            ),
        )
        .merge(
            weather[weather_cols].rename(columns={'id':'game_id'}),
            how='left',
            on=['game_id']
        )
        .assign(
            distance=lambda x: np.where(
                x.yards_to_goal - x.distance < 0,
                x.yards_to_goal,
                x.distance
            )
        )
        .merge(
            lines[spread_cols].rename(columns={'id': 'game_id'}),
            on=['game_id'],
            how='left'
        )
    )

    data['pressure_rating'] = np.select(
        [
            # tie or take the lead, last 2 min
            (data['pct_game_played'] >= (58 / 60)) & (data['score_diff'] >= -3) & (data['score_diff'] <= 0),

            # tie or take the lead, last 5 - 2 min
            (data['pct_game_played'] >= (55 / 60)) & (data['score_diff'] >= -3) & (data['score_diff'] <= 0),

            # tie or take the lead, last 10 - 5 min
            (data['pct_game_played'] >= (50 / 60)) & (data['score_diff'] >= -3) & (data['score_diff'] <= 0),
            
            # tie or take the lead, last 15 - 10 min
            (data['pct_game_played'] >= (45 / 60)) & (data['score_diff'] >= -3) & (data['score_diff'] <= 0),
        ],
        [4, 3, 2, 1],
        default=0
    )

    data = data.assign(
        offense_division = np.where(
            data['offense'] == data['home_team'], 
            data['home_division'], 
            data['away_division']
        ),
        defense_division = np.where(
            data['offense'] == data['home_team'], 
            data['away_division'], 
            data['home_division']
        ),
        grass = data.grass.fillna(False),
        game_indoors = data.game_indoors.fillna(False),
        temperature = data.temperature.fillna(int(data.temperature.mean())),
        wind_speed = np.where(
            data.game_indoors, 
            0, 
            data.wind_speed.fillna(int(data.wind_speed.mean()))
        ),
        elevation = data.elevation.fillna(int(data.elevation.astype(float).mean())),
        precipitation = np.where(
            data.game_indoors, 
            0, 
            data.precipitation.fillna(int(data.precipitation.mean()))
        ),
        defense_timeouts = np.minimum(np.maximum(data.defense_timeouts, 0), 3),
        offense_timeouts = np.minimum(np.maximum(data.offense_timeouts, 0), 3),
    )
    data['elevation'] = data['elevation'].astype(float)   

    data = _impute_point_spread(data) 

    data = (
        data
        .assign(
            game_seconds_remaining = (
                (4 * 15 * 60) - 
                ((data['period'] - 1) * 15 * 60 + (15 * 60) - 
                 (data['clock_minutes'] * 60 + data['clock_seconds']))
            ),
            pregame_spread = np.where(
                data['offense'] == data['home_team'], 
                data['home_spread'], 
                -data['home_spread']
            ),
        )

        .assign(
            diff_time_ratio = lambda x: (
                x['score_diff'] * np.exp(4 * (3600 - x['game_seconds_remaining']) / 3600)
            ),
            spread_time_ratio = lambda x: (
                x['pregame_spread'] * np.exp(4 * (3600 - x['game_seconds_remaining']) / 3600)
            ),
        )
    )

    data = (
        data.merge(
            team_strengths
            [['team', 'offensive_strength', 'season', 'week', 'season_type']]
            .rename(columns={
                'team': 'offense', 
                'offensive_strength': 'offense_strength',
            }),
            on=['season', 'week', 'offense', 'season_type'],
            how='left'
        ).merge(
            team_strengths
            [['team', 'defensive_strength', 'season', 'week', 'season_type']]
            .rename(columns={
                'team': 'defense', 
                'defensive_strength': 'defense_strength',
            }),
            on=['season', 'week', 'defense', 'season_type'],
            how='left'
        )
    )

    return data

def add_decision(data: pd.DataFrame) -> pd.DataFrame:
    rush = (
        data['play_text'].str.contains('rush', case=False) |
        data['play_text'].str.contains('run', case=False) |
        data['play_text'].str.contains('scramble', case=False) |
        (data['play_type'] == 'Rush')
    )
    pass_ = (
        data['play_text'].str.contains('pass', case=False) |
        data['play_text'].str.contains('throw', case=False) |
        data['play_text'].str.contains('sack', case=False) |
        data['play_text'].str.contains('intercept', case=False) |
        (data['play_type'].str.contains('Pass', case=False)) |
        (data['play_type'].str.contains('Sack', case=False)) 
    )
    punt = (
        data['play_text'].str.contains('punt', case=False) |
        data['play_type'].str.contains('Punt', case=False) | 
        (data.play_type == 'Punt')
    )
    field_goal = (
        data['play_text'].str.contains('field goal', case=False) |
        data['play_text'].str.contains('fg', case=False) |
        data['play_type'].str.contains('Field Goal', case=False)
    )
    penalty = (
        data['play_text'].str.contains('penalty', case=False) |
        data['play_type'].str.contains('Penalty', case=False) |
        data['play_text'].str.contains('illegal', case=False) |
        data['play_text'].str.contains('offside', case=False) |
        data['play_text'].str.contains('false start', case=False) |
        data['play_text'].str.contains('delay of game', case=False) |
        data['play_text'].str.contains('pass interference', case=False) |
        data['play_text'].str.contains('holding', case=False) |
        data['play_text'].str.contains('personal foul', case=False) |
        data['play_text'].str.contains('roughing', case=False) |
        data['play_text'].str.contains('unsportsmanlike', case=False) |
        data['play_text'].str.contains('taunting', case=False)
    )
    kneel = (
        data['play_text'].str.contains('kneel', case=False) |
        data['play_type'].str.contains('Kneel', case=False)
    )
    timeout = (
        data['play_text'].str.contains('timeout', case=False) |
        data['play_type'].str.contains('Timeout', case=False)
    )
    kickoff = (
        data['play_text'].str.contains('kickoff', case=False) |
        data['play_type'].str.contains('Kickoff', case=False)
    )
    end_of_period = (
        data['play_text'].str.contains('end ', case=False) |
        data['play_type'].str.contains('end ', case=False)
    )
    safety = (
        data['play_text'].str.contains('safety', case=False) |
        data['play_type'].str.contains('Safety', case=False)
    )

    data = data.assign(
        action=np.select(
            [
                penalty, end_of_period, timeout, kneel, kickoff, field_goal, 
                punt, rush, pass_, safety
            ],
            [
                'penalty', 'end_of_period', 'timeout', 'kneel', 'kickoff', 
                'field_goal', 'punt', 'rush', 'pass', 'safety'
            ],
            default='other'
        )
    )

    drop_actions = [
        'penalty', 'timeout', 'kickoff', 'end_of_period', 'safety', 'kneel', 'other'
    ]
    data = data.query('action not in @drop_actions').reset_index(drop=True)

    data['decision'] = np.select(
        [
            data['action'].isin(['rush', 'pass']),
            data['action'] == 'punt',
            data['action'] == 'field_goal'
        ],
        [
            'go', 'punt', 'field_goal'
        ],
        default=-1
    )

    return data

def _impute_point_spread(data: pd.DataFrame) -> pd.DataFrame:
    """ Impute missing point spreads using linear regression based on ELO ratings.
    
    Args:
        data (pd.DataFrame): DataFrame containing point spreads with possible 
        missing values along with pregame ELO ratings.

    Returns:
        pd.DataFrame: DataFrame with missing point spreads imputed.
    """

    model = pickle.load(open(
        join(MODEL_DIR, 'point_spread_imputation', 'regressor.pkl'), 'rb'
    ))

    mask = data['home_spread'].isna()
    if mask.sum() != 0:
        data.loc[mask, 'home_spread'] = model.predict(
            data.loc[mask, ['home_pregame_elo', 'away_pregame_elo']]
        )
    return data