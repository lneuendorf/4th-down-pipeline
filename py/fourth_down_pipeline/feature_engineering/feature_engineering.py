import logging
from os.path import join
import pickle
import json
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np

MODEL_DIR = 'models'

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
    level=logging.INFO
)
LOG = logging.getLogger(__name__)

def engineer_features(
    games: pd.DataFrame,
    plays: pd.DataFrame,
    weather: pd.DataFrame,
    venues: pd.DataFrame,
    lines: pd.DataFrame,
    elo: pd.DataFrame,
    team_strengths: pd.DataFrame,
    advanced_team_stats: pd.DataFrame
) -> pd.DataFrame:
    """ Engineer features for 4th down decision making model.

    Args:
        games (pd.DataFrame): DataFrame containing game information
        plays (pd.DataFrame): DataFrame containing play-by-play data
        weather (pd.DataFrame): DataFrame containing weather information
        venues (pd.DataFrame): DataFrame containing venue information
        lines (pd.DataFrame): DataFrame containing betting lines
        elo (pd.DataFrame): DataFrame containing ELO ratings
        team_strengths (pd.DataFrame): DataFrame containing team strength metrics'
        advanced_team_stats (pd.DataFrame): DataFrame containing advanced team stats

    Returns:
        pd.DataFrame: DataFrame with engineered features for 4th down decision making
    """

    LOG.info("Engineering features for 4th down decision making model")
    
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
            is_goal_to_go=lambda x: x['yards_to_goal'] <= x['distance']
        )
        .assign(
            pct_game_played=lambda x: (((x['period'] - 1) * 15 * 60) + (15* 60) - 
                                    (x['clock_minutes'] * 60 + x['clock_seconds'])
                                    ) / (15 * 60 * 4),
            score_diff=lambda x: x['offense_score'] - x['defense_score'],
        )
        .query('0 < period <= 4') # only regulation
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
            pct_half_played=lambda x: np.where(
                x.period <= 2,
                x.pct_game_played,
                x.pct_game_played - 0.5
            )
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
        temperature = np.where(
            data.game_indoors,
            70,
            data.temperature.fillna(int(data.temperature.mean()))
        ),
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
    ).assign(
        offense_timeouts=lambda x: np.select(
            [
                x.offense_timeouts.isna() & x.period.isin([1, 2, 3, 4]),
                x.offense_timeouts.isna() & ~x.period.isin([1, 2, 3, 4])
            ],
            [
                np.floor(-2.4387 * x.pct_half_played + 3.1631).astype(int),
                0
            ],
            default=x.offense_timeouts
        ),
        defense_timeouts=lambda x: np.select(
            [
                x.defense_timeouts.isna() & x.period.isin([1, 2, 3, 4]),
                x.defense_timeouts.isna() & ~x.period.isin([1, 2, 3, 4])
            ],
            [
                np.floor(-2.4387 * x.pct_half_played + 3.1631).astype(int),
                0
            ],
            default=x.defense_timeouts
        )
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
            seconds_left_in_half = lambda x: np.where(
                x['period'].isin([1,2]), 
                (2 - x['period']) * 15 * 60 + (x['clock_minutes'] * 60 + x['clock_seconds']),
                (4 - x['period']) * 15 * 60 + (x['clock_minutes'] * 60 + x['clock_seconds'])
            ),
        )

        .assign(
            diff_time_ratio = lambda x: (
                x['score_diff'] * np.exp(4 * (3600 - x['game_seconds_remaining']) / 3600)
            ),
            spread_time_ratio = lambda x: (
                x['pregame_spread'] * np.exp(-4 * (3600 - x['game_seconds_remaining']) / 3600)
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

    data = add_kneel_features(data)

    data = add_fg_pressure_rating(data)

    data = add_offense_success_rates(data, games, elo, advanced_team_stats)

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

def add_kneel_features(data: pd.DataFrame) -> pd.DataFrame:
    def simulate_kneelout(
        seconds,
        defense_timeouts,
        down=1,
        play_clock=40,
        kneel_duration=2
    ):
        timeouts = defense_timeouts
        downs_remaining = 4 - down
        prior_to_two_minute = seconds > 120

        while seconds > 0 and downs_remaining > 0:
            # Kneel
            seconds -= kneel_duration

            # Clock stoppage due to two-minute warning
            if prior_to_two_minute and seconds <= 120:
                prior_to_two_minute = False
            # Defensive timeout to stop clock
            elif timeouts > 0:
                timeouts -= 1
            else:
                # Clock stops after kneel down at two minute mark - Defense has no timeouts left
                if prior_to_two_minute and seconds <= (120 + play_clock):
                    prior_to_two_minute = False
                    seconds = 120
                # Normal clock runoff
                else:
                    seconds -= play_clock

            downs_remaining -= 1

        return max(seconds, 0)


    def seconds_after_punt_and_opponent_kneelout(
        row,
        avg_punt_time=7,
        play_clock=40,
        kneel_duration=2
    ):
        # Time after punt
        seconds = row["game_seconds_remaining"] - avg_punt_time
        if seconds <= 0:
            return 0

        # Now opponent has ball and tries to kneel it out
        seconds = simulate_kneelout(
            seconds=seconds,
            defense_timeouts=row["offense_timeouts"],
            down=1,
            play_clock=play_clock,
            kneel_duration=kneel_duration
        )

        # Punt back to original offense
        seconds -= avg_punt_time
        return max(seconds, 0)
    
    data["seconds_after_kneelout"] = data.apply(
        lambda r: simulate_kneelout(
            seconds=r["game_seconds_remaining"],
            defense_timeouts=r["defense_timeouts"],
            down=r["down"]
        ),
        axis=1
    )

    data["seconds_after_punt_and_opponent_kneelout"] = data.apply(
        seconds_after_punt_and_opponent_kneelout,
        axis=1
    )

    data['can_kneel_out'] = data.seconds_after_kneelout <= 0
    data['can_kneel_out_30'] = data.seconds_after_kneelout <= 30
    data['can_kneel_out_60'] = data.seconds_after_kneelout <= 60
    data['can_kneel_out_90'] = data.seconds_after_kneelout <= 90

    return data

def add_fg_pressure_rating(data: pd.DataFrame) -> pd.DataFrame:
    data['pressure_rating'] = np.select(
        [
            # tie or take the lead, last 2 min or OT
            (
                ((data['game_seconds_remaining'] <= 120) | (data['period'] > 4)) & 
                (data['score_diff'] >= -3) & (data['score_diff'] <= 0)
            ), 

            # (tie or take the lead, last 5 - 2 min) or (stay within one score, last 2 min)
            (
                ((data['game_seconds_remaining'] <= 300) & (data['score_diff'] >= -3) & (data['score_diff'] <= 0)) |
                ((data['game_seconds_remaining'] <= 120) & (data['score_diff'] >= -11) & (data['score_diff'] <= -4))
            ),

            # (tie or take the lead, last 10 - 5 min) or (stay within one score, last 5 min)
            (
                ((data['game_seconds_remaining'] <= 600) & (data['score_diff'] >= -3) & (data['score_diff'] <= 0)) |
                ((data['game_seconds_remaining'] <= 300) & (data['score_diff'] >= -11) & (data['score_diff'] <= -4))
            ),

            # (tie or take the lead, last 15 - 10 min) or (stay within one score, last 10 min)
            (
                ((data['game_seconds_remaining'] <= 900) & (data['score_diff'] >= -3) & (data['score_diff'] <= 0)) |
                ((data['game_seconds_remaining'] <= 600) & (data['score_diff'] >= -11) & (data['score_diff'] <= -4))
            ),

            # stay within one score, last 15 min
            (
                (data['game_seconds_remaining'] <= 900) & (data['score_diff'] >= -11) & (data['score_diff'] <= -4)
            )
        ],
        [4, 3, 2, 1, 0.5],
        default=0
    )

    return data

def add_offense_success_rates(
    data: pd.DataFrame,
    games: pd.DataFrame,
    elo: pd.DataFrame,
    advanced_team_stats: pd.DataFrame
) -> pd.DataFrame:
    """ Add offense pass and rush success rates, adjusted with priors from ELO ratings. """

    advanced_team_stats = (
        # only keep teams that had a game that week
        games.dropna(subset=['home_points','away_points'])
        .melt(
            id_vars=['season','week', 'season_type'], 
            value_vars=['home_team','away_team'], 
            var_name='home_away', 
            value_name='team'
        ).drop(columns=['home_away'])
        # Merge in advanced stats for that week
        .merge(
            advanced_team_stats[['season','week','team','offense_pass_success','offense_rush_success']], 
            on=['season','week','team'], 
            how='left'
        )
        # Merge in pre-game ELO for that week, will be used to approximate priors for success rates for teams
        .merge(elo[['season','week','team','elo']], on=['season','week','team'], how='left')
    )

    advanced_team_stats['max_week'] = advanced_team_stats.groupby(['season', 'season_type'])['week'].transform('max')

    # --- Regress each success rate on ELO for the season ---
    success_cols = ['offense_pass_success', 'offense_rush_success']
    for col in success_cols:
        model_path = join(MODEL_DIR, 'offense_success_rate_model', f'{col}_regression_coefficients.json')
        with open(model_path, 'r') as f:
            model_wieghts = json.load(f)

        reg = LinearRegression()
        reg.coef_ = np.atleast_1d(model_wieghts['coef']).astype(float)
        reg.intercept_ = float(model_wieghts['intercept'])
        reg.n_features_in_ = 1
        reg.feature_names_in_ = np.array(['elo'])
        
        # Prior from ELO
        X = advanced_team_stats[['elo']].astype(float)
        advanced_team_stats[f'{col}_prior'] = reg.predict(X)
        
        # Fill missing success rates with prior (for teams with no plays in that week)
        advanced_team_stats[col] = advanced_team_stats[col].fillna(advanced_team_stats[f'{col}_prior'])

        # Update observed success rate with weighted prior
        # Weight increases as season progresses: week/max_week
        advanced_team_stats[f'{col}_adjusted'] = (
            advanced_team_stats[f'{col}_prior'] * (1 - advanced_team_stats['week']/(advanced_team_stats['max_week'] + 1)) +
            advanced_team_stats[col] * (advanced_team_stats['week']/(advanced_team_stats['max_week'] + 1))
        )

    # Keep relevant columns
    keep_cols = ['season', 'week', 'season_type', 'team', 'elo'] + \
                success_cols + \
                [f'{c}_prior' for c in success_cols] + \
                [f'{c}_adjusted' for c in success_cols]
    advanced_team_stats = advanced_team_stats[keep_cols]

    data = data.merge(
        advanced_team_stats.rename(columns={
            'team': 'offense',
            'offense_pass_success_adjusted': 'offense_pass_success_adjusted',
            'offense_rush_success_adjusted': 'offense_rush_success_adjusted'
        })[['season', 'week', 'offense', 'offense_pass_success_adjusted', 'offense_rush_success_adjusted']],
        on=['season', 'week', 'offense'],
        how='left'
    )

    return data.drop_duplicates()

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