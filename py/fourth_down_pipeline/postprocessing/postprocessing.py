from os.path import join
import pandas as pd
import numpy as np

APP_DATA_DIR = '../app/data'

def postprocess_results(
    data: pd.DataFrame,
    teams: pd.DataFrame,
    coaches: pd.DataFrame,
):
    
    data = data.drop_duplicates(subset=['season', 'week', 'season_type', 'game_id','play_id'], ignore_index=True)
    breakpoint()
    teams.drop_duplicates(subset=['season', 'id', 'color', 'alternate_color'], inplace=True, ignore_index=True)
    coaches.drop_duplicates(inplace=True, ignore_index=True)

    # Get the recommendations
    data = recommend(data)

    # Merge teams to data
    team_cols = ['season', 'id', 'color', 'alternate_color', 'logos']
    data_cols = ['start_date', 'season', 'week', 'season_type', 'game_id', 'play_id', 
        'offense', 'offense_id', 'offense_division', 'offense_conference', 'offense_timeouts', 'offense_score', 'pregame_offense_elo',
        'defense', 'defense_id', 'defense_division', 'defense_conference', 'defense_timeouts', 'defense_score', 'pregame_defense_elo',
        'period', 'clock_minutes', 'clock_seconds',
        'play_text', 
        'yards_to_goal', 'down', 'distance', 
        'action','decision', 'recommendation', 'eWP_diff',
        'exp_wp_fg','exp_wp_go', 'exp_wp_punt']
    data = (
        data.assign(
            offense_id = np.where(data.home_team == data.offense, data.home_id, data.away_id),
            offense_conference = np.where(data.home_team == data.offense, data.home_conference, data.away_conference),
            defense_id = np.where(data.home_team == data.defense, data.home_id, data.away_id),
            defense_conference = np.where(data.home_team == data.defense, data.home_conference, data.away_conference),
        )[data_cols]
        .merge(
            teams[team_cols].add_prefix('offense_'),
            left_on=['season', 'offense_id'],
            right_on=['offense_season', 'offense_id'],
            how='left',
        ).drop(columns=['offense_season'])
        .merge(
            teams[team_cols].add_prefix('defense_'),
            left_on=['season', 'defense_id'],
            right_on=['defense_season', 'defense_id'],
            how='left',
        ).drop(columns=['defense_season'])
        .rename(columns={
            'offense': 'offense_team',
            'defense': 'defense_team',
        })
        .assign(
            offense_logos=lambda x: x['offense_logos'].str[0],
            defense_logos=lambda x: x['defense_logos'].str[0],
        )
        .assign(
            offense_color=lambda x: x['offense_color'].astype(str),
            offense_alternate_color=lambda x: x['offense_alternate_color'].astype(str),
            offense_logos=lambda x: x['offense_logos'].astype(str),
        )
    )

    # Generate Team Tendencies Dataset
    generate_team_tendencies(data)

    # Create Coach Tendencies Dataset
    generate_coach_tendencies(data, coaches)

    # Create Game Decision Dataset
    generate_game_decisions(data, coaches)
    

def recommend(data: pd.DataFrame) -> pd.DataFrame:
    data = data.assign(
        eWP_diff_go_fg = data['exp_wp_go'] - data['exp_wp_fg'],
        eWP_diff_go_punt = data['exp_wp_go'] - data['exp_wp_punt'],
        eWP_diff_fg_punt = data['exp_wp_fg'] - data['exp_wp_punt'],
    )

    choices =['No Recommendation', 'Go', 'Field Goal', 'Punt',
                'Go or Field Goal', 'Go or Punt', 'Field Goal or Punt']
    conditions = [
        ( # No Recommendation if all options are within 0.5% eWP
            (np.abs(data['eWP_diff_go_fg']) == 0) & 
            (np.abs(data['eWP_diff_go_punt']) == 0) & 
            (np.abs(data['eWP_diff_fg_punt']) == 0)
        ),
        ( # Go Recommendation
            (data['eWP_diff_go_fg'] > 0) & (data['eWP_diff_go_punt'] > 0)
        ),
        ( # Field Goal Recommendation
            (data['eWP_diff_go_fg'] < 0) & (data['eWP_diff_fg_punt'] > 0)
        ),
        ( # Punt Recommendation
            (data['eWP_diff_go_punt'] < 0) & (data['eWP_diff_fg_punt'] < 0)
        ),
        ( # Go and Field Goal Recommendation (equal with punt worse)
            (data['eWP_diff_go_fg'] == 0) & (data['eWP_diff_go_punt'] > 0) & (data['eWP_diff_fg_punt'] > 0)
        ),
        ( # Go and Punt Recommendation (equal with field goal worse)
            (data['eWP_diff_go_punt'] == 0) & (data['eWP_diff_go_fg'] > 0) & (data['eWP_diff_fg_punt'] < 0)
        ),
        ( # Field Goal and Punt Recommendation (equal with go worse)
            (data['eWP_diff_fg_punt'] == 0) & (data['eWP_diff_go_fg'] < 0) & (data['eWP_diff_go_punt'] < 0)
        ),
    ]

    data['recommendation'] = np.select(conditions, choices, default='No Recommendation')
    
    data['eWP_diff'] = np.where(
        data.recommendation != 'No Recommendation',
        np.where(
            data.recommendation.isin(['Go', 'Field Goal', 'Punt']),
            np.abs(data[['eWP_diff_go_fg', 'eWP_diff_go_punt', 'eWP_diff_fg_punt']]).min(axis=1),
            np.where(
                data.recommendation == 'Go or Field Goal',
                np.abs(data['eWP_diff_go_punt']),
                np.where(
                    data.recommendation == 'Go or Punt',
                    np.abs(data['eWP_diff_go_fg']),
                    np.abs(data['eWP_diff_go_fg'])
                )
            ),
        ),
        0
    )

    return data

def generate_team_tendencies(
    data: pd.DataFrame,
) -> None:
    cols = [
        'season', 'week', 'season_type', 'offense_id', 'offense_team',
        'offense_division', 'offense_conference', 'offense_color', 'offense_alternate_color',
        'offense_logos', 'defense_id', 'defense_team', 'defense_division', 'defense_conference',
        'defense_color', 'defense_alternate_color', 'defense_logos',
        'exp_wp_go', 'exp_wp_fg', 'exp_wp_punt', 'decision', 'recommendation'
    ]
    team_tendencies_go = (
        data.query('recommendation == "Go"').copy()
        [cols]
        .assign(
            wp_lost = lambda x: np.select(
                [x.decision == 'field_goal', x.decision == 'punt'],
                [x.exp_wp_go - x.exp_wp_fg, x.exp_wp_go - x.exp_wp_punt],
                default=0
            ),
        )
        .reset_index(drop=True)
    )
    team_tendencies_res = (
        team_tendencies_go
        .groupby(['season','offense_team', 'offense_conference', 'offense_color', 
                  'offense_alternate_color', 'offense_logos'])
        .agg(
            n_go=('decision', lambda x: (x == 'go').sum()),
            n_go_rec=('offense_team', 'count'),
            net_wp_lost=('wp_lost', 'sum'),
        )
        .sort_values('n_go', ascending=False)
        .reset_index()
    )

    team_tendencies_res = (
        team_tendencies_res
        .replace(
            {
                "offense_color": {
                    "#null": "#FFFFFF",
                },
                "offense_alternate_color": {
                    "#null": "#000",
                },
            }
        )
    )

    # Fix: make the the darker color the bourder color column, and the lighter color the fill color
    team_tendencies_res["fill_color"] = np.where(
        team_tendencies_res.offense_color > team_tendencies_res.offense_alternate_color,
        team_tendencies_res.offense_alternate_color,
        team_tendencies_res.offense_color
    )
    team_tendencies_res["border_color"] = np.where(
        team_tendencies_res.offense_color < team_tendencies_res.offense_alternate_color, 
        team_tendencies_res.offense_alternate_color, 
        team_tendencies_res.offense_color
    )

    # if bourder color is white, make light grey
    team_tendencies_res["border_color"] = np.where(
        team_tendencies_res["border_color"] > "#fafafa",
        "#ebebeb",
        team_tendencies_res["border_color"]
    )

    team_tendencies_res.drop(
        columns=['offense_color', 'offense_alternate_color'],
        inplace=True
    )

    team_tendencies_res.to_parquet(
        join(APP_DATA_DIR, 'team_tendencies.parquet'),
        index=False,
    )

def generate_coach_tendencies(
    data: pd.DataFrame,
    coaches: pd.DataFrame,
) -> None:
    cols = [
        'season', 'week', 'offense_id', 'offense_team',
        'offense_division', 'offense_conference', 'offense_color', 'offense_alternate_color',
        'offense_logos', 'defense_id', 'defense_team', 'defense_division', 'defense_conference',
        'defense_color', 'defense_alternate_color', 
        'exp_wp_go', 'exp_wp_fg', 'exp_wp_punt', 'decision', 'recommendation', 'start_date', 'row_id'
    ]
    coach_tendencies_go = (
        data.query('recommendation == "Go"').copy()
        .assign(
            row_id=lambda x: x.index,
        )
        [cols]
        .assign(
            wp_lost = lambda x: np.select(
                [x.decision == 'field_goal', x.decision == 'punt'],
                [x.exp_wp_go - x.exp_wp_fg, x.exp_wp_go - x.exp_wp_punt],
                default=0
            ),
        )
        .merge(
            coaches,
            left_on=['season', 'offense_team'],
            right_on=['season', 'school'],
            how='left'
        ).drop(columns=['school'])
        # Keep the most recenlty hired coach
        .query('start_date >= hire_date')
        .sort_values('hire_date', ascending=True)
        .drop_duplicates(subset=['row_id'])
        .drop(columns=['row_id','hire_date'])
        .reset_index(drop=True)
    )

    coach_tendencies_res = (
        coach_tendencies_go
        .groupby(['season','offense_team', 'offense_conference', 'offense_color', 
                  'offense_alternate_color', 'coach_name'])
        .agg(
            n_go=('decision', lambda x: (x == 'go').sum()),
            n_go_rec=('offense_team', 'count'),
            net_wp_lost=('wp_lost', 'sum'),
        )
        .sort_values('n_go', ascending=False)
        .reset_index()
    )
    coach_tendencies_res = (
        coach_tendencies_res
        .replace(
            {
                "offense_color": {
                    "#null": "#FFFFFF",
                },
                "offense_alternate_color": {
                    "#null": "#000",
                },
            }
        )
    )

    # Fix: make the the darker color the bourder color column, and the lighter color the fill color
    coach_tendencies_res["fill_color"] = np.where(
        coach_tendencies_res.offense_color > coach_tendencies_res.offense_alternate_color,
        coach_tendencies_res.offense_alternate_color,
        coach_tendencies_res.offense_color
    )
    coach_tendencies_res["border_color"] = np.where(
        coach_tendencies_res.offense_color < coach_tendencies_res.offense_alternate_color, 
        coach_tendencies_res.offense_alternate_color, 
        coach_tendencies_res.offense_color
    )

    # if bourder color is white, make light grey
    coach_tendencies_res["border_color"] = np.where(
        coach_tendencies_res["border_color"] > "#fafafa",
        "#ebebeb",
        coach_tendencies_res["border_color"]
    )

    coach_tendencies_res.drop(
        columns=['offense_color', 'offense_alternate_color'], 
        inplace=True
    )

    coach_tendencies_res.to_parquet(
        join(APP_DATA_DIR, 'coach_tendencies.parquet'),
        index=False,
    )

def generate_game_decisions(
    data: pd.DataFrame,
    coaches: pd.DataFrame,
) -> None:
    data['time'] = (
        'Q' + data['period'].astype(str) + ' ' +
        data['clock_minutes'].astype(str).str.zfill(2) + ':' +
        data['clock_seconds'].astype(str).str.zfill(2)
    )

    data = data.dropna().reset_index(drop=True)

    # merge coaches
    data = (
        data.assign(
            row_id=lambda x: x.index
        )
        .merge(
            coaches,
            left_on=['season', 'offense_team'],
            right_on=['season', 'school'],
            how='left'
        ).drop(columns=['school'])
        # Keept the most recenlty hired coach
        .query('start_date >= hire_date')
        .sort_values('hire_date', ascending=True)
        .drop_duplicates(subset=['row_id'])
        .drop(columns=['row_id', 'start_date','hire_date'])
    )

    data['distance'] = data.distance.replace(0, 1)

    cols = [
        'season', 'week', 
        'offense_team', 'offense_conference', 'offense_division', 'offense_logos', 
        'offense_score', 'coach_name',
        'defense_team', 'defense_logos', 'defense_score',
        'exp_wp_go', 'exp_wp_fg', 'exp_wp_punt',
        'time','pregame_offense_elo','pregame_defense_elo','down','distance','yards_to_goal', 
        'recommendation', 'decision', 'play_text'
    ]

    rename_dict = {
        'season': 'Season',
        'week': 'Week',
        'offense_team': 'Offense Team',
        'offense_conference': 'Offense Conference',
        'offense_division': 'Offense Division',
        'offense_logos': 'Offense Logo',
        'offense_score': 'Offense Score',
        'coach_name': 'Offense Coach Name',
        'defense_team': 'Defense Team',
        'defense_logos': 'Defense Logo',
        'defense_score': 'Defense Score',
        'wp_diff': 'Win Probability Diff',
        'time': 'Time',
        'pregame_offense_elo': 'Pregame Offense Elo',
        'pregame_defense_elo': 'Pregame Defense Elo',
        'down': 'Down',
        'distance': 'Distance',
        'yards_to_goal': 'Yards to Goal',
        'recommendation': 'Recommendation',
        'decision': 'Decision',
        'play_text': 'Desc',
        'exp_wp_go': 'Win Probability Go',
        'exp_wp_fg': 'Win Probability Field Goal',
        'exp_wp_punt': 'Win Probability Punt',
    }

    data_decisions = (
        data.query('offense_division == "fbs"')
        .assign(
            pregame_defense_elo=lambda x: x['pregame_defense_elo'].astype(int),
            pregame_offense_elo=lambda x: x['pregame_offense_elo'].astype(int),
        )
        [cols]
        .rename(columns=rename_dict)
        .drop(columns=['Offense Division'])
        .assign(
            Decision=lambda x: np.select([x.Decision=="field_goal", x.Decision=="go", x.Decision=="punt"],
                ['Field Goal', 'Go', 'Punt'],
                default='Unknown'
            ),
        )
    )        

    down_map = {
        1: '1st',
        2: '2nd',
        3: '3rd',
        4: '4th'
    }

    data_decisions['Down & Distance'] = (
        data_decisions['Down'].map(down_map) + ' & ' + data_decisions['Distance'].astype(str)
    )

    data_decisions.to_parquet(
        join(APP_DATA_DIR, 'game_decisions.parquet'),
        index=False,
    )