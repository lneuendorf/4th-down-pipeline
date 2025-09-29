import pandas as pd
import argparse

import recommender
from data_loader.data_loader import load_games, load_teams, load_coaches
from postprocessing.postprocessing import postprocess

def get_recommendations_for_week(
    year: int, 
    week: int, 
    season_type: str, 
    force_data_update: bool = False
) -> None:
    result = recommender.generate_recommendations(year, week, season_type, force_data_update)
    return result

def get_recommendations_for_season_range(
    start_year: int, 
    end_year: int, 
    force_data_update: bool = False
) -> None:
    
    # Fetch all weeks of each season in the range
    games = []
    for year in range(start_year, end_year + 1):
        games.append(load_games(year, force_data_update=force_data_update))
        cols = ['season','week','season_type']
    games = (
        pd.concat(games, ignore_index=True)
        .query('completed == True and season_type.isin(["regular", "postseason"])')
        .query('home_division == "fbs" or away_division == "fbs"')
        [cols]
        .drop_duplicates(ignore_index=True)
    )

    results = []
    for _, row in games.iterrows():
        results.append(
            get_recommendations_for_week(
                year=row['season'], 
                week=row['week'], 
                season_type=row['season_type'], 
                force_data_update=force_data_update
            )
        )
    results = pd.concat(results, ignore_index=True)

    postprocess_all_recommendations(results, start_year, end_year)

def postprocess_all_recommendations(
        results: pd.DataFrame,
        start_year: int = None,
        end_year: int = None
    ) -> None:
    teams = []
    coaches = []
    for year in range(start_year, end_year + 1):
        teams.append(load_teams(year))
        coaches.append(load_coaches(year))
    teams = pd.concat(teams, ignore_index=True)
    coaches = pd.concat(coaches, ignore_index=True)

    results = postprocess(results, teams, coaches)

def main():
    parser = argparse.ArgumentParser(description='Fourth Down Pipeline Jobs')
    parser.add_argument('--jobname', required=True, choices=['week', 'season_range'], 
                       help='Name of the job to run')
    
    # Arguments for week job
    parser.add_argument('--year', type=int, help='Year for week job')
    parser.add_argument('--week', type=int, help='Week number for week job')
    parser.add_argument('--season_type', choices=['regular', 'postseason'], help='Season type for week job')
    
    # Arguments for season_range job
    parser.add_argument('--start_year', type=int, help='Start year for season_range job')
    parser.add_argument('--end_year', type=int, help='End year for season_range job')
    
    # Common argument
    parser.add_argument('--force', action='store_true', help='Force data update')
    
    args = parser.parse_args()
    
    if args.jobname == "week":
        if not all([args.year, args.week, args.season_type]):
            parser.error("--jobname week requires --year, --week, and --season_type")
        
        get_recommendations_for_week(
            year=args.year,
            week=args.week,
            season_type=args.season_type,
            force_data_update=args.force
        )
    
    elif args.jobname == "season_range":
        if not all([args.start_year, args.end_year]):
            parser.error("--jobname season_range requires --start_year and --end_year")
        
        get_recommendations_for_season_range(
            start_year=args.start_year,
            end_year=args.end_year,
            force_data_update=args.force
        )
    
    else:
        raise ValueError(f"Unknown job name: {args.jobname}")

if __name__ == "__main__":
    main()