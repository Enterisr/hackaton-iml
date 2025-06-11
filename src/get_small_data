from utils import read_jsonl
import json
import os
import random
import pandas as pd

def extract_and_save_sample_seasons(num_seasons=3, output_dir="sample_seasons"):
    """
    Extract a sample of seasons from the training data and save in regular JSON format.
    
    Args:
        num_seasons (int): Number of seasons to extract (default is now 1)
        output_dir (str): Output directory path
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the training data
    file_path = "data/train_test_splits/train.jsonl"
    seasons = read_jsonl(file_path)
    
    print(f"Total seasons in dataset: {len(seasons)}")
    
    # Take just one season (or the specified number)
    if len(seasons) > num_seasons:
        sample_seasons = random.sample(seasons, num_seasons)
    else:
        sample_seasons = seasons[:num_seasons]
        
    print(f"Selected {len(sample_seasons)} season(s)")
    
    # Convert DataFrames to dictionaries for JSON serialization
    for i, season in enumerate(sample_seasons):
        # Select five random teams
        selected_team_names = random.sample(list(season['input']['teams'].keys()), 5)
        selected_teams = {team: season['input']['teams'][team] for team in selected_team_names}

        # Get all players from the selected teams
        selected_players = set()
        for team in selected_teams.values():
            selected_players.update(team)

        # Filter last season DataFrame to only include players from selected teams
        season['input']['last season'] = season['input']['last season'][
            season['input']['last season']['personId'].isin(selected_players)
        ]

        # Filter next season DataFrame to only include players from selected teams
        season['next season'] = season['next season'][
            season['next season']['personId'].isin(selected_players)
        ]

        # Filter draft class to only include players who are in the selected teams
        season['input']['draft class'] = [
            player for player in season['input']['draft class']
            if player in selected_players
        ]

        # Update teams to only include the selected teams
        season['input']['teams'] = selected_teams
        serializable_season = {
            
            'input': {
                "uid":season["input"]["uid"],
                'last season': season['input']['last season'].to_dict(orient='records'),
                'draft class': season['input']['draft class'],
                'teams': season['input']['teams']
            },
            'next season': season['next season'].to_dict(orient='records')
        }
        
        # Save each season as a separate JSON file
        output_file = os.path.join(output_dir, f"sample_season_{i+1}.json")
        with open(output_file, "w") as f:
            json.dump(serializable_season, f, indent=2)  # Pretty print with indentation
        
        print(f"Season {i+1} saved to {output_file}")
        
        # Print statistics about this season
        print(f"\n=== Season {i+1} Statistics ===")
        print(f"Last Season DataFrame Shape: {season['input']['last season'].shape}")
        print(f"Draft Class Count: {len(season['input']['draft class'])}")
        print(f"Number of Teams: {len(season['input']['teams'])}")
        print(f"Next Season DataFrame Shape: {season['next season'].shape}")
        
        # Print sample of the data structure
        print("\nSample of last season data (3 records):")
        print(json.dumps(serializable_season['input']['last season'][:3], indent=2))
        
        print("\nSample of draft class (5 players):")
        draft_sample = serializable_season['input']['draft class'][:5]
        print(json.dumps(draft_sample, indent=2))
        
        print("\nSample of teams (first 2):")
        teams_sample = dict(list(serializable_season['input']['teams'].items())[:2])
        print(json.dumps(teams_sample, indent=2))

def extract_and_save_combined_samples(num_seasons=3, output_dir="sample_data"):
    """
    Extract samples from multiple seasons using the same teams and save to a single JSON file.
    
    Args:
        num_seasons (int): Number of seasons to extract (default is 3)
        output_dir (str): Output directory path
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the training data
    file_path = "data/train_test_splits/train.jsonl"
    seasons = read_jsonl(file_path)
    
    print(f"Total seasons in dataset: {len(seasons)}")
    
    # Take the specified number of seasons
    if len(seasons) > num_seasons:
        sample_seasons = random.sample(seasons, num_seasons)
    else:
        sample_seasons = seasons[:num_seasons]
        
    print(f"Selected {len(sample_seasons)} season(s)")
    
    # Select five random teams from the first season to use across all seasons
    first_season_teams = list(sample_seasons[0]['input']['teams'].keys())
    selected_team_names = random.sample(first_season_teams, 5)
    print(f"Selected teams: {selected_team_names}")
    
    combined_data = {
        "seasons": []
    }
    
    # Process each season
    for i, season in enumerate(sample_seasons):
        # Filter teams to only include the selected teams
        # Note: Some teams might not exist in all seasons, so we need to check
        selected_teams = {}
        for team in selected_team_names:
            if team in season['input']['teams']:
                selected_teams[team] = season['input']['teams'][team]
        
        # Get all players from the selected teams
        selected_players = set()
        for team in selected_teams.values():
            selected_players.update(team)
        
        # Filter last season DataFrame to only include players from selected teams
        filtered_last_season = season['input']['last season'][
            season['input']['last season']['personId'].isin(selected_players)
        ]
        
        # Filter next season DataFrame to only include players from selected teams
        filtered_next_season = season['next season'][
            season['next season']['personId'].isin(selected_players)
        ]
        
        # Filter draft class to only include players who are in the selected teams
        filtered_draft_class = [
            player for player in season['input']['draft class']
            if player in selected_players
        ]
        
        # Add this season's data to the combined structure
        season_data = {
            'input': {
                'uid': season["input"]["uid"],
                'last season': filtered_last_season.to_dict(orient='records'),
                'draft class': filtered_draft_class,
                'teams': selected_teams
                
            },
            'next season': filtered_next_season.to_dict(orient='records')
        }
        
        combined_data["seasons"].append(season_data)
        
        # Print statistics about this season
        print(f"\n=== Season {i+1} Statistics ===")
        print(f"Last Season DataFrame Shape: {filtered_last_season.shape}")
        print(f"Draft Class Count: {len(filtered_draft_class)}")
        print(f"Number of Teams: {len(selected_teams)}")
        print(f"Next Season DataFrame Shape: {filtered_next_season.shape}")
    
    # Save the combined data to a single JSON file
    output_file = os.path.join(output_dir, "combined_seasons.json")
    with open(output_file, "w") as f:
        json.dump(combined_data, f, indent=2)
        
    print(f"\nAll {num_seasons} seasons combined and saved to {output_file}")
    
    # Print sample of the combined data structure
    print("\nSample of the combined data structure:")
    if combined_data["seasons"]:
        first_season = combined_data["seasons"][0]
        print(f"Number of seasons: {len(combined_data['seasons'])}")
        print(f"First season teams: {list(first_season['input']['teams'].keys())}")
        
        if first_season['input']['last season']:
            print("\nSample of last season data (first record):")
            print(json.dumps(first_season['input']['last season'][0], indent=2))

if __name__ == "__main__":
    extract_and_save_sample_seasons(5)  # Extract and save five seasons separately
    extract_and_save_combined_samples(5)  # Extract five seasons and combine them
    
    