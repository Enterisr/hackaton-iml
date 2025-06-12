import json
import pandas as pd
import random
from utils import score_team_on_season, score_prediction

# Load the gold data (actual outcomes)
with open('/home/or_israeli/hacktaon-iml/iml-hack-nba-draft/sample_data/combined_seasons.json', 'r') as f:
    combined_data = json.load(f)
    all_gold_instances = combined_data["seasons"]  # Get all seasons array

# Create prediction instances and filter gold instances accordingly
prediction_instances = []
filtered_gold_instances = []  # Only keep gold instances that have valid predictions

for gold_inst in all_gold_instances:
    if 'input' not in gold_inst or 'last season' not in gold_inst['input']:
        continue  # Skip entries without required structure
        
    # Get the last season data
    last_season_df = pd.DataFrame(gold_inst['input']['last season'])
    
    # Skip if last_season is empty
    if last_season_df.empty:
        continue
    
    # Get all player IDs from the last season
    all_players = last_season_df['personId'].unique().tolist()
    
    # If we don't have enough players, skip this instance
    if len(all_players) < 10:
        continue
    
    # Select 10 random players as our draft
    drafted_players = random.sample(all_players, 10)
    
    # Create the prediction instance with the same UID
    pred_inst = {
        "input": {"uid": gold_inst['input']['uid']},
        "output": drafted_players
    }
    
    # Add to our lists
    prediction_instances.append(pred_inst)
    filtered_gold_instances.append(gold_inst)  # Keep track of gold instances we're actually using

# Score the predictions (only if we have valid predictions)
if prediction_instances:
    # Use filtered gold instances that correspond to our predictions
    score, individual_scores = score_prediction(filtered_gold_instances, prediction_instances)
    print(f"Total score: {score}")
    print(f"Individual scores: {individual_scores}")
else:
    print("No valid predictions could be created from the data")

# Load the combined seasons data
with open('/home/or_israeli/hacktaon-iml/iml-hack-nba-draft/sample_data/combined_seasons.json', 'r') as f:
    combined_seasons = json.load(f)["seasons"]

# Choose a season to evaluate
season_id = "0dbb66da9c" 
next_season_id = "828444e8b8"

# Convert to pandas DataFrames
season_df = pd.DataFrame(combined_seasons[season_id])
next_season_df = pd.DataFrame(combined_seasons[next_season_id])

# Get a list of all player IDs from the current season
all_players = season_df['personId'].unique().tolist()

# Select 10 random players for our draft
random_draft = random.sample(all_players, 10)

# Evaluate how this random draft would perform in the next season
draft_score = score_team_on_season(random_draft, next_season_df)
print(f"Random draft score: {draft_score}")

# Let's select another random set of players
random_draft_2 = random.sample(all_players, 10)
draft_score_2 = score_team_on_season(random_draft_2, next_season_df)
print(f"Another random draft score: {draft_score_2}")

# Compare with a third random draft
random_draft_3 = random.sample(all_players, 10)
draft_score_3 = score_team_on_season(random_draft_3, next_season_df)
print(f"Third random draft score: {draft_score_3}")

# Find which random draft performed best
best_score = max(draft_score, draft_score_2, draft_score_3)
print(f"Best random draft score: {best_score}")

