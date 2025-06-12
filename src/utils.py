import pdb
from io import StringIO
import pandas as pd
import json
from collections import defaultdict
import hashlib
import numpy as np
from tqdm import tqdm
import os

player_categories = {
    'fieldGoalsAttempted': {"weight": -1},
    'fieldGoalsMade': {"weight": 2},
    'threePointersAttempted': {"weight": -0.5},
    'threePointersMade': {"weight": 3},
    'freeThrowsAttempted': {"weight": -0.5},
    'freeThrowsMade': {"weight": 1},
    'reboundsDefensive': {"weight": 1},
    'reboundsOffensive': {"weight": 1.5},
    'turnovers': {"weight": -2},
    "win": {"weight": 1}
}

DEFAULT_VALUES_MAP = {
        'numMinutes': 0.0,
        'points': 0,
        'assists': 0,
        'blocks': 0,
        'steals': 0,
        'fieldGoalsAttempted': 0,
        'fieldGoalsMade': 0,
        'fieldGoalsPercentage': 0.0,
        'threePointersAttempted': 0,
        'threePointersMade': 0,
        'threePointersPercentage': 0.0,
        'freeThrowsAttempted': 0,
        'freeThrowsMade': 0,
        'freeThrowsPercentage': 0.0,
        'reboundsDefensive': 0,
        'reboundsOffensive': 0,
        'reboundsTotal': 0,
        'foulsPersonal': 0,
        'turnovers': 0
    }

NUM_OF_DRAFTED_PLAYERS = len(player_categories)

def hash_id(s):
    return hashlib.sha256(s.encode()).hexdigest()[:10] 

def read_jsonl(inp_fn):
    """
    return a json lines format from a file
    """
    lines = []
    for line in tqdm(open(inp_fn, encoding = "utf8"), desc = f"reading {inp_fn}"):
        cur_dict = json.loads(line)
        cur_dict["input"]["last season"] = pd.read_json(StringIO(cur_dict["input"]["last season"]))
        if "next season" in cur_dict:
            cur_dict["next season"] = pd.read_json(StringIO(cur_dict["next season"]))
        lines.append(cur_dict)
    return lines


class BaseModel:
    """
    A base class for all models
    """
    counter = 0 # a static variable that counts number of models

    def __init__(self):
        BaseModel.counter += 1
        self.name = self.__class__.__name__ + f"_{self.counter}"

    def predict(self, input_json):
        """
        Implment this function in inheriting classes
        """
        raise NotImplementedError("Subclasses must implement this method.")

def season_by_category(season_df, category, weight):
    """
    Return a dictionary from player to their stats in a certain category
    """
    player_stats = season_df.groupby('personId')[category].sum() * weight
    return player_stats

def calc_game_started(season, pid):
    """
    Count the number of games in which a player played in
    a given season
    """
    games_started = len(season[season.personId == pid])
    return games_started


def score_team_on_season(team, season):
    """
    Get the score of a given team in a praticular season
    returns a dictionary of score by category.
    """

    cat_dicts = dict([(cat, season_by_category(season, cat, cat_dict["weight"]))
                      for cat, cat_dict in player_categories.items()])

    games_started = dict([(pid, calc_game_started(season, pid))
                           for pid in team])

    cat_dicts["gamesStarted"] = games_started
    score = 0
    for cat, cat_dict in cat_dicts.items():
        for pid in team:
            score += cat_dict[pid]

    return score


def score_prediction(gold_insts, pred_insts):
    """
    Score a predicted draft according to a gold reference
    according to all categories - each category gets a score between 0-1
    as a fraction of the performance of the top players in that category
    """
    scores = []
    num_of_seasons = len(gold_insts)
    assert(len(pred_insts) == num_of_seasons)
    
    for gold_inst, pred_inst in tqdm(zip(gold_insts, pred_insts)):
        # make sure that pred and gold refer to the same instance
        gold_uid = gold_inst["input"]["uid"]
        pred_uid = pred_inst["input"]["uid"]
        if gold_uid != pred_uid:
            raise Exception(f"""There's a mismatch between predicted and gold files.
            gold_uid: {gold_uid} 
            pred_uid: {pred_uid}""")


        season = gold_inst["next season"]
        predicted_draft = pred_inst["output"]
        cur_score = score_team_on_season(predicted_draft, season)
        scores.append(cur_score)

    # normalize by seasons
    score = np.average(scores)
        
    return score, scores



def get_rookies(season, draft_class):
    """
    Get a list of players appearing in the draft that did not
    appear in a given season, represented as dataframe
    """
    s1_players = set(season.personId)
    new_players = [pid for pid in draft_class
                   if pid not in s1_players]
    return new_players


def get_veterans(season, draft_class):
    """
    Get a list of players appearing in the draft that also
    appear in a given season, represented as a dataframe
    """
    s1_players = set(season.personId)
    
    old_players = [pid for pid in draft_class
                   if pid in s1_players]
    return old_players

#--------------preprocess checks------------------------------------------

def _load_json_data(path_to_data: str) -> pd.DataFrame:
    """Loads JSON data from the specified path into a pandas DataFrame."""
    with open(path_to_data, 'r') as f:
        full_data = json.load(f)
    
    if 'input' in full_data and 'last season' in full_data['input']:
        data = pd.DataFrame(full_data['input']['last season'])
    else:
        print("Warning: JSON structure 'input' -> 'last season' not found. Attempting direct load.")
        data = pd.read_json(path_to_data)
    return data

def _coerce_numerical_types(df: pd.DataFrame) -> pd.DataFrame:
    """Converts specified columns to numeric, coercing errors to NaN."""
    numerical_cols_to_coerce = [
        'numMinutes', 'points', 'assists', 'blocks', 'steals', 'fieldGoalsAttempted',
        'fieldGoalsMade', 'fieldGoalsPercentage', 'threePointersAttempted', 'threePointersMade',
        'threePointersPercentage', 'freeThrowsAttempted', 'freeThrowsMade', 'freeThrowsPercentage',
        'reboundsDefensive', 'reboundsOffensive', 'reboundsTotal', 'foulsPersonal', 'turnovers',
        'plusMinusPoints', 'win', 'home'
    ]
    for col in numerical_cols_to_coerce:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def _fill_initial_nans(df: pd.DataFrame) -> pd.DataFrame:
    """Fills NaN values with 0 across the DataFrame."""
    nan_count_before_fill = df.isnull().sum().sum()
    if nan_count_before_fill > 0:
        df.fillna(0, inplace=True)
        print(f"\nFilled {nan_count_before_fill} NaN values with 0 across the DataFrame.")
    return df

def _correct_negative_values_to_zero(df: pd.DataFrame) -> pd.DataFrame:
    """Sets negative values to 0 for logically non-negative numerical columns."""
    numerical_cols_to_correct_to_zero = [
        'numMinutes', 'points', 'assists', 'blocks', 'steals', 'fieldGoalsAttempted',
        'fieldGoalsMade', 'threePointersAttempted', 'threePointersMade',
        'freeThrowsAttempted', 'freeThrowsMade', 'reboundsDefensive',
        'reboundsOffensive', 'reboundsTotal', 'foulsPersonal', 'turnovers'
    ]
    for col in numerical_cols_to_correct_to_zero:
        if col in df.columns and (df[col] < 0).any():
            count_negative = (df[col] < 0).sum()
            print(f"  Correcting {count_negative} negative values in '{col}' to 0.")
            df.loc[df[col] < 0, col] = 0
    return df

def _correct_percentage_discrepancies(df: pd.DataFrame) -> pd.DataFrame:
    """Recalculates and corrects percentage fields based on attempts and made shots."""
    percentage_cols = ['fieldGoalsPercentage', 'threePointersPercentage', 'freeThrowsPercentage']
    attempt_cols = ['fieldGoalsAttempted', 'threePointersAttempted', 'freeThrowsAttempted']
    made_cols = ['fieldGoalsMade', 'threePointersMade', 'freeThrowsMade']

    for p_col, a_col, m_col in zip(percentage_cols, attempt_cols, made_cols):
        if p_col in df.columns and a_col in df.columns and m_col in df.columns:
            df.loc[df[a_col] == 0, p_col] = 0.0
            
            valid_attempts_mask = df[a_col] > 0
            calculated_pct = np.where(valid_attempts_mask, df[m_col] / df[a_col], 0.0)
            
            if not np.allclose(df.loc[valid_attempts_mask, p_col], calculated_pct[valid_attempts_mask], rtol=1e-05, atol=1e-08):
                print(f"Warning: Correcting discrepancies in '{p_col}'.")
                df.loc[valid_attempts_mask, p_col] = calculated_pct[valid_attempts_mask]
    return df

def _engineer_per_minute_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Adds per-minute statistical features."""
    df['numMinutes_for_per_minute_calc'] = df['numMinutes'].replace(0, np.nan)
    
    per_minute_stats = [
        'points', 'assists', 'blocks', 'steals', 'reboundsTotal', 'turnovers', 'foulsPersonal'
    ]
    for col in per_minute_stats:
        df[f'{col}_per_minute'] = df[col] / df['numMinutes_for_per_minute_calc']
        df[f'{col}_per_minute'] = df[f'{col}_per_minute'].fillna(0)

    df.drop(columns=['numMinutes_for_per_minute_calc'], inplace=True)
    return df

def _standardize_team_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardizes casing for team names."""
    if 'playerteamName' in df.columns:
        df['playerteamName'] = df['playerteamName'].str.title()
    if 'opponentteamName' in df.columns:
        df['opponentteamName'] = df['opponentteamName'].str.title()
    return df

def _cap_minutes_played(df: pd.DataFrame, max_minutes: float = 70.0) -> pd.DataFrame:
    """Caps numMinutes at a specified maximum threshold."""
    if 'numMinutes' in df.columns:
        count_above_max = (df['numMinutes'] > max_minutes).sum()
        if count_above_max > 0:
            print(f"  Capping {count_above_max} 'numMinutes' values above {max_minutes} to {max_minutes}.")
            df.loc[df['numMinutes'] > max_minutes, 'numMinutes'] = max_minutes
        
        count_zero_minutes = (df['numMinutes'] == 0).sum()
        if count_zero_minutes > 0:
            print(f"  Note: {count_zero_minutes} 'numMinutes' values are 0 (player did not play or were corrected from negative).")
    return df

def _validate_win_column(df: pd.DataFrame) -> pd.DataFrame:
    """Validates 'win' column values, rounding and clipping to 0 or 1."""
    if 'win' in df.columns:
        original_win_values = df['win'].copy()
        df['win'] = df['win'].astype(float)
        df['win'] = np.round(df['win'])
        df['win'] = np.clip(df['win'], 0, 1)
        df['win'] = df['win'].astype(int)
        
        changes_in_win = (original_win_values != df['win']).sum()
        if changes_in_win > 0:
            print(f"  Corrected {changes_in_win} 'win' values to nearest 0 or 1.")
    return df

def _validate_rebounds_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """Corrects 'reboundsTotal' if inconsistent with sum of defensive/offensive rebounds."""
    if 'reboundsTotal' in df.columns and 'reboundsDefensive' in df.columns and 'reboundsOffensive' in df.columns:
        df['reboundsDefensive'] = df['reboundsDefensive'].astype(float)
        df['reboundsOffensive'] = df['reboundsOffensive'].astype(float)
        df['reboundsTotal'] = df['reboundsTotal'].astype(float)

        inconsistent_rebounds_mask = ~np.isclose(
            df['reboundsTotal'],
            df['reboundsDefensive'] + df['reboundsOffensive'],
            rtol=1e-05, atol=1e-08
        )
        if inconsistent_rebounds_mask.any():
            count_inconsistent_rebounds = inconsistent_rebounds_mask.sum()
            print(f"  Correcting {count_inconsistent_rebounds} inconsistent 'reboundsTotal' entries.")
            df.loc[inconsistent_rebounds_mask, 'reboundsTotal'] = \
                df.loc[inconsistent_rebounds_mask, 'reboundsDefensive'] + \
                df.loc[inconsistent_rebounds_mask, 'reboundsOffensive']
    return df

def _print_df_summary(df: pd.DataFrame, stage_name: str):
    """Prints concise info and descriptive statistics for a DataFrame at a given stage."""
    print(f"\n--- {stage_name} Data Info ---")
    df.info()
    print(f"\n--- {stage_name} Descriptive Statistics ---")
    print(df.describe())
    print(f"\n--- {stage_name} Min/Max of Numerical Features ---")
    print(df.min(numeric_only=True))
    print(df.max(numeric_only=True))


# --- Main Orchestrating Functions ---

def load_and_initial_preprocess(path_to_data: str) -> pd.DataFrame:
    """
    Orchestrates the loading of data and initial preprocessing steps.
    """
    print("--- Starting Data Loading and Initial Preprocessing ---")
    try:
        data = _load_json_data(path_to_data)
        _print_df_summary(data, "Initial")

        data = _coerce_numerical_types(data)
        data = _fill_initial_nans(data)
        data = _correct_negative_values_to_zero(data)
        data = _correct_percentage_discrepancies(data)
        data = _engineer_per_minute_stats(data)
        data = _standardize_team_names(data)

        _print_df_summary(data, "Processed (before filtering)")
        print("--- Initial Preprocessing Complete ---")
        return data

    except Exception as e:
        print(f"An unexpected error occurred during initial preprocessing: {e}")
        return pd.DataFrame()

def apply_data_quality_checks(data_df: pd.DataFrame) -> pd.DataFrame:
    """
    Orchestrates the application of comprehensive data quality checks
    by modifying values in place rather than removing rows.
    """
    initial_rows = len(data_df)
    print(f"\n--- Starting Comprehensive Filtering (Value Correction Mode) ---")
    print(f"Initial number of rows: {initial_rows}")

    filtered_df = data_df.copy()
    filtered_df = _cap_minutes_played(filtered_df)
    filtered_df = _validate_win_column(filtered_df)
    filtered_df = _validate_rebounds_consistency(filtered_df)

    _print_df_summary(filtered_df, "Final Corrected")

    print(f"--- Comprehensive Filtering (Value Correction Mode) Complete ---")
    return filtered_df


def pre_proc_orchestrator(file_path):
    df = load_and_initial_preprocess(file_path)
    return apply_data_quality_checks(df)
    