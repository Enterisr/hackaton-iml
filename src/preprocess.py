from utils import read_jsonl, score_team_on_season    
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split

INTRESTING_FEATURES = ['win', 'home', 'numMinutes', 'points', 'assists', 'blocks', 'steals', 
                      'fieldGoalsAttempted', 'fieldGoalsMade', 'fieldGoalsPercentage', 
                      'threePointersAttempted', 'threePointersMade', 'threePointersPercentage', 
                      'freeThrowsAttempted', 'freeThrowsMade', 'freeThrowsPercentage', 
                      'reboundsDefensive', 'reboundsOffensive', 'reboundsTotal', 
                      'foulsPersonal', 'turnovers', 'plusMinusPoints']

def preprocess(player_stats):
    """
    Extracts features from a player's stats dataframe.
    
    Args:
        player_stats (pd.DataFrame): DataFrame containing a player's statistics
        
    Returns:
        list: Feature vector with values for each feature in INTRESTING_FEATURES
    """
    feats = []
    for col in INTRESTING_FEATURES:
        if col in player_stats:
            v = player_stats[col].mean()
            feats.append(float(v) if pd.notnull(v) else 0.0)
        else:
            feats.append(0.0)
    return feats

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
    seasons =read_jsonl(path_to_data)
    seaons_procceseed = []
    for season in seasons:
        data = pd.DataFrame(season['input']['last season'])
        _print_df_summary(data, "Initial")

        data = _coerce_numerical_types(data)
        data = _fill_initial_nans(data)
        data = _correct_negative_values_to_zero(data)
        data = _correct_percentage_discrepancies(data)
        data = _engineer_per_minute_stats(data)
        data = _standardize_team_names(data)

        _print_df_summary(data, "Processed (before filtering)")
        print("--- Initial Preprocessing Complete ---")
        seaons_procceseed.append(data)
    return seaons_procceseed

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


def pre_proc_orchestrator(instances):
    """
    Process all seasons data for quality and consistency.
    
    Args:
        instances (list): List of instances, each containing season data
        
    Returns:
        list: List of processed instances with cleaned data
    """
    processed_instances = []
    
    for inst in instances:
        # Process last season data
        if "input" in inst and "last season" in inst["input"]:
            last_season_df = inst["input"]["last season"]
            if isinstance(last_season_df, pd.DataFrame):
                last_season_df = _coerce_numerical_types(last_season_df)
                last_season_df = _fill_initial_nans(last_season_df)
                last_season_df = _correct_negative_values_to_zero(last_season_df)
                last_season_df = _correct_percentage_discrepancies(last_season_df)
                last_season_df = _engineer_per_minute_stats(last_season_df)
                last_season_df = _standardize_team_names(last_season_df)
                last_season_df = _cap_minutes_played(last_season_df)
                last_season_df = _validate_win_column(last_season_df)
                last_season_df = _validate_rebounds_consistency(last_season_df)
                inst["input"]["last season"] = last_season_df
        
        # Process next season data if available
        if "next season" in inst:
            next_season_df = inst["next season"]
            if isinstance(next_season_df, pd.DataFrame):
                next_season_df = _coerce_numerical_types(next_season_df)
                next_season_df = _fill_initial_nans(next_season_df)
                next_season_df = _correct_negative_values_to_zero(next_season_df)
                next_season_df = _correct_percentage_discrepancies(next_season_df)
                next_season_df = _engineer_per_minute_stats(next_season_df)
                next_season_df = _standardize_team_names(next_season_df)
                next_season_df = _cap_minutes_played(next_season_df)
                next_season_df = _validate_win_column(next_season_df)
                next_season_df = _validate_rebounds_consistency(next_season_df)
                inst["next season"] = next_season_df
        
        processed_instances.append(inst)
        
    return processed_instances

def split_train_test(instances, test_size=0.2, random_state=42):
    """
    Splits player data within each season into train and test sets.
    
    Args:
        file_path (str): Path to the JSONL file containing season data
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (train_X, train_y, test_X, test_y, test_meta)
    """
    
    
    # Process the instances for data quality
    processed_instances = pre_proc_orchestrator(instances)
    
    X_train, y_train, X_test, y_test = [], [], [], []
    test_meta = []  # (season_idx, player_id)

    for season_idx, inst in enumerate(processed_instances):
        last_season = inst["input"]["last season"]
        next_season = inst["next season"]
        draft_class = inst["input"]["draft class"]

        features = []
        targets = []
        pids = []
        
        for pid in draft_class:
            player_stats = last_season[last_season.personId == pid]
            if player_stats.empty:
                feat = [0] * len(INTRESTING_FEATURES)
            else:
                # Extract features using the preprocess function
                feat = preprocess(player_stats)
            
            features.append(feat)
            # Calculate target score for this player
            targets.append(score_team_on_season([pid], next_season))
            pids.append(pid)

        # Split players for this season
        if len(features) > 1:
            X_tr, X_te, y_tr, y_te, pids_tr, pids_te = train_test_split(
                features, targets, pids, test_size=test_size, random_state=random_state
            )
        else:
            X_tr, X_te, y_tr, y_te, pids_tr, pids_te = features, [], targets, [], pids, []

        # Print mean features for this season's training set
        if X_tr:
            print(f"Season {season_idx} train set mean features:")
            mean_features = np.mean(X_tr, axis=0)
            for i, feat_name in enumerate(INTRESTING_FEATURES):
                print(f"  {feat_name}: {mean_features[i]:.4f}")
            
        X_train.extend(X_tr)
        y_train.extend(y_tr)
        X_test.extend(X_te)
        y_test.extend(y_te)
        test_meta.extend([(season_idx, pid) for pid in pids_te])

    # Print overall mean features for the complete training set
    if X_train:
        print("\nOverall train set mean features:")
        overall_mean = np.mean(X_train, axis=0)
        for i, feat_name in enumerate(INTRESTING_FEATURES):
            print(f"  {feat_name}: {overall_mean[i]:.4f}")

    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test), test_meta

def process_unlabeled_data(instances):
    """
    Processes player data from instances without labels (no next season data).
    
    Args:
        instances (list): List of instances, each containing only current season data
        
    Returns:
        tuple: (features, meta_data)
            - features: numpy array of player features
            - meta_data: list of tuples (season_idx, player_id)
    """
    # Process the instances for data quality
    processed_instances = pre_proc_orchestrator(instances)
    
    all_features = []
    meta_data = []  # (season_idx, player_id)

    for season_idx, inst in enumerate(processed_instances):
        last_season = inst["input"]["last season"]
        draft_class = inst["input"]["draft class"]

        season_features = []
        season_pids = []
        
        for pid in draft_class:
            player_stats = last_season[last_season.personId == pid]
            if player_stats.empty:
                feat = [0] * len(INTRESTING_FEATURES)
            else:
                # Extract features using the preprocess function
                feat = preprocess(player_stats)
            
            season_features.append(feat)
            season_pids.append(pid)
        
        # Print mean features for this season
        if season_features:
            print(f"Season {season_idx} mean features:")
            mean_features = np.mean(season_features, axis=0)
            for i, feat_name in enumerate(INTRESTING_FEATURES):
                print(f"  {feat_name}: {mean_features[i]:.4f}")
        
        all_features.extend(season_features)
        meta_data.extend([(season_idx, pid) for pid in season_pids])

    # Print overall mean features
    if all_features:
        print("\nOverall mean features:")
        overall_mean = np.mean(all_features, axis=0)
        for i, feat_name in enumerate(INTRESTING_FEATURES):
            print(f"  {feat_name}: {overall_mean[i]:.4f}")

    return np.array(all_features), meta_data