from utils import read_jsonl    
import numpy as np
import pandas as pd 
INTRESTING_FEATURES =  ['win', 'home', 'numMinutes', 'points', 'assists', 'blocks', 'steals', 'fieldGoalsAttempted', 'fieldGoalsMade', 'fieldGoalsPercentage', 'threePointersAttempted', 'threePointersMade', 'threePointersPercentage', 'freeThrowsAttempted', 'freeThrowsMade', 'freeThrowsPercentage', 'reboundsDefensive', 'reboundsOffensive', 'reboundsTotal', 'foulsPersonal', 'turnovers', 'plusMinusPoints']
#replace with real big file
def preprocess(player_stats):
    feats = []
    for col in INTRESTING_FEATURES:
        if col in player_stats:
            v = player_stats[col].mean()
            feats.append(float(v) if pd.notnull(v) else 0.0)
        else:
            feats.append(0.0)
    return feats
def split_train_test(instances, test_size=0.2, random_state=42):
    """
    Splits player data within each season into train and test sets.
    Returns (train_X, train_y, test_X, test_y, test_meta)
    test_meta: list of (season_idx, player_id) for each test sample, in order.
    """
    from sklearn.model_selection import train_test_split
    from utils import score_team_on_season

    X_train, y_train, X_test, y_test = [], [], [], []
    test_meta = []  # (season_idx, player_id)

    for season_idx, inst in enumerate(instances):
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
               feat = preprocess(player_stats)
            features.append(feat)
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
