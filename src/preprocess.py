from utils import read_jsonl    
import numpy as np

def preprocess(player_stats):
    # Group by personId first to get one row per player
    if len(player_stats) > 1:
        player_stats = player_stats.groupby('personId').agg('median').reset_index()
    
    # Select only numeric columns
    numeric_cols = player_stats.select_dtypes(include=[np.number]).columns
    # Drop personId and gameId columns if they exist in numeric_cols
    numeric_cols = [col for col in numeric_cols if col not in ['personId', 'gameId']]
    
    # Get feature vector
    feat = player_stats[numeric_cols].values.tolist()[0]
    return feat

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
                numeric_cols = last_season.select_dtypes(include=[np.number]).columns
                feat = [0] * len(numeric_cols)
            else:
               feat =  preprocess(player_stats)
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

        X_train.extend(X_tr)
        y_train.extend(y_tr)
        X_test.extend(X_te)
        y_test.extend(y_te)
        test_meta.extend([(season_idx, pid) for pid in pids_te])

    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test), test_meta
