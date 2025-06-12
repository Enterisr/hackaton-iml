""" Usage:
    <file-name> --gold=GOLD_FILE [--debug]
"""
# External imports
import logging
import pdb
from pprint import pprint
from pprint import pformat
from docopt import docopt
from pathlib import Path
from tqdm import tqdm
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Local imports
from utils import read_jsonl, score_prediction
from preprocess import preprocess, split_train_test
from LinearRegression import LinearRegression
PLAYERS_TO_CHOOSE = 10
if __name__ == "__main__":
    # Parse command line arguments
    args = docopt(__doc__)
    gold_fn = Path(args["--gold"]) 

    # Determine logging level
    debug = args["--debug"]
    if debug:
        logging.basicConfig(level = logging.DEBUG)
    else:
        logging.basicConfig(level = logging.INFO)
    # Start computation
    gold_insts = read_jsonl(gold_fn)

    # Split gold_insts into train and test sets
    X_train, y_train, X_test, y_test, test_meta = split_train_test(gold_insts, test_size=0.2, random_state=42)

    # Fit LinearRegression model using train_insts
    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict({'X': X_test})

    assert(len(y_test) == len(predictions))

    # --- Convert predictions to season-level structure for score_prediction ---
    # Group predictions by season
    from collections import defaultdict
    season_pred_dict = defaultdict(list)
    for (season_idx, pid), pred_score in zip(test_meta, predictions):
        season_pred_dict[season_idx].append((pid, pred_score))

    pred_insts = []
    for season_idx in sorted(season_pred_dict.keys()):
        # Sort players by predicted score (descending)
        sorted_players = [pid for pid, _ in sorted(season_pred_dict[season_idx], key=lambda x: -x[1])]
        # Copy input and uid from gold_insts
        pred_insts.append({
            "input": gold_insts[season_idx]["input"],
            "output": sorted_players[:PLAYERS_TO_CHOOSE]
        })

    # Only keep test seasons for gold_insts
    test_season_idxs = sorted(season_pred_dict.keys())
    gold_test_insts = [gold_insts[i] for i in test_season_idxs]

    score, scores = score_prediction(gold_test_insts, pred_insts)
    logging.info(f"{scores} \n Combined score: {score}")
    print("selected players for last season: " +str(sorted_players[:PLAYERS_TO_CHOOSE]))

