""" Usage:
    <file-name> --gold=INPUT_FILE [--debug]
"""
# External imports
import logging
import pdb
from pprint import pprint
from pprint import pformat
from docopt import docopt
from pathlib import Path
from tqdm import tqdm
import random
import json
import pandas as pd
from io import StringIO
from collections import defaultdict
import json
import os
import numpy as np
# Remove sklearn import and use your own implementation
# from sklearn.linear_model import LinearRegression
                 
# Local imports
from utils import NUM_OF_DRAFTED_PLAYERS, BaseModel, score_prediction
from preprocess import split_train_test,pre_proc_orchestrator
# Import your custom LinearRegression
from LinearRegression import LinearRegression

#----
PLAYERS_TO_CHOOSE = 10

def average_stats_matrix(data):
    exclude_keys = {
        "gameId", "playerteamName", "opponentteamName",
        "season", "next season uid"
    }

    player_entries = defaultdict(list)
    for entry in data:
        player_entries[entry["personId"]].append(entry)

    rows = []
    all_keys = set()

    for person_id, entries in player_entries.items():
        sums = defaultdict(float)
        count = len(entries)

        for entry in entries:
            for key, value in entry.items():
                if key not in exclude_keys and isinstance(value, (int, float)):
                    sums[key] += value
                    all_keys.add(key)

        row = {"personId": person_id}
        for key in all_keys:
            if key != "personId":
                row[key] = sums[key] / count
        rows.append(row)

    sorted_columns = ["personId"] + sorted(k for k in all_keys if k != "personId")
    df = pd.DataFrame(rows)[sorted_columns]
    return df



class RandomStaticBaseline(BaseModel):
    """
    Simple baseline container
    """
    def __init__(self):
        super().__init__()



    def predict(self, input_json):
        """
        Make a random full draft.
        """
        draft_class = input_json["draft class"]
        random_static_draft = random.sample(draft_class, NUM_OF_DRAFTED_PLAYERS)
        return random_static_draft


# model_router = {
#     "random": RandomStaticBaseline,
# }
    


if __name__ == "__main__":
    args = docopt(__doc__)
    gold_path = args["--gold"] if args["--gold"] else None
    out_folder = Path("./tmp.out")
    
    # Check if gold path is provided
    if not gold_path:
        logging.error("No gold instances provided. Use --gold=INPUT_FILE")
        exit(1)
    
    # Load gold instances for later use
    from utils import read_jsonl
    gold_insts = read_jsonl(gold_path)
    gold_insts = pre_proc_orchestrator(gold_insts)
    
    # Split gold_insts into train and test sets
    X_train, y_train, X_test, y_test, test_meta = split_train_test(gold_insts, test_size=0.2, random_state=42)

    ##use your custom random forest model
    # #Fit RandomForestRegression model
    # model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=8, min_samples_split=8,
    #                               min_samples_leaf=4, max_features='sqrt')
    # model.fit(X_train, y_train)
    # predictions = model.predict(X_test)

    #Fit adaboost
    # model = AdaBoostRegressor(
    #     n_estimators=100,
    #     learning_rate=1.0,
    #     random_state=42
    # )
    # model.fit(X_train, y_train)
    # predictions = model.predict(X_test)

    # Use your custom LinearRegression model
    model = LinearRegression()
    model.fit(X_train, y_train)
 
    predictions = model.predict({'X': X_test})  # Use the format expected by your custom class

    assert(len(y_test) == len(predictions))

    # --- Convert predictions to season-level structure for score_prediction ---
    # Group predictions by season
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
    print("Selected players for last season: " + str(sorted_players[:PLAYERS_TO_CHOOSE]))




    # # Determine logging level
    # debug = args["--debug"]
    # if debug:
    #     logging.basicConfig(level = logging.DEBUG)
    # else:
    #     logging.basicConfig(level = logging.INFO)
    #
    # # determine model
    # model = model_router[model_name]()
    #
    # # Read all test instances
    # test_insts = [json.loads(line) for line in open(inp_fn, encoding = "utf8")]
    #
    # # Make predictions
    # for test_inst in tqdm(test_insts):
    #     test_inst["output"] = model.predict(test_inst["input"])
    #
    # # Write to file
    # out_str = "\n".join(map(json.dumps, test_insts))
    # out_fn = out_folder / f"{model.name}.jsonl"
    # logging.info(f"writing to {out_fn}")
    # with open(out_fn, "w", encoding = "utf8") as fout:
    #     fout.write(out_str)
    #
    # # End
    # logging.info("DONE")
