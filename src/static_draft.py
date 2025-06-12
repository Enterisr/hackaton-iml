""" Usage:
    <file-name> --in=INPUT_FILE --model=MODEL_NAME --out=OUTPUT_FILE [--debug]
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
from preprocess import process_unlabeled_data, split_train_test,pre_proc_orchestrator
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

def load_model_and_train(gold_path,model_name):
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
    # Use your custom LinearRegression model
    model = model_router[model_name](f"./weights_{model_name}/weights")
    model.fit(X_train, y_train,f"./weights_{model_name}/weights")
    return model,gold_insts,X_test,y_test,test_meta
def use_model(inp_fn, model_name):
    """
    Load model, process unlabeled data, and make predictions.
    
    Args:
        inp_fn (Path): Path to input data file
        model_name (str): Name of the model to use
    
    Returns:
        BaseModel: Trained model instance
    """
    # Check if input path is provided
    if not inp_fn:
        logging.error("No input instances provided. Use --in=INPUT_FILE")
        exit(1)
    
    # Load input instances
    from utils import read_jsonl
    input_insts = read_jsonl(inp_fn)
    
    # Process the data for prediction
    features, meta_data = process_unlabeled_data(input_insts)
    
    # Initialize the model
    model = model_router[model_name](f"./weights_{model_name}/weights")
    
    # Make predictions
    predictions = model.predict({'X': features})
    
    # Group predictions by season
    season_pred_dict = defaultdict(list)
    for (season_idx, pid), pred_score in zip(meta_data, predictions):
        season_pred_dict[season_idx].append((pid, pred_score))
    
    # Build the prediction instances
    pred_insts = []
    for season_idx in sorted(season_pred_dict.keys()):
        # Sort players by predicted score (descending)
        sorted_players = [pid for pid, _ in sorted(season_pred_dict[season_idx], key=lambda x: -x[1])]
        # Copy input from input_insts
        pred_insts.append({
            "input": input_insts[season_idx]["input"],
            "output": sorted_players[:PLAYERS_TO_CHOOSE]
        })
    
    return  model,pred_insts

def debug_model(inp_fn,model_name):
    model,gold_insts,X_test,y_test,test_meta = load_model_and_train(inp_fn,model_name,is_debug)
    predictions = model.predict({'X': X_test})
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
    return model,pred_insts
    
model_router = {
    "linear": LinearRegression,
    "random":RandomStaticBaseline,
    #"adaboost":AdaBoostRegressor
 #   "randomForest":RandomForestRegressor
}

if __name__ == "__main__":
    args = docopt(__doc__)
    inp_fn = Path(args["--in"]) if args["--in"] else None
    model_name = args["--model"]
    out_folder = Path(args["--out"]) if args["--out"] else Path("./tmp.out")
    is_debug = args["--debug"]
    if is_debug:           
        model,pred_insts = debug_model(inp_fn,model_name)
    else:   
        model,pred_insts = use_model(inp_fn,model_name)
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
        

    # Write to file
    # Convert any non-serializable objects to serializable format
    clean_pred_insts = []
    for inst in pred_insts:
        clean_inst = {}
        
        # Handle input field
        input_data = inst["input"]
        if isinstance(input_data, dict):
            clean_input = {}
            for k, v in input_data.items():
                if isinstance(v, pd.DataFrame):
                    # Convert DataFrame to a JSON string instead of dict of records
                    clean_input[k] = v.to_json(orient='records')
                elif isinstance(v, np.ndarray):
                    clean_input[k] = v.tolist()
                else:
                    clean_input[k] = v
            clean_inst["input"] = clean_input
        else:
            clean_inst["input"] = input_data
            
        # Handle output field
        output_data = inst["output"]
        if isinstance(output_data, (pd.DataFrame, np.ndarray)):
            clean_inst["output"] = output_data.tolist() if isinstance(output_data, np.ndarray) else output_data.to_json(orient='records')
        else:
            clean_inst["output"] = output_data
            
        clean_pred_insts.append(clean_inst)
    # Now convert to JSON
    out_str = "\n".join(json.dumps(inst) for inst in clean_pred_insts)
    out_fn = out_folder / f"{model.name}.jsonl"
    logging.info(f"writing to {out_fn}")
    os.makedirs(out_fn.parent,exist_ok=True)
    with open(out_fn, "w", encoding="utf8") as fout:
        fout.write(out_str)

