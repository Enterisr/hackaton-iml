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
#from sklearn import LinearRegression
from collections import defaultdict
import json
import os
import numpy as np
from sklearn.linear_model import LinearRegression
from utils import pre_proc_orchestrator
                 
# Local imports
from utils import NUM_OF_DRAFTED_PLAYERS, BaseModel
#----

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


model_router = {
    "random": RandomStaticBaseline,
}
    


if __name__ == "__main__":
    file_path = os.path.join(os.path.dirname(__file__), 'sample_season_1.json')
    processed_df = pre_proc_orchestrator(file_path)
    print("\nSuccessfully processed data (first 5 rows):")
    print(processed_df.head())
    print("\nFinal shape of filtered DataFrame:", processed_df.shape)

    # file_path = os.path.join(os.path.dirname(__file__), 'train.sample.jsonl')
    # with open(file_path, 'r', encoding='utf-8') as f:
    #     data = json.load(f)
    #     data = data["input"]["last season"]
    #     data_y= data["input"]["next season"]

    # matrix_x = average_stats_matrix(data)
    # matrix_y = average_stats_matrix(data_y)
    # print(f'matrix_X is: {matrix_x.head(3)}')
    # print(f'matrix_y is: {matrix_y.head(3)}')


    # Parse command line arguments
    # args = docopt(__doc__)
    # inp_fn = Path(args["--in"]) if args["--in"] else None
    # model_name = args["--model"]
    # out_folder = Path(args["--out"]) if args["--out"] else Path("./tmp.out")
    #
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
