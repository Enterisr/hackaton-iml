""" Usage:
    <file-name> --in=INPUT_FILE --models=MODEL_NAME --out=OUTPUT_FILE [--debug]
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
                 
# Local imports
from utils import NUM_OF_DRAFTED_PLAYERS, BaseModel
#----


class RandomDynamicBaseline(BaseModel):
    """
    Random static assignement - no inner state.
    """
    def __init__(self):
        super().__init__()
    
    def predict(self, input_json):
        """
        Make a single random draft.
        Can also show the expected format for a prediction.
        """
        draft_class = input_json["draft class"]
        random_single_draft = random.sample(draft_class, 1)[0]
        return random_single_draft

model_router = {
    "random": RandomDynamicBaseline,
    # @TODO: Add here a prediction class for your own models following the same
    # input and output format as the provided baselines
}

if __name__ == "__main__":

    # Parse command line arguments
    args = docopt(__doc__)
    inp_fn = Path(args["--in"]) if args["--in"] else None
    model_names = args["--models"].split(",")
    out_folder = Path(args["--out"]) if args["--out"] else Path("./tmp.out")

    # Determine logging level
    debug = args["--debug"]
    if debug:
        logging.basicConfig(level = logging.DEBUG)
    else:
        logging.basicConfig(level = logging.INFO)

    # Read all test instances
    test_insts = [json.loads(line) for line in open(inp_fn, encoding = "utf8")]

    # Instansiate models
    models = [model_router[model_name]() for model_name in model_names]
    inst_names = [model.name for model in models]
    
    # Build snake draft - switching directions after each full round
    snake_draft_names = []
    for i in range(NUM_OF_DRAFTED_PLAYERS):
        snake_draft_names += inst_names[::1 - 2*(i % 2)]
    snake_draft_models = []
    for i in range(NUM_OF_DRAFTED_PLAYERS):
        snake_draft_models += models[::1 - 2*(i % 2)]
    
    
    # Make prediction
    predictions = defaultdict(list)
    
    for test_inst in tqdm(test_insts):
        outputs = defaultdict(list)
        inp = test_inst["input"]
        drft_class = inp["draft class"]
        for model_name, model in zip(snake_draft_names, snake_draft_models):
            # let the current model draft a player
            cur_draft = model.predict(inp)
            outputs[model_name].append(cur_draft)
            # remove the drafted player from the draft class
            drft_class.remove(cur_draft)
            if not drft_class:
                # all available players were drafted, possibly
                # before all teams finished drafting
                break

        # draft finished for this instance
        for (inst_name, cur_pred) in outputs.items():
            cur_inst = {"input": inp,
                        "output": cur_pred}
            predictions[inst_name].append(cur_inst)

    # Write to file
    for inst_name, test_insts in predictions.items():
        out_str = "\n".join(map(json.dumps, test_insts))
        out_fn = str( out_folder / f"{inst_name}.jsonl")
        with open(out_fn, "w", encoding = "utf8") as fout:
            fout.write(out_str)
        

    # End
    logging.info("DONE")
