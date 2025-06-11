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
                 
# Local imports
from utils import NUM_OF_DRAFTED_PLAYERS, BaseModel
#----


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
    # Parse command line arguments
    args = docopt(__doc__)
    inp_fn = Path(args["--in"]) if args["--in"] else None
    model_name = args["--model"]
    out_folder = Path(args["--out"]) if args["--out"] else Path("./tmp.out")

    # Determine logging level
    debug = args["--debug"]
    if debug:
        logging.basicConfig(level = logging.DEBUG)
    else:
        logging.basicConfig(level = logging.INFO)

    # determine model
    model = model_router[model_name]()

    # Read all test instances
    test_insts = [json.loads(line) for line in open(inp_fn, encoding = "utf8")]

    # Make predictions
    for test_inst in tqdm(test_insts):
        test_inst["output"] = model.predict(test_inst["input"])

    # Write to file
    out_str = "\n".join(map(json.dumps, test_insts))
    out_fn = out_folder / f"{model.name}.jsonl"
    logging.info(f"writing to {out_fn}")
    with open(out_fn, "w", encoding = "utf8") as fout:
        fout.write(out_str)        

    # End
    logging.info("DONE")
