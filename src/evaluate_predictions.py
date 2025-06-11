""" Usage:
    <file-name> --gold=GOLD_FILE --pred=PRED_FILE [--debug]
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
                 
# Local imports
from utils import read_jsonl, score_prediction

#----

if __name__ == "__main__":
    # Parse command line arguments
    args = docopt(__doc__)
    gold_fn = Path(args["--gold"]) 
    pred_fn = Path(args["--pred"]) 

    # Determine logging level
    debug = args["--debug"]
    if debug:
        logging.basicConfig(level = logging.DEBUG)
    else:
        logging.basicConfig(level = logging.INFO)

    # Start computation
    gold_insts = read_jsonl(gold_fn)
    pred_insts = read_jsonl(pred_fn)

    assert(len(gold_insts) == len(pred_insts))

    score, scores = score_prediction(gold_insts, pred_insts)

    logging.info(f"{scores} \n Combined score: {score}")


    # End
    logging.info("DONE")
