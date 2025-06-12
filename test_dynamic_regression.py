import logging
from collections import defaultdict
from pathlib import Path
import json
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import random

from utils import read_jsonl, score_prediction, NUM_OF_DRAFTED_PLAYERS
from preprocess import pre_proc_orchestrator
from dynamic_draft import model_router

logging.basicConfig(level=logging.INFO)

INPUT_FILE = "data/sample_seasons/test.jsonl"  # Change to your test file
MODEL_NAME = "linear"  # Can be "linear" or "random"
OUTPUT_FILE = Path(f"./tmp.out/{MODEL_NAME}.jsonl")
WEIGHTS_PATH = f"./weights_{MODEL_NAME}/weights"

if __name__ == "__main__":
    logging.info("Loading and preprocessing test data...")
    test_insts = read_jsonl(INPUT_FILE)
    test_insts = pre_proc_orchestrator(test_insts)

    gold_insts = test_insts.copy()

    # Instantiate model
    model_class = model_router[MODEL_NAME]
    model = model_class(weights_path=WEIGHTS_PATH)
    inst_name = model.name

    predictions = []

    for test_inst in tqdm(test_insts, desc="Predicting drafts"):
        inp = test_inst["input"]
        draft_class = inp["draft class"].copy()

        # Reset model cache/state before new prediction
        if hasattr(model, "player_scores"):
            model.player_scores = {}

        outputs = []
        for _ in range(NUM_OF_DRAFTED_PLAYERS):
            if not draft_class:
                break
            pick = model.predict({"draft class": draft_class, "last season": inp["last season"]})
            outputs.append(pick)
            draft_class.remove(pick)

        predictions.append({
            "input": {"uid": inp["uid"]},
            "output": outputs
        })

    logging.info("Scoring predictions...")
    score, scores = score_prediction(gold_insts, predictions)
    logging.info(f"Per-season scores: {scores}\nTotal score: {score}")

    logging.info(f"Saving predictions to {OUTPUT_FILE}...")
    os.makedirs(OUTPUT_FILE.parent, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf8") as fout:
        for inst in predictions:
            fout.write(json.dumps(inst) + "\n")

    logging.info("DONE")
