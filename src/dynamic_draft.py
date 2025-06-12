""" Usage:
    dynamic_draft.py --in=INPUT_FILE --models=MODEL_NAME --out=OUTPUT_FOLDER [--debug]
"""
import logging
from collections import defaultdict
from pathlib import Path
import json
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import random
from docopt import docopt

from utils import read_jsonl, score_prediction, NUM_OF_DRAFTED_PLAYERS
from preprocess import pre_proc_orchestrator
import numpy as np
from utils import BaseModel
from LinearRegression import LinearRegression
from preprocess import preprocess, INTRESTING_FEATURES


    # Define models directly (no import from dynamic_draft)

class RandomDynamicBaseline(BaseModel):
    def __init__(self, weights_path=None):
        super().__init__()

    def predict(self, input_json):
        draft_class = input_json["draft class"]
        return random.sample(draft_class, 1)[0]

class DynamicLinearRegression(LinearRegression):
    def __init__(self, weights_path=None):
        super().__init__(weights_path)
        self.player_scores = {}

    def fit(self, X, y, weights_path=None):
        return super().fit(X, y, weights_path)

    def predict(self, input_json):
        draft_class = input_json["draft class"]
        last_season = input_json["last season"]

        if not self.player_scores:
            features = []
            pids = []
            for pid in draft_class:
                player_stats = last_season[last_season.personId == pid]
                if player_stats.empty:
                    feat = [0] * len(INTRESTING_FEATURES)
                else:
                    feat = preprocess(player_stats)
                features.append(feat)
                pids.append(pid)

            predictions = super().predict({'X': np.array(features)})
            for pid, score in zip(pids, predictions):
                self.player_scores[pid] = score

        best_player = max(draft_class, key=lambda pid: self.player_scores.get(pid, -float('inf')))
        return best_player

model_router = {
    "random": RandomDynamicBaseline,
    "linear": DynamicLinearRegression
}


if __name__ == "__main__":
    args = docopt(__doc__)
    inp_fn = Path(args["--in"])
    model_names = args["--models"].split(",")
    out_folder = Path(args["--out"])

    debug = args["--debug"]
    logging.basicConfig(level=logging.DEBUG if debug else logging.INFO)

    logging.info("Loading and preprocessing test data...")
    test_insts = read_jsonl(inp_fn)
    test_insts = pre_proc_orchestrator(test_insts)
    gold_insts = test_insts.copy()

    models = []
    for model_name in model_names:
        weights_path = f"./weights_{model_name}/weights"
        model_class = model_router[model_name]
        model = model_class(weights_path=weights_path)
        models.append(model)

    inst_names = [model.name for model in models]
    
    snake_draft_names = []
    for i in range(NUM_OF_DRAFTED_PLAYERS):
        snake_draft_names += inst_names[::1 - 2 * (i % 2)]
    snake_draft_models = []
    for i in range(NUM_OF_DRAFTED_PLAYERS):
        snake_draft_models += models[::1 - 2 * (i % 2)]

    predictions = defaultdict(list)

    for test_inst in tqdm(test_insts, desc="Predicting drafts"):
        outputs = defaultdict(list)
        inp = test_inst["input"]
        draft_class = inp["draft class"].copy()

        for model in models:
            if hasattr(model, 'player_scores'):
                model.player_scores = {}

        for model_name, model in zip(snake_draft_names, snake_draft_models):
            if not draft_class:
                break
            pick = model.predict({"draft class": draft_class, "last season": inp["last season"]})
            outputs[model_name].append(pick)
            draft_class.remove(pick)
            inp["draft class"] = draft_class

        for inst_name, output in outputs.items():
            predictions[inst_name].append({
                "input": {"uid": inp["uid"]},
                "output": output
            })

    for inst_name, pred_insts in predictions.items():
        logging.info(f"Scoring model {inst_name}...")
        score, scores = score_prediction(gold_insts, pred_insts)
        logging.info(f"Model {inst_name} scores: {scores}\nTotal: {score}")

        out_fn = out_folder / f"{inst_name}.jsonl"
        os.makedirs(out_fn.parent, exist_ok=True)
        with open(out_fn, "w", encoding="utf8") as fout:
            for inst in pred_insts:
                fout.write(json.dumps(inst) + "\n")

    logging.info("DONE")