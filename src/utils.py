import pdb
from io import StringIO
import pandas as pd
import json
from collections import defaultdict
import hashlib
import numpy as np
from tqdm import tqdm

player_categories = {
    'fieldGoalsAttempted': {"weight": -1},
    'fieldGoalsMade': {"weight": 2},
    'threePointersAttempted': {"weight": -0.5},
    'threePointersMade': {"weight": 3},
    'freeThrowsAttempted': {"weight": -0.5},
    'freeThrowsMade': {"weight": 1},
    'reboundsDefensive': {"weight": 1},
    'reboundsOffensive': {"weight": 1.5},
    'turnovers': {"weight": -2},
    "win": {"weight": 1}
}

NUM_OF_DRAFTED_PLAYERS = len(player_categories)

def hash_id(s):
    return hashlib.sha256(s.encode()).hexdigest()[:10] 

def read_jsonl(inp_fn):
    """
    return a json lines format from a file
    """
    lines = []
    for line in tqdm(open(inp_fn, encoding = "utf8"), desc = f"reading {inp_fn}"):
        cur_dict = json.loads(line)
        cur_dict["input"]["last season"] = pd.read_json(StringIO(cur_dict["input"]["last season"]))
        if "next season" in cur_dict:
            cur_dict["next season"] = pd.read_json(StringIO(cur_dict["next season"]))
        lines.append(cur_dict)
    return lines


class BaseModel:
    """
    A base class for all models
    """
    counter = 0 # a static variable that counts number of models

    def __init__(self):
        BaseModel.counter += 1
        self.name = self.__class__.__name__ + f"_{self.counter}"

    def predict(self, input_json):
        """
        Implment this function in inheriting classes
        """
        raise NotImplementedError("Subclasses must implement this method.")

def season_by_category(season_df, category, weight):
    """
    Return a dictionary from player to their stats in a certain category
    """
    player_stats = season_df.groupby('personId')[category].sum() * weight
    return player_stats

def calc_game_started(season, pid):
    """
    Count the number of games in which a player played in
    a given season
    """
    games_started = len(season[season.personId == pid])
    return games_started


def score_team_on_season(team, season):
    """
    Get the score of a given team in a praticular season
    returns a dictionary of score by category.
    """

    cat_dicts = dict([(cat, season_by_category(season, cat, cat_dict["weight"]))
                      for cat, cat_dict in player_categories.items()])

    # games started is different than other metrics as it counts appearances
    games_started = dict([(pid, calc_game_started(season, pid))
                           for pid in team])

    cat_dicts["gamesStarted"] = games_started
    score = 0
    for cat, cat_dict in cat_dicts.items():
        for pid in team:
            score += cat_dict[pid]

    return score


def score_prediction(gold_insts, pred_insts):
    """
    Score a predicted draft according to a gold reference
    according to all categories - each category gets a score between 0-1
    as a fraction of the performance of the top players in that category
    """
    scores = []
    num_of_seasons = len(gold_insts)
    assert(len(pred_insts) == num_of_seasons)
    
    for gold_inst, pred_inst in tqdm(zip(gold_insts, pred_insts)):
        # make sure that pred and gold refer to the same instance
        gold_uid = gold_inst["input"]["uid"]
        pred_uid = pred_inst["input"]["uid"]
        if gold_uid != pred_uid:
            raise Exception(f"""There's a mismatch between predicted and gold files.
            gold_uid: {gold_uid} 
            pred_uid: {pred_uid}""")


        season = gold_inst["next season"]
        predicted_draft = pred_inst["output"]
        cur_score = score_team_on_season(predicted_draft, season)
        scores.append(cur_score)

    # normalize by seasons
    score = np.average(scores)
        
    return score, scores



def get_rookies(season, draft_class):
    """
    Get a list of players appearing in the draft that did not
    appear in a given season, represented as dataframe
    """
    s1_players = set(season.personId)
    new_players = [pid for pid in draft_class
                   if pid not in s1_players]
    return new_players


def get_veterans(season, draft_class):
    """
    Get a list of players appearing in the draft that also
    appear in a given season, represented as a dataframe
    """
    s1_players = set(season.personId)
    
    old_players = [pid for pid in draft_class
                   if pid in s1_players]
    return old_players

        

    




                
    

    
