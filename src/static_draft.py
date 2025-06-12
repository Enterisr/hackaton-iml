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

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
                 
# Local imports
from utils import NUM_OF_DRAFTED_PLAYERS, BaseModel, score_prediction
from preprocess import split_train_test,pre_proc_orchestrator
# Import your custom LinearRegression
from LinearRegression import LinearRegression

#----
PLAYERS_TO_CHOOSE = 10


def plot_confusion_matrix(y_true, y_pred_labels, title='Confusion Matrix'):
    """
    Plots a confusion matrix using seaborn's heatmap.
    
    Args:
        y_true (list or np.array): True labels (0 for Low-Value, 1 for High-Value).
        y_pred_labels (list or np.array): Predicted labels (0 for Not Selected, 1 for Selected).
        title (str): The title for the plot.
    """
    cm = confusion_matrix(y_true, y_pred_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not Selected', 'Selected'], 
                yticklabels=['Low-Value', 'High-Value'])
    plt.title(title)
    plt.ylabel('True Player Value')
    plt.xlabel('Model Selection')
    plt.show()

def plot_error_analysis(y_true, y_pred):
    """
    Generates and displays error analysis plots for a regression model.
    
    Args:
        y_true (np.array): The true target values.
        y_pred (np.array): The predicted values from the model.
    """
    residuals = y_true - y_pred

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 1. Predicted vs. True Values Plot
    sns.scatterplot(x=y_true, y=y_pred, ax=ax1, alpha=0.6)
    ax1.set_title('Predicted vs. True Values')
    ax1.set_xlabel('True Scores')
    ax1.set_ylabel('Predicted Scores')
    # Add a y=x line for reference
    perfect_line_lim = [min(min(y_true), min(y_pred)), max(max(y_true), max(y_pred))]
    ax1.plot(perfect_line_lim, perfect_line_lim, color='red', linestyle='--', lw=2)
    ax1.grid(True)

    # 2. Residuals Plot
    sns.scatterplot(x=y_pred, y=residuals, ax=ax2, alpha=0.6)
    ax2.set_title('Residuals vs. Predicted Values')
    ax2.set_xlabel('Predicted Scores')
    ax2.set_ylabel('Residuals (True - Predicted)')
    # Add a horizontal line at y=0
    ax2.axhline(0, color='red', linestyle='--', lw=2)
    ax2.grid(True)
    
    plt.suptitle('Model Error Analysis', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_pca_visualization(X_test_df, y_true, y_pred, title='PCA of Player Stats with Prediction Error'):
    """
    Performs PCA on the test data and visualizes the results in 2D,
    colored by the prediction error.
    
    Args:
        X_test_df (pd.DataFrame): The feature matrix for the test set.
        y_true (np.array): The true target values.
        y_pred (np.array): The predicted values.
        title (str): The title for the plot.
    """
    # Ensure X_test_df contains only numeric data
    X_numeric = X_test_df.select_dtypes(include=np.number)
    
    # Perform PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_numeric)
    
    residuals = y_true - y_pred
    
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=residuals, cmap='coolwarm', alpha=0.7)
    plt.colorbar(scatter, label='Prediction Residual (True - Predicted)')
    plt.title(title)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)
    plt.show()


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
    X_train, y_train, X_test, y_test, test_meta, feature_names = split_train_test(gold_insts, test_size=0.2, random_state=42)

    ##use your custom random forest model
    # #Fit RandomForestRegression model
    # model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=8, min_samples_split=8,
    #                               min_samples_leaf=4, max_features='sqrt')
    # model.fit(X_train, y_train)
    # predictions = model.predict(X_test)

    #Fit adaboost
    model = AdaBoostRegressor(
        n_estimators=100,
        learning_rate=1.0,
        random_state=42
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Use your custom LinearRegression model
    # model = LinearRegression()
    # model.fit(X_train, y_train)
 
    # predictions = model.predict({'X': X_test})  # Use the format expected by your custom class

    assert(len(y_test) == len(predictions))

    # --- Convert predictions to season-level structure for score_prediction ---
    # Group predictions by season
    season_pred_dict = defaultdict(list)
    for (season_idx, pid), pred_score in zip(test_meta, predictions):
        season_pred_dict[season_idx].append((pid, pred_score))

    pred_insts = []
    selected_player_pids = []
    for season_idx in sorted(season_pred_dict.keys()):
        # Sort players by predicted score (descending)
        sorted_players = [pid for pid, _ in sorted(season_pred_dict[season_idx], key=lambda x: -x[1])]
        draft_selection = sorted_players[:PLAYERS_TO_CHOOSE]
        selected_player_pids = selected_player_pids + draft_selection
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

    
    print("\n--- Starting Model Analysis ---")
    
    X_test_df = pd.DataFrame(X_test, columns=feature_names)

    median_true_score = np.median(y_test)
    true_labels = (y_test >= median_true_score).astype(int) # 1 if High-Value, 0 if Low-Value

    test_pids = [pid for _, pid in test_meta]

    predicted_selection_labels = [1 if pid in selected_player_pids else 0 for pid in test_pids]
    
    plot_confusion_matrix(true_labels, predicted_selection_labels, title='Adapted Confusion Matrix: Player Value vs. Model Selection')
    
    plot_error_analysis(y_test, predictions)
    
    plot_pca_visualization(X_test_df, y_test, predictions)



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
