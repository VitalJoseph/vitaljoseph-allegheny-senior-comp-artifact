from flask import Flask, render_template
import pandas as pd
import numpy as np
from collections import Counter

import sys
import os

# Suppress output from imported modules
class NullWriter:
    def write(self, _): pass
    def flush(self): pass

sys.stdout = NullWriter() 


# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models import manual_mccv, cluster_mccv, cl

# Restore stdout so Flask work properly
sys.stdout = sys.__stdout__

app = Flask(__name__)

# Extract player names
players = list(manual_mccv.average_predicted_success_scores.keys())

# Convert data into a structured format for the webpage
player_data = [
    {
        "name": player,
        "actual_category": manual_mccv.final_actual_categories.get(player, "Unknown"),
        "predicted_category": manual_mccv.final_predicted_categories.get(player, "Unknown"),
        "actual_score": manual_mccv.target[manual_mccv.merged_data['Player'] == player].values[0] if player in manual_mccv.merged_data['Player'].values else None,
        "predicted_score": manual_mccv.average_predicted_success_scores.get(player, None),
    }
    for player in players
]

@app.route("/")
def index():
    return render_template("index.html", players=player_data)

if __name__ == "__main__":
    app.run(debug=True)