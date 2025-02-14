from flask import Flask, render_template
import sys
import os

# Suppress output from imported modules
class NullWriter:
    def write(self, _): pass
    def flush(self): pass

sys.stdout = NullWriter() 

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models import manual_mccv, cluster_mccv

# Restore stdout so Flask works properly
sys.stdout = sys.__stdout__

app = Flask(__name__)

# Extract player names from both models
players_manual = list(manual_mccv.average_predicted_success_scores.keys())
players_cluster = list(cluster_mccv.average_predicted_success_scores.keys())

# Convert data into a structured format for the webpage
player_data_manual = [
    {
        "name": player,
        "actual_category": manual_mccv.final_actual_categories.get(player, "Unknown"),
        "predicted_category": manual_mccv.final_predicted_categories.get(player, "Unknown"),
        "actual_score": manual_mccv.target[manual_mccv.merged_data['Player'] == player].values[0] if player in manual_mccv.merged_data['Player'].values else None,
        "predicted_score": manual_mccv.average_predicted_success_scores.get(player, None),
    }
    for player in players_manual
]

player_data_cluster = [
    {
        "name": player,
        "actual_category": cluster_mccv.final_actual_categories.get(player, "Unknown"),
        "predicted_category": cluster_mccv.final_predicted_categories.get(player, "Unknown"),
        "actual_score": cluster_mccv.target[cluster_mccv.merged_data['Player'] == player].values[0] if player in cluster_mccv.merged_data['Player'].values else None,
        "predicted_score": cluster_mccv.average_predicted_success_scores.get(player, None),
    }
    for player in players_cluster
]

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/manual")
def manual():
    return render_template("manual.html", players=player_data_manual)

@app.route("/cluster")
def cluster():
    return render_template("cluster.html", players=player_data_cluster)

if __name__ == "__main__":
    app.run(debug=True)
