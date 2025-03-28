from flask import Flask, render_template, jsonify
import sys
import os

# Suppress output from imported modules
class NullWriter:
    def write(self, _): pass
    def flush(self): pass

sys.stdout = NullWriter() 

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.mccv import linear_threshold, linear_threshold_weighted, linear_cluster, linear_cluster_weighted

# Restore stdout so Flask works properly
sys.stdout = sys.__stdout__

app = Flask(__name__)

# Extract player names from all models
lt_player_list = list(linear_threshold.player_avg_success_scores.keys())
ltw_player_list = list(linear_threshold_weighted.player_avg_success_scores.keys())
lc_player_list = list(linear_cluster.player_avg_success_scores.keys())
lcw_player_list = list(linear_cluster_weighted.player_avg_success_scores.keys())

# Convert data into a structured format for the webpage
linear_threshold_data = [
    {
        "name": player,
        "actual_category": linear_threshold.player_actual_categories.get(player, "Unknown"),
        "predicted_category": linear_threshold.player_predicted_categories.get(player, "Unknown"),
        "actual_score": linear_threshold.target[linear_threshold.merged_data['Player'] == player].values[0] if player in linear_threshold.merged_data['Player'].values else None,
        "predicted_score": linear_threshold.player_avg_success_scores.get(player, None),
        "match": linear_threshold.player_actual_categories.get(player) == linear_threshold.player_predicted_categories.get(player)
    }
    for player in lt_player_list
]

linear_threshold_weighted_data = [
    {
        "name": player,
        "actual_category": linear_threshold_weighted.player_actual_categories.get(player, "Unknown"),
        "predicted_category": linear_threshold_weighted.player_predicted_categories.get(player, "Unknown"),
        "actual_score": linear_threshold_weighted.target[linear_threshold_weighted.merged_data['Player'] == player].values[0] if player in linear_threshold_weighted.merged_data['Player'].values else None,
        "predicted_score": linear_threshold_weighted.player_avg_success_scores.get(player, None),
        "match": linear_threshold_weighted.player_actual_categories.get(player) == linear_threshold_weighted.player_predicted_categories.get(player)
    }
    for player in ltw_player_list
]

linear_cluster_data = [
    {
        "name": player,
        "actual_category": linear_cluster.player_actual_categories.get(player, "Unknown"),
        "predicted_category": linear_cluster.player_predicted_categories.get(player, "Unknown"),
        "actual_score": linear_cluster.target[linear_cluster.merged_data['Player'] == player].values[0] if player in linear_cluster.merged_data['Player'].values else None,
        "predicted_score": linear_cluster.player_avg_success_scores.get(player, None),
        "match": linear_cluster.player_actual_categories.get(player) == linear_cluster.player_predicted_categories.get(player)
    }
    for player in lc_player_list
]

linear_cluster_weighted_data = [
    {
        "name": player,
        "actual_category": linear_cluster_weighted.player_actual_categories.get(player, "Unknown"),
        "predicted_category": linear_cluster_weighted.player_predicted_categories.get(player, "Unknown"),
        "actual_score": linear_cluster_weighted.target[linear_cluster_weighted.merged_data['Player'] == player].values[0] if player in linear_cluster_weighted.merged_data['Player'].values else None,
        "predicted_score": linear_cluster_weighted.player_avg_success_scores.get(player, None),
        "match": linear_cluster_weighted.player_actual_categories.get(player) == linear_cluster_weighted.player_predicted_categories.get(player)
    }
    for player in lcw_player_list
]

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/linear_threshold")
def lt():
    return render_template("linear_threshold.html", players=linear_threshold_data)

@app.route("/linear_threshold_weighted")
def ltw():
    return render_template("linear_threshold_weighted.html", players=linear_threshold_weighted_data)

@app.route("/linear_cluster")
def lc():
    return render_template("linear_cluster.html", players=linear_cluster_data)

@app.route("/linear_cluster_weighted")
def lcw():
    return render_template("linear_cluster_weighted.html", players=linear_cluster_weighted_data)

@app.route("/threshold_data")
def threshold_data():
    return jsonify(linear_threshold_data)

@app.route("/threshold_weighted_data")
def threshold_weighted_data():
    return jsonify(linear_threshold_weighted_data)

@app.route("/cluster_data")
def cluster_data():
    return jsonify(linear_cluster_data)

@app.route("/cluster_weighted_data")
def cluster_weighted_data():
    return jsonify(linear_cluster_weighted_data)

if __name__ == "__main__":
    app.run(debug=True)
