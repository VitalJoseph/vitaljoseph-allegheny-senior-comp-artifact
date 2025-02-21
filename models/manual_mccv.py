"""Prototype for manual categorization with Monte Carlo cross validation technique."""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from collections import Counter
from collections import defaultdict
from sklearn.metrics import classification_report



# File paths for the datasets
file_paths_nfl = [
    "data/historical-nfl/2019wr.csv",
    "data/historical-nfl/2020wr.csv",
    "data/historical-nfl/2021wr.csv",
    "data/historical-nfl/2022wr.csv"
]

# Load and combine the datasets
combined_data_nfl = pd.concat([pd.read_csv(path) for path in file_paths_nfl], ignore_index=True)

# Save the combined data to a new CSV file
output_path_nfl = "data/historical-nfl/combined_wr_data.csv"
combined_data_nfl.to_csv(output_path_nfl, index=False)

# Load and preprocess nfl stats
nfl_stats = pd.read_csv("data/historical-nfl/combined_wr_data.csv")
nfl_stats['nflYears'] = nfl_stats['nflYears'].astype(float)

# List of columns to normalize and average
nfl_metrics_to_normalize = ['nflRec', 'nflYds', 'nflTD', 'AP1', 'St', 'PB']

# Average the metrics by dividing by nflYears
for metric in nfl_metrics_to_normalize:
    nfl_stats[metric + '_avg'] = nfl_stats[metric] / nfl_stats['nflYears']

# Standardize the averaged metrics
scaler = StandardScaler()

# Generate the list of column names for the normalized columns
normalized = [metric + '_avg_normalized' for metric in nfl_metrics_to_normalize]

# Standardize the averaged metrics (normalize by standard deviation)
nfl_stats[normalized] = scaler.fit_transform(nfl_stats[[metric + '_avg' for metric in nfl_metrics_to_normalize]])

# Standardize the un-averaged metrics
nfl_stats['wAV_normalized'] = scaler.fit_transform(nfl_stats[['wAV']])

# Define success metric using the normalized averaged metrics
nfl_stats['SuccessMetric'] = (nfl_stats['nflYds_avg_normalized'] + nfl_stats['nflTD_avg_normalized'] +
                               nfl_stats['nflRec_avg_normalized'] + nfl_stats['AP1_avg_normalized'] +
                               nfl_stats['St_avg_normalized'] + nfl_stats['PB_avg_normalized'] +
                               nfl_stats['wAV_normalized'])


# File paths for the datasets
file_paths_measurements = [
    "data/historical-measurements/2019wr.csv",
    "data/historical-measurements/2020wr.csv",
    "data/historical-measurements/2021wr.csv",
    "data/historical-measurements/2022wr.csv"
]

file_paths_combine = [
    "data/historical-combine/2019wr.csv",
    "data/historical-combine/2020wr.csv",
    "data/historical-combine/2021wr.csv",
    "data/historical-combine/2022wr.csv"
]

file_paths_college = [
    "data/historical-college/2019wr.csv",
    "data/historical-college/2020wr.csv",
    "data/historical-college/2021wr.csv",
    "data/historical-college/2022wr.csv"
]

# Load and combine the datasets
combined_data_measurements = pd.concat([pd.read_csv(path) for path in file_paths_measurements], ignore_index=True)
combined_data_combine = pd.concat([pd.read_csv(path) for path in file_paths_combine], ignore_index=True)
combined_data_college = pd.concat([pd.read_csv(path) for path in file_paths_college], ignore_index=True)

# Save the combined data to a new CSV file
output_path_measurements = "data/historical-measurements/combined_wr_data.csv"
combined_data_measurements.to_csv(output_path_measurements, index=False)

output_path_combine = "data/historical-combine/combined_wr_data.csv"
combined_data_combine.fillna(np.nan, inplace=True)
combined_data_combine.to_csv(output_path_combine, index=False)

output_path_college = "data/historical-college/combined_wr_data.csv"
combined_data_college.to_csv(output_path_college, index=False)

# Select features and target variable
measurements = pd.read_csv("data/historical-measurements/combined_wr_data.csv")
combine_stats = pd.read_csv("data/historical-combine/combined_wr_data.csv")
college_stats = pd.read_csv("data/historical-college/combined_wr_data.csv")

conference_rankings = {
    'SEC': 10,
    'Big Ten': 9,
    'Big 12': 8,
    'Pac-12': 7,
    'ACC': 6,
    'AAC': 5,
    'MWC': 4,
    'C-USA': 3,
    'Ind': 2,
    'CAA': 1
}

# Map conference names to their rankings, default to 1 for any conference not listed
college_stats['ConfRank'] = college_stats['Conf'].map(conference_rankings).fillna(1)

# Merge datasets on Player column
data = college_stats.merge(measurements, on="Player").merge(combine_stats, on="Player")

# Ensure that the players in the target and features match
merged_data = data.merge(nfl_stats[['Player', 'SuccessMetric']], on='Player', how='inner')

# Features
features = merged_data[['Rec', 'Yds', 'Y/R', 'TD', 'Y/G', 'G', 'ConfRank', '40yd', 'Height(in)', 'Weight', 'Hand(in)', 'Arm(in)', 'Wingspan(in)']]
                        
                        #, 'Height(in)', 'Weight', 'Hand(in)', 'Arm(in)', 'Wingspan(in)']]

# Normalize the feature metrics (standard deviation normalization)
features_normalized = scaler.fit_transform(features)

# Convert the normalized features back into a DataFrame with the same column names
features_normalized_df = pd.DataFrame(features_normalized, columns=features.columns)

# Target
target = merged_data['SuccessMetric']

# Define categorize_player function
def categorize_player(success_metric):
    if success_metric >= 15:  
        return "All-Pro (Elite)"
    elif success_metric >= 5:  
        return "Pro Bowler (Great)"
    elif success_metric >= 1:  
        return "Starter (Good)"
    elif success_metric >= -2:  
        return "Backup (Average)"
    else:
        return "Practice Squad (Below Average)"  


# Monte Carlo Cross Validation
num_iterations = 100

# Storage containers
player_success_scores = defaultdict(list)
player_actual_categories = defaultdict(list)
player_predicted_categories = defaultdict(list)


for i in range(num_iterations):
    # random_state=i: Ensures a different random split for each iteration.
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.1, random_state=i)
    
    # Trains (fit) the model using the training data 
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict success scores
    predictions = model.predict(X_test)

    
    # Loops through each player's name and their predicted success score. Saves each player’s predicted score to player_success_scores.
    for player, score in zip(merged_data.loc[X_test.index, 'Player'], predictions):
        player_success_scores[player].append(score)

    # Converts the predicted and actual success scores into categories using categorize_player().
    predicted_categories = [categorize_player(pred) for pred in predictions]
    actual_categories = [categorize_player(actual) for actual in y_test]

    # Iterates through each player’s name, actual category, and predicted category and then stores them
    for player, actual, predicted in zip(merged_data.loc[X_test.index, 'Player'], actual_categories, predicted_categories):
        player_actual_categories[player].append(actual)
        player_predicted_categories[player].append(predicted)

# Create two empty lists to store actual and predicted categories
all_actuals = []
all_predictions = []

# Loops through each player
for player in player_actual_categories:
    #  Retrieves the list of actual categories for the player and adds them to all_actuals, all at once
    all_actuals.extend(player_actual_categories[player])  
    #  Retrieves the list of predicted categories for the player and adds them to all_predictions, all at once
    all_predictions.extend(player_predicted_categories[player]) 

# Converts the classification report into a dictionary using output_dict=True, allowing easy access to the values.
report = classification_report(all_actuals, all_predictions, digits=4, zero_division=0, output_dict=True)

# Generate and print classification report
print("\n\033[1;33m=== Classification Report ===\033[0m") 

# Explanation of classification metrics
print("\n\033[1mUnderstanding the Classification Report:\033[0m")
print("- Precision: Of all the times the model predicted a category, how often was it correct?")
print("- Recall: Out of all the actual cases for a category, how many did the model correctly identify?")
print("- F1-score: A balance between precision and recall (higher is better).")
print("- Support: The number of actual occurrences of the category in the dataset.\n")

# Loop through each category (excluding overall accuracy tests)
for category, metrics in report.items():
    if category not in ['accuracy', 'macro avg', 'weighted avg']:
        precision = metrics["precision"] * 100
        recall = metrics["recall"] * 100
        f1_score = metrics["f1-score"] * 100
        support = metrics["support"]

        print(f"\nCategory: {category}")
        print(f"  - Precision ({precision:.2f}%) -> When the model predicted \"{category},\" it was correct {precision:.2f}% of the time.")
        print(f"  - Recall ({recall:.2f}%) -> The model correctly classified {recall:.2f}% of actual \"{category}\" players.")
        print(f"  - F1-score ({f1_score:.2f}%) -> A balance between precision and recall.")
        print(f"  - Support ({support}) -> There were {support} actual \"{category}\" players.\n")

# overall accuracy
accuracy = report["accuracy"] * 100
print(f"Overall Accuracy: {accuracy:.2f}% -> The model correctly classified {accuracy:.2f}% of all players.\n")


# Finds the most frequently occurring category for each player. This gives us the final category prediction and actual category for each player.
final_actual_categories = {player: Counter(categories).most_common(1)[0][0] for player, categories in player_actual_categories.items()}
final_predicted_categories = {player: Counter(categories).most_common(1)[0][0] for player, categories in player_predicted_categories.items()}

# Holds the predicted success metric for each player over multiple iterations. Computes the average success metric for each player.
average_predicted_success_scores = {player: np.mean(scores) for player, scores in player_success_scores.items()}

print("\n\033[1;36m=== Monte Carlo Cross-Validation Results ===\033[0m")

# Sorts players in descending order based on their average_predicted_success_scores. If a player is missing a success score, they default to 0.
sorted_players = sorted(final_actual_categories.keys(), key=lambda x: average_predicted_success_scores.get(x, 0), reverse=True)

# Define how many players to show from the top and bottom
num_to_show = 5 

# Select the first and last few players
top_players = sorted_players[:num_to_show]
bottom_players = sorted_players[-num_to_show:]

# top players
print(f"\n--- Top {num_to_show} Players ---")
for player in top_players:
    print(f"{player}:")
    print(f"   - Actual Category: {final_actual_categories.get(player, 'Unknown')}")
    print(f"   - Predicted Category: {final_predicted_categories.get(player, 'Unknown')}")
    print(f"   - Average Predicted Success Score: {average_predicted_success_scores.get(player, 'N/A'):.4f}")

# separator 
print("\n... (skipping middle rows) ...")

# bottom players
print(f"\n--- Bottom {num_to_show} Players ---")
for player in bottom_players:
    print(f"{player}:")
    print(f"   - Actual Category: {final_actual_categories.get(player, 'Unknown')}")
    print(f"   - Predicted Category: {final_predicted_categories.get(player, 'Unknown')}")
    print(f"   - Average Predicted Success Score: {average_predicted_success_scores.get(player, 'N/A'):.4f}")

# Counts how many players fall into each actual and predicted category using Counter().
print("\n--- Category Distribution Summary ---")
actual_counts = Counter(final_actual_categories.values())
predicted_counts = Counter(final_predicted_categories.values())

print("\nActual Category Distribution:")
for category, count in actual_counts.items():
    print(f"  {category}: {count} players")

print("\nPredicted Category Distribution:")
for category, count in predicted_counts.items():
    print(f"  {category}: {count} players")

print("\n\033[1;31m=== End of Monte Carlo Results ===\033[0m")
