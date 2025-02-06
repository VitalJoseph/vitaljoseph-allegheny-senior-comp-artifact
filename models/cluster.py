"""Prototype for KMeans Clustering."""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from collections import Counter


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

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.7,)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and categorize players in the test set
predictions = model.predict(X_test)

# Reshape predictions to be a 2D array (n_samples, 1)
predictions_reshaped = predictions.reshape(-1, 1)

# Use kmeans to fit and predict categories for all players
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(np.concatenate((y_test.to_numpy().reshape(-1, 1), predictions_reshaped), axis=0))

# Separate cluster labels for predicted and actual categories
num_predictions = len(predictions_reshaped)
predicted_clusters = cluster_labels[num_predictions:]
actual_clusters = cluster_labels[:len(y_test)]

# Add success metrics to associate cluster with category
y_combined = pd.concat([y_test.reset_index(drop=True), pd.Series(predictions)], axis=0)
cluster_avg_scores = pd.DataFrame({
    "Cluster": cluster_labels,
    "Score": y_combined
}).groupby("Cluster")["Score"].mean()

# Sort clusters by average score
sorted_clusters = cluster_avg_scores.sort_values(ascending=False)

# Map sorted clusters to categories
cluster_to_category = {
    sorted_clusters.index[0]: "All-Pro",
    sorted_clusters.index[1]: "Pro Bowler",
    sorted_clusters.index[2]: "Starter",
    sorted_clusters.index[3]: "Backup",
    sorted_clusters.index[4]: "Practice Squad"
}

# Function to categorize player based on cluster
def categorize_player_automated(cluster_label: int):
    return cluster_to_category.get(cluster_label, "Unknown")

# Apply the categorization
predicted_categories = [categorize_player_automated(cluster) for cluster in predicted_clusters]
actual_categories = [categorize_player_automated(cluster) for cluster in actual_clusters]

# Calculate the distribution of predicted categories
predicted_category_counts = Counter(predicted_categories)

# Calculate the distribution of actual categories
actual_category_counts = Counter(actual_categories)

# Print the distributions in a readable format
print("\nPredicted Category Distribution:")
for category, count in predicted_category_counts.items():
    print(f"{category}: {count}")

print("\nActual Category Distribution:")
for category, count in actual_category_counts.items():
    print(f"{category}: {count}")

print()
print()


# Output results with aligned columns
for player, pred, actual, pred_category, actual_category in zip(
    merged_data['Player'][X_test.index], 
    predictions, 
    y_test, 
    predicted_categories, 
    actual_categories
):
    print(f"Player: {player:<20} Predicted Success: {pred:>10.2f}  Actual Success: {actual:>10.2f}  Predicted Category: {pred_category:<15} Actual Category: {actual_category:<15}")