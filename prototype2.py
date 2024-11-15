import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Load and preprocess nfl stats
nfl_stats1 = pd.read_csv("data/historical-nfl/2020wr.csv")
nfl_stats1['nflYears'] = nfl_stats1['nflYears'].astype(float)

# List of columns to normalize and average
nfl_metrics_to_normalize = ['nflRec', 'nflYds', 'nflTD', 'AP1', 'St', 'PB', 'wAV']

# Average the metrics by dividing by nflYears
for metric in nfl_metrics_to_normalize:
    nfl_stats1[metric + '_avg'] = nfl_stats1[metric] / nfl_stats1['nflYears']

# Standardize the averaged metrics
scaler = StandardScaler()

# Generate the list of column names for the normalized columns
normalized = [metric + '_avg_normalized' for metric in nfl_metrics_to_normalize]

# Standardize the averaged metrics (normalize by standard deviation)
nfl_stats1[normalized] = scaler.fit_transform(nfl_stats1[[metric + '_avg' for metric in nfl_metrics_to_normalize]])

# Define success metric using the normalized averaged metrics
nfl_stats1['SuccessMetric'] = (nfl_stats1['nflYds_avg_normalized'] + nfl_stats1['nflTD_avg_normalized'] +
                               nfl_stats1['nflRec_avg_normalized'] + nfl_stats1['AP1_avg_normalized'] +
                               nfl_stats1['St_avg_normalized'] + nfl_stats1['PB_avg_normalized'] +
                               nfl_stats1['wAV_avg_normalized'])

# Calculate basic statistics for SuccessMetric
nfl_stats1['SuccessMetric'].describe(percentiles=[.25, .5, .75, .9])

def categorize_player(success_metric):
    if success_metric >= 15:  # Top ~10%
        return "All-Pro"
    elif success_metric >= 5:  # Top ~25%
        return "Pro Bowler"
    elif success_metric >= 1:  # Above average players
        return "Starter"
    elif success_metric >= -2:  # Backup-level players
        return "Backup"
    else:
        return "Practice Squad"  # Players with lower scores

nfl_stats1['Category'] = nfl_stats1['SuccessMetric'].apply(categorize_player)

# Check thresholds distribution
print(nfl_stats1['Category'].value_counts())

# Select features and target variable
measurements1 = pd.read_csv("data/historical-measurements/2020wr.csv")
combine_stats1 = pd.read_csv("data/historical-combine/2020wr.csv")
college_stats1 = pd.read_csv("data/historical-college/2020wr.csv")

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
college_stats1['ConfRank'] = college_stats1['Conf'].map(conference_rankings).fillna(1)

# Merge datasets on Player column
data = college_stats1.merge(measurements1, on="Player").merge(combine_stats1, on="Player")

# Ensure that the players in the target and features match
merged_data = data.merge(nfl_stats1[['Player', 'SuccessMetric']], on='Player', how='inner')

# Features
features = merged_data[['Rec', 'Yds', 'Y/R', 'TD', 'Y/G', 'G', 'ConfRank',]] 
                        
                        #'40yd', 'Height(in)', 'Weight', 'Hand(in)', 'Arm(in)', 'Wingspan(in)']]

# Normalize the feature metrics (standard deviation normalization)
features_normalized = scaler.fit_transform(features)

# Convert the normalized features back into a DataFrame with the same column names
features_normalized_df = pd.DataFrame(features_normalized, columns=features.columns)

# Target
target = merged_data['SuccessMetric']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.5,)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and categorize players in the test set
predictions = model.predict(X_test)
predicted_categories = [categorize_player(pred) for pred in predictions]


# Apply the categorize_player function to the actual success metric values
actual_categories = [categorize_player(actual) for actual in y_test]

# Output results with aligned columns
for player, pred, actual, pred_category, actual_category in zip(merged_data['Player'][X_test.index], predictions, y_test, predicted_categories, actual_categories):
    print(f"Player: {player:<20} Predicted Success: {pred:>10.2f}  Actual Success: {actual:>10.2f}  Predicted Category: {pred_category:<15} Actual Category: {actual_category:<15}")
