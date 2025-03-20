#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 


# In[ ]:


#dataset setup for cleaning 
import os
import zipfile
import pandas as pd

# Path to your zip file
zip_file_path = 'march-madness-2025-cleaned.zip'

# Create a folder to extract the files to (if it doesn't exist)
extract_folder = 'march-madness-2025-cleaned'
if not os.path.exists(extract_folder):
    os.makedirs(extract_folder)

# Extract all files from the zip
print(f"Extracting files from {zip_file_path}...")
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_folder)

# List all CSV files in the extracted folder
csv_files = [f for f in os.listdir(extract_folder) if f.endswith('.csv')]
print(f"Found {len(csv_files)} CSV files")

# Create an empty dictionary to store dataframes
dfs = {}

# Read each CSV into a dictionary; key is the filename (without .csv)
for file in csv_files:
    file_path = os.path.join(extract_folder, file)
    df = pd.read_csv(file_path)
    key = file.replace('.csv', '')
    dfs[key] = df
    # Print out a brief acknowledgement
    print(f'Read {file}: {df.shape[0]} rows x {df.shape[1]} columns')

# Now you can access any dataframe using the dictionary
# For example: dfs['MTeams'] will give you the Men's Teams dataframe

# Example: Print the first few rows of a dataframe
print("\nExample: First few rows of MTeams dataframe:")
if 'MTeams.csv' in csv_files:
    print(dfs['MTeams'].head())
else:
    print("MTeams.csv not found in the zip file")

# The dfs dictionary now contains all your dataframes ready to use
print("\nAll dataframes are loaded and ready to use in the 'dfs' dictionary")


# In[ ]:


#In game setup training data
import os
import pandas as pd

# Folder containing cleaned csv files
cleaned_folder = 'march-madness-2025-cleaned'

# Paths for men's tournament results and teams
men_results_path = os.path.join(cleaned_folder, 'MNCAATourneyDetailedResults.csv')
men_teams_path = os.path.join(cleaned_folder, 'MTeams.csv')

# Paths for women's tournament results and teams
women_results_path = os.path.join(cleaned_folder, 'WNCAATourneyDetailedResults.csv')
women_teams_path = os.path.join(cleaned_folder, 'WTeams.csv')

# Read men's data if available
if os.path.exists(men_results_path) and os.path.exists(men_teams_path):
    df_men = pd.read_csv(men_results_path)
    df_mteams = pd.read_csv(men_teams_path)
    # Merge to include team names
    # For winners, merge team name from df_mteams
    df_men = df_men.merge(df_mteams[['TeamID', 'TeamName']], left_on='WTeamID', right_on='TeamID', how='left')
    df_men.rename(columns={'TeamName': 'WTeamName'}, inplace=True)
    df_men.drop('TeamID', axis=1, inplace=True)
    # For losers, merge separately
    df_men = df_men.merge(df_mteams[['TeamID', 'TeamName']], left_on='LTeamID', right_on='TeamID', how='left')
    df_men.rename(columns={'TeamName': 'LTeamName'}, inplace=True)
    df_men.drop('TeamID', axis=1, inplace=True)
    
    # Create combined features
    # Create columns for lower and higher team IDs and target
    def process_game(row):
        # lower team id is the smaller of WTeamID and LTeamID
        lower_id = min(row['WTeamID'], row['LTeamID'])
        higher_id = max(row['WTeamID'], row['LTeamID'])
        # target: 1 if the lower team won, 0 otherwise
        if lower_id == row['WTeamID']:
            target = 1
        else:
            target = 0
        # Add absolute score difference as an example feature
        score_margin = abs(row['WScore'] - row['LScore'])
        return pd.Series({'Season': row['Season'], 'LowerTeamID': lower_id, 'HigherTeamID': higher_id,
                          'Target': target, 'ScoreMargin': score_margin, 'Tournament': 'Men',
                          'WTeamID': row['WTeamID'], 'LTeamID': row['LTeamID'],
                          'WTeamName': row['WTeamName'], 'LTeamName': row['LTeamName']})
        
    df_men_processed = df_men.apply(process_game, axis=1)
else:
    print('Men tournament data not available.')
    df_men_processed = pd.DataFrame()

# Read women's data if available
if os.path.exists(women_results_path) and os.path.exists(women_teams_path):
    df_women = pd.read_csv(women_results_path)
    df_wteams = pd.read_csv(women_teams_path)
    # Merge to include team names
    df_women = df_women.merge(df_wteams[['TeamID', 'TeamName']], left_on='WTeamID', right_on='TeamID', how='left')
    df_women.rename(columns={'TeamName': 'WTeamName'}, inplace=True)
    df_women.drop('TeamID', axis=1, inplace=True)
    df_women = df_women.merge(df_wteams[['TeamID', 'TeamName']], left_on='LTeamID', right_on='TeamID', how='left')
    df_women.rename(columns={'TeamName': 'LTeamName'}, inplace=True)
    df_women.drop('TeamID', axis=1, inplace=True)
    
    def process_game_w(row):
        lower_id = min(row['WTeamID'], row['LTeamID'])
        higher_id = max(row['WTeamID'], row['LTeamID'])
        if lower_id == row['WTeamID']:
            target = 1
        else:
            target = 0
        score_margin = abs(row['WScore'] - row['LScore'])
        return pd.Series({'Season': row['Season'], 'LowerTeamID': lower_id, 'HigherTeamID': higher_id,
                          'Target': target, 'ScoreMargin': score_margin, 'Tournament': 'Women',
                          'WTeamID': row['WTeamID'], 'LTeamID': row['LTeamID'],
                          'WTeamName': row['WTeamName'], 'LTeamName': row['LTeamName']})
    
    df_women_processed = df_women.apply(process_game_w, axis=1)
else:
    print('Women tournament data not available.')
    df_women_processed = pd.DataFrame()

# Combine men's and women's processed data
df_train = pd.concat([df_men_processed, df_women_processed], ignore_index=True)

# Optionally, further feature engineering can be done here
# For now, we have Season, LowerTeamID, HigherTeamID, Target, ScoreMargin and Tournament identifier

# Save the combined training data
output_path = 'train_data.csv'
df_train.to_csv(output_path, index=False)
print('Combined training data saved as ' + output_path)
print('Shape of combined training data:', df_train.shape)

# Preview the first few rows
print(df_train.head())


# In[ ]:


#initial gradient boosting in game
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import brier_score_loss
import matplotlib.pyplot as plt
import seaborn as sns

# Load the training dataset that we created
df_train = pd.read_csv('train_data.csv')

# For this example, we will use the features from the training data we created
# We'll use ScoreMargin and Season as features. In a real scenario, you can add more engineered features.
# Also, note that features like Season may need scaling or more processing. This is a baseline example.

# Use ScoreMargin as a numeric feature, but Season may be considered as categorical. Here, we'll try a simple model using ScoreMargin only

# Define feature set X and target y
# For illustration, we'll use a subset of columns as features. Let's use Season and ScoreMargin
X = df_train[['Season', 'ScoreMargin']]
y = df_train['Target']

# Split the data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("Training data shape:", X_train.shape)
print("Validation data shape:", X_valid.shape)

# Initialize an XGBoost classifier
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

# Set up a parameter grid. You can expand this grid to search over more parameters.
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2]
}

# Run a grid search with 3-fold cross-validation
grid_search = GridSearchCV(xgb_model, param_grid, scoring='neg_log_loss', cv=3, verbose=1)
grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)
print("Best score (neg_log_loss):", grid_search.best_score_)

# Use the best estimator from the grid search
best_model = grid_search.best_estimator_

# Predict probabilities on the validation set for class 1
y_pred_proba = best_model.predict_proba(X_valid)[:, 1]

# Evaluate using Brier score (lower is better)
brier = brier_score_loss(y_valid, y_pred_proba)
print("Brier score on validation set:", brier)

# Plot feature importance
plt.figure(figsize=(8, 5))
sns.barplot(x=best_model.feature_importances_, y=X_train.columns)
plt.title('Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

print("Gradient Boosting Model training and evaluation complete.")


# In[7]:


import pandas as pd
import numpy as np
import os

# Folder containing cleaned csv files
cleaned_folder = 'march-madness-2025-cleaned'

# Paths for men's tournament results and teams
men_results_path = os.path.join(cleaned_folder, 'MNCAATourneyDetailedResults.csv')
men_teams_path = os.path.join(cleaned_folder, 'MTeams.csv')

# Paths for women's tournament results and teams
women_results_path = os.path.join(cleaned_folder, 'WNCAATourneyDetailedResults.csv')
women_teams_path = os.path.join(cleaned_folder, 'WTeams.csv')

# Check if files exist
files_exist = True
for path in [men_results_path, men_teams_path, women_results_path, women_teams_path]:
    if not os.path.exists(path):
        print(f"File not found: {path}")
        files_exist = False

if not files_exist:
    print("Listing files in the cleaned folder:")
    print(os.listdir(cleaned_folder))

# Function to calculate historical team statistics
def calculate_team_stats(df_results, gender='M'):
    # Create separate dataframes for wins and losses
    wins = df_results[['Season', 'WTeamID', 'WScore', 'LScore']].copy()
    wins.rename(columns={'WTeamID': 'TeamID', 'WScore': 'TeamScore', 'LScore': 'OpponentScore'}, inplace=True)
    wins['Win'] = 1
    
    losses = df_results[['Season', 'LTeamID', 'LScore', 'WScore']].copy()
    losses.rename(columns={'LTeamID': 'TeamID', 'LScore': 'TeamScore', 'WScore': 'OpponentScore'}, inplace=True)
    losses['Win'] = 0
    
    # Combine wins and losses
    games = pd.concat([wins, losses], ignore_index=True)
    games['ScoreMargin'] = games['TeamScore'] - games['OpponentScore']
    games['Tournament'] = gender
    
    # Sort by Season to ensure chronological order
    games = games.sort_values('Season')
    
    # Initialize lists to store historical stats
    seasons = []
    team_ids = []
    hist_games = []
    hist_wins = []
    hist_win_pcts = []
    hist_avg_margins = []
    
    # Get unique team IDs and seasons
    unique_teams = games['TeamID'].unique()
    unique_seasons = games['Season'].unique()
    
    # For each team and season, calculate historical stats
    for team_id in unique_teams:
        team_games = games[games['TeamID'] == team_id]
        
        for season in unique_seasons:
            # Get games before this season
            past_games = team_games[team_games['Season'] < season]
            
            if len(past_games) > 0:
                # Calculate historical stats
                hist_game_count = len(past_games)
                hist_win_count = past_games['Win'].sum()
                hist_win_pct = hist_win_count / hist_game_count
                hist_avg_margin = past_games['ScoreMargin'].mean()
                
                # Append to lists
                seasons.append(season)
                team_ids.append(team_id)
                hist_games.append(hist_game_count)
                hist_wins.append(hist_win_count)
                hist_win_pcts.append(hist_win_pct)
                hist_avg_margins.append(hist_avg_margin)
    
    # Create dataframe with historical stats
    hist_stats = pd.DataFrame({
        'Season': seasons,
        'TeamID': team_ids,
        'HistGames': hist_games,
        'HistWins': hist_wins,
        'HistWinPct': hist_win_pcts,
        'HistAvgMargin': hist_avg_margins,
        'Tournament': gender
    })
    
    return hist_stats

# Process men's tournament data
if os.path.exists(men_results_path) and os.path.exists(men_teams_path):
    print("Processing men's tournament data...")
    df_men = pd.read_csv(men_results_path)
    df_mteams = pd.read_csv(men_teams_path)
    
    # Calculate historical stats for men's teams
    men_hist_stats = calculate_team_stats(df_men, 'Men')
    
    # Create matchup data
    def process_game(row, hist_stats):
        # Get lower and higher team IDs
        lower_id = min(row['WTeamID'], row['LTeamID'])
        higher_id = max(row['WTeamID'], row['LTeamID'])
        
        # Target: 1 if lower team won, 0 otherwise
        target = 1 if lower_id == row['WTeamID'] else 0
        
        # Get historical stats for both teams
        lower_stats = hist_stats[(hist_stats['Season'] == row['Season']) & 
                                (hist_stats['TeamID'] == lower_id)]
        higher_stats = hist_stats[(hist_stats['Season'] == row['Season']) & 
                                 (hist_stats['TeamID'] == higher_id)]
        
        # Extract historical stats if available
        lower_hist_games = lower_stats['HistGames'].values[0] if len(lower_stats) > 0 else np.nan
        lower_hist_wins = lower_stats['HistWins'].values[0] if len(lower_stats) > 0 else np.nan
        lower_hist_win_pct = lower_stats['HistWinPct'].values[0] if len(lower_stats) > 0 else np.nan
        lower_hist_avg_margin = lower_stats['HistAvgMargin'].values[0] if len(lower_stats) > 0 else np.nan
        
        higher_hist_games = higher_stats['HistGames'].values[0] if len(higher_stats) > 0 else np.nan
        higher_hist_wins = higher_stats['HistWins'].values[0] if len(higher_stats) > 0 else np.nan
        higher_hist_win_pct = higher_stats['HistWinPct'].values[0] if len(higher_stats) > 0 else np.nan
        higher_hist_avg_margin = higher_stats['HistAvgMargin'].values[0] if len(higher_stats) > 0 else np.nan
        
        # Calculate differences in historical stats
        win_pct_diff = lower_hist_win_pct - higher_hist_win_pct if not (np.isnan(lower_hist_win_pct) or np.isnan(higher_hist_win_pct)) else np.nan
        avg_margin_diff = lower_hist_avg_margin - higher_hist_avg_margin if not (np.isnan(lower_hist_avg_margin) or np.isnan(higher_hist_avg_margin)) else np.nan
        
        # Add score margin as a feature
        score_margin = abs(row['WScore'] - row['LScore'])
        
        return pd.Series({
            'Season': row['Season'],
            'LowerTeamID': lower_id,
            'HigherTeamID': higher_id,
            'Target': target,
            'ScoreMargin': score_margin,
            'Tournament': 'Men',
            'LowerTeam_HistGames': lower_hist_games,
            'LowerTeam_HistWins': lower_hist_wins,
            'LowerTeam_HistWinPct': lower_hist_win_pct,
            'LowerTeam_HistAvgMargin': lower_hist_avg_margin,
            'HigherTeam_HistGames': higher_hist_games,
            'HigherTeam_HistWins': higher_hist_wins,
            'HigherTeam_HistWinPct': higher_hist_win_pct,
            'HigherTeam_HistAvgMargin': higher_hist_avg_margin,
            'WinPctDiff': win_pct_diff,
            'AvgMarginDiff': avg_margin_diff
        })
    
    # Apply the processing function to each game
    df_men_processed = df_men.apply(lambda row: process_game(row, men_hist_stats), axis=1)
else:
    print("Men's tournament data not available.")
    df_men_processed = pd.DataFrame()

# Process women's tournament data
if os.path.exists(women_results_path) and os.path.exists(women_teams_path):
    print("Processing women's tournament data...")
    df_women = pd.read_csv(women_results_path)
    df_wteams = pd.read_csv(women_teams_path)
    
    # Calculate historical stats for women's teams
    women_hist_stats = calculate_team_stats(df_women, 'Women')
    
    # Apply the processing function to each game
    df_women_processed = df_women.apply(lambda row: process_game(row, women_hist_stats), axis=1)
else:
    print("Women's tournament data not available.")
    df_women_processed = pd.DataFrame()

# Combine men's and women's processed data
df_train = pd.concat([df_men_processed, df_women_processed], ignore_index=True)

# Drop rows with missing historical stats
df_train_clean = df_train.dropna(subset=['LowerTeam_HistWinPct', 'HigherTeam_HistWinPct'])
print(f"Original dataset size: {len(df_train)} rows")
print(f"After dropping rows with missing historical stats: {len(df_train_clean)} rows")

# Save the combined training data
output_path = 'train_data_with_pregame_features.csv'
df_train_clean.to_csv(output_path, index=False)
print(f'Enriched training data saved as {output_path}')

# Preview the first few rows
print("\
Preview of the enriched training data:")
print(df_train_clean.head())


# In[9]:


# Install the shap package
get_ipython().run_line_magic('pip', 'install shap')
print("SHAP package installed successfully.")


# In[10]:


import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectFromModel
import shap

# Load the enriched training dataset
df_train = pd.read_csv('train_data_with_pregame_features.csv')

# Display basic information about the dataset
print("Dataset shape:", df_train.shape)
print("\
Feature columns:")
for col in df_train.columns:
    if col != 'Target':
        print(f"- {col}")

# Define features and target
# Exclude non-predictive columns like ScoreMargin (post-game feature)
feature_cols = [
    'Season', 
    'LowerTeam_HistGames', 'LowerTeam_HistWins', 'LowerTeam_HistWinPct', 'LowerTeam_HistAvgMargin',
    'HigherTeam_HistGames', 'HigherTeam_HistWins', 'HigherTeam_HistWinPct', 'HigherTeam_HistAvgMargin',
    'WinPctDiff', 'AvgMarginDiff'
]

X = df_train[feature_cols]
y = df_train['Target']

# Split the data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("\
Training data shape:", X_train.shape)
print("Validation data shape:", X_valid.shape)

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)

# Convert back to DataFrame to keep column names
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_valid_scaled = pd.DataFrame(X_valid_scaled, columns=X_valid.columns)

# Train a baseline XGBoost model
print("\
Training baseline XGBoost model...")
baseline_model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)
baseline_model.fit(X_train_scaled, y_train)

# Evaluate baseline model
y_pred_proba = baseline_model.predict_proba(X_valid_scaled)[:, 1]
baseline_log_loss = log_loss(y_valid, y_pred_proba)
baseline_brier = brier_score_loss(y_valid, y_pred_proba)
baseline_auc = roc_auc_score(y_valid, y_pred_proba)
baseline_acc = accuracy_score(y_valid, y_pred_proba > 0.5)

print(f"Baseline model performance:")
print(f"- Log Loss: {baseline_log_loss:.4f}")
print(f"- Brier Score: {baseline_brier:.4f}")
print(f"- ROC AUC: {baseline_auc:.4f}")
print(f"- Accuracy: {baseline_acc:.4f}")

# Plot feature importance for baseline model
plt.figure(figsize=(10, 6))
feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': baseline_model.feature_importances_
}).sort_values('Importance', ascending=False)

sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('XGBoost Feature Importance')
plt.tight_layout()
plt.show()

# Feature selection using SelectFromModel
print("\
Performing feature selection...")
selector = SelectFromModel(baseline_model, threshold='median', prefit=True)
selected_features_mask = selector.get_support()
selected_features = X_train.columns[selected_features_mask].tolist()
print("Selected features:")
for feature in selected_features:
    print(f"- {feature}")

# Use only selected features
X_train_selected = X_train_scaled[selected_features]
X_valid_selected = X_valid_scaled[selected_features]

# Hyperparameter tuning with GridSearchCV
print("\
Performing hyperparameter tuning...")
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [2, 3, 4, 5],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'min_child_weight': [1, 3, 5]
}

grid_search = GridSearchCV(
    XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    param_grid,
    scoring='neg_log_loss',
    cv=5,
    verbose=1,
    n_jobs=-1
)

grid_search.fit(X_train_selected, y_train)

print("\
Best parameters:", grid_search.best_params_)
print("Best cross-validation score (neg_log_loss):", grid_search.best_score_)

# Get the best model
best_model = grid_search.best_estimator_

# Evaluate the best model on validation set
y_pred_proba_best = best_model.predict_proba(X_valid_selected)[:, 1]
best_log_loss = log_loss(y_valid, y_pred_proba_best)
best_brier = brier_score_loss(y_valid, y_pred_proba_best)
best_auc = roc_auc_score(y_valid, y_pred_proba_best)
best_acc = accuracy_score(y_valid, y_pred_proba_best > 0.5)

print("\
Best model performance:")
print(f"- Log Loss: {best_log_loss:.4f}")
print(f"- Brier Score: {best_brier:.4f}")
print(f"- ROC AUC: {best_auc:.4f}")
print(f"- Accuracy: {best_acc:.4f}")

# Compare baseline and best model
print("\
Improvement over baseline:")
print(f"- Log Loss: {(baseline_log_loss - best_log_loss) / baseline_log_loss * 100:.2f}%")
print(f"- Brier Score: {(baseline_brier - best_brier) / baseline_brier * 100:.2f}%")
print(f"- ROC AUC: {(best_auc - baseline_auc) / baseline_auc * 100:.2f}%")
print(f"- Accuracy: {(best_acc - baseline_acc) / baseline_acc * 100:.2f}%")

# Plot feature importance for best model
plt.figure(figsize=(10, 6))
feature_importance_best = pd.DataFrame({
    'Feature': selected_features,
    'Importance': best_model.feature_importances_
}).sort_values('Importance', ascending=False)

sns.barplot(x='Importance', y='Feature', data=feature_importance_best)
plt.title('Best XGBoost Model Feature Importance')
plt.tight_layout()
plt.show()

# Calculate SHAP values for model interpretability
print("\
Calculating SHAP values for model interpretability...")
explainer = shap.Explainer(best_model)
shap_values = explainer(X_valid_selected)

# Plot SHAP summary
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_valid_selected, plot_type="bar")
plt.tight_layout()
plt.show()

# Plot SHAP summary dot plot
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_valid_selected)
plt.tight_layout()
plt.show()

# Save the best model
import pickle
with open('best_xgboost_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

# Save the scaler and selected features for future use
with open('feature_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('selected_features.pkl', 'wb') as f:
    pickle.dump(selected_features, f)

print("\
Model, scaler, and selected features saved to disk.")
print("- best_xgboost_model.pkl")
print("- feature_scaler.pkl")
print("- selected_features.pkl")

print("\
Gradient Boosting model training and evaluation complete.")


# In[11]:


# Let's focus on improving the Brier score through probability calibration
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data and use the previously selected features
df_train = pd.read_csv('train_data_with_pregame_features.csv')

# Define features and target based on our previous feature selection
selected_features = [
    'LowerTeam_HistGames', 
    'LowerTeam_HistAvgMargin',
    'HigherTeam_HistGames', 
    'HigherTeam_HistWinPct',
    'WinPctDiff', 
    'AvgMarginDiff'
]

X = df_train[selected_features]
y = df_train['Target']

# Split the data into training, calibration, and test sets
from sklearn.model_selection import train_test_split

# First split: 80% train+calibration, 20% test
X_train_cal, X_test, y_train_cal, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Second split: 75% train, 25% calibration (from the 80% train+calibration)
X_train, X_cal, y_train, y_cal = train_test_split(
    X_train_cal, y_train_cal, test_size=0.25, random_state=42, stratify=y_train_cal
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Calibration set: {X_cal.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Scale the features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_cal_scaled = scaler.transform(X_cal)
X_test_scaled = scaler.transform(X_test)

# Train the base XGBoost model with the best parameters from our previous grid search
base_model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=3,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss'
)

base_model.fit(X_train_scaled, y_train)

# Get predictions on calibration and test sets
y_cal_pred_proba = base_model.predict_proba(X_cal_scaled)[:, 1]
y_test_pred_proba = base_model.predict_proba(X_test_scaled)[:, 1]

# Evaluate the base model
base_brier_cal = brier_score_loss(y_cal, y_cal_pred_proba)
base_brier_test = brier_score_loss(y_test, y_test_pred_proba)
base_log_loss_cal = log_loss(y_cal, y_cal_pred_proba)
base_log_loss_test = log_loss(y_test, y_test_pred_proba)
base_auc_cal = roc_auc_score(y_cal, y_cal_pred_proba)
base_auc_test = roc_auc_score(y_test, y_test_pred_proba)
base_acc_cal = accuracy_score(y_cal, y_cal_pred_proba > 0.5)
base_acc_test = accuracy_score(y_test, y_test_pred_proba > 0.5)

print("\
Base model performance on calibration set:")
print(f"- Brier Score: {base_brier_cal:.4f}")
print(f"- Log Loss: {base_log_loss_cal:.4f}")
print(f"- ROC AUC: {base_auc_cal:.4f}")
print(f"- Accuracy: {base_acc_cal:.4f}")

print("\
Base model performance on test set:")
print(f"- Brier Score: {base_brier_test:.4f}")
print(f"- Log Loss: {base_log_loss_test:.4f}")
print(f"- ROC AUC: {base_auc_test:.4f}")
print(f"- Accuracy: {base_acc_test:.4f}")

# Plot calibration curve for the base model
plt.figure(figsize=(10, 6))
prob_true, prob_pred = calibration_curve(y_cal, y_cal_pred_proba, n_bins=10)
plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
plt.plot(prob_pred, prob_true, 's-', label=f'Base XGBoost (Brier: {base_brier_cal:.4f})')
plt.xlabel('Mean predicted probability')
plt.ylabel('Fraction of positives')
plt.title('Calibration curve (reliability diagram) - Base Model')
plt.legend()
plt.grid(True)
plt.show()

# Now let's try different calibration methods
# 1. Platt Scaling (Logistic Regression)
platt_model = CalibratedClassifierCV(
    base_model, 
    method='sigmoid', 
    cv='prefit'
)
platt_model.fit(X_cal_scaled, y_cal)
y_cal_pred_proba_platt = platt_model.predict_proba(X_cal_scaled)[:, 1]
y_test_pred_proba_platt = platt_model.predict_proba(X_test_scaled)[:, 1]

# 2. Isotonic Regression
isotonic_model = CalibratedClassifierCV(
    base_model, 
    method='isotonic', 
    cv='prefit'
)
isotonic_model.fit(X_cal_scaled, y_cal)
y_cal_pred_proba_isotonic = isotonic_model.predict_proba(X_cal_scaled)[:, 1]
y_test_pred_proba_isotonic = isotonic_model.predict_proba(X_test_scaled)[:, 1]

# Evaluate calibrated models on calibration set
platt_brier_cal = brier_score_loss(y_cal, y_cal_pred_proba_platt)
isotonic_brier_cal = brier_score_loss(y_cal, y_cal_pred_proba_isotonic)

# Evaluate calibrated models on test set
platt_brier_test = brier_score_loss(y_test, y_test_pred_proba_platt)
platt_log_loss_test = log_loss(y_test, y_test_pred_proba_platt)
platt_auc_test = roc_auc_score(y_test, y_test_pred_proba_platt)
platt_acc_test = accuracy_score(y_test, y_test_pred_proba_platt > 0.5)

isotonic_brier_test = brier_score_loss(y_test, y_test_pred_proba_isotonic)
isotonic_log_loss_test = log_loss(y_test, y_test_pred_proba_isotonic)
isotonic_auc_test = roc_auc_score(y_test, y_test_pred_proba_isotonic)
isotonic_acc_test = accuracy_score(y_test, y_test_pred_proba_isotonic > 0.5)

print("\
Platt Scaling performance on test set:")
print(f"- Brier Score: {platt_brier_test:.4f}")
print(f"- Log Loss: {platt_log_loss_test:.4f}")
print(f"- ROC AUC: {platt_auc_test:.4f}")
print(f"- Accuracy: {platt_acc_test:.4f}")

print("\
Isotonic Regression performance on test set:")
print(f"- Brier Score: {isotonic_brier_test:.4f}")
print(f"- Log Loss: {isotonic_log_loss_test:.4f}")
print(f"- ROC AUC: {isotonic_auc_test:.4f}")
print(f"- Accuracy: {isotonic_acc_test:.4f}")

# Plot calibration curves for all models
plt.figure(figsize=(10, 6))
# Base model
prob_true_base, prob_pred_base = calibration_curve(y_test, y_test_pred_proba, n_bins=10)
plt.plot(prob_pred_base, prob_true_base, 's-', label=f'Base XGBoost (Brier: {base_brier_test:.4f})')

# Platt scaling
prob_true_platt, prob_pred_platt = calibration_curve(y_test, y_test_pred_proba_platt, n_bins=10)
plt.plot(prob_pred_platt, prob_true_platt, 'o-', label=f'Platt Scaling (Brier: {platt_brier_test:.4f})')

# Isotonic regression
prob_true_isotonic, prob_pred_isotonic = calibration_curve(y_test, y_test_pred_proba_isotonic, n_bins=10)
plt.plot(prob_pred_isotonic, prob_true_isotonic, '^-', label=f'Isotonic Regression (Brier: {isotonic_brier_test:.4f})')

# Perfect calibration
plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')

plt.xlabel('Mean predicted probability')
plt.ylabel('Fraction of positives')
plt.title('Calibration curves comparison on test set')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

# Let's also look at the probability distributions
plt.figure(figsize=(12, 6))

# Create a subplot for each model
plt.subplot(1, 3, 1)
sns.histplot(y_test_pred_proba, bins=20, kde=True)
plt.title(f'Base Model\
Brier: {base_brier_test:.4f}')
plt.xlabel('Predicted Probability')
plt.xlim(0, 1)

plt.subplot(1, 3, 2)
sns.histplot(y_test_pred_proba_platt, bins=20, kde=True)
plt.title(f'Platt Scaling\
Brier: {platt_brier_test:.4f}')
plt.xlabel('Predicted Probability')
plt.xlim(0, 1)

plt.subplot(1, 3, 3)
sns.histplot(y_test_pred_proba_isotonic, bins=20, kde=True)
plt.title(f'Isotonic Regression\
Brier: {isotonic_brier_test:.4f}')
plt.xlabel('Predicted Probability')
plt.xlim(0, 1)

plt.tight_layout()
plt.show()

# Determine the best calibration method based on Brier score
best_method = "Platt Scaling" if platt_brier_test < isotonic_brier_test else "Isotonic Regression"
best_model = platt_model if platt_brier_test < isotonic_brier_test else isotonic_model
best_brier = min(platt_brier_test, isotonic_brier_test)

print(f"\
Best calibration method: {best_method} with Brier score: {best_brier:.4f}")
print(f"Improvement over base model: {(base_brier_test - best_brier) / base_brier_test * 100:.2f}%")

# Save the best calibrated model
import pickle
with open('best_calibrated_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

with open('feature_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('selected_features.pkl', 'wb') as f:
    pickle.dump(selected_features, f)

print("\
Best calibrated model saved to 'best_calibrated_model.pkl'")
print("Feature scaler saved to 'feature_scaler.pkl'")
print("Selected features saved to 'selected_features.pkl'")


# In[24]:


# Let's continue with our analysis
import pandas as pd
import numpy as np
from tqdm import tqdm

# Load our base training data
train_df = pd.read_csv('train_data_with_pregame_features.csv')
print('Original training data shape:', train_df.shape)

# Load Massey Ordinals data
massey_df = pd.read_csv('MMasseyOrdinals.csv')
print('Massey Ordinals data shape:', massey_df.shape)

# Find unique ranking systems
unique_systems = massey_df['SystemName'].unique()
print(f'Found {len(unique_systems)} unique ranking systems. First 10:')
print(unique_systems[:10])

# Let's select a few popular ranking systems to use
selected_systems = ['POM', 'SAG', 'MOR', 'DOL']  # Pomeroy, Sagarin, Massey, Dolphin
print('Using these ranking systems:')
print(selected_systems)

# Function to process rankings for a specific system
def process_system_rankings(system_name):
    # Get the latest rankings before tournament (RankingDayNum close to 133)
    system_data = massey_df[massey_df['SystemName'] == system_name]
    
    # Group by season and get the maximum ranking day (closest to tournament)
    max_days = system_data.groupby('Season')['RankingDayNum'].max().reset_index()
    
    # Merge to get the latest rankings for each season
    latest_rankings = pd.merge(system_data, max_days, on=['Season', 'RankingDayNum'])
    
    # Keep only essential columns
    latest_rankings = latest_rankings[['Season', 'TeamID', 'OrdinalRank']]
    latest_rankings.columns = ['Season', 'TeamID', f'{system_name}_Rank']
    
    return latest_rankings

# Process each ranking system
rankings_dfs = {}
for system in selected_systems:
    rankings_dfs[system] = process_system_rankings(system)
    print(f'Processed {system} rankings, shape: {rankings_dfs[system].shape}')

# Add rankings to training data
for system in selected_systems:
    # Add lower team rankings
    train_df = pd.merge(
        train_df,
        rankings_dfs[system],
        left_on=['Season', 'LowerTeamID'],
        right_on=['Season', 'TeamID'],
        how='left'
    )
    train_df.drop('TeamID', axis=1, inplace=True)
    train_df.rename(columns={f'{system}_Rank': f'LowerTeam_{system}_Rank'}, inplace=True)
    
    # Add higher team rankings
    train_df = pd.merge(
        train_df,
        rankings_dfs[system],
        left_on=['Season', 'HigherTeamID'],
        right_on=['Season', 'TeamID'],
        how='left'
    )
    train_df.drop('TeamID', axis=1, inplace=True)
    train_df.rename(columns={f'{system}_Rank': f'HigherTeam_{system}_Rank'}, inplace=True)
    
    # Calculate ranking difference
    train_df[f'{system}_RankDiff'] = train_df[f'HigherTeam_{system}_Rank'] - train_df[f'LowerTeam_{system}_Rank']
    
    print(f'Added {system} rankings to training data')

# Now let's add team statistics from detailed results
detailed_results_df = pd.read_csv('MRegularSeasonDetailedResults.csv')
print('Detailed results shape:', detailed_results_df.shape)

# Create team statistics dataframe
team_stats = pd.DataFrame()

# Process each season to create team statistics
for season in tqdm(train_df['Season'].unique(), desc="Processing seasons"):
    season_data = detailed_results_df[detailed_results_df['Season'] == season]
    
    # Calculate offensive stats for winning teams
    w_offensive_cols = ['WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst', 'WTO', 'WStl', 'WBlk', 'WPF']
    w_team_stats = season_data.groupby('WTeamID')[w_offensive_cols].mean().reset_index()
    w_team_stats.columns = ['TeamID'] + [f'Off_{col[1:]}' for col in w_offensive_cols]
    
    # Calculate defensive stats for winning teams (against losing teams)
    l_defensive_cols = ['LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF']
    w_team_def_stats = season_data.groupby('WTeamID')[l_defensive_cols].mean().reset_index()
    w_team_def_stats.columns = ['TeamID'] + [f'Def_{col[1:]}' for col in l_defensive_cols]
    
    # Calculate offensive stats for losing teams
    l_offensive_cols = ['LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF']
    l_team_stats = season_data.groupby('LTeamID')[l_offensive_cols].mean().reset_index()
    l_team_stats.columns = ['TeamID'] + [f'Off_{col[1:]}' for col in l_offensive_cols]
    
    # Calculate defensive stats for losing teams (against winning teams)
    w_defensive_cols = ['WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst', 'WTO', 'WStl', 'WBlk', 'WPF']
    l_team_def_stats = season_data.groupby('LTeamID')[w_defensive_cols].mean().reset_index()
    l_team_def_stats.columns = ['TeamID'] + [f'Def_{col[1:]}' for col in w_defensive_cols]
    
    # Merge winning team offensive and defensive stats
    w_team_combined = pd.merge(w_team_stats, w_team_def_stats, on='TeamID', how='outer')
    
    # Merge losing team offensive and defensive stats
    l_team_combined = pd.merge(l_team_stats, l_team_def_stats, on='TeamID', how='outer')
    
    # Combine winning and losing team stats
    season_team_stats = pd.concat([w_team_combined, l_team_combined], ignore_index=True)
    
    # Group by TeamID and average the stats (in case a team appears in both winning and losing datasets)
    season_team_stats = season_team_stats.groupby('TeamID').mean().reset_index()
    
    # Add season column
    season_team_stats['Season'] = season
    
    # Append to overall team stats
    team_stats = pd.concat([team_stats, season_team_stats], ignore_index=True)

print('Team statistics shape:', team_stats.shape)

# Calculate derived statistics
team_stats['FG_Pct'] = team_stats['Off_FGM'] / team_stats['Off_FGA']
team_stats['FG3_Pct'] = team_stats['Off_FGM3'] / team_stats['Off_FGA3']
team_stats['FT_Pct'] = team_stats['Off_FTM'] / team_stats['Off_FTA']
team_stats['Def_FG_Pct'] = team_stats['Def_FGM'] / team_stats['Def_FGA']
team_stats['Def_FG3_Pct'] = team_stats['Def_FGM3'] / team_stats['Def_FGA3']
team_stats['Def_FT_Pct'] = team_stats['Def_FTM'] / team_stats['Def_FTA']
team_stats['Reb_Diff'] = (team_stats['Off_OR'] + team_stats['Off_DR']) - (team_stats['Def_OR'] + team_stats['Def_DR'])
team_stats['TO_Diff'] = team_stats['Def_TO'] - team_stats['Off_TO']
team_stats['Ast_Diff'] = team_stats['Off_Ast'] - team_stats['Def_Ast']
team_stats['Stl_Diff'] = team_stats['Off_Stl'] - team_stats['Def_Stl']
team_stats['Blk_Diff'] = team_stats['Off_Blk'] - team_stats['Def_Blk']
team_stats['PF_Diff'] = team_stats['Def_PF'] - team_stats['Off_PF']
team_stats['Scoring_Margin'] = (team_stats['Off_FGM']*2 + team_stats['Off_FGM3'] + team_stats['Off_FTM']) - (team_stats['Def_FGM']*2 + team_stats['Def_FGM3'] + team_stats['Def_FTM'])
team_stats['Possessions'] = team_stats['Off_FGA'] - team_stats['Off_OR'] + team_stats['Off_TO'] + 0.4*team_stats['Off_FTA']

# Select derived statistics for the model
derived_stats = team_stats[['Season', 'TeamID', 'FG_Pct', 'FG3_Pct', 'FT_Pct', 'Def_FG_Pct', 'Def_FG3_Pct', 'Def_FT_Pct', 
                           'Reb_Diff', 'TO_Diff', 'Ast_Diff', 'Stl_Diff', 'Blk_Diff', 'PF_Diff', 'Scoring_Margin']]
print('Derived team statistics shape:', derived_stats.shape)

# Add team statistics to training data
# Add lower team stats
train_df = pd.merge(
    train_df,
    derived_stats,
    left_on=['Season', 'LowerTeamID'],
    right_on=['Season', 'TeamID'],
    how='left'
)
train_df.drop('TeamID', axis=1, inplace=True)
# Rename columns to indicate lower team
for col in derived_stats.columns:
    if col not in ['Season', 'TeamID']:
        train_df.rename(columns={col: f'LowerTeam_{col}'}, inplace=True)

# Add higher team stats
train_df = pd.merge(
    train_df,
    derived_stats,
    left_on=['Season', 'HigherTeamID'],
    right_on=['Season', 'TeamID'],
    how='left'
)
train_df.drop('TeamID', axis=1, inplace=True)
# Rename columns to indicate higher team
for col in derived_stats.columns:
    if col not in ['Season', 'TeamID']:
        train_df.rename(columns={col: f'HigherTeam_{col}'}, inplace=True)

# Calculate differences between team stats
for col in derived_stats.columns:
    if col not in ['Season', 'TeamID']:
        train_df[f'{col}_Diff'] = train_df[f'LowerTeam_{col}'] - train_df[f'HigherTeam_{col}']

# Handle any NaN values in the final dataset - THIS IS WHERE THE ERROR OCCURRED BEFORE
# Fix: Use numeric_only=True to avoid issues with non-numeric columns
train_df = train_df.fillna(train_df.mean(numeric_only=True))

print('Final enhanced training data shape:', train_df.shape)
print('New columns added:', len(train_df.columns) - 16)  # 16 is the original number of columns

# Display a sample of the enhanced data
print('\
Sample of enhanced training data:')
print(train_df.head(2))


# In[25]:


# Continue with model building and evaluation
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Load our enhanced training data
train_df = pd.read_csv('train_data_with_pregame_features.csv')
print('Training data shape:', train_df.shape)

# Check if we have the enhanced features from our previous analysis
if train_df.shape[1] < 50:  # If we don't have the enhanced features
    print("Enhanced features not found in the CSV. Re-running feature engineering...")
    
    # Load Massey Ordinals data
    massey_df = pd.read_csv('MMasseyOrdinals.csv')
    
    # Find unique ranking systems
    unique_systems = massey_df['SystemName'].unique()
    
    # Select a few popular ranking systems to use
    selected_systems = ['POM', 'SAG', 'MOR', 'DOL']  # Pomeroy, Sagarin, Massey, Dolphin
    
    # Function to process rankings for a specific system
    def process_system_rankings(system_name):
        # Get the latest rankings before tournament (RankingDayNum close to 133)
        system_data = massey_df[massey_df['SystemName'] == system_name]
        
        # Group by season and get the maximum ranking day (closest to tournament)
        max_days = system_data.groupby('Season')['RankingDayNum'].max().reset_index()
        
        # Merge to get the latest rankings for each season
        latest_rankings = pd.merge(system_data, max_days, on=['Season', 'RankingDayNum'])
        
        # Keep only essential columns
        latest_rankings = latest_rankings[['Season', 'TeamID', 'OrdinalRank']]
        latest_rankings.columns = ['Season', 'TeamID', f'{system_name}_Rank']
        
        return latest_rankings
    
    # Process each ranking system
    rankings_dfs = {}
    for system in selected_systems:
        rankings_dfs[system] = process_system_rankings(system)
        print(f'Processed {system} rankings, shape: {rankings_dfs[system].shape}')
    
    # Add rankings to training data
    for system in selected_systems:
        # Add lower team rankings
        train_df = pd.merge(
            train_df,
            rankings_dfs[system],
            left_on=['Season', 'LowerTeamID'],
            right_on=['Season', 'TeamID'],
            how='left'
        )
        train_df.drop('TeamID', axis=1, inplace=True)
        train_df.rename(columns={f'{system}_Rank': f'LowerTeam_{system}_Rank'}, inplace=True)
        
        # Add higher team rankings
        train_df = pd.merge(
            train_df,
            rankings_dfs[system],
            left_on=['Season', 'HigherTeamID'],
            right_on=['Season', 'TeamID'],
            how='left'
        )
        train_df.drop('TeamID', axis=1, inplace=True)
        train_df.rename(columns={f'{system}_Rank': f'HigherTeam_{system}_Rank'}, inplace=True)
        
        # Calculate ranking difference
        train_df[f'{system}_RankDiff'] = train_df[f'HigherTeam_{system}_Rank'] - train_df[f'LowerTeam_{system}_Rank']
    
    # Now let's add team statistics from detailed results
    detailed_results_df = pd.read_csv('MRegularSeasonDetailedResults.csv')
    
    # Create team statistics dataframe
    team_stats = pd.DataFrame()
    
    # Process each season to create team statistics
    for season in tqdm(train_df['Season'].unique(), desc="Processing seasons"):
        season_data = detailed_results_df[detailed_results_df['Season'] == season]
        
        # Calculate offensive stats for winning teams
        w_offensive_cols = ['WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst', 'WTO', 'WStl', 'WBlk', 'WPF']
        w_team_stats = season_data.groupby('WTeamID')[w_offensive_cols].mean().reset_index()
        w_team_stats.columns = ['TeamID'] + [f'Off_{col[1:]}' for col in w_offensive_cols]
        
        # Calculate defensive stats for winning teams (against losing teams)
        l_defensive_cols = ['LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF']
        w_team_def_stats = season_data.groupby('WTeamID')[l_defensive_cols].mean().reset_index()
        w_team_def_stats.columns = ['TeamID'] + [f'Def_Against_{col[1:]}' for col in l_defensive_cols]
        
        # Calculate offensive stats for losing teams
        l_offensive_cols = ['LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF']
        l_team_stats = season_data.groupby('LTeamID')[l_offensive_cols].mean().reset_index()
        l_team_stats.columns = ['TeamID'] + [f'Off_{col[1:]}' for col in l_offensive_cols]
        
        # Calculate defensive stats for losing teams (against winning teams)
        w_defensive_cols = ['WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst', 'WTO', 'WStl', 'WBlk', 'WPF']
        l_team_def_stats = season_data.groupby('LTeamID')[w_defensive_cols].mean().reset_index()
        l_team_def_stats.columns = ['TeamID'] + [f'Def_Against_{col[1:]}' for col in w_defensive_cols]
        
        # Merge winning team offensive and defensive stats
        w_team_combined = pd.merge(w_team_stats, w_team_def_stats, on='TeamID', how='outer')
        
        # Merge losing team offensive and defensive stats
        l_team_combined = pd.merge(l_team_stats, l_team_def_stats, on='TeamID', how='outer')
        
        # Combine winning and losing team stats
        season_team_stats = pd.concat([w_team_combined, l_team_combined], ignore_index=True)
        
        # Group by TeamID and average the stats (in case a team appears in both winning and losing datasets)
        season_team_stats = season_team_stats.groupby('TeamID').mean().reset_index()
        
        # Add season column
        season_team_stats['Season'] = season
        
        # Append to overall team stats
        team_stats = pd.concat([team_stats, season_team_stats], ignore_index=True)
    
    # Calculate derived statistics
    team_stats['FG_Pct'] = team_stats['Off_FGM'] / team_stats['Off_FGA']
    team_stats['FG3_Pct'] = team_stats['Off_FGM3'] / team_stats['Off_FGA3']
    team_stats['FT_Pct'] = team_stats['Off_FTM'] / team_stats['Off_FTA']
    team_stats['Def_FG_Pct'] = team_stats['Def_Against_FGM'] / team_stats['Def_Against_FGA']
    team_stats['Def_FG3_Pct'] = team_stats['Def_Against_FGM3'] / team_stats['Def_Against_FGA3']
    team_stats['Def_FT_Pct'] = team_stats['Def_Against_FTM'] / team_stats['Def_Against_FTA']
    team_stats['Reb_Diff'] = (team_stats['Off_OR'] + team_stats['Off_DR']) - (team_stats['Def_Against_OR'] + team_stats['Def_Against_DR'])
    team_stats['TO_Diff'] = team_stats['Def_Against_TO'] - team_stats['Off_TO']
    team_stats['Ast_Diff'] = team_stats['Off_Ast'] - team_stats['Def_Against_Ast']
    team_stats['Stl_Diff'] = team_stats['Off_Stl'] - team_stats['Def_Against_Stl']
    team_stats['Blk_Diff'] = team_stats['Off_Blk'] - team_stats['Def_Against_Blk']
    team_stats['PF_Diff'] = team_stats['Def_Against_PF'] - team_stats['Off_PF']
    team_stats['Scoring_Margin'] = (team_stats['Off_FGM']*2 + team_stats['Off_FGM3'] + team_stats['Off_FTM']) - (team_stats['Def_Against_FGM']*2 + team_stats['Def_Against_FGM3'] + team_stats['Def_Against_FTM'])
    
    # Select derived statistics for the model
    derived_stats = team_stats[['Season', 'TeamID', 'FG_Pct', 'FG3_Pct', 'FT_Pct', 'Def_FG_Pct', 'Def_FG3_Pct', 'Def_FT_Pct', 
                               'Reb_Diff', 'TO_Diff', 'Ast_Diff', 'Stl_Diff', 'Blk_Diff', 'PF_Diff', 'Scoring_Margin']]
    
    # Add team statistics to training data
    # Add lower team stats
    train_df = pd.merge(
        train_df,
        derived_stats,
        left_on=['Season', 'LowerTeamID'],
        right_on=['Season', 'TeamID'],
        how='left'
    )
    train_df.drop('TeamID', axis=1, inplace=True)
    # Rename columns to indicate lower team
    for col in derived_stats.columns:
        if col not in ['Season', 'TeamID']:
            train_df.rename(columns={col: f'LowerTeam_{col}'}, inplace=True)
    
    # Add higher team stats
    train_df = pd.merge(
        train_df,
        derived_stats,
        left_on=['Season', 'HigherTeamID'],
        right_on=['Season', 'TeamID'],
        how='left'
    )
    train_df.drop('TeamID', axis=1, inplace=True)
    # Rename columns to indicate higher team
    for col in derived_stats.columns:
        if col not in ['Season', 'TeamID']:
            train_df.rename(columns={col: f'HigherTeam_{col}'}, inplace=True)
    
    # Calculate differences between team stats
    for col in derived_stats.columns:
        if col not in ['Season', 'TeamID']:
            train_df[f'{col}_Diff'] = train_df[f'LowerTeam_{col}'] - train_df[f'HigherTeam_{col}']
    
    # Handle any NaN values in the final dataset
    train_df = train_df.fillna(train_df.mean(numeric_only=True))
    
    print('Enhanced training data shape:', train_df.shape)

# Now let's prepare for modeling
print("\
Preparing data for modeling...")

# Check for any remaining NaN values
print("Number of NaN values in dataset:", train_df.isna().sum().sum())

# If there are NaN values, fill them
if train_df.isna().sum().sum() > 0:
    train_df = train_df.fillna(train_df.mean(numeric_only=True))
    print("Filled NaN values with mean")

# Drop non-numeric columns for correlation analysis
numeric_df = train_df.select_dtypes(include=[np.number])

# Calculate correlation with target
target_corr = numeric_df.corr()['Target'].sort_values(ascending=False)
print("\
Top 10 features correlated with Target:")
print(target_corr.head(11))  # 11 because Target itself will be included
print("\
Bottom 10 features correlated with Target:")
print(target_corr.tail(10))

# Visualize correlation with target
plt.figure(figsize=(12, 8))
top_features = target_corr.drop('Target').abs().sort_values(ascending=False).head(15).index
correlation_data = numeric_df[list(top_features) + ['Target']].corr()['Target'].drop('Target')
sns.barplot(x=correlation_data.values, y=correlation_data.index)
plt.title('Top 15 Features by Correlation with Target')
plt.xlabel('Correlation Coefficient')
plt.tight_layout()
plt.show()

# Prepare features and target
X = numeric_df.drop('Target', axis=1)
y = numeric_df['Target']

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("\
Training set shape:", X_train.shape)
print("Validation set shape:", X_val.shape)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Train a Random Forest model
print("\
Training Random Forest model...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred_rf = rf_model.predict(X_val_scaled)
y_pred_proba_rf = rf_model.predict_proba(X_val_scaled)[:, 1]

# Evaluate the model
print("\
Random Forest Model Evaluation:")
print("Accuracy:", accuracy_score(y_val, y_pred_rf))
print("ROC AUC:", roc_auc_score(y_val, y_pred_proba_rf))
print("Log Loss:", log_loss(y_val, y_pred_proba_rf))
print("\
Confusion Matrix:")
print(confusion_matrix(y_val, y_pred_rf))
print("\
Classification Report:")
print(classification_report(y_val, y_pred_rf))

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\
Top 15 most important features:")
print(feature_importance.head(15))

# Visualize feature importance
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance.head(15))
plt.title('Random Forest Feature Importance')
plt.tight_layout()
plt.show()

# Train a Gradient Boosting model
print("\
Training Gradient Boosting model...")
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred_gb = gb_model.predict(X_val_scaled)
y_pred_proba_gb = gb_model.predict_proba(X_val_scaled)[:, 1]

# Evaluate the model
print("\
Gradient Boosting Model Evaluation:")
print("Accuracy:", accuracy_score(y_val, y_pred_gb))
print("ROC AUC:", roc_auc_score(y_val, y_pred_proba_gb))
print("Log Loss:", log_loss(y_val, y_pred_proba_gb))
print("\
Confusion Matrix:")
print(confusion_matrix(y_val, y_pred_gb))
print("\
Classification Report:")
print(classification_report(y_val, y_pred_gb))

# Feature importance for Gradient Boosting
gb_feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': gb_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\
Top 15 most important features (Gradient Boosting):")
print(gb_feature_importance.head(15))

# Visualize feature importance
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=gb_feature_importance.head(15))
plt.title('Gradient Boosting Feature Importance')
plt.tight_layout()
plt.show()

# Compare model performance
models = ['Random Forest', 'Gradient Boosting']
accuracy = [accuracy_score(y_val, y_pred_rf), accuracy_score(y_val, y_pred_gb)]
roc_auc = [roc_auc_score(y_val, y_pred_proba_rf), roc_auc_score(y_val, y_pred_proba_gb)]
log_loss_values = [log_loss(y_val, y_pred_proba_rf), log_loss(y_val, y_pred_proba_gb)]

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.barplot(x=models, y=accuracy)
plt.title('Accuracy')
plt.ylim(0.5, 1.0)

plt.subplot(1, 3, 2)
sns.barplot(x=models, y=roc_auc)
plt.title('ROC AUC')
plt.ylim(0.5, 1.0)

plt.subplot(1, 3, 3)
sns.barplot(x=models, y=log_loss_values)
plt.title('Log Loss')

plt.tight_layout()
plt.show()

# Save the best model and scaler for future use
import pickle

if roc_auc_score(y_val, y_pred_proba_rf) > roc_auc_score(y_val, y_pred_proba_gb):
    best_model = rf_model
    print("\
Random Forest is the better model. Saving...")
else:
    best_model = gb_model
    print("\
Gradient Boosting is the better model. Saving...")

# Save the model
with open('best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

# Save the scaler
with open('feature_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Save the feature names
selected_features = X.columns.tolist()
with open('selected_features.pkl', 'wb') as f:
    pickle.dump(selected_features, f)

print("Model, scaler, and feature names saved successfully.")


# In[29]:


# Continue with model building and evaluation
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Load our training data
train_df = pd.read_csv('train_data_with_pregame_features.csv')
print('Training data shape:', train_df.shape)
print('Columns:', train_df.columns.tolist())

# Load Massey Ordinals data
massey_df = pd.read_csv('MMasseyOrdinals.csv')
print('Massey Ordinals data shape:', massey_df.shape)

# Find unique ranking systems
unique_systems = massey_df['SystemName'].unique()
print('Found', len(unique_systems), 'unique ranking systems. First 10:')
print(unique_systems[:10])

# Select a few popular ranking systems to use
selected_systems = ['POM', 'SAG', 'MOR', 'DOL']  # Pomeroy, Sagarin, Massey, Dolphin
print('Using these ranking systems:')
print(selected_systems)

# Function to process rankings for a specific system
def process_system_rankings(system_name):
    # Get the latest rankings before tournament (RankingDayNum close to 133)
    system_data = massey_df[massey_df['SystemName'] == system_name]
    
    # Group by season and get the maximum ranking day (closest to tournament)
    max_days = system_data.groupby('Season')['RankingDayNum'].max().reset_index()
    
    # Merge to get the latest rankings for each season
    latest_rankings = pd.merge(system_data, max_days, on=['Season', 'RankingDayNum'])
    
    # Keep only essential columns
    latest_rankings = latest_rankings[['Season', 'TeamID', 'OrdinalRank']]
    latest_rankings.columns = ['Season', 'TeamID', f'{system_name}_Rank']
    
    return latest_rankings

# Process each ranking system
rankings_dfs = {}
for system in selected_systems:
    rankings_dfs[system] = process_system_rankings(system)
    print(f'Processed {system} rankings, shape: {rankings_dfs[system].shape}')

# Add rankings to training data
for system in selected_systems:
    # Add lower team rankings
    train_df = pd.merge(
        train_df,
        rankings_dfs[system],
        left_on=['Season', 'LowerTeamID'],
        right_on=['Season', 'TeamID'],
        how='left'
    )
    train_df.drop('TeamID', axis=1, inplace=True)
    train_df.rename(columns={f'{system}_Rank': f'LowerTeam_{system}_Rank'}, inplace=True)
    
    # Add higher team rankings
    train_df = pd.merge(
        train_df,
        rankings_dfs[system],
        left_on=['Season', 'HigherTeamID'],
        right_on=['Season', 'TeamID'],
        how='left'
    )
    train_df.drop('TeamID', axis=1, inplace=True)
    train_df.rename(columns={f'{system}_Rank': f'HigherTeam_{system}_Rank'}, inplace=True)
    
    # Calculate ranking difference
    train_df[f'{system}_RankDiff'] = train_df[f'HigherTeam_{system}_Rank'] - train_df[f'LowerTeam_{system}_Rank']
    print(f'Added {system} rankings to training data')

# Now let's add team statistics from detailed results
detailed_results_df = pd.read_csv('MRegularSeasonDetailedResults.csv')
print('\
Detailed results shape:')
print(detailed_results_df.shape)

# Create team statistics dataframe
team_stats = pd.DataFrame()

# Process each season to create team statistics
for season in tqdm(train_df['Season'].unique(), desc="Processing seasons"):
    season_data = detailed_results_df[detailed_results_df['Season'] == season]
    
    # Calculate offensive stats for winning teams
    w_offensive_cols = ['WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst', 'WTO', 'WStl', 'WBlk', 'WPF']
    w_team_stats = season_data.groupby('WTeamID')[w_offensive_cols].mean().reset_index()
    w_team_stats.columns = ['TeamID'] + [f'Off_{col[1:]}' for col in w_offensive_cols]
    
    # Calculate defensive stats for winning teams (against losing teams)
    l_defensive_cols = ['LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF']
    w_team_def_stats = season_data.groupby('WTeamID')[l_defensive_cols].mean().reset_index()
    w_team_def_stats.columns = ['TeamID'] + [f'Def_Against_{col[1:]}' for col in l_defensive_cols]
    
    # Calculate offensive stats for losing teams
    l_offensive_cols = ['LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF']
    l_team_stats = season_data.groupby('LTeamID')[l_offensive_cols].mean().reset_index()
    l_team_stats.columns = ['TeamID'] + [f'Off_{col[1:]}' for col in l_offensive_cols]
    
    # Calculate defensive stats for losing teams (against winning teams)
    w_defensive_cols = ['WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst', 'WTO', 'WStl', 'WBlk', 'WPF']
    l_team_def_stats = season_data.groupby('LTeamID')[w_defensive_cols].mean().reset_index()
    l_team_def_stats.columns = ['TeamID'] + [f'Def_Against_{col[1:]}' for col in w_defensive_cols]
    
    # Merge winning team offensive and defensive stats
    w_team_combined = pd.merge(w_team_stats, w_team_def_stats, on='TeamID', how='outer')
    
    # Merge losing team offensive and defensive stats
    l_team_combined = pd.merge(l_team_stats, l_team_def_stats, on='TeamID', how='outer')
    
    # Combine winning and losing team stats
    season_team_stats = pd.concat([w_team_combined, l_team_combined], ignore_index=True)
    
    # Group by TeamID and average the stats (in case a team appears in both winning and losing datasets)
    season_team_stats = season_team_stats.groupby('TeamID').mean().reset_index()
    
    # Add season column
    season_team_stats['Season'] = season
    
    # Append to overall team stats
    team_stats = pd.concat([team_stats, season_team_stats], ignore_index=True)

print('\
Team statistics shape:')
print(team_stats.shape)

# Calculate derived statistics
team_stats['FG_Pct'] = team_stats['Off_FGM'] / team_stats['Off_FGA']
team_stats['FG3_Pct'] = team_stats['Off_FGM3'] / team_stats['Off_FGA3']
team_stats['FT_Pct'] = team_stats['Off_FTM'] / team_stats['Off_FTA']
team_stats['Def_FG_Pct'] = team_stats['Def_Against_FGM'] / team_stats['Def_Against_FGA']
team_stats['Def_FG3_Pct'] = team_stats['Def_Against_FGM3'] / team_stats['Def_Against_FGA3']
team_stats['Def_FT_Pct'] = team_stats['Def_Against_FTM'] / team_stats['Def_Against_FTA']
team_stats['Reb_Diff'] = (team_stats['Off_OR'] + team_stats['Off_DR']) - (team_stats['Def_Against_OR'] + team_stats['Def_Against_DR'])
team_stats['TO_Diff'] = team_stats['Def_Against_TO'] - team_stats['Off_TO']
team_stats['Ast_Diff'] = team_stats['Off_Ast'] - team_stats['Def_Against_Ast']
team_stats['Stl_Diff'] = team_stats['Off_Stl'] - team_stats['Def_Against_Stl']
team_stats['Blk_Diff'] = team_stats['Off_Blk'] - team_stats['Def_Against_Blk']
team_stats['PF_Diff'] = team_stats['Def_Against_PF'] - team_stats['Off_PF']
team_stats['Scoring_Margin'] = (team_stats['Off_FGM']*2 + team_stats['Off_FGM3'] + team_stats['Off_FTM']) - (team_stats['Def_Against_FGM']*2 + team_stats['Def_Against_FGM3'] + team_stats['Def_Against_FTM'])

# Select derived statistics for the model
derived_stats = team_stats[['Season', 'TeamID', 'FG_Pct', 'FG3_Pct', 'FT_Pct', 'Def_FG_Pct', 'Def_FG3_Pct', 'Def_FT_Pct', 
                           'Reb_Diff', 'TO_Diff', 'Ast_Diff', 'Stl_Diff', 'Blk_Diff', 'PF_Diff', 'Scoring_Margin']]

print('\
Derived team statistics shape:')
print(derived_stats.shape)

# Add team statistics to training data
# Add lower team stats
train_df = pd.merge(
    train_df,
    derived_stats,
    left_on=['Season', 'LowerTeamID'],
    right_on=['Season', 'TeamID'],
    how='left'
)
train_df.drop('TeamID', axis=1, inplace=True)
# Rename columns to indicate lower team
for col in derived_stats.columns:
    if col not in ['Season', 'TeamID']:
        train_df.rename(columns={col: f'LowerTeam_{col}'}, inplace=True)

# Add higher team stats
train_df = pd.merge(
    train_df,
    derived_stats,
    left_on=['Season', 'HigherTeamID'],
    right_on=['Season', 'TeamID'],
    how='left'
)
train_df.drop('TeamID', axis=1, inplace=True)
# Rename columns to indicate higher team
for col in derived_stats.columns:
    if col not in ['Season', 'TeamID']:
        train_df.rename(columns={col: f'HigherTeam_{col}'}, inplace=True)

# Calculate differences between team stats
for col in derived_stats.columns:
    if col not in ['Season', 'TeamID']:
        train_df[f'{col}_Diff'] = train_df[f'LowerTeam_{col}'] - train_df[f'HigherTeam_{col}']

# Handle any NaN values in the final dataset
train_df = train_df.fillna(train_df.mean(numeric_only=True))

print('\
Final enhanced training data shape:')
print(train_df.shape)

print('New columns added:')
print(train_df.shape[1] - 16)  # 16 was the original number of columns

# Save the enhanced training data
train_df.to_csv('train_data_enhanced.csv', index=False)
print('\
Enhanced training data saved to train_data_enhanced.csv')

# Display a sample of the enhanced data
print('\
Sample of enhanced training data:')
print(train_df.head(2).T)  # Transpose to see more columns


# In[1]:


# This cell performs model training & evaluation using Gradient Boosting and Random Forest on the enhanced training data.
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load the enhanced training data
df = pd.read_csv('train_data_enhanced.csv')
print('Enhanced training data shape:', df.shape)

# Define predictor features and target
# Exclude identifying columns: Season, LowerTeamID, HigherTeamID, Tournament
X = df.drop(columns=['Season', 'LowerTeamID', 'HigherTeamID', 'Tournament', 'Target'])

y = df['Target']

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print('Training set:', X_train.shape, 'Test set:', X_test.shape)

# Initialize models
gb_model = GradientBoostingClassifier(random_state=42)
rf_model = RandomForestClassifier(random_state=42)

# Train Gradient Boosting
gb_model.fit(X_train, y_train)
gb_pred_proba = gb_model.predict_proba(X_test)[:,1]
gb_pred = gb_model.predict(X_test)

# Train Random Forest
rf_model.fit(X_train, y_train)
rf_pred_proba = rf_model.predict_proba(X_test)[:,1]
rf_pred = rf_model.predict(X_test)

# Define a function to compute metrics
def evaluate_model(model_name, y_true, pred, pred_proba):
    acc = accuracy_score(y_true, pred)
    ll = log_loss(y_true, pred_proba)
    roc = roc_auc_score(y_true, pred_proba)
    # Gini coefficient: 2*AUC - 1
    gini = 2 * roc - 1
    print(f'--- {model_name} Evaluation ---')
    print('Accuracy:', acc)
    print('Log Loss:', ll)
    print('ROC AUC:', roc)
    print('Gini Coefficient:', gini)
    print('Confusion Matrix:\
', confusion_matrix(y_true, pred))
    print('Classification Report:\
', classification_report(y_true, pred))
    return acc, ll, roc, gini

print('Gradient Boosting Metrics:')
gb_metrics = evaluate_model('Gradient Boosting', y_test, gb_pred, gb_pred_proba)

print('Random Forest Metrics:')
rf_metrics = evaluate_model('Random Forest', y_test, rf_pred, rf_pred_proba)

# Plot feature importance for both models
# For Gradient Boosting
gb_importance = pd.Series(gb_model.feature_importances_, index=X.columns).sort_values(ascending=False)
rf_importance = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
sns.barplot(x=gb_importance[:10], y=gb_importance.index[:10])
plt.title('Top 10 Features - Gradient Boosting')

plt.subplot(1,2,2)
sns.barplot(x=rf_importance[:10], y=rf_importance.index[:10])
plt.title('Top 10 Features - Random Forest')

plt.tight_layout()
plt.show()

print('done')


# In[3]:


from sklearn.metrics import brier_score_loss

# Compute Brier Score for both models

gb_brier = brier_score_loss(y_test, gb_pred_proba)
rf_brier = brier_score_loss(y_test, rf_pred_proba)

print('Gradient Boosting Brier Score:', gb_brier)
print('Random Forest Brier Score:', rf_brier)


# In[ ]:


# This cell implements a stacking ensemble using Gradient Boosting, Random Forest as base models and Logistic Regression as meta-model.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, confusion_matrix, classification_report, brier_score_loss

# Load enhanced training data
df = pd.read_csv('train_data_enhanced.csv')

# Prepare features and target
X = df.drop(columns=['Season', 'LowerTeamID', 'HigherTeamID', 'Tournament', 'Target'])
y = df['Target']

# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define base models
estimators = [
    ('gb', GradientBoostingClassifier(random_state=42)),
    ('rf', RandomForestClassifier(random_state=42))
]

# Define meta-model
meta_model = LogisticRegression(max_iter=1000, random_state=42)

# Create Stacking Classifier
stack_model = StackingClassifier(estimators=estimators, final_estimator=meta_model, cv=5, passthrough=False)

# Fit the stacking model
stack_model.fit(X_train, y_train)

# Predict probabilities and classes
stack_pred_proba = stack_model.predict_proba(X_test)[:, 1]
stack_pred = stack_model.predict(X_test)

# Evaluate the stacking ensemble
acc = accuracy_score(y_test, stack_pred)
ll = log_loss(y_test, stack_pred_proba)
roc = roc_auc_score(y_test, stack_pred_proba)
gini = 2 * roc - 1
brier = brier_score_loss(y_test, stack_pred_proba)

print('--- Stacking Ensemble Evaluation ---')
print('Accuracy:', acc)
print('Log Loss:', ll)
print('ROC AUC:', roc)
print('Gini Coefficient:', gini)
print('Brier Score:', brier)
print('Confusion Matrix:\
', confusion_matrix(y_test, stack_pred))
print('Classification Report:\
', classification_report(y_test, stack_pred))

print('done')


# In[4]:


# This cell implements a stacking ensemble using Gradient Boosting, Random Forest as base models and Logistic Regression as meta-model.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, confusion_matrix, classification_report, brier_score_loss

# Load enhanced training data
df = pd.read_csv('train_data_enhanced.csv')

# Prepare features and target
X = df.drop(columns=['Season', 'LowerTeamID', 'HigherTeamID', 'Tournament', 'Target'])
y = df['Target']

# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define base models
estimators = [
    ('gb', GradientBoostingClassifier(random_state=42)),
    ('rf', RandomForestClassifier(random_state=42))
]

# Define meta-model
meta_model = LogisticRegression(max_iter=1000, random_state=42)

# Create Stacking Classifier
stack_model = StackingClassifier(estimators=estimators, final_estimator=meta_model, cv=5, passthrough=False)

# Fit the stacking model
stack_model.fit(X_train, y_train)

# Predict probabilities and classes
stack_pred_proba = stack_model.predict_proba(X_test)[:, 1]
stack_pred = stack_model.predict(X_test)

# Evaluate the stacking ensemble
acc = accuracy_score(y_test, stack_pred)
ll = log_loss(y_test, stack_pred_proba)
roc = roc_auc_score(y_test, stack_pred_proba)
gini = 2 * roc - 1
brier = brier_score_loss(y_test, stack_pred_proba)

print('--- Stacking Ensemble Evaluation ---')
print('Accuracy:', acc)
print('Log Loss:', ll)
print('ROC AUC:', roc)
print('Gini Coefficient:', gini)
print('Brier Score:', brier)
print('Confusion Matrix:\
', confusion_matrix(y_test, stack_pred))
print('Classification Report:\
', classification_report(y_test, stack_pred))

print('done')


# In[7]:


# Let's try a different approach for calibration - using a separate calibration set

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, confusion_matrix, classification_report, brier_score_loss
import matplotlib.pyplot as plt

# Load enhanced training data
df = pd.read_csv('train_data_enhanced.csv')

# Prepare features and target
X = df.drop(columns=['Season', 'LowerTeamID', 'HigherTeamID', 'Tournament', 'Target'])
y = df['Target']

# Split data into train, calibration, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_calib, X_test, y_calib, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"Train set: {X_train.shape}, Calibration set: {X_calib.shape}, Test set: {X_test.shape}")

# Define base models for stacking ensemble
estimators = [
    ('gb', GradientBoostingClassifier(random_state=42)),
    ('rf', RandomForestClassifier(random_state=42))
]

# Define meta-model 
meta_model = LogisticRegression(max_iter=1000, random_state=42)

# Create and train stacking ensemble
stack_model = StackingClassifier(estimators=estimators, final_estimator=meta_model, cv=5, passthrough=False)
stack_model.fit(X_train, y_train)

# Get uncalibrated predictions on calibration and test sets
uncalib_calib_proba = stack_model.predict_proba(X_calib)[:, 1]
uncalib_test_proba = stack_model.predict_proba(X_test)[:, 1]
uncalib_test_pred = stack_model.predict(X_test)

# Calculate uncalibrated metrics
uncalib_brier = brier_score_loss(y_test, uncalib_test_proba)
uncalib_acc = accuracy_score(y_test, uncalib_test_pred)
uncalib_roc = roc_auc_score(y_test, uncalib_test_proba)

print("--- Uncalibrated Stacking Ensemble ---")
print(f"Accuracy: {uncalib_acc:.4f}")
print(f"ROC AUC: {uncalib_roc:.4f}")
print(f"Brier Score: {uncalib_brier:.4f}")

# Perform isotonic calibration manually
from sklearn.isotonic import IsotonicRegression

# Train isotonic regression on calibration set
iso_reg = IsotonicRegression(out_of_bounds='clip')
iso_reg.fit(uncalib_calib_proba, y_calib)

# Apply calibration to test set
calib_test_proba = iso_reg.predict(uncalib_test_proba)
calib_test_pred = (calib_test_proba >= 0.5).astype(int)

# Calculate calibrated metrics
calib_brier = brier_score_loss(y_test, calib_test_proba)
calib_acc = accuracy_score(y_test, calib_test_pred)
calib_roc = roc_auc_score(y_test, calib_test_proba)
calib_gini = 2 * calib_roc - 1

print("\
--- Calibrated Stacking Ensemble (Isotonic) ---")
print(f"Accuracy: {calib_acc:.4f}")
print(f"ROC AUC: {calib_roc:.4f}")
print(f"Gini Coefficient: {calib_gini:.4f}")
print(f"Brier Score: {calib_brier:.4f}")
print(f"Confusion Matrix:\
 {confusion_matrix(y_test, calib_test_pred)}")
print(f"Classification Report:\
 {classification_report(y_test, calib_test_pred)}")

# Plot calibration curves
plt.figure(figsize=(10, 6))
# Plot perfectly calibrated
plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')

# Plot calibration curves
prob_true, prob_pred = calibration_curve(y_test, uncalib_test_proba, n_bins=10)
plt.plot(prob_pred, prob_true, 's-', label=f'Uncalibrated (Brier: {uncalib_brier:.4f})')

prob_true, prob_pred = calibration_curve(y_test, calib_test_proba, n_bins=10)
plt.plot(prob_pred, prob_true, 's-', label=f'Isotonic Calibration (Brier: {calib_brier:.4f})')

plt.xlabel('Mean predicted probability')
plt.ylabel('Fraction of positives')
plt.title('Calibration Curve')
plt.legend()
plt.grid(True)
plt.show()

print('done')


# In[8]:


# This cell adds XGBoost to the existing stacking ensemble to improve overall performance and calibration.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, confusion_matrix, classification_report, brier_score_loss

# For XGBoost
import xgboost as xgb

# Load enhanced training data

df = pd.read_csv('train_data_enhanced.csv')

# Prepare features and target
X = df.drop(columns=['Season', 'LowerTeamID', 'HigherTeamID', 'Tournament', 'Target'])
y = df['Target']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define base models
estimators = [
    ('gb', GradientBoostingClassifier(random_state=42)),
    ('rf', RandomForestClassifier(random_state=42)),
    ('xgb', xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'))
]

# Define meta-model
meta_model = LogisticRegression(max_iter=1000, random_state=42)

# Create Stacking ensemble
stack_model = StackingClassifier(estimators=estimators, final_estimator=meta_model, cv=5, passthrough=False)

# Train the stacking model
stack_model.fit(X_train, y_train)

# Predict on test set
pred_proba = stack_model.predict_proba(X_test)[:, 1]
pred = stack_model.predict(X_test)

# Evaluate the model
acc = accuracy_score(y_test, pred)
ll = log_loss(y_test, pred_proba)
roc = roc_auc_score(y_test, pred_proba)
gini = 2 * roc - 1
brier = brier_score_loss(y_test, pred_proba)

print('--- Enhanced Stacking Ensemble with XGBoost Evaluation ---')
print('Accuracy:', acc)
print('Log Loss:', ll)
print('ROC AUC:', roc)
print('Gini Coefficient:', gini)
print('Brier Score:', brier)
print('Confusion Matrix:\
', confusion_matrix(y_test, pred))
print('Classification Report:\
', classification_report(y_test, pred))

print('done')


# In[9]:


# This cell adds XGBoost to the existing stacking ensemble to improve overall performance and calibration.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, confusion_matrix, classification_report, brier_score_loss

# For XGBoost
import xgboost as xgb

# Load enhanced training data

df = pd.read_csv('train_data_enhanced.csv')

# Prepare features and target
X = df.drop(columns=['Season', 'LowerTeamID', 'HigherTeamID', 'Tournament', 'Target'])
y = df['Target']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define base models
estimators = [
    ('gb', GradientBoostingClassifier(random_state=42)),
    ('rf', RandomForestClassifier(random_state=42)),
    ('xgb', xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'))
]

# Define meta-model
meta_model = LogisticRegression(max_iter=1000, random_state=42)

# Create Stacking ensemble
stack_model = StackingClassifier(estimators=estimators, final_estimator=meta_model, cv=5, passthrough=False)

# Train the stacking model
stack_model.fit(X_train, y_train)

# Predict on test set
pred_proba = stack_model.predict_proba(X_test)[:, 1]
pred = stack_model.predict(X_test)

# Evaluate the model
acc = accuracy_score(y_test, pred)
ll = log_loss(y_test, pred_proba)
roc = roc_auc_score(y_test, pred_proba)
gini = 2 * roc - 1
brier = brier_score_loss(y_test, pred_proba)

print('--- Enhanced Stacking Ensemble with XGBoost Evaluation ---')
print('Accuracy:', acc)
print('Log Loss:', ll)
print('ROC AUC:', roc)
print('Gini Coefficient:', gini)
print('Brier Score:', brier)
print('Confusion Matrix:\
', confusion_matrix(y_test, pred))
print('Classification Report:\
', classification_report(y_test, pred))

print('done')

