import os
import zipfile
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, confusion_matrix, classification_report, brier_score_loss
import xgboost as xgb
import lightgbm as lgb

def extract_data(zip_file_path, extract_folder):
    """Extract data from zip file to folder"""
    # Create folder if not exists
    if not os.path.exists(extract_folder):
        os.makedirs(extract_folder)
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_folder)
    print('Extracted files from ' + zip_file_path)

def prepare_training_data(cleaned_folder):
    """Prepare training data from raw files"""
    print("Preparing training data from raw files...")
    
    # Paths for men's tournament results and teams
    men_results_path = os.path.join(cleaned_folder, 'MNCAATourneyDetailedResults.csv')
    men_teams_path = os.path.join(cleaned_folder, 'MTeams.csv')
    
    # Check if files exist
    if not os.path.exists(men_results_path) or not os.path.exists(men_teams_path):
        print(f"Required files not found. Checking contents of {cleaned_folder}:")
        if os.path.exists(cleaned_folder):
            print(os.listdir(cleaned_folder))
        else:
            print(f"{cleaned_folder} directory does not exist")
        raise FileNotFoundError(f"Required files not found in {cleaned_folder}")
    
    # Load tournament results and teams
    results_df = pd.read_csv(men_results_path)
    teams_df = pd.read_csv(men_teams_path)
    
    # Create features for each game
    print("Creating features for each game...")
    
    # Initialize an empty dataframe for training data
    train_data = []
    
    # Process each game
    for _, row in results_df.iterrows():
        season = row['Season']
        wteam = row['WTeamID']
        lteam = row['LTeamID']
        
        # Determine higher and lower seed teams (using team ID as proxy if seed info not available)
        higher_team = max(wteam, lteam)
        lower_team = min(wteam, lteam)
        
        # Create features (simplified example - expand as needed)
        features = {
            'Season': season,
            'HigherTeamID': higher_team,
            'LowerTeamID': lower_team,
            'Tournament': 1,  # 1 for tournament games
            'ScoreDiff': abs(row['WScore'] - row['LScore']),
            'TotalScore': row['WScore'] + row['LScore'],
            'WLoc': 1 if row['WLoc'] == 'H' else (0 if row['WLoc'] == 'A' else 0.5),  # Home/Away/Neutral
            'NumOT': row['NumOT'],
            'WFGM': row['WFGM'],
            'WFGA': row['WFGA'],
            'WFGM3': row['WFGM3'],
            'WFGA3': row['WFGA3'],
            'WFTM': row['WFTM'],
            'WFTA': row['WFTA'],
            'WOR': row['WOR'],
            'WDR': row['WDR'],
            'WAst': row['WAst'],
            'WTO': row['WTO'],
            'WStl': row['WStl'],
            'WBlk': row['WBlk'],
            'WPF': row['WPF'],
            'LFGM': row['LFGM'],
            'LFGA': row['LFGA'],
            'LFGM3': row['LFGM3'],
            'LFGA3': row['LFGA3'],
            'LFTM': row['LFTM'],
            'LFTA': row['LFTA'],
            'LOR': row['LOR'],
            'LDR': row['LDR'],
            'LAst': row['LAst'],
            'LTO': row['LTO'],
            'LStl': row['LStl'],
            'LBlk': row['LBlk'],
            'LPF': row['LPF'],
            'Target': 1 if wteam == higher_team else 0  # 1 if higher seed team won, 0 otherwise
        }
        
        train_data.append(features)
    
    # Convert to dataframe
    train_df = pd.DataFrame(train_data)
    
    # Save the prepared data
    train_file = os.path.join(cleaned_folder, 'train_data_enhanced.csv')
    train_df.to_csv(train_file, index=False)
    print(f"Training data saved to {train_file}")
    
    return train_df

def load_and_prepare_data(cleaned_folder):
    """Load and prepare data for training"""
    # Try to load the preprocessed file
    train_file = os.path.join(cleaned_folder, 'train_data_enhanced.csv')
    
    if os.path.exists(train_file):
        print(f"Loading preprocessed training data from {train_file}")
        df = pd.read_csv(train_file)
    else:
        print(f"Preprocessed file {train_file} not found. Preparing from raw data...")
        df = prepare_training_data(cleaned_folder)
    
    # Create features and target
    X = df.drop(columns=['Season', 'LowerTeamID', 'HigherTeamID', 'Tournament', 'Target'])
    y = df['Target']
    return X, y

def train_model(X_train, y_train):
    """Train a stacking ensemble model"""
    print("Training stacking ensemble model...")
    # Define base models
    estimators = [
        ('gb', GradientBoostingClassifier(random_state=42)),
        ('rf', RandomForestClassifier(random_state=42)),
        ('xgb', xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')),
        ('lgb', lgb.LGBMClassifier(random_state=42))
    ]
    
    # Define meta-model
    meta_model = LogisticRegression(max_iter=1000, random_state=42)
    
    # Create Stacking ensemble
    stack_model = StackingClassifier(estimators=estimators, final_estimator=meta_model, cv=5, passthrough=False)
    
    # Train the stacking model
    stack_model.fit(X_train, y_train)
    print("Model training complete")
    
    return stack_model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model on test data"""
    print("Evaluating model on test data...")
    # Predict on test set
    pred_proba = model.predict_proba(X_test)[:, 1]
    pred = model.predict(X_test)
    
    # Evaluate the model
    acc = accuracy_score(y_test, pred)
    ll = log_loss(y_test, pred_proba)
    roc = roc_auc_score(y_test, pred_proba)
    gini = 2 * roc - 1
    brier = brier_score_loss(y_test, pred_proba)
    
    print('--- Enhanced Stacking Ensemble Evaluation ---')
    print('Accuracy:', acc)
    print('Log Loss:', ll)
    print('ROC AUC:', roc)
    print('Gini Coefficient:', gini)
    print('Brier Score:', brier)
    print('Confusion Matrix:\n', confusion_matrix(y_test, pred))
    print('Classification Report:\n', classification_report(y_test, pred))

def prepare_test_data(cleaned_folder, sample_submission):
    """Prepare test data from sample submission IDs"""
    print("Preparing test data from sample submission IDs...")
    
    # Extract information from sample submission IDs
    # Format is assumed to be: {Season}{Team1ID}{Team2ID}
    test_data = []
    
    for id_val in sample_submission['ID']:
        # Convert to string and extract components
        id_str = str(int(id_val))
        
        # Extract season and team IDs (adjust parsing logic as needed)
        if len(id_str) >= 11:  # Ensure ID is long enough
            season = int(id_str[:4])
            team1 = int(id_str[4:8])
            team2 = int(id_str[8:])
            
            # Determine higher and lower seed teams
            higher_team = max(team1, team2)
            lower_team = min(team1, team2)
            
            # Create basic features (expand as needed)
            features = {
                'ID': id_val,
                'ScoreDiff': 0,  # Placeholder
                'TotalScore': 0,  # Placeholder
                'WLoc': 0.5,  # Neutral location for tournament games
                'NumOT': 0,  # Placeholder
                # Add more features as needed, matching the training data structure
                # Use average values or other reasonable defaults
            }
            
            test_data.append(features)
    
    # Convert to dataframe
    test_df = pd.DataFrame(test_data)
    
    # Save the prepared data
    test_file = os.path.join(cleaned_folder, 'test_data_enhanced.csv')
    test_df.to_csv(test_file, index=False)
    print(f"Test data saved to {test_file}")
    
    return test_df

def create_submission_file(model, submission_template_path, submission_filename, cleaned_folder=None):
    """Create submission file for the competition"""
    print(f"Creating submission file using template: {submission_template_path}")
    
    # Load the sample submission file
    sample_submission = pd.read_csv(submission_template_path)
    
    # Check if test data is available
    test_file = os.path.join(cleaned_folder, 'test_data_enhanced.csv') if cleaned_folder else None
    
    if os.path.exists(test_file):
        # Load test data
        print(f"Loading test data from {test_file}")
        test_df = pd.read_csv(test_file)
        # Prepare test features (adjust as needed)
        X_test_submission = test_df.drop(columns=['ID'])
        # Generate predictions
        preds = model.predict_proba(X_test_submission)[:, 1]
        sample_submission['Pred'] = preds
    else:
        # If test data is not available, prepare it
        print(f"Test data not found at {test_file}. Preparing test data...")
        try:
            test_df = prepare_test_data(cleaned_folder, sample_submission)
            X_test_submission = test_df.drop(columns=['ID'])
            preds = model.predict_proba(X_test_submission)[:, 1]
            sample_submission['Pred'] = preds
        except Exception as e:
            print(f"Error preparing test data: {e}")
            print("Using default predictions (0.5)")
            sample_submission['Pred'] = 0.5
    
    # Save the submission file
    sample_submission.to_csv(submission_filename, index=False)
    print(f'Submission file created as {submission_filename}')
    print(sample_submission.head())

def main():
    """Main function to run the entire pipeline"""
    print("Starting March Madness prediction pipeline...")
    
    # Set file and folder paths. In Kaggle, adjust input paths as needed.
    base_path = '../input/march-machine-learning-mania-2025'
    if not os.path.exists(base_path):
        # If not running on Kaggle, use local paths
        base_path = '.'
        print(f"Using local path: {base_path}")
    else:
        print(f"Using Kaggle path: {base_path}")
    
    # List files in the base path
    print(f"Files in {base_path}:")
    if os.path.exists(base_path):
        print(os.listdir(base_path))
    
    zip_file_path = os.path.join(base_path, 'march-machine-learning-mania-2025.zip')
    cleaned_folder = 'march-madness-2025-cleaned'
    
    # Check if extraction is needed
    if not os.path.exists(cleaned_folder):
        if os.path.exists(zip_file_path):
            print(f"Extracting data from {zip_file_path}...")
            extract_data(zip_file_path, cleaned_folder)
        else:
            # If zip file doesn't exist, check if individual files are directly available
            print(f"Zip file {zip_file_path} not found. Checking for direct file access...")
            cleaned_folder = base_path
    
    # Load and prepare data
    try:
        X, y = load_and_prepare_data(cleaned_folder)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = train_model(X_train, y_train)
        
        # Evaluate model
        evaluate_model(model, X_test, y_test)
        
        # Create submission file
        submission_template_path = os.path.join(base_path, 'SampleSubmissionStage2.csv')
        if not os.path.exists(submission_template_path):
            # Try to find the sample submission file
            for root, dirs, files in os.walk(base_path):
                for file in files:
                    if 'samplesubmission' in file.lower() and 'stage2' in file.lower():
                        submission_template_path = os.path.join(root, file)
                        break
        
        if not os.path.exists(submission_template_path):
            submission_template_path = 'SampleSubmissionStage2.csv'
        
        submission_filename = 'submission.csv'
        create_submission_file(model, submission_template_path, submission_filename, cleaned_folder)
        
    except Exception as e:
        print(f"Error in pipeline: {e}")
        # Create a simple submission with default values
        try:
            submission_template_path = os.path.join(base_path, 'SampleSubmissionStage2.csv')
            if not os.path.exists(submission_template_path):
                for root, dirs, files in os.walk(base_path):
                    for file in files:
                        if 'samplesubmission' in file.lower():
                            submission_template_path = os.path.join(root, file)
                            break
            
            if os.path.exists(submission_template_path):
                print(f"Creating default submission using template: {submission_template_path}")
                sample_submission = pd.read_csv(submission_template_path)
                sample_submission['Pred'] = 0.5
                submission_filename = 'submission.csv'
                sample_submission.to_csv(submission_filename, index=False)
                print(f'Default submission file created as {submission_filename}')
                print(sample_submission.head())
            else:
                print("Could not find sample submission template")
        except Exception as sub_error:
            print(f"Error creating default submission: {sub_error}")

if __name__ == '__main__':
    main()
