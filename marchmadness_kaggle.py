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

def load_and_prepare_data(cleaned_folder):
    """Load and prepare data for training"""
    # Example: load a preprocessed CSV file for training
    # Update the file path as necessary (this example assumes the training data is in the cleaned folder)
    train_file = os.path.join(cleaned_folder, 'train_data_enhanced.csv')
    df = pd.read_csv(train_file)
    # Create features and target (example, adjust as needed)
    X = df.drop(columns=['Season', 'LowerTeamID', 'HigherTeamID', 'Tournament', 'Target'])
    y = df['Target']
    return X, y

def train_model(X_train, y_train):
    """Train a stacking ensemble model"""
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
    
    return stack_model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model on test data"""
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

def create_submission_file(model, submission_template_path, submission_filename, cleaned_folder=None):
    """Create submission file for the competition"""
    # Load the sample submission file
    sample_submission = pd.read_csv(submission_template_path)
    
    # Check if test data is available
    test_file = os.path.join(cleaned_folder, 'test_data_enhanced.csv') if cleaned_folder else None
    
    if os.path.exists(test_file):
        # Load test data
        test_df = pd.read_csv(test_file)
        # Prepare test features (adjust as needed)
        X_test_submission = test_df.drop(columns=['ID'])
        # Generate predictions
        preds = model.predict_proba(X_test_submission)[:, 1]
        sample_submission['Pred'] = preds
    else:
        # If test data is not available, assign default probability
        print("Test data not found. Using default predictions.")
        sample_submission['Pred'] = 0.5
    
    # Save the submission file
    sample_submission.to_csv(submission_filename, index=False)
    print('Submission file created as ' + submission_filename)
    print(sample_submission.head())

def main():
    """Main function to run the entire pipeline"""
    # Set file and folder paths. In Kaggle, adjust input paths as needed.
    # For example, if your zip file is in ../input/march-madness-2025/ then:
    base_path = '../input/march-machine-learning-mania-2025'
    if not os.path.exists(base_path):
        # If not running on Kaggle, use local paths
        base_path = '.'
    
    zip_file_path = os.path.join(base_path, 'march-machine-learning-mania-2025.zip')
    cleaned_folder = 'march-madness-2025-cleaned'
    
    # Check if extraction is needed
    if not os.path.exists(cleaned_folder) and os.path.exists(zip_file_path):
        extract_data(zip_file_path, cleaned_folder)
    
    # Load and prepare data
    X, y = load_and_prepare_data(cleaned_folder)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate model
    evaluate_model(model, X_test, y_test)
    
    # Create submission file
    submission_template_path = os.path.join(base_path, 'SampleSubmissionStage2.csv')
    if not os.path.exists(submission_template_path):
        submission_template_path = 'SampleSubmissionStage2.csv'
    
    submission_filename = 'submission.csv'
    create_submission_file(model, submission_template_path, submission_filename, cleaned_folder)

if __name__ == '__main__':
    main()
