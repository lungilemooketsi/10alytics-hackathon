"""
Model Engine for African Sovereign Debt Crisis Early Warning System
Trains Random Forest Classifier to predict fiscal crises
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import os
import warnings
warnings.filterwarnings('ignore')


def load_data(filepath):
    """Load cleaned fiscal data."""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def create_target_variable(df):
    """
    Create binary target variable: Will Debt_to_GDP exceed 70% next year?
    
    Args:
        df: DataFrame with Debt_to_GDP column
    
    Returns:
        DataFrame with 'Crisis_Next_Year' target variable
    """
    print("Creating target variable (Debt_to_GDP > 70% next year)...")
    
    # Sort by Country and Year
    df = df.sort_values(['Country', 'Year'])
    
    # Create next year's Debt_to_GDP using shift
    df['Next_Year_Debt_to_GDP'] = df.groupby('Country')['Debt_to_GDP'].shift(-1)
    
    # Binary target: 1 if next year's debt exceeds 70%, 0 otherwise
    df['Crisis_Next_Year'] = (df['Next_Year_Debt_to_GDP'] > 70).astype(int)
    
    # Remove rows where we don't have next year's data
    df_model = df[df['Crisis_Next_Year'].notna()].copy()
    
    print(f"Target distribution:")
    print(df_model['Crisis_Next_Year'].value_counts())
    print(f"Crisis rate: {df_model['Crisis_Next_Year'].mean():.2%}")
    
    return df_model


def prepare_features(df):
    """
    Select and prepare features for modeling.
    
    Args:
        df: DataFrame with all features
    
    Returns:
        X (features), y (target), feature_names
    """
    print("Preparing features...")
    
    # Define feature columns (exclude identifiers and target)
    exclude_cols = ['Country', 'Year', 'Crisis_Next_Year', 'Next_Year_Debt_to_GDP']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Remove columns with too many missing values (>50%)
    missing_pct = df[feature_cols].isnull().sum() / len(df)
    valid_features = missing_pct[missing_pct < 0.5].index.tolist()
    
    print(f"Selected {len(valid_features)} features with <50% missing values")
    
    X = df[valid_features].copy()
    y = df['Crisis_Next_Year'].copy()
    
    # Fill remaining missing values with median
    X = X.fillna(X.median())
    
    # Store feature names
    feature_names = valid_features
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Features: {feature_names}")
    
    return X, y, feature_names, df[['Country', 'Year']]


def train_model(X_train, y_train):
    """
    Train Random Forest Classifier.
    
    Args:
        X_train: Training features
        y_train: Training target
    
    Returns:
        Trained model
    """
    print("Training Random Forest Classifier...")
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced'  # Handle class imbalance
    )
    
    model.fit(X_train, y_train)
    
    print("Model training complete!")
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
    """
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    if len(np.unique(y_test)) > 1:
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        print(f"\nROC-AUC Score: {roc_auc:.4f}")
    
    print("="*60 + "\n")


def plot_feature_importance(model, feature_names, output_path):
    """
    Plot and save feature importance.
    
    Args:
        model: Trained model
        feature_names: List of feature names
        output_path: Path to save plot
    """
    print(f"Creating feature importance plot...")
    
    # Get feature importances
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:15]  # Top 15 features
    
    # Create plot
    plt.figure(figsize=(12, 8))
    plt.title('Top 15 Feature Importances - Fiscal Crisis Prediction', fontsize=16, fontweight='bold')
    plt.barh(range(len(indices)), importances[indices], color='steelblue')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Feature importance plot saved to {output_path}")
    plt.close()
    
    # Print top features
    print("\nTop 10 Most Important Features:")
    for i in range(min(10, len(indices))):
        idx = indices[i]
        print(f"{i+1}. {feature_names[idx]}: {importances[idx]:.4f}")


def predict_at_risk_countries(model, df, X, feature_names, metadata, year=2025):
    """
    Identify top at-risk countries for a specific year with detailed metrics.
    
    Args:
        model: Trained model
        df: Full dataframe
        X: Feature matrix
        feature_names: List of feature names
        metadata: DataFrame with Country and Year
        year: Year to predict for
    
    Returns:
        DataFrame with comprehensive risk assessment
    """
    print(f"\n" + "="*60)
    print(f"TOP AT-RISK COUNTRIES FOR {year}")
    print("="*60)
    
    # Get predictions for all data
    df_pred = metadata.copy()
    df_pred['Risk_Score'] = model.predict_proba(X)[:, 1] * 100
    df_pred['Crisis_Prediction'] = model.predict(X)
    
    # Add comprehensive metrics
    if 'Debt_to_GDP' in df.columns:
        df_pred['Current_Debt_to_GDP'] = df['Debt_to_GDP'].values
    if 'Deficit_to_GDP' in df.columns:
        df_pred['Deficit_to_GDP'] = df['Deficit_to_GDP'].values
    if 'Tax_to_GDP' in df.columns:
        df_pred['Tax_to_GDP'] = df['Tax_to_GDP'].values
    if 'GDP Growth Rate' in df.columns:
        df_pred['GDP_Growth_Rate'] = df['GDP Growth Rate'].values
    if 'Inflation Rate ' in df.columns:
        df_pred['Inflation_Rate'] = df['Inflation Rate '].values
    if 'Unemployment Rate' in df.columns:
        df_pred['Unemployment_Rate'] = df['Unemployment Rate'].values
    
    # Calculate 3-year trends where possible
    df_pred['Debt_Trend'] = df_pred.groupby('Country')['Current_Debt_to_GDP'].diff(3)
    
    # Risk categories
    df_pred['Risk_Category'] = pd.cut(
        df_pred['Risk_Score'],
        bins=[0, 20, 40, 60, 80, 100],
        labels=['Low', 'Moderate', 'Elevated', 'High', 'Critical']
    )
    
    # Filter for the specified year or latest available year
    if year in df_pred['Year'].values:
        df_year = df_pred[df_pred['Year'] == year]
    else:
        max_year = df_pred['Year'].max()
        print(f"Year {year} not found. Using latest available year: {max_year}")
        df_year = df_pred[df_pred['Year'] == max_year]
    
    # Sort by risk score and get all countries
    df_risk = df_year.sort_values('Risk_Score', ascending=False)
    
    print(f"\nDetailed Risk Assessment:\n")
    print(f"{'Rank':<6}{'Country':<20}{'Risk':<8}{'Category':<12}{'Debt/GDP':<12}{'Deficit/GDP':<14}{'Growth%':<10}")
    print("-" * 90)
    
    for idx, (i, row) in enumerate(df_risk.iterrows(), 1):
        country = row['Country'][:18]
        risk = f"{row['Risk_Score']:.1f}%"
        category = str(row['Risk_Category'])
        debt = f"{row.get('Current_Debt_to_GDP', 0):.1f}%" if pd.notna(row.get('Current_Debt_to_GDP')) else "N/A"
        deficit = f"{row.get('Deficit_to_GDP', 0):.1f}%" if pd.notna(row.get('Deficit_to_GDP')) else "N/A"
        growth = f"{row.get('GDP_Growth_Rate', 0):.1f}%" if pd.notna(row.get('GDP_Growth_Rate')) else "N/A"
        
        print(f"{idx:<6}{country:<20}{risk:<8}{category:<12}{debt:<12}{deficit:<14}{growth:<10}")
    
    print("="*60 + "\n")
    
    return df_risk


def main():
    """Main execution function."""
    # Define paths
    data_path = "data/cleaned_fiscal_data.csv"
    output_plot = "outputs/feature_importance.png"
    
    # Check if cleaned data exists
    if not os.path.exists(data_path):
        print(f"Error: Cleaned data not found at {data_path}")
        print("Please run data_cleaner.py first to generate the cleaned data.")
        return
    
    # Load data
    df = load_data(data_path)
    
    # Create target variable
    df_model = create_target_variable(df)
    
    # Prepare features
    X, y, feature_names, metadata = prepare_features(df_model)
    
    # Split data
    print("\nSplitting data into train and test sets (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate model
    evaluate_model(model, X_test, y_test)
    
    # Plot feature importance
    os.makedirs('outputs', exist_ok=True)
    plot_feature_importance(model, feature_names, output_plot)
    
    # Predict at-risk countries for 2025
    at_risk = predict_at_risk_countries(model, df_model, X, feature_names, metadata, year=2025)
    
    # Save comprehensive predictions
    predictions_path = "outputs/at_risk_countries_2025.csv"
    at_risk.to_csv(predictions_path, index=False)
    print(f"At-risk countries saved to {predictions_path}")
    
    # Save detailed model metrics
    print("\nGenerating additional analytics...")
    
    # Save all predictions with probabilities
    all_predictions = metadata.copy()
    all_predictions['Risk_Score'] = model.predict_proba(X)[:, 1] * 100
    all_predictions['Crisis_Prediction'] = model.predict(X)
    all_predictions['Current_Debt_to_GDP'] = df_model['Debt_to_GDP'].values
    all_predictions_path = "outputs/all_countries_risk_scores.csv"
    all_predictions.to_csv(all_predictions_path, index=False)
    print(f"Complete risk scores saved to {all_predictions_path}")
    
    # Save feature importance data
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    feature_importance_path = "outputs/feature_importance_scores.csv"
    feature_importance_df.to_csv(feature_importance_path, index=False)
    print(f"Feature importance scores saved to {feature_importance_path}")
    
    print("\n✅ Model training and prediction complete!")


if __name__ == "__main__":
    main()
