"""
Pioneer Simulation - Stress Testing and Contagion Analysis
Simulates revenue shocks and analyzes systemic risk across African nations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import pickle
import os
import warnings
warnings.filterwarnings('ignore')


def load_data_and_model():
    """Load cleaned data and train a quick model if needed."""
    print("Loading data...")
    df = pd.read_csv("data/cleaned_fiscal_data.csv")
    
    # Prepare model data
    df = df.sort_values(['Country', 'Year'])
    df['Next_Year_Debt_to_GDP'] = df.groupby('Country')['Debt_to_GDP'].shift(-1)
    df['Crisis_Next_Year'] = (df['Next_Year_Debt_to_GDP'] > 70).astype(int)
    df_model = df[df['Crisis_Next_Year'].notna()].copy()
    
    # Prepare features
    exclude_cols = ['Country', 'Year', 'Crisis_Next_Year', 'Next_Year_Debt_to_GDP']
    feature_cols = [col for col in df_model.columns if col not in exclude_cols]
    missing_pct = df_model[feature_cols].isnull().sum() / len(df_model)
    valid_features = missing_pct[missing_pct < 0.5].index.tolist()
    
    X = df_model[valid_features].fillna(df_model[valid_features].median())
    y = df_model['Crisis_Next_Year']
    
    # Train model
    print("Training model for simulation...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X, y)
    
    return df_model, model, valid_features


def simulate_revenue_shock(df, country, shock_pct=-20):
    """
    Simulate a revenue shock for a specific country.
    
    Args:
        df: DataFrame with fiscal data
        country: Country to apply shock to
        shock_pct: Percentage change in revenue (negative = decrease)
    
    Returns:
        DataFrame with shocked values
    """
    print(f"\n{'='*70}")
    print(f"STRESS TEST: {country.upper()} REVENUE SHOCK")
    print(f"{'='*70}")
    print(f"Simulating {shock_pct}% revenue shock (e.g., oil price crash)...\n")
    
    df_shocked = df.copy()
    
    # Find revenue-related columns
    revenue_cols = [col for col in df.columns if 'revenue' in col.lower() or 'tax' in col.lower()]
    
    if not revenue_cols:
        print("Warning: No revenue columns found. Using 'Tax_to_GDP' adjustment.")
        # Adjust based on Tax_to_GDP ratio
        country_mask = df_shocked['Country'] == country
        if 'Tax_to_GDP' in df_shocked.columns:
            df_shocked.loc[country_mask, 'Tax_to_GDP'] = df_shocked.loc[country_mask, 'Tax_to_GDP'] * (1 + shock_pct/100)
    else:
        # Apply shock to revenue columns
        country_mask = df_shocked['Country'] == country
        for col in revenue_cols:
            if col in df_shocked.columns:
                original_val = df_shocked.loc[country_mask, col].mean()
                df_shocked.loc[country_mask, col] = df_shocked.loc[country_mask, col] * (1 + shock_pct/100)
                new_val = df_shocked.loc[country_mask, col].mean()
                print(f"  {col}: {original_val:.2f} → {new_val:.2f} ({shock_pct}% change)")
    
    # Recalculate Deficit_to_GDP if possible
    if 'Deficit_to_GDP' in df_shocked.columns:
        # Revenue drop typically increases deficit
        country_mask = df_shocked['Country'] == country
        original_deficit = df_shocked.loc[country_mask, 'Deficit_to_GDP'].mean()
        # Approximate: revenue drop increases deficit by same amount
        df_shocked.loc[country_mask, 'Deficit_to_GDP'] = df_shocked.loc[country_mask, 'Deficit_to_GDP'] - shock_pct
        new_deficit = df_shocked.loc[country_mask, 'Deficit_to_GDP'].mean()
        print(f"  Deficit_to_GDP: {original_deficit:.2f}% → {new_deficit:.2f}%")
    
    return df_shocked


def compare_risk_scores(df_original, df_shocked, model, feature_cols, country, year=None):
    """
    Compare risk scores before and after shock.
    
    Args:
        df_original: Original dataframe
        df_shocked: Shocked dataframe
        model: Trained model
        feature_cols: List of feature columns
        country: Country being analyzed
        year: Specific year to analyze (if None, use latest)
    """
    # Prepare features for both scenarios
    X_original = df_original[feature_cols].fillna(df_original[feature_cols].median())
    X_shocked = df_shocked[feature_cols].fillna(df_shocked[feature_cols].median())
    
    # Get predictions
    risk_original = model.predict_proba(X_original)[:, 1]
    risk_shocked = model.predict_proba(X_shocked)[:, 1]
    
    # Add to dataframes
    df_original['Risk_Score'] = risk_original
    df_shocked['Risk_Score'] = risk_shocked
    
    # Filter for specific country
    if year:
        df_orig_country = df_original[(df_original['Country'] == country) & (df_original['Year'] == year)]
        df_shock_country = df_shocked[(df_shocked['Country'] == country) & (df_shocked['Year'] == year)]
    else:
        df_orig_country = df_original[df_original['Country'] == country].tail(1)
        df_shock_country = df_shocked[df_shocked['Country'] == country].tail(1)
    
    if len(df_orig_country) == 0:
        print(f"No data found for {country}")
        return
    
    orig_risk = df_orig_country['Risk_Score'].values[0]
    shocked_risk = df_shock_country['Risk_Score'].values[0]
    risk_increase = shocked_risk - orig_risk
    
    print(f"\n{'─'*70}")
    print(f"RISK ASSESSMENT FOR {country.upper()}")
    print(f"{'─'*70}")
    print(f"  Baseline Risk Score:      {orig_risk:.1%}")
    print(f"  Post-Shock Risk Score:    {shocked_risk:.1%}")
    print(f"  Risk Increase:            {risk_increase:+.1%}")
    print(f"  Risk Multiplier:          {shocked_risk/orig_risk:.2f}x")
    print(f"{'='*70}\n")


def analyze_contagion(df):
    """
    Analyze economic contagion risk through debt correlation.
    
    Args:
        df: DataFrame with Debt_to_GDP column
    """
    print(f"\n{'='*70}")
    print("CONTAGION ANALYSIS - Debt Correlation Matrix")
    print(f"{'='*70}\n")
    
    # Pivot data to get countries as columns, years as rows
    debt_pivot = df.pivot_table(
        index='Year',
        columns='Country',
        values='Debt_to_GDP'
    )
    
    print(f"Analyzing correlations across {len(debt_pivot.columns)} countries...")
    print(f"Time period: {debt_pivot.index.min()} - {debt_pivot.index.max()}\n")
    
    # Calculate correlation matrix
    correlation_matrix = debt_pivot.corr()
    
    # Find high correlation pairs (>0.8)
    high_corr_pairs = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            corr_value = correlation_matrix.iloc[i, j]
            if corr_value > 0.8:
                country1 = correlation_matrix.columns[i]
                country2 = correlation_matrix.columns[j]
                high_corr_pairs.append((country1, country2, corr_value))
    
    # Sort by correlation
    high_corr_pairs.sort(key=lambda x: x[2], reverse=True)
    
    print("HIGH-RISK CONTAGION PAIRS (Correlation > 0.8):")
    print(f"{'─'*70}")
    
    if high_corr_pairs:
        print(f"{'Rank':<6}{'Country 1':<25}{'Country 2':<25}{'Correlation':<15}")
        print(f"{'─'*70}")
        for idx, (c1, c2, corr) in enumerate(high_corr_pairs[:10], 1):
            print(f"{idx:<6}{c1:<25}{c2:<25}{corr:.3f}")
        
        print(f"\n⚠️  Found {len(high_corr_pairs)} country pairs with >0.8 correlation")
        print("    These pairs face elevated systemic risk - a crisis in one")
        print("    country could trigger contagion in correlated countries.")
    else:
        print("No country pairs found with correlation > 0.8")
        print("Analyzing moderate correlation pairs (> 0.6)...")
        
        moderate_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if corr_value > 0.6:
                    country1 = correlation_matrix.columns[i]
                    country2 = correlation_matrix.columns[j]
                    moderate_pairs.append((country1, country2, corr_value))
        
        moderate_pairs.sort(key=lambda x: x[2], reverse=True)
        
        if moderate_pairs:
            print(f"\n{'Rank':<6}{'Country 1':<25}{'Country 2':<25}{'Correlation':<15}")
            print(f"{'─'*70}")
            for idx, (c1, c2, corr) in enumerate(moderate_pairs[:10], 1):
                print(f"{idx:<6}{c1:<25}{c2:<25}{corr:.3f}")
    
    print(f"{'='*70}\n")
    
    # Save correlation heatmap
    plot_correlation_heatmap(correlation_matrix)
    
    return correlation_matrix, high_corr_pairs


def plot_correlation_heatmap(correlation_matrix):
    """
    Create and save correlation heatmap.
    
    Args:
        correlation_matrix: Correlation matrix of Debt_to_GDP
    """
    print("Creating correlation heatmap...")
    
    plt.figure(figsize=(14, 12))
    
    # Use mask for upper triangle to avoid redundancy
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    
    sns.heatmap(
        correlation_matrix,
        mask=mask,
        annot=False,
        cmap='RdYlGn_r',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Correlation Coefficient"}
    )
    
    plt.title('Debt-to-GDP Correlation Matrix\nContagion Risk Analysis', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Country', fontsize=12)
    plt.ylabel('Country', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    output_path = "outputs/contagion_correlation_heatmap.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Correlation heatmap saved to {output_path}\n")
    plt.close()


def main():
    """Main execution function."""
    # Check if data exists
    if not os.path.exists("data/cleaned_fiscal_data.csv"):
        print("Error: Cleaned data not found. Please run data_cleaner.py first.")
        return
    
    # Load data and model
    df, model, feature_cols = load_data_and_model()
    
    # Stress Test: Nigeria revenue shock
    target_country = "Nigeria"
    shock_percentage = -20
    
    # Check if Nigeria exists in data
    if target_country not in df['Country'].values:
        print(f"Warning: {target_country} not found in data. Using first available country.")
        target_country = df['Country'].unique()[0]
    
    df_shocked = simulate_revenue_shock(df, target_country, shock_percentage)
    
    # Compare risk scores
    compare_risk_scores(df, df_shocked, model, feature_cols, target_country)
    
    # Contagion Analysis
    correlation_matrix, high_corr_pairs = analyze_contagion(df)
    
    # Save results
    os.makedirs('outputs', exist_ok=True)
    
    # Save stress test results
    stress_results = pd.DataFrame({
        'Country': [target_country],
        'Shock_Type': ['Revenue Shock'],
        'Shock_Magnitude': [f'{shock_percentage}%'],
        'Analysis': ['Oil price crash simulation']
    })
    stress_results.to_csv('outputs/stress_test_results.csv', index=False)
    
    # Save high correlation pairs
    if high_corr_pairs:
        contagion_df = pd.DataFrame(high_corr_pairs, columns=['Country_1', 'Country_2', 'Correlation'])
        contagion_df.to_csv('outputs/high_contagion_pairs.csv', index=False)
        print(f"High contagion pairs saved to outputs/high_contagion_pairs.csv")
    
    print("\n✅ Stress testing and contagion analysis complete!")
    print("\nGenerated outputs:")
    print("  - outputs/stress_test_results.csv")
    print("  - outputs/contagion_correlation_heatmap.png")
    if high_corr_pairs:
        print("  - outputs/high_contagion_pairs.csv")


if __name__ == "__main__":
    main()
