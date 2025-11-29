"""
Data Cleaner for African Sovereign Debt Crisis Project
Handles unit normalization, data pivoting, and feature engineering
"""

import pandas as pd
import numpy as np
import os


def normalize_unit(value, unit):
    """
    Normalize financial values to Billions.
    
    Args:
        value: The numeric value
        unit: The unit as string ('Million', 'Billion', 'Trillion')
    
    Returns:
        Normalized value in Billions
    """
    if pd.isna(value) or pd.isna(unit):
        return np.nan
    
    unit = str(unit).strip().lower()
    
    if 'million' in unit:
        return value / 1000  # Convert millions to billions
    elif 'billion' in unit:
        return value
    elif 'trillion' in unit:
        return value * 1000  # Convert trillions to billions
    else:
        # If no unit specified, assume it's already in billions
        return value


def clean_and_transform_data(input_path, output_path):
    """
    Clean raw fiscal data and transform from long to wide format.
    
    Args:
        input_path: Path to raw Excel file
        output_path: Path to save cleaned CSV file
    """
    print("Loading raw data...")
    df = pd.read_excel(input_path, sheet_name='Data')
    
    print("Initial data shape:", df.shape)
    print("Columns:", df.columns.tolist())
    
    # Check if Unit column exists, if not create a default
    if 'Unit' not in df.columns:
        print("Warning: 'Unit' column not found. Assuming values are in Billions.")
        df['Unit'] = 'Billion'
    
    # Normalize units to Billions
    print("Normalizing units to Billions...")
    # The actual column name is 'Amount' in the dataset
    value_column = 'Amount' if 'Amount' in df.columns else 'Value'
    
    if value_column not in df.columns:
        # Try to find value column with different name
        value_cols = [col for col in df.columns if 'value' in col.lower() or 'amount' in col.lower()]
        if value_cols:
            value_column = value_cols[0]
        else:
            raise ValueError("No 'Value' or 'Amount' column found in the data")
    
    df['Value_Normalized'] = df.apply(
        lambda row: normalize_unit(row[value_column], row['Unit']), 
        axis=1
    )
    
    # Extract Year from Time column
    if 'Time' in df.columns and 'Year' not in df.columns:
        print("Extracting Year from Time column...")
        df['Year'] = pd.to_datetime(df['Time']).dt.year
    
    # Identify key columns for pivoting
    required_cols = ['Country', 'Year', 'Indicator']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"Warning: Missing columns {missing_cols}. Attempting to infer from data...")
        # Try to find similar column names
        for col in missing_cols:
            similar = [c for c in df.columns if col.lower() in c.lower()]
            if similar:
                print(f"Using '{similar[0]}' for '{col}'")
    
    # Pivot from long to wide format
    print("Pivoting data from long to wide format...")
    df_wide = df.pivot_table(
        index=['Country', 'Year'],
        columns='Indicator',
        values='Value_Normalized',
        aggfunc='first'
    ).reset_index()
    
    print("Data after pivoting:", df_wide.shape)
    print("Available indicators:", [col for col in df_wide.columns if col not in ['Country', 'Year']])
    
    # Feature Engineering
    print("Engineering features...")
    
    # Calculate Debt_to_GDP (with zero-division protection)
    # Prioritize actual GDP columns (Nominal GDP, Real GDP) over GDP Growth Rate
    gdp_col = None
    if 'Nominal GDP' in df_wide.columns:
        gdp_col = 'Nominal GDP'
    elif 'Real GDP' in df_wide.columns:
        gdp_col = 'Real GDP'
    elif 'GDP' in df_wide.columns:
        gdp_col = 'GDP'
    else:
        # Find GDP columns, but exclude "GDP Growth Rate"
        gdp_cols = [col for col in df_wide.columns if 'gdp' in col.lower() and 'growth' not in col.lower() and 'per capita' not in col.lower()]
        if gdp_cols:
            gdp_col = gdp_cols[0]
    
    if 'Total_Debt' in df_wide.columns and gdp_col:
        df_wide['Debt_to_GDP'] = (df_wide['Total_Debt'] / df_wide[gdp_col].replace(0, np.nan)) * 100
    elif 'Debt' in df_wide.columns and gdp_col:
        df_wide['Debt_to_GDP'] = (df_wide['Debt'] / df_wide[gdp_col].replace(0, np.nan)) * 100
    else:
        # Try to find debt-related columns
        debt_cols = [col for col in df_wide.columns if 'debt' in col.lower()]
        if debt_cols and gdp_col:
            df_wide['Debt_to_GDP'] = (df_wide[debt_cols[0]] / df_wide[gdp_col].replace(0, np.nan)) * 100
            print(f"Using {debt_cols[0]} and {gdp_col} for Debt_to_GDP")
    
    # Calculate Deficit_to_GDP (with zero-division protection)
    deficit_cols = [col for col in df_wide.columns if 'deficit' in col.lower() or 'surplus' in col.lower()]
    if deficit_cols and gdp_col:
        df_wide['Deficit_to_GDP'] = (df_wide[deficit_cols[0]] / df_wide[gdp_col].replace(0, np.nan)) * 100
        print(f"Using {deficit_cols[0]} and {gdp_col} for Deficit_to_GDP")
    
    # Calculate Tax_to_GDP (with zero-division protection)
    revenue_cols = [col for col in df_wide.columns if ('revenue' in col.lower() or 'tax' in col.lower()) and 'vat' not in col.lower()]
    if revenue_cols and gdp_col:
        df_wide['Tax_to_GDP'] = (df_wide[revenue_cols[0]] / df_wide[gdp_col].replace(0, np.nan)) * 100
        print(f"Using {revenue_cols[0]} and {gdp_col} for Tax_to_GDP")
    
    # Remove rows with too many missing values
    threshold = len(df_wide.columns) * 0.5
    df_wide = df_wide.dropna(thresh=threshold)
    
    print(f"Final data shape: {df_wide.shape}")
    print(f"Countries: {df_wide['Country'].nunique()}")
    print(f"Year range: {df_wide['Year'].min()} - {df_wide['Year'].max()}")
    
    # Save cleaned data
    print(f"Saving cleaned data to {output_path}...")
    df_wide.to_csv(output_path, index=False)
    print("Data cleaning complete!")
    
    return df_wide


if __name__ == "__main__":
    # Define paths
    input_file = "data/raw_fiscal_data.xlsx"
    output_file = "data/cleaned_fiscal_data.csv"
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        print("Please place your raw fiscal data in the 'data/' folder with the name 'raw_fiscal_data.xlsx'")
        print("\nExpected data format:")
        print("- Columns: Country, Year, Indicator, Value, Unit")
        print("- Indicators: Total_Debt, GDP, Budget_Deficit, Tax_Revenue, etc.")
        print("- Units: Million, Billion, or Trillion")
    else:
        df_cleaned = clean_and_transform_data(input_file, output_file)
        print(f"\nCleaned data preview:")
        print(df_cleaned.head())
        print(f"\nEngineered features:")
        engineered = [col for col in df_cleaned.columns if '_to_' in col]
        print(engineered)
