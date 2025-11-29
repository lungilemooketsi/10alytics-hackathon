# African Sovereign Debt Crisis - Hackathon Project

## Project Overview
Machine Learning Early Warning System for predicting fiscal crises and simulating economic contagion across African nations.

## Project Structure
- `data/` - Raw Excel and CSV files
- `src/` - Python scripts for data processing, modeling, and simulation
- `outputs/` - Generated plots and risk reports
- `notebooks/` - Jupyter notebooks for experimentation

## Setup Instructions
1. Install Python dependencies: `pip install -r requirements.txt`
2. Place raw fiscal data in `data/raw_fiscal_data.xlsx`
3. Run data cleaning: `python src/data_cleaner.py`
4. Train model: `python src/model_engine.py`
5. Run simulations: `python src/pioneer_simulation.py`

## Key Features
- Unit normalization (Million/Billion/Trillion to Billions)
- Feature engineering: Debt_to_GDP, Deficit_to_GDP, Tax_to_GDP
- Random Forest classifier for fiscal crisis prediction
- Stress testing and contagion analysis

## Status
✅ Workspace structure created
✅ Requirements file added
✅ Data cleaning script implemented
✅ Model engine implemented
✅ Stress testing simulation implemented
