import pandas as pd

# Load Excel file
xls = pd.ExcelFile('data/raw_fiscal_data.xlsx')

print("Sheet names:", xls.sheet_names)
print("\n" + "="*80)

# Inspect each sheet
for sheet in xls.sheet_names:
    print(f"\n=== Sheet: {sheet} ===")
    df = pd.read_excel(xls, sheet_name=sheet)
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)[:10]}")
    print(f"\nFirst 3 rows:")
    print(df.head(3))
    print("\n" + "-"*80)
