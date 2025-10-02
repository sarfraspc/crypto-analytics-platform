from core.database import get_timescale_engine
import pandas as pd

engine = get_timescale_engine()

long_df = pd.read_sql("""
    SELECT time, symbol, feature_name, feature_value
    FROM ohlcv_features_panel
    ORDER BY time, symbol, feature_name
""", engine, parse_dates=['time'])

wide_df = long_df.pivot_table(  
    index=['time', 'symbol'],
    columns='feature_name',
    values='feature_value',
    aggfunc='first'  
).reset_index()

wide_df.to_csv('panel_wide.csv', index=False)  
print(f"Exported wide panel: {wide_df.shape} rows/cols")
print(wide_df.head())