import pandas as pd
import numpy as np
from geopy.distance import geodesic
import time
import traceback

def preprocess_data(df_input, cat_encoder, gen_encoder):
    """Applies all preprocessing"""

    # Copy the input DataFrame
    df = df_input.copy()

    # Check for essential base columns
    base_required = ['cc_num', 'amt', 'category', 'gender', 'lat', 'long',
                     'city_pop', 'unix_time', 'merch_lat', 'merch_long']

    missing_base = [col for col in base_required if col not in df.columns]

    if missing_base:
        raise ValueError(f"Input data is missing essential base columns: {missing_base}")

    # 1. Time Features
    try:
        df['trans_datetime'] = pd.to_datetime(df['unix_time'], unit='s')
        df['hour_of_day'] = df['trans_datetime'].dt.hour
        df['day_of_week'] = df['trans_datetime'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

    except Exception as e:
        raise RuntimeError(f"Error in Time Features: {e}")

    # 2. Time Since Last Transaction
    try:
        df = df.sort_values(by=['cc_num', 'unix_time'], ascending=True)
        df['time_since_last'] = df.groupby('cc_num')['unix_time'].diff()
        df['time_since_last'] = df['time_since_last'].fillna(1e9)

    except Exception as e:
        raise RuntimeError(f"Error in Time Since Last Transaction: {e}")

    # 3. Amount Features
    try:
        amt_stats = df.groupby('cc_num')['amt'].agg(['mean', 'std']).reset_index()
        amt_stats.columns = ['cc_num', 'amt_mean', 'amt_std']
        amt_stats['amt_std'].fillna(0, inplace=True)
        df = pd.merge(df, amt_stats, on='cc_num', how='left')
        df['amt_standardized'] = (df['amt'] - df['amt_mean']) / df['amt_std'].replace(0, 1)
        df['amt_standardized'].fillna(0, inplace=True)
        df['amt_deviation'] = abs(df['amt'] - df['amt_mean'])
        df['amt_deviation'].fillna(0, inplace=True)
        df.drop(['amt_mean', 'amt_std'], axis=1, inplace=True)

    except Exception as e:
        raise RuntimeError(f"Error in Amount Features: {e}")

    # 4. Location Features
    try:
        df['lat'] = df['lat'].astype(float)
        df['long'] = df['long'].astype(float)
        df['merch_lat'] = df['merch_lat'].astype(float)
        df['merch_long'] = df['merch_long'].astype(float)
        distances = []

        for row in df.itertuples():
            try:
                dist = geodesic((row.lat, row.long), (row.merch_lat, row.merch_long)).km
                distances.append(dist)

            except ValueError:
                distances.append(np.nan)

        df['distance_from_home'] = distances
        threshold = 50  # kms
        df['is_in_home_city'] = (df['distance_from_home'] <= threshold).astype(int)
        df.loc[df['distance_from_home'].isnull(), 'is_in_home_city'] = 0

    except Exception as e:
        raise RuntimeError(f"Error in Location Features: {e}")

    # 5. Frequency Features (Transactions in Last Hour)
    if 'trans_datetime' not in df.columns or not pd.api.types.is_datetime64_any_dtype(df['trans_datetime']):
        raise ValueError("'trans_datetime' column is missing or not datetime type.")

    try:
        df = df.sort_values(by=['cc_num', 'trans_datetime'], ascending=True)
        df_indexed = df.set_index('trans_datetime')
        rolling_counts = df_indexed.groupby('cc_num')['amt'].rolling('1H').count()
        rolling_counts = rolling_counts.reset_index()
        rolling_counts.rename(columns={'amt': 'trans_last_hour'}, inplace=True)
        df = pd.merge(df, rolling_counts, on=['cc_num', 'trans_datetime'], how='left')
        df['trans_last_hour'].fillna(1, inplace=True)
        df['trans_last_hour'] = df['trans_last_hour'].astype(int)

    except Exception as e:
        raise RuntimeError(f"Error in Frequency Features: {e}")

    # 6. Encode Categorical Features
    try:
        if cat_encoder is None or gen_encoder is None:
            raise ValueError("Encoders are not loaded properly.")
        df['category_encoded'] = df['category'].apply(
            lambda x: cat_encoder.transform([x])[0] if x in cat_encoder.classes_ else -1
        )
        df['gender_encoded'] = df['gender'].apply(
            lambda x: gen_encoder.transform([x])[0] if x in gen_encoder.classes_ else -1
        )

    except Exception as e:
        raise RuntimeError(f"Error in Encoding Categorical Features: {e}")

    # 6.5 Convert cc_num to numeric
    try:
        df['cc_num'] = pd.to_numeric(df['cc_num'], errors='coerce')
        if df['cc_num'].isnull().any():
            df['cc_num'].fillna(0, inplace=True)
        df['cc_num'] = df['cc_num'].astype(np.int64)

    except Exception as e:
        raise RuntimeError(f"Error converting cc_num to numeric: {e}")

    # 7. Final Feature Selection
    final_features_ordered = [
        'cc_num', 'amt', 'city_pop', 'hour_of_day', 'day_of_week', 'is_weekend',
        'time_since_last', 'amt_standardized', 'amt_deviation', 'distance_from_home',
        'is_in_home_city', 'trans_last_hour', 'category_encoded', 'gender_encoded'
    ]

    missing_final_features = [f for f in final_features_ordered if f not in df.columns]

    if missing_final_features:
        raise ValueError(f"Missing final features: {missing_final_features}")

    df_final_features = df[final_features_ordered].copy()

    # Impute any remaining NaNs
    if df_final_features.isnull().values.any():
        df_final_features.fillna(0, inplace=True)

    # Check for non-numeric/bool types
    final_dtypes = df_final_features.dtypes
    allowed_dtypes = [np.int64, np.int32, np.int16, np.int8,
                      np.float64, np.float32, np.float16,
                      int, float, bool]
    bad_dtype_cols = [col for col, dtype in final_dtypes.items() if dtype.type not in allowed_dtypes]
    
    if bad_dtype_cols:
        raise ValueError(f"Non-numeric/non-bool columns in final features: {bad_dtype_cols}")

    return df_final_features, df