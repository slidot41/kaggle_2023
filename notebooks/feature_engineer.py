import pandas as pd
import numpy as np
import gc
from numba import njit, prange
from itertools import combinations


def reduce_mem_usage(df, verbose=0):
    """
    Iterate through all numeric columns of a dataframe and modify the data type
    to reduce memory usage.
    """
    # Calculate the initial memory usage of the DataFrame
    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                # Check if the column's data type is a float
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float32)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float32)
                    
    if verbose:
        print(f"Memory usage of dataframe is {start_mem:.2f} MB")
        end_mem = df.memory_usage().sum() / 1024**2
        print(f"Memory usage after optimization is: {end_mem:.2f} MB")
        decrease = 100 * (start_mem - end_mem) / start_mem
        print(f"Decreased by {decrease:.2f}%")

    return df


# Function to compute triplet imbalance in parallel using Numba
@njit(parallel=True)
def compute_triplet_imbalance(df_values, comb_indices):
    num_rows = df_values.shape[0]
    num_combinations = len(comb_indices)
    imbalance_features = np.empty((num_rows, num_combinations))

    # Loop through all combinations of triplets
    for i in prange(num_combinations):
        a, b, c = comb_indices[i]
        
        # Loop through rows of the DataFrame
        for j in range(num_rows):
            max_val = max(df_values[j, a], df_values[j, b], df_values[j, c])
            min_val = min(df_values[j, a], df_values[j, b], df_values[j, c])
            mid_val = df_values[j, a] + df_values[j, b] + df_values[j, c] - min_val - max_val
            
            # Prevent division by zero
            if mid_val == min_val:
                imbalance_features[j, i] = np.nan
            else:
                imbalance_features[j, i] = (max_val - mid_val) / (mid_val - min_val)

    return imbalance_features

# Function to calculate triplet imbalance for given price data and a DataFrame
def calculate_triplet_imbalance_numba(price, df):
    # Convert DataFrame to numpy array for Numba compatibility
    df_values = df[price].values
    comb_indices = [(price.index(a), price.index(b), price.index(c)) for a, b, c in combinations(price, 3)]

    # Calculate the triplet imbalance using the Numba-optimized function
    features_array = compute_triplet_imbalance(df_values, comb_indices)

    # Create a DataFrame from the results
    columns = [f"{a}_{b}_{c}_imb2" for a, b, c in combinations(price, 3)]
    features = pd.DataFrame(features_array, columns=columns)

    return features

# Function to generate imbalance features
def imbalance_features(df):
    # Define lists of price and size-related column names
    prices = ["reference_price", "far_price", "near_price", "ask_price", "bid_price", "wap"]
    sizes = ["matched_size", "bid_size", "ask_size", "imbalance_size"]

    df['session_label'] = df['seconds_in_bucket'] // 300 + 1

    # V1 features
    # Calculate various features using Pandas eval function
    df["volume"] = df.eval("ask_size + bid_size")
    df["mid_price"] = df.eval("ask_price + bid_price")/2
    df["liquidity_imbalance"] = df.eval("(bid_size-ask_size)/(bid_size+ask_size)")
    df["matched_imbalance"] = df.eval("imbalance_size - matched_size")/df.eval("matched_size+imbalance_size")
    df["size_imbalance"] = df.eval("bid_size / ask_size")
    df["imbalance_intensity"] = df.eval("imbalance_size / volume")
    df["matched_intensity"] = df.eval("matched_size / volume")

    # Create features for pairwise price imbalances
    for c in combinations(prices, 2):
        df[f"{c[0]}_{c[1]}_imb"] = df.eval(f"({c[0]} - {c[1]})/({c[0]} + {c[1]})")
        
    df["price_spread"] = df["ask_price"] - df["bid_price"]
    df['market_urgency'] = df['price_spread'] * df['liquidity_imbalance']
    df['depth_pressure'] = (df['ask_size'] - df['bid_size']) * (df['far_price'] - df['near_price'])
    df['price_pressure'] = df['imbalance_size'] * (df['ask_price'] - df['bid_price'])
    df['imbalance_with_flag'] = df['imbalance_size'] * df['imbalance_buy_sell_flag']
    
    # V2 features
    # Calculate additional features for each stock on each day
    group_by_stock_date = df.groupby(['stock_id', 'date_id'])
    df["imbalance_momentum"] = group_by_stock_date['imbalance_size'].diff(periods=1) / df['matched_size']
    df["spread_intensity"] = group_by_stock_date['price_spread'].pct_change()

    for col in [
        'matched_size', 'imbalance_size', 'bid_size', 'ask_size', 
        'reference_price', 'ask_price', 'bid_price', 
        'market_urgency', 'imbalance_momentum', 'size_imbalance', 
        ]:
        for window in range(1, 7):
            df[f"{col}_ret_{window}"] = group_by_stock_date[col].pct_change(window)
    

    for window in range(1, 7):
        df['imbalance_flag_diff'] = group_by_stock_date['imbalance_buy_sell_flag'].diff(window)
    

    # V2 features
    # Calculate additional features
    # df["imbalance_momentum"] = df.groupby(['stock_id', 'date_id'])['imbalance_size'].diff(periods=1) / df['matched_size']
    # df["spread_intensity"] = df.groupby(['stock_id', 'date_id'])['price_spread'].diff()
    
    # V3 features
    # Calculate shifted and return features for specific columns
    # for col in ['matched_size', 'imbalance_size', 'reference_price', 'imbalance_with_flag']:
    #     for window in [1, 2, 3, 10]:
    #         df[f"{col}_shift_{window}"] = df.groupby(['stock_id', 'date_id'])[col].shift(window)
    #         df[f"{col}_ret_{window}"] = df.groupby(['stock_id', 'date_id'])[col].pct_change(window)
    
    # # Calculate diff features for specific columns
    # for col in ['ask_price', 'bid_price', 'ask_size', 'bid_size',
    #             'market_urgency', 'imbalance_momentum', 'size_imbalance']:
    #     for window in [1, 2, 3, 10]:
    #         df[f"{col}_diff_{window}"] = df.groupby("stock_id")[col].diff(window)
    # Replace infinite values with 0
    return df.replace([np.inf, -np.inf], 0)


def numba_imb_features(df):
    prices = ["reference_price", "far_price", "near_price", "ask_price", "bid_price", "wap"]
    sizes = ["matched_size", "bid_size", "ask_size", "imbalance_size"]
    
    for func in ["mean", "std", "skew", "kurt"]:
        df[f"all_prices_{func}"] = df[prices].agg(func, axis=1)
        df[f"all_sizes_{func}"] = df[sizes].agg(func, axis=1)
        
    # Calculate triplet imbalance features using the Numba-optimized function
    for c in [['ask_price', 'bid_price', 'wap', 'reference_price'], sizes]:
        triplet_feature = calculate_triplet_imbalance_numba(c, df)
        df[triplet_feature.columns] = triplet_feature.values
    return df


# Function to generate time and stock-related features
def other_features(df, global_stock_id_feats):
    df["dow"] = df["date_id"] % 5                 # Day of the week
    df["seconds"] = df["seconds_in_bucket"] % 60  # Seconds
    df["minute"] = df["seconds_in_bucket"] // 60  # Minutes

    # Map global features to the DataFrame
    for key, value in global_stock_id_feats.items():
        df[f"global_{key}"] = df["stock_id"].map(value.to_dict())

    return df


# Function to generate all features by combining imbalance and other features
def generate_all_features(df, global_stock_id_feats):
    # Select relevant columns for feature generation
    # cols = [c for c in df.columns if c not in ["row_id", "time_id", "target"]]
    # df = df[cols]
    
    # Generate imbalance features
    df = imbalance_features(df)
    df = numba_imb_features(df)
    # Generate time and stock-related features
    df = other_features(df, global_stock_id_feats)
    gc.collect()
    
    # Select and return the generated features
    feature_name = [i for i in df.columns if i not in ["row_id", "time_id"]]
    
    return df[feature_name]