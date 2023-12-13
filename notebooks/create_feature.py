import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import talib as ta
from itertools import combinations
# import seaborn as sns
# import os, sys, warnings
from time import time 


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

defualt_drop_cols = [
        'min_matched_imbalance', 
        'min_imbalance_size', 
        'min_imbalance_intensity', 
        'min_price_pressure'
        ]

default_pct_features = [
    'imbalance_with_flag', 
    'matched_imbalance',
    'imbalance_size',
    'wap', 
    'bid_price',
    'ask_price'
    ]

default_roll_features = [
    'ask_price',
    'ask_price_wap_imbalance',
    'ask_size',
    'bid_price',
    'bid_price_wap_imbalance',
    'bid_size',
    'far_price_near_price_imbalance',
    'imbalance_intensity',
    'imbalance_with_flag',
    'liquidity_imbalance',
    'market_urgency',
    'matched_imbalance',
    'matched_intensity',
    'matched_size',
    'price_pressure',
    'reference_price_ask_price_imbalance',
    'reference_price_bid_price_imbalance',
    'reference_price_wap_imbalance',
    'volume',
    'wap'
    ]


def gen_v1v2_features(
    df, prices, sizes, 
    stock_labels=None,
    v2_stats=['mean', 'median', 'std', 'min', 'max'],
    drop_cols=defualt_drop_cols
    ):
    
    if stock_labels is not None:
        df = pd.merge(df, stock_labels, on='stock_id', how='left')
    
    ## V1 features: row-wise features
    v1_features = {
        "volume": "ask_size + bid_size",
        "mid_price": "(ask_price + bid_price)/2",
        "liquidity_imbalance": "(bid_size-ask_size)/(bid_size+ask_size)",
        "matched_imbalance": "(imbalance_size - matched_size)/(matched_size+imbalance_size)",
        "size_imbalance": "bid_size / ask_size",
        "imbalance_intensity": "imbalance_size / volume",
        "matched_intensity": "matched_size / volume",
        "price_spread": "ask_price - bid_price",
        'market_urgency': 'price_spread * liquidity_imbalance',
        'depth_pressure': '(ask_size - bid_size) * (far_price - near_price)',
        'price_pressure': 'imbalance_size * (ask_price - bid_price)',
        'imbalance_with_flag': 'imbalance_size * imbalance_buy_sell_flag',
        'far_price_near_price_imbalance': '(far_price - near_price) / (far_price + near_price)',
    }

    # include pair-wise price imbalances
    for c in combinations(["reference_price", "ask_price", "bid_price", "wap"], 2):
        v1_features[f"{c[0]}_{c[1]}_imbalance"] = f"({c[0]} - {c[1]}) / ({c[0]} + {c[1]})"
    
    for k, v in v1_features.items():
        df[k] = df.eval(v)
    
    v1_features = list(v1_features.keys())
    
    # -----------------------------------
    # V2 features: cross-section features
    # -----------------------------------
    
    if df['date_id'].nunique() > 1:
        group_key = ['date_id', 'seconds_in_bucket']
    else:
        group_key = ['seconds_in_bucket']
        
    group_by_seconds = df.groupby(group_key)
    v2_feat_cols = prices + sizes + v1_features
    
    # cross-section statistics of row-wise features
    df_stats = group_by_seconds[v2_feat_cols].agg(v2_stats)
    df_stats.columns = df_stats.columns.map(lambda x: '_'.join([x[1], x[0]])).to_series()
    df = pd.merge(df, df_stats, left_on=group_key, right_index=True, how='left')
    
    # cross-section ranks of row-wise features
    df_rank = group_by_seconds[v2_feat_cols].rank(pct=True).add_prefix('rank_')
    df = pd.concat([df, df_rank], axis=1)

    df = df.drop(columns=drop_cols)
    v2_features = df_stats.columns.tolist() + df_rank.columns.tolist()
    v2_features = [f for f in v2_features if f not in drop_cols]
    
    df.fillna(0, inplace=True)
    df.replace([np.inf, -np.inf], 0, inplace=True)
    df = reduce_mem_usage(df, verbose=0)
        
    return df, v1_features, v2_features


def gen_v3_features(
    df, 
    pct_features=default_pct_features,
    roll_window=5,
    roll_features=default_roll_features, 
    roll_stats=['mean', 'std', 'max', 'min'],
    add_ta=False
    ):
    
    # V3 features: rolling on timeseries features 
    if df['date_id'].nunique() > 1:
        group_key = ['date_id', 'stock_id']
    else:
        group_key = ['stock_id']
        
    group_by_stock = df.groupby(group_key)
    
    df_change = group_by_stock[pct_features].pct_change(1, fill_method=None).add_prefix('pct_')
    
    df_rolling = group_by_stock[roll_features].rolling(roll_window).agg(roll_stats).droplevel(group_key)
    df_rolling.columns = df_rolling.columns.map(
        lambda x: '_'.join([x[1], x[0], str(roll_window)])
        ).to_series()
    
    concat_list = [df, df_change, df_rolling]
    v3_features = df_change.columns.tolist() + df_rolling.columns.tolist()
    
    if add_ta:
        df_ta = calc_TA_indicators(group_by_stock)
        df_ta = df_ta.droplevel(group_key)
        concat_list.append(df_ta)
        v3_features += df_ta.columns.tolist() 
    
    df = pd.concat(concat_list, axis=1)
    df.fillna(0, inplace=True)
    df.replace([np.inf, -np.inf], 0, inplace=True)
    df = reduce_mem_usage(df, verbose=0)
    
    return df, v3_features


def calc_TA_indicators(grouped_df):
    
    import talib as ta
    
    def composite_ta(df):

        ad_osc = ta.ADOSC(df['ask_price'], df['bid_price'], df['wap'], df['volume'], fastperiod=3, slowperiod=5)
        macd, macdsignal, macdhist = ta.MACD(df['wap'], fastperiod=5, slowperiod=11, signalperiod=3)
        
        return pd.DataFrame({
            'ema': ta.EMA(df['wap'], timeperiod=5),
            'rsi': ta.RSI(df['wap'], timeperiod=5),
            'cci': ta.CCI(df['ask_price'], df['bid_price'], df['wap'], timeperiod=5),
            'mfi': ta.MFI(df['ask_price'], df['bid_price'], df['wap'], df['volume'], timeperiod=5),
            'ad_osc': ad_osc,
            'macd': macd,
            'macdsignal': macdsignal,
            'macdhist': macdhist
        })
        
    df_ta = grouped_df.apply(composite_ta)
    
    return df_ta


def gen_daily_stats_features(df, on_cols=['target', 'wap', 'volume'], stats=['mean', 'std'], n_days=5):
    
    if df['seconds_in_bucket'].nunique() > 1:
        group_key = ['stock_id', 'seconds_in_bucket']
    else:
        group_key = ['stock_id']
        
    daily_stats = df.groupby(group_key)[on_cols].rolling(n_days).agg(stats).droplevel(group_key)
    daily_stats.columns = daily_stats.columns.map(
        lambda x: '_'.join([x[1], x[0], 'daily', str(n_days)])
        ).to_series()
    
    df = pd.concat([df, daily_stats], axis=1)
    
    daily_features = daily_stats.columns.tolist()
    
    df.fillna(0, inplace=True)
    df.replace([np.inf, -np.inf], 0, inplace=True)
    df = reduce_mem_usage(df, verbose=0)
    
    return df, daily_features


if __name__ == "__main__":
    
    now = time()
    
    prices =  ["reference_price", "far_price", "near_price", "ask_price", "bid_price", "wap"]
    sizes = ["matched_size", "bid_size", "ask_size", "imbalance_size"]
    category = ["stock_id", "seconds_in_bucket", 'imbalance_buy_sell_flag', 'stock_label']

    csv_train = "./data/train.csv"
    df = pd.read_csv(csv_train, nrows=None)
    df = df[~df['target'].isnull()] 
    df.fillna(0, inplace=True)
    
    # df = df[df['date_id'] >= 430]

    print(df.shape)
    print(f"Trading days: {df['date_id'].nunique()}")
    print(f"Stocks: {df['stock_id'].nunique()}")
    
    stock_labels = pd.read_csv("/home/lishi/projects/Competition/kaggle_2023/stock_labels.csv")
    stock_labels.columns = ['stock_id', 'stock_label']
    
    df_v1v2, v1_features, v2_features = gen_v1v2_features(df, prices, sizes, stock_labels=stock_labels)
    
    df_v3, v3_features = gen_v3_features(df_v1v2, add_ta=True)
    
    df_daily, daily_features = gen_daily_stats_features(df_v3)
    
    print(df_daily.shape)
    
    print(f"V1 features: {len(v1_features)}")
    print(f"V2 features: {len(v2_features)}")
    print(f"V3 features: {len(v3_features)}")
    print(f"Daily features: {len(daily_features)}")
    
    print(f"Time elapsed: {(time()-now)/60:.2f} min.")
    print("Saving to csv...")
    
    df_daily.to_parquet(
        "/home/lishi/projects/Competition/kaggle_2023/data/train_full_features.parquet", 
        index=False)
    
    print(f"Time elapsed: {(time()-now)/60:.2f} min.")