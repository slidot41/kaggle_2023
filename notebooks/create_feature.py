import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import talib as ta
from itertools import combinations
# import seaborn as sns
# import os, sys, warnings
# from time import time 


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


def gen_v1_features(df, prices):
    # V1 features: directly apply formula to a single row
    
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
    for c in combinations(prices, 2):
        if ('far_price' in c) or ('near_price' in c):
            pass 
        else:
            v1_features[f"{c[0]}_{c[1]}_imbalance"] = f"({c[0]} - {c[1]}) / ({c[0]} + {c[1]})"
    
    for k, v in v1_features.items():
        df[k] = df.eval(v)
        
    # v1_feature_category = {
    #     # 'minute': 'seconds_in_bucket // 60',
    #     'imb_buy_side': "(imbalance_buy_sell_flag == 1)",
    #     'imb_sell_side': "(imbalance_buy_sell_flag == -1)",
    #     # 'first_half_session': '(seconds_in_bucket <= 240)',
    #     # 'second_half_session': '(seconds_in_bucket > 240)'
    # }
    
    # for k, v in v1_feature_category.items():
    #     df[k] = df.eval(v).astype(np.int8)
        
    df = reduce_mem_usage(df, verbose=0)
        
    return df, list(v1_features.keys())#, list(v1_feature_category.keys())
        
def gen_v2_features(df, v2_feat_cols):
    
    # V2 features: cross-section features
    # V2 features are generated on the groupby(['date_id', 'seconds_in_bucket'])
    # These features includes:
    # 1. statistics of V1 features (non-categorical)
    # 2. rank of V1 features for each stocks (non-categorical)
    
    group = df.groupby(['date_id', 'seconds_in_bucket'])

    v2_features_stats = ['mean', 'median', 'std', 'min', 'max']

    # calculate statistics of V1 features for each stock
    df_v2 = group[v2_feat_cols].agg(v2_features_stats).reset_index()
    df_v2.columns = ['date_id', 'seconds_in_bucket'] + [f"{c[1]}_{c[0]}" for c in df_v2.columns[2:]]
    df = df.merge(df_v2, on=['date_id', 'seconds_in_bucket'], how='left')
    
    # calculate rank of V1 features for each stock
    df_v2 = group[v2_feat_cols].rank(pct=True).add_prefix('rank_')
    df = df.merge(df_v2, left_index=True, right_index=True, how='left')
    
    df = reduce_mem_usage(df, verbose=0)
    
    drop_cols = [
        'min_matched_imbalance', 'min_imbalance_size', 
        'min_imbalance_intensity', 'min_price_pressure'
        ]
    
    df = df.drop(columns=drop_cols)
    
    v2_features = \
        [f"{s}_{c}" for c in v2_feat_cols for s in v2_features_stats] + \
        [f"rank_{c}" for c in v2_feat_cols]
        
    v2_features = [f for f in v2_features if f not in drop_cols]
        
    return df, v2_features


# !!! Requrires at least 11 timesteps to calculate all rolling statistics
def gen_v3_features(df, prices, sizes, v1_features):
    # V3 features: rolling statistics of V1 features (non-categorical)
    # V3 features are generated on the groupby(['date_id', 'stock_id'])
    # here we introduce ta-lib functions to calculate TA indicators

    # V3.1 relative change of V1 features by shift(1)
    # for prices, we calculate the change in basis points (*1e4)
    # for other features, we calculate the change in percentage (*1e2)
    group_by_stock = df.groupby(['date_id', 'stock_id'])
    
    pct_features = ['imbalance_with_flag', 'wap', 'matched_imbalance']
    
    relative_change = group_by_stock[pct_features].pct_change(1, fill_method=None).add_prefix('pct_')

    relative_change[ 'pct_wap' ] *= 1e4
    relative_change[ ['pct_imbalance_with_flag', 'pct_matched_imbalance'] ] *= 1e2
    
    # skip_v1_feat = [
    #     'ask_price_wap_imbalance', 'liquidity_imbalance',
    #     'size_imbalance', 'market_urgency',
    #     'imbalance_intensity', 'price_spread',
    #     'ask_price_bid_price_imbalance', 'volume',
    #     'matched_intensity', 'depth_pressure',
    #     'far_price_near_price_imbalance']
    
    # others_feats = [f for f in sizes+v1_features if f not in skip_v1_feat]
    
    df = pd.concat([df, relative_change], axis=1)
    v3_features = relative_change.columns.tolist()
    
    # V3.2 Simple TA indicators
    # Those are simple TA indicators that use only one feature
    
    # skip_rolling = ['size_imbalance', 'ask_price_bid_price_imbalance']
    # rolling_features = [f for f in prices + sizes + v1_features if f not in skip_rolling]
    rolling_features = [
        'ask_size',
        'bid_size',
        'imbalance_with_flag',
        'liquidity_imbalance',
        'matched_imbalance',
        'reference_price_ask_price_imbalance',
        'reference_price_bid_price_imbalance',
        'reference_price_wap_imbalance',
        'far_price_near_price_imbalance',
        ]
    
    df_v3 = group_by_stock[rolling_features].rolling(5).agg(['mean', 'std', 'max', 'min']).reset_index()
    stats_cols = [f"{c[1]}_{c[0]}_5" for c in df_v3.columns[2:]]
    df_v3.columns = ['date_id', 'stock_id'] + stats_cols
    df_v3.set_index('_level_2_5', inplace=True)
    df_v3.drop(columns=['date_id', 'stock_id'], inplace=True)
    
    df = df.merge(df_v3, left_index=True, right_index=True, how='left')
    v3_features += df_v3.columns.tolist()
        
    # # # V3.3 TA indicators that use multiple features
    # def composite_ta(df):

    #     ad_osc = ta.ADOSC(df['ask_price'], df['bid_price'], df['wap'], df['volume'], fastperiod=3, slowperiod=5)
    #     macd, macdsignal, macdhist = ta.MACD(df['wap'], fastperiod=5, slowperiod=11, signalperiod=3)
        
    #     return pd.DataFrame({
    #         'ema': ta.EMA(df['wap'], timeperiod=5),
    #         'rsi': ta.RSI(df['wap'], timeperiod=5),
    #         'cci': ta.CCI(df['ask_price'], df['bid_price'], df['wap'], timeperiod=5),
    #         'mfi': ta.MFI(df['ask_price'], df['bid_price'], df['wap'], df['volume'], timeperiod=5),
    #         'ad_osc': ad_osc,
    #         'macd': macd,
    #         'macdsignal': macdsignal,
    #         'macdhist': macdhist
    #     })
    
    # df_v3 = group_by_stock.apply(composite_ta) 
    # v3_features += df_v3.columns.tolist()
    
    # df_v3.reset_index(inplace=True)
    # df_v3.set_index('level_2', inplace=True)
    # df_v3.drop(columns=['date_id', 'stock_id'], inplace=True)
    
    # df = pd.concat([df, df_v3], axis=1)
    
    return df, v3_features


def gen_feature_cols(feature_dicts):
    
    feature_cols = []
    category_cols = []
    
    for k, v in feature_dicts.items():
        feature_cols += v
        if k in ['category', 'v1_feature_category']:
            category_cols += v
            
    return feature_cols, category_cols


def gen_features(df_train, feature_dicts):
    
    df_train.fillna(0, inplace=True)
    df_train = reduce_mem_usage(df_train, verbose=0)

    df_v1, v1_feat, v1_feat_cat = gen_v1_features(df_train, feature_dicts['prices'])
    feature_dicts['v1_features'] = v1_feat
    feature_dicts['v1_feature_category'] = v1_feat_cat
    
    v2_feat_cols = feature_dicts['prices'] + feature_dicts['sizes'] + feature_dicts['v1_features']
    df_v2, v2_features = gen_v2_features(df_v1, v2_feat_cols)
    feature_dicts['v2_features'] = v2_features
    
    df_v3, v3_features = gen_v3_features(
        df_v2, 
        feature_dicts['prices'],
        feature_dicts['sizes'],
        feature_dicts['v1_features']
        )
    
    feature_dicts['v3_features'] = v3_features
    
    df_v3.fillna(0, inplace=True)
    df_v3.replace([np.inf, -np.inf], 0, inplace=True)
    df_v3 = reduce_mem_usage(df_v3, verbose=0)
    
    return df_v3, feature_dicts


if __name__ == "__main__":
    
    feature_dicts = {
        'prices': ["reference_price", "far_price", "near_price", "ask_price", "bid_price", "wap"],
        'sizes':  ["matched_size", "bid_size", "ask_size", "imbalance_size"],
        "category": ["stock_id", "seconds_in_bucket", 'imbalance_buy_sell_flag']
    }

    csv_train = "./data/train.csv"
    df = pd.read_csv(csv_train)
    df = df[~df['target'].isnull()] 

    print(df.shape)
    print(f"Trading days: {df['date_id'].nunique()}")
    print(f"Stocks: {df['stock_id'].nunique()}")
    
    # test with only 30 days of data
    sub_df = df[df['date_id'] <= 30].copy()
    df_final, feature_dicts = gen_features(sub_df, feature_dicts)
    
    print(df_final.shape)
    print(df_final.columns[:20])
    