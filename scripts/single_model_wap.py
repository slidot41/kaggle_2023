import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import catboost as ctb 
import joblib, gc, os, sys
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.model_selection import KFold, train_test_split
from itertools import combinations

import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

sys.path.append("/home/lishi/projects/Competition/kaggle_2023/notebooks")

from create_feature import ( reduce_mem_usage )


def load_csv(csv_file):
    df = pd.read_csv(csv_file)
    df = reduce_mem_usage(df)
    df = df[df['target'].notnull()].reset_index(drop=True)
    df = df.drop(columns=['row_id', 'time_id'])
    df['imbalance_buy_sell_flag'] = df['imbalance_buy_sell_flag'].replace({-1: 0, 1: 1})
    return df

def gen_features(df):

    prices =  ["reference_price", "far_price", "near_price", "ask_price", "bid_price", "wap"]
    sizes = ["matched_size", "bid_size", "ask_size", "imbalance_size"]

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

    # time-seires shifts
    roll_window = 5
    gp = df.groupby(['date_id', 'stock_id'])
    row_shifts = [
        gp[prices+sizes].shift(i).add_prefix(f"prev_{i}_") for i in range(1, roll_window+1) 
        ]

    future_wap = gp['wap'].shift(-6).add_prefix("future_")
    df = pd.concat([df, future_wap]+row_shifts, axis=1)
    df['wap_chg'] = df['future_wap'] / df['wap'] - 1

    # This feature shall be pre-calculated and loaded from an external csv. We don't use it for now.
    # wap_chg_std = df.groupby(['date_id', 'stock_id'])['wap_chg'].std().reset_index().rename(columns={'wap_chg': 'wap_chg_std'})
    # df = df.merge(wap_chg_std, on=['date_id', 'stock_id'], how='left')
   
    df.fillna(0, inplace=True)
    df.replace([np.inf, -np.inf], 0, inplace=True)
    df = reduce_mem_usage(df, verbose=0)

    return df


def split_dataset(df, feature_cols, target_col, nsplit=5):

    dates_list = df['date_id'].unique()

    k_fold = KFold(n_splits=nsplit, shuffle=False, random_state=None)
    kf_split = list(k_fold.split(dates_list))

    datasets = {}

    for fold, (train_idx, valid_idx) in enumerate(kf_split):

        df_train = df[df['date_id'].isin(dates_list[train_idx])],
        x_train, x_test = train_test_split(df_train, test_size=0.2, random_state=42)

        df_valid = df[df['date_id'].isin(dates_list[valid_idx])]

        datasets[f"fold_{fold}"] = {
            'x_train': x_train[feature_cols],
            'y_train': x_train[target_col],
            'x_test': x_test[feature_cols],
            'y_test': x_test[target_col],
            'x_valid': df_valid[feature_cols],
            'y_valid': df_valid[target_col],
        }

    return datasets


def train_lgbm(lgb_params, x_train, y_train, x_test, y_test, feature_cols, category_cols):

    lgb_model = lgb.LGBMRegressor(**lgb_params)
    lgb_model.fit(
        x_train, 
        y_train,
        eval_set=[(x_test, y_test)],
        feature_name = feature_cols,
        categorical_feature = category_cols,
        verbose=0,
        early_stopping_rounds=10,
    )
    
    return lgb_model


def valid_lgbm(lgb_model, x_valid, y_valid):
    y_pred = lgb_model.predict(x_valid)
    mae = mean_absolute_error(y_valid, y_pred)
    return mae


if __name__ == "__main__":

    save_dir = "/home/lishi/projects/Competition/kaggle_2023/wap_pred_models"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    train_csv = "/home/lishi/projects/Competition/kaggle_2023/data/train.csv"
    df_all = load_csv(train_csv)
    df_all = gen_features(df_all)

    print("Load Data of Shape: ", df_all.shape)
    print("Training days: ", df_all['date_id'].nunique())
    print("Stocks: ", df_all['stock_id'].nunique())

    feature_cols = [x for x in df_all.columns if x not in ['target', 'date_id', 'future_wap', 'wap_chg']]
    category_cols = ['stock_id', 'seconds_in_bucket', 'imbalance_buy_sell_flag' ]

    target_col = 'wap_chg'

    scale_cols = [x for x in feature_cols if x not in category_cols]

    lgb_params = {
        'boosting_type': 'gbdt',
        'learning_rate': 0.012, #0.009,#0.018,
        'max_depth': 16,#9,
        'n_estimators': 500,#600,
        'num_leaves': 600,#440,
        'objective': 'mae',
        'random_state': 42,
        'reg_alpha': 0.01,
        'reg_lambda': 0.01,
        'early_stopping_rounds': 50,
        'num_threads': 16,
        'importance_type': 'gain',
        'verbose': -1,
        }
    
    # loop over all stocks and train a model for each stock
    mae_df = pd.DataFrame(columns=['stock_id']+['mae_fold_'+str(x+1) for x in range(5)])
    
    for stock_id in df_all['stock_id'].unique():

        df = df_all[df_all['stock_id']==stock_id].copy()

        scaler_features = StandardScaler()
        df[scale_cols] = scaler_features.fit_transform(df[scale_cols])
    
        scaler_target = StandardScaler()
        df[target_col] = scaler_target.fit_transform(df[target_col])

        joblib.dump(scaler_features, f"{save_dir}/scaler_features_{stock_id}.pkl")
        joblib.dump(scaler_target, f"{save_dir}/scaler_target_{stock_id}.pkl")

        datasets = split_dataset(df, feature_cols, target_col, nsplit=5)
        print(f"Dataset shape for stock_id {stock_id}:", df.shape)
        print(f"Training days: {df['date_id'].nunique()}")

        model_name = f"lgbm_{stock_id}"

        mae_list = []
        for fold, data in datasets.items():

            print(f"Training model {model_name} for fold {fold} ...")
            lgb_model = train_lgbm(
                lgb_params, 
                data['x_train'], data['y_train'], 
                data['x_test'], data['y_test'], 
                feature_cols, 
                category_cols
                )
            
            mae = valid_lgbm(lgb_model, data['x_valid'], data['y_valid'])
            print(f"MAE for fold {fold}: {mae}")
            mae_list.append(mae)

            joblib.dump(lgb_model, f"{save_dir}/{model_name}_fold_{fold}.pkl")

            del lgb_model
            gc.collect()

        mae_df.loc[len(mae_df)] = [stock_id] + mae_list
        print(f"MAE for stock_id {stock_id}: {np.mean(mae_list)}")
        print(f"Current Overall MAE: {mae_df[['mae_fold_1', 'mae_fold_2', 'mae_fold_3', 'mae_fold_4', 'mae_fold_5']].mean(axis=1).mean()}")

    mae_df.to_csv(f"{save_dir}/valid_mae_log.csv", index=False)
