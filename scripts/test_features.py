import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import lightgbm as lgb
import joblib, gc
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

import os, sys, warnings
from time import time 
from glob import glob

sys.path.append("/home/lishi/projects/Competition/kaggle_2023/notebooks")

from create_feature import (
    reduce_mem_usage, gen_v1_features, gen_v2_features, gen_v3_features, gen_feature_cols )

warnings.filterwarnings('ignore')


def prepare_data(csv_file, feature_dicts, feature_versions=['v1', 'v2', 'v3'], nrows=None, save_csv=None):
    
    df = pd.read_csv(csv_file, nrows=nrows)
    df = df[~df['target'].isnull()] 
    
    df.reset_index(drop=True, inplace=True)
    
    df.fillna(0, inplace=True)
    df = reduce_mem_usage(df, verbose=0)
    
    print(df.shape)
    print(f"Trading days: {df['date_id'].nunique()}")
    print(f"Stocks: {df['stock_id'].nunique()}")
    
    if 'v1' in feature_versions:
        df, v1_feat = gen_v1_features(df, feature_dicts['prices'])
        feature_dicts['v1_features'] = v1_feat
        # feature_dicts['v1_feature_category'] = v1_feat_cat
    
    if 'v2' in feature_versions:
        v2_feat_cols = feature_dicts['prices'] + feature_dicts['sizes'] + feature_dicts['v1_features']
        df, v2_features = gen_v2_features(df, v2_feat_cols)
        feature_dicts['v2_features'] = v2_features
        
    if 'v3' in feature_versions:
        df, v3_features = gen_v3_features(
            df, 
            feature_dicts['prices'],
            feature_dicts['sizes'],
            feature_dicts['v1_features']
            )
        
        feature_dicts['v3_features'] = v3_features
    
    df.fillna(0, inplace=True)
    df.replace([np.inf, -np.inf], 0, inplace=True)
    df = reduce_mem_usage(df, verbose=0)
    
    return df, feature_dicts


def train_and_cv(df, feature_cols, category_cols, lgb_params, model_name, save_dir, scaler_file=None, n_splits=5):
    
    scale_cols = [x for x in feature_cols if x not in category_cols]
    scaler = StandardScaler().fit(df[scale_cols])
    
    if scaler_file:
        joblib.dump(scaler, scaler_file)

    df[scale_cols] = scaler.transform(df[scale_cols])
    
    check_invalids = pd.DataFrame(columns=['null', 'inf'])
    for col in df.columns:
        try:
            check_invalids.loc[col] = [df[col].isnull().sum(), np.isinf(df[col]).sum()]
        except:
            print("Skip column: ", col)
            pass
        
    has_invalids = check_invalids.T[check_invalids.sum(axis=0)!=0]
    
    if len(has_invalids) > 0:
        print("Invalid values were found in dataframe.")
        print(has_invalids)
        raise Exception("Invalid values in dataframe")
    
    dates_list = df['date_id'].unique()
    
    k_fold = KFold(n_splits=n_splits, shuffle=False, random_state=None)
    kf_split = k_fold.split(dates_list)
    
    mae_scores = []
    models = []
    
    print("Start Cross-validation...")
    for fold, (train_idx, valid_idx) in enumerate(kf_split):
        
        print(f"Fold {fold+1}")
        fold_start = time()
        
        train_dates = dates_list[train_idx]
        
        half_valid = int(len(valid_idx)/2)
        valid_dates_1 = dates_list[valid_idx[:half_valid]]
        valid_dates_2 = dates_list[valid_idx[half_valid:]]
        
        print(f"Valid Dates 1: {valid_dates_1[0]} - {valid_dates_1[-1]}")
        print(f"Valid Dates 2: {valid_dates_2[0]} - {valid_dates_2[-1]}")
        
        # split train and valid set
        df_train_fold = df[df["date_id"].isin(train_dates)]
        
        df_valid_fold_1 = df[df["date_id"].isin(valid_dates_1)]
        df_valid_fold_2 = df[df["date_id"].isin(valid_dates_2)]
        
        print(f"Train  : {df_train_fold[feature_cols].shape}")
        print(f"Valid 1: {df_valid_fold_1[feature_cols].shape}")
        print(f"Valid 2: {df_valid_fold_2[feature_cols].shape}")
        
        print(f"Data preparation finished. Start training...")
        
        training_start = time()
        
        lgb_model = lgb.LGBMRegressor(**lgb_params)
        
        lgb_model.fit(
            df_train_fold[feature_cols], 
            df_train_fold['target'],
            eval_set=[(df_valid_fold_1[feature_cols], df_valid_fold_1['target'])],
            feature_name = feature_cols,
            categorical_feature = category_cols,
            callbacks=[lgb.callback.log_evaluation(period=100)],
            )
        
        models.append(lgb_model)
        
        model_file = f"{save_dir}/{model_name}_fold_{fold+1}.pkl" 
        joblib.dump(lgb_model, model_file)
        
        print(f"Fold {fold+1} Trainning finished. Time elapsed: {time()-training_start:.2f}s")
        
        y_pred_valid = lgb_model.predict(df_valid_fold_2[feature_cols])
        mae = mean_absolute_error(df_valid_fold_2['target'].values, y_pred_valid)
        mae_scores.append(mae)

        print(f"Fold {fold+1} MAE: {mae}")
        print(f"Fold {fold+1} Time elapsed: {time()-fold_start:.2f}s")
        
        del df_train_fold, df_valid_fold_1, df_valid_fold_2, y_pred_valid
        gc.collect()
        
    return models, mae_scores


def calc_feature_importance(models, feature_cols):

    df_importance = []
    
    for model in models:
        feature_imp = pd.DataFrame({'Value':model.feature_importances_,'Feature':feature_cols})
        feature_imp.sort_values(by='Value', ascending=False, inplace=True)
        df_importance.append(feature_imp)
        
    df_importance = pd.concat(df_importance)
    df_importance = df_importance.groupby('Feature').mean().reset_index()

    df_importance.sort_values(by='Value', ascending=False, inplace=True)
    df_importance = df_importance.reset_index(drop=True)
    
    return df_importance


if __name__ == "__main__":
    
    train_csv = "/home/lishi/projects/Competition/kaggle_2023/data/train.csv"

    feature_dicts = {
        'prices': ["reference_price", "far_price", "near_price", "ask_price", "bid_price", "wap"],
        'sizes':  ["matched_size", "bid_size", "ask_size", "imbalance_size"],
        "category": ["stock_id", "seconds_in_bucket", 'imbalance_buy_sell_flag']
        }
    
    # ta_indicators = ['ema', 'rsi', 'cci', 'mfi', 'ad_osc', 'macd', 'macdhist', 'macdsignal']
    
    reduce_feature = False
    percentile_thred = 20
    csv_importance = "/home/lishi/projects/Competition/kaggle_2023/data/lgb_models/lgb_v1v2v3/lgb_v1v2v3_feature_importance.csv"
    
    feat_version = ['v1', 'v2', 'v3']#, 'v3']
    model_name = "lgb_v1v2v3_no_ta_reduce"
    
    print(f"Feature version: {feat_version}")
    print("Preparing Data...")
    
    df, feature_dicts = prepare_data(train_csv, feature_dicts, feat_version, nrows=None, save_csv=None)

    feature_cols, category_cols = gen_feature_cols(feature_dicts)
    
    # feature_cols = [x for x in feature_cols if x not in ta_indicators]

    print("Number of features:", len(feature_cols))
    print("Number of category features:", len(category_cols))

    model_name += f"_{len(feature_cols)+len(category_cols)}"
    
    if reduce_feature:
        feat_importance = pd.read_csv(csv_importance)
        imp_thred = np.percentile(feat_importance['Value'].values, percentile_thred)
        print(f"Importance threshold: {imp_thred}")
        less_imp_cols = feat_importance[feat_importance['Value']<imp_thred]['Feature'].values
        print(f"Number of less important features: {len(less_imp_cols)}")
        feature_cols = [x for x in feature_cols if x not in less_imp_cols]
        print(f"Feature reduced from {len(feat_importance)} to {len(feature_cols)}")
        category_cols = [x for x in category_cols if x in feature_cols]
        print(f"Category reduced from {len(feat_importance)} to {len(category_cols)}")
        
        model_name = f"{model_name}_reduce_{percentile_thred}"
    
    save_dir = f"/home/lishi/projects/Competition/kaggle_2023/data/lgb_models/{model_name}"
    scaler_file = f"{save_dir}/{model_name}_scaler.pkl"
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    lgb_params = {
        'boosting_type': 'gbdt',
        'learning_rate': 0.015, #0.009,#0.018,
        'max_depth': 12,#9,
        'n_estimators': 800,#600,
        'num_leaves': 1024,#440,
        'objective': 'mae',
        'random_state': 42,
        'reg_alpha': 0.01,
        'reg_lambda': 0.01,
        'early_stopping_rounds': 50,
        'num_threads': 24,
        'importance_type': 'gain',
        'verbose': -1,
        }
        
    cv_results = train_and_cv(
        df,
        feature_cols, 
        category_cols, 
        lgb_params, 
        model_name=model_name, 
        save_dir=save_dir, 
        scaler_file=scaler_file, 
        n_splits=5)
    
    fig = plt.figure(figsize=(6, 5))
    mae_scores = cv_results[1]
    plt.plot(mae_scores, marker='o', color='blue', label='MAE')
    plt.title(f'MAE Scores(Overall: {np.mean(mae_scores):.4f})')
    # add number to data points
    for i, v in enumerate(mae_scores):
        plt.text(i, v, f'{v:.4f}', color='blue', fontweight='bold')
    plt.xlabel('Fold')
    plt.ylabel('MAE')
    fig.savefig(f"{save_dir}/{model_name}_MAE.png")
    plt.show()
    
    df_importance = calc_feature_importance(cv_results[0], feature_cols)
    df_importance.to_csv(f"{save_dir}/{model_name}_feature_importance.csv", index=False)
    