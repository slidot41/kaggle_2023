# kaggle_2023

# Meeting Notes

## 28. NOV

Visualize on Tableau to detect data patterns and correlations.
Produce 128 features based on original features.

QYQ: 
Applied naive Gaussian Process Regression with Matern, Rational Quadratic, White Kernel, memory explosion
Extract non-linear feature map with kernel approximation (reduction from 45 to 40 features), applied Bayesian Ridge Regression, validation score = 5.96
Directly apply Hist Gradient Boosting Regressor, validation score = 5.89

LB & LKL
Built a fully connected neural network with 2 layers and train, validation score = 5.96

```
split_day = 435
df_train = df[df["date_id"] <= split_day]
df_valid = df[df["date_id"] > split_day]
print(f"train : {df_train.shape}, valid : {df_valid.shape}")
```

### To do:
Add feature selection.

# Set up env with conda:

```
conda create -n dsml python=3.10
conda activate dsml
# install pytorch
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia 
# install other packages
conda install -c conda-forge kaggle pandas scikit-learn scipy 
conda install -c conda-forge lightgbm catboost xgboost gplearn 
conda install -c conda-forge sympy ta-lib
conda install -c conda-forge matplotlib seaborn
conda install -c conda-forge numba
```

# Download Kaggle dataset

First authenticate using an API token. Go to the 'Account' tab of your Kaggle user profile and select 'Create New Token'. This will trigger the download of kaggle.json, a file containing your API credentials.

If you are using the Kaggle CLI tool, the tool will look for this token at ~/.kaggle/kaggle.json on Linux, OSX, and other UNIX-based operating systems, and at C:\Users\<Windows-username>\.kaggle\kaggle.json on Windows. 

Run the following commands using the kaggle command-line-tool:

```
# list files related to the comepition:
kaggle c files optiver-trading-at-the-close 

# download files:
kaggle competitions download -c optiver-trading-at-the-close 
unzip optiver-trading-at-the-close.zip 
rm optiver-trading-at-the-close.zip 

# submit to LB
kaggle competitions submit -c optiver-trading-at-the-close -f [FILE] -m [MESSAGE]

# check previous submissions
kaggle competitions submissions -c optiver-trading-at-the-close
```