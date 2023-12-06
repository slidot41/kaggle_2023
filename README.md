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