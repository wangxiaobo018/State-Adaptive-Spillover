
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize
from numba import jit
from concurrent.futures import ThreadPoolExecutor
import warnings


from scipy.stats import norm
from scipy.special import logsumexp
warnings.filterwarnings('ignore')
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from sklearn.linear_model import LinearRegression, LassoCV, Lasso
from scipy.ndimage import uniform_filter1d
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import mstats, norm
from scipy.special import gamma
from tqdm import tqdm
from sklearn.model_selection import TimeSeriesSplit, KFold
import matplotlib.pyplot as plt
import os


from datetime import date
# Read the data
df = pd.read_csv("c:/Users/lenovo/Desktop/spillover/crypto_5min_data/BTCUSDT_5m.csv")


data_filtered = df[df['code'] == "BTC"].copy()


def get_RV_BV(data, alpha=0.05, times=True):

    idx = 100 if times else 1

    df = data.copy()


    df['datetime'] = pd.to_datetime(df['time'])
    df['day'] = df['datetime'].dt.date

    results = []
    for day, group in df.groupby('day'):

        group = group.sort_values('datetime')


        group['Ret'] = (np.log(group['close']) - np.log(group['close'].shift(1))) * idx


        group = group.dropna(subset=['Ret'])
        n = len(group)

        if n < 5:
            continue

        # 计算RV
        RV = np.sum(group['Ret'] ** 2)

        # 计算BV
        abs_ret = np.abs(group['Ret'])
        BV = (np.pi / 2) * np.sum(abs_ret.shift(1) * abs_ret.shift(-1).dropna())


        TQ_coef = n * (2 ** (2 / 3) * gamma(7 / 6) / gamma(0.5)) ** (-3) * (n / (n - 4))


        term1 = abs_ret.iloc[4:].values  # Ret[5:n()]
        term2 = abs_ret.iloc[2:-2].values  # Ret[3:(n-2)]
        term3 = abs_ret.iloc[:-4].values  # Ret[1:(n-4)]

        min_len = min(len(term1), len(term2), len(term3))
        if min_len > 0:
            TQ = TQ_coef * np.sum((term1[:min_len] ** (4 / 3)) *
                                  (term2[:min_len] ** (4 / 3)) *
                                  (term3[:min_len] ** (4 / 3)))
        else:
            continue

        # Z_test
        Z_test = ((RV - BV) / RV) / np.sqrt(((np.pi / 2) ** 2 + np.pi - 5) *
                                            (1 / n) * max(1, TQ / (BV ** 2)))

        # calculate JV
        q_alpha = norm.ppf(1 - alpha)
        JV = (RV - BV) * (Z_test > q_alpha)
        C_t = (Z_test <= q_alpha) * RV + (Z_test > q_alpha) * BV

        results.append({

            'BV': BV,
            'JV': JV,
            'C_t': C_t
        })


    result_df = pd.DataFrame(results)
    return result_df[['BV', 'JV', 'C_t']]

har_cj = get_RV_BV(data_filtered, alpha=0.05, times=True)
print(har_cj)


# Read the data
df_data = pd.read_csv("c:/Users/lenovo/Desktop/spillover/crypto_5min_data/BTCUSDT_5m.csv")

# Get group summary
group_summary = df_data.groupby('code').size().reset_index(name='NumObservations')

# Create data_ret DataFrame with renamed columns first
data_ret = df_data[['time', 'code', 'close']].copy()
data_ret.columns = ['DT', 'id', 'PRICE']
data_ret = data_ret.dropna()

# Calculate returns for each group using the new formula
def calculate_returns(prices):
    # Compute daily returns using the given formula
    returns = (prices / prices.shift(1) - 1) * 100  # (Pt - Pt-1) / Pt-1 * 100
    returns.iloc[0] = 0  # First return is 0
    returns[prices.shift(1) == 0] = np.nan  # Handle division by zero
    return returns

# Calculate returns by group
data_ret['Ret'] = data_ret.groupby('id')['PRICE'].transform(calculate_returns)

# Get group summary for data_ret
group_summary_ret = data_ret.groupby('id').size().reset_index(name='NumObservations')

# Filter for "000001.XSHG" and remove unnecessary columns
data_filtered = data_ret[data_ret['id'] == "BTC"].copy()
data_filtered = data_filtered.drop('id', axis=1)

# Convert DT to datetime and calculate daily RV
data_filtered['DT'] = pd.to_datetime(data_filtered['DT']).dt.date

RV = (data_filtered
      .groupby('DT')['Ret']
      .apply(lambda x: np.sum(x**2))
      .reset_index())

# Ensure RV has the correct column names
RV.columns = ['DT', 'RV']

# Convert DT to datetime for consistency with har_cj
RV['DT'] = pd.to_datetime(RV['DT'])


data_get_cj = pd.merge(RV, har_cj, left_index=True, right_index=True)
print(data_get_cj)


data_get_cj_log = data_get_cj.copy()

# 获取需要取log的列（除了DT）
log_columns = [col for col in data_get_cj.columns if col != 'DT']

# 取log变换
data_get_cj_log[log_columns] = data_get_cj_log[log_columns].apply(np.log)

# 特殊处理JV列：原来<=0的值，log后设为0
jv_negative_mask = data_get_cj['JV'] <= 0
data_get_cj_log.loc[jv_negative_mask, 'JV'] = 0

data_get_cj_log.to_csv('har-cj-data.csv', index=False)
JV_lag1 = data_get_cj_log['JV'].shift(1)
C_t_lag1 = data_get_cj_log['C_t'].shift(1)
JV_lag5 = data_get_cj_log['JV'].rolling(window=5).mean().shift(1)
C_t_lag5 = data_get_cj_log['C_t'].rolling(window=5).mean().shift(1)
JV_lag22 = data_get_cj_log['JV'].rolling(window=22).mean().shift(1)
C_t_lag22 = data_get_cj_log['C_t'].rolling(window=22).mean().shift(1)

model_data= pd.DataFrame({
    'RV':data_get_cj_log['RV'],
    'Jv_lag1': JV_lag1,
    'Jv_lag5': JV_lag5,
    'Jv_lag22': JV_lag22,
    'C_t_lag1': C_t_lag1,
    'C_t_lag5': C_t_lag5,
    'C_t_lag22': C_t_lag22
})
model_data = model_data.dropna()
#
# # 定义测试集和窗口大小
test_size = 300
window_size = 1800
#
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# 分割训练集和测试集
train_end = len(model_data) - test_size
X_train = model_data[['Jv_lag1', 'Jv_lag5', 'Jv_lag22', 'C_t_lag1', 'C_t_lag5', 'C_t_lag22']].iloc[:train_end]
X_test = model_data[['Jv_lag1', 'Jv_lag5', 'Jv_lag22', 'C_t_lag1', 'C_t_lag5', 'C_t_lag22']].iloc[train_end:]
y_train = model_data['RV'].iloc[:train_end]
y_test = model_data['RV'].iloc[train_end:]


# 初始化预测、实际值和日期列表
predictions_lr1 = []
predictions_lr5 = []
predictions_lr22 = []
actuals_lr1 = []
actuals_lr5 = []
actuals_lr22 = []
prediction_dates = []

# 初始化滚动窗口
rolling_X = X_train.copy()
rolling_y = y_train.copy()

# 滚动时间窗预测主循环
for i in range(len(X_test)):
    # 准备训练数据
    if isinstance(rolling_X, np.ndarray):
        X_train_loop = rolling_X
        y_train_loop = rolling_y
    else:
        X_train_loop = rolling_X.values
        y_train_loop = rolling_y.values

    # 训练线性回归模型
    model = LinearRegression()
    model.fit(X_train_loop, y_train_loop)

    # 1步预测
    pred_1 = model.predict(X_test.iloc[i:i + 1])[0]
    predictions_lr1.append(pred_1)
    actuals_lr1.append(y_test.iloc[i])


    # 5步预测
    if i + 5 <= len(X_test):
        pred_5 = pred_1
        temp_X = X_test.iloc[i:i + 1].copy()
        for j in range(4):  # Predict additional 4 steps to reach 5
            temp_X['Jv_lag1'] = pred_5
            temp_X['Jv_lag5'] = temp_X['Jv_lag1'].shift(4).fillna(pred_1)
            temp_X['Jv_lag22'] = temp_X['Jv_lag1'].shift(21).fillna(pred_1)
            temp_X['C_t_lag1'] = pred_5
            temp_X['C_t_lag5'] = temp_X['C_t_lag1'].shift(4).fillna(pred_1)
            temp_X['C_t_lag22'] = temp_X['C_t_lag1'].shift(21).fillna(pred_1)
            pred_5 = model.predict(temp_X)[0]
        predictions_lr5.append(pred_5)
        actuals_lr5.append(y_test.iloc[i + 4])
    else:
        predictions_lr5.append(np.nan)
        actuals_lr5.append(np.nan)

    # 22步预测
    if i + 22 <= len(X_test):
        pred_22 = pred_1
        temp_X = X_test.iloc[i:i + 1].copy()
        for j in range(21):  # Predict additional 21 steps to reach 22
            temp_X['Jv_lag1'] = pred_22
            temp_X['Jv_lag5'] = temp_X['Jv_lag1'].shift(4).fillna(pred_1)
            temp_X['Jv_lag22'] = temp_X['Jv_lag1'].shift(21).fillna(pred_1)
            temp_X['C_t_lag1'] = pred_22
            temp_X['C_t_lag5'] = temp_X['C_t_lag1'].shift(4).fillna(pred_1)
            temp_X['C_t_lag22'] = temp_X['C_t_lag1'].shift(21).fillna(pred_1)
            pred_22 = model.predict(temp_X)[0]
        predictions_lr22.append(pred_22)
        actuals_lr22.append(y_test.iloc[i + 21])
    else:
        predictions_lr22.append(np.nan)
        actuals_lr22.append(np.nan)

    # 更新滚动窗口
    if isinstance(rolling_X, pd.DataFrame):
        new_obs_X = X_test.iloc[i:i+1]
        new_obs_y = y_test.iloc[i:i+1]
        rolling_X = pd.concat([rolling_X.iloc[1:], new_obs_X], ignore_index=True)
        rolling_y = pd.concat([rolling_y.iloc[1:], new_obs_y], ignore_index=True)
    else:  # 如果是numpy数组
        new_obs_X = X_test.iloc[i:i + 1].values
        rolling_X = np.vstack((rolling_X[1:], new_obs_X))
        rolling_y = np.append(rolling_y[1:], y_test.iloc[i])

# 创建结果DataFrame
df_predictions_lr = pd.DataFrame({
    'Prediction_1': predictions_lr1,
    'Actual_1': actuals_lr1,
    'Prediction_5': predictions_lr5,
    'Actual_5': actuals_lr5,
    'Prediction_22': predictions_lr22,
    'Actual_22': actuals_lr22
})
df_predictions_lr.to_csv('har-cj.csv', index=False)
import numpy as np

# 假设 df_predictions_lr 包含预测值和实际值
predictions = df_predictions_lr['Prediction_1'].values
actuals = df_predictions_lr['Actual_1'].values

# MSE
mse = np.mean((predictions - actuals) ** 2)

# MAE
mae = np.mean(np.abs(predictions - actuals))

# HMSE
hmse = np.mean((1 - predictions / actuals) ** 2)

# HMAE
hmae = np.mean(np.abs(1 - predictions / actuals))

# RMSE
rmse = np.sqrt(np.mean((predictions - actuals) ** 2))

# 打印结果
print(f"1-Step Prediction Loss Metrics:")
print(f"MSE: {mse:.6f}")
print(f"MAE: {mae:.6f}")
print(f"HMSE: {hmse:.6f}")
print(f"HMAE: {hmae:.6f}")
print(f"RMSE: {rmse:.6f}")