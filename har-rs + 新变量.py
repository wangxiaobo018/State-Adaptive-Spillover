import pandas as pd
import numpy as np
from pathlib import Path

from scipy.stats import norm

from scipy.stats import norm
from scipy.special import gamma
from scipy.optimize import differential_evolution, minimize
from concurrent.futures import ThreadPoolExecutor
import warnings
import seaborn as sns
from statsmodels.tsa.vector_ar.var_model import VAR
from scipy.stats import norm
from scipy.special import logsumexp
import statsmodels.tools.numdiff as nd
from statsmodels.tools.eval_measures import aic, bic, hqic
from torch.fx.experimental.graph_gradual_typechecker import all_eq

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
import matplotlib.dates as mdates
from sklearn.preprocessing import StandardScaler
import os
from scipy.stats import norm
from scipy.special import logsumexp
import json
from numpy.linalg import inv
from scipy.stats import t, kendalltau, spearmanr
from statsmodels.regression.quantile_regression import QuantReg


def calculate_returns(prices):
    """Calculate returns from price series."""
    returns = np.zeros(len(prices))
    for i in range(1, len(prices)):
        if prices[i - 1] == 0:
            returns[i] = np.nan
        else:
            returns[i] = ((prices[i] - prices[i - 1]) / prices[i - 1]) * 100
    returns[0] = 0
    return returns


def calculate_RS(data):
    """Calculate RS+ and RS- from returns."""
    positive_returns = np.where(data['Ret'] > 0, data['Ret'], 0)
    negative_returns = np.where(data['Ret'] < 0, data['Ret'], 0)

    RS_plus = np.sum(np.square(positive_returns))
    RS_minus = np.sum(np.square(negative_returns))

    return pd.Series({
        'RS_plus': RS_plus,
        'RS_minus': RS_minus
    })


def process_har_rs_model(data_idx_path, crypto_id):
    """Process data for HAR-RS model for a single cryptocurrency."""
    # Read the index data
    df_idx = pd.read_csv(data_idx_path)

    # Process index data
    data_ret_idx = (
        df_idx[['time', 'code', 'close']]
        .rename(columns={'time': 'DT', 'code': 'id', 'close': 'PRICE'})
        .dropna()
    )

    # Calculate returns for index data
    grouped_idx = data_ret_idx.groupby('id')
    returns_list_idx = []

    for name, group in grouped_idx:
        group_returns = pd.DataFrame({
            'DT': group['DT'],
            'id': group['id'],
            'Ret': calculate_returns(group['PRICE'].values)
        })
        returns_list_idx.append(group_returns)

    data_ret_idx = pd.concat(returns_list_idx, ignore_index=True)

    # Filter for specific index
    data_cj = data_ret_idx.query(f'id == "{crypto_id}"').copy()

    # Calculate RS statistics by date
    result = (
        data_cj.groupby(pd.to_datetime(data_cj['DT']).dt.date)
        .apply(calculate_RS)
        .reset_index()
    )

    # Rename columns for consistency
    result = result.rename(columns={'level_0': 'DT'})
    return result


def combine_rs_data(data_files):
    """Combine RS_plus and RS_minus for multiple cryptocurrencies."""
    rs_plus_list = []
    rs_minus_list = []

    for crypto_id, file_path in data_files.items():
        # Process each cryptocurrency
        result = process_har_rs_model(file_path, crypto_id)
        # Convert DT to datetime for consistency
        result['DT'] = pd.to_datetime(result['DT'])
        # Create DataFrames for RS_plus and RS_minus
        rs_plus = result[['DT', 'RS_plus']].rename(columns={'RS_plus': crypto_id})
        rs_minus = result[['DT', 'RS_minus']].rename(columns={'RS_minus': crypto_id})
        rs_plus_list.append(rs_plus)
        rs_minus_list.append(rs_minus)

    # Merge all RS_plus DataFrames on DT
    rs_plus_combined = rs_plus_list[0]
    for df in rs_plus_list[1:]:
        rs_plus_combined = rs_plus_combined.merge(df, on='DT', how='outer')

    # Merge all RS_minus DataFrames on DT
    rs_minus_combined = rs_minus_list[0]
    for df in rs_minus_list[1:]:
        rs_minus_combined = rs_minus_combined.merge(df, on='DT', how='outer')

    # Set DT as index
    rs_plus_combined.set_index('DT', inplace=True)
    rs_minus_combined.set_index('DT', inplace=True)

    return rs_plus_combined, rs_minus_combined


# Example usage
if __name__ == "__main__":
    data_files = {
        'BTC': "c:/Users/lenovo/Desktop/spillover/crypto_5min_data/BTCUSDT_5m.csv",
        'DASH': "c:/Users/lenovo/Desktop/spillover/crypto_5min_data/DASHUSDT_5m.csv",
        'ETH': "c:/Users/lenovo/Desktop/spillover/crypto_5min_data/ETHUSDT_5m.csv",
        'LTC': "c:/Users/lenovo/Desktop/spillover/crypto_5min_data/LTCUSDT_5m.csv",
        'XLM': "c:/Users/lenovo/Desktop/spillover/crypto_5min_data/XLMUSDT_5m.csv",
        'XRP': "c:/Users/lenovo/Desktop/spillover/crypto_5min_data/XRPUSDT_5m.csv"
    }

    # Process all cryptocurrencies and combine results
    rs_plus_df, rs_minus_df = combine_rs_data(data_files)

    print("\nRS_plus Combined Data Sample:")
    print(rs_plus_df.head())
    print("\nRS_minus Combined Data Sample:")
    print(rs_minus_df.head())


# 对 rs_plus_df 进行对数变换
rs_plus_log = rs_plus_df.copy()
rs_plus_log[rs_plus_log.columns.difference(['DT'])] = np.log1p(rs_plus_log[rs_plus_log.columns.difference(['DT'])])

# 对 rs_minus_df 进行对数变换
rs_minus_log = rs_minus_df.copy()
rs_minus_log[rs_minus_log.columns.difference(['DT'])] = np.log1p(rs_minus_log[rs_minus_log.columns.difference(['DT'])])

############################ rs_plus_log 和 rs_minus_log  都选择2 已经是对数变换后的数据


# 读取数据
data_files = {
    'BTC': "c:/Users/lenovo/Desktop/spillover/crypto_5min_data/BTCUSDT_5m.csv",
    'DASH': "c:/Users/lenovo/Desktop/spillover/crypto_5min_data/DASHUSDT_5m.csv",
    'ETH': "c:/Users/lenovo/Desktop/spillover/crypto_5min_data/ETHUSDT_5m.csv",
    'LTC': "c:/Users/lenovo/Desktop/spillover/crypto_5min_data/LTCUSDT_5m.csv",
    'XLM': "c:/Users/lenovo/Desktop/spillover/crypto_5min_data/XLMUSDT_5m.csv",
    'XRP': "c:/Users/lenovo/Desktop/spillover/crypto_5min_data/XRPUSDT_5m.csv"
}


# 定义计算收益率的函数
def calculate_returns(prices):
    # 计算日收益率：(P_t / P_{t-1} - 1) * 100
    returns = (prices / prices.shift(1) - 1) * 100
    returns.iloc[0] = 0  # 第一个收益率设为0
    returns[prices.shift(1) == 0] = np.nan  # 处理除零情况
    return returns


# 定义计算RV的函数
def calculate_rv(df, coin_name):
    # 复制数据并重命名列
    data_ret = df[['time', 'close']].copy()
    data_ret.columns = ['DT', 'PRICE']
    data_ret = data_ret.dropna()  # 删除缺失值

    # 计算收益率
    data_ret['Ret'] = calculate_returns(data_ret['PRICE'])

    # 将DT转换为日期
    data_ret['DT'] = pd.to_datetime(data_ret['DT']).dt.date

    # 计算日度RV：收益率平方的日度总和
    RV = (data_ret
          .groupby('DT')['Ret']
          .apply(lambda x: np.sum(x ** 2))
          .reset_index())

    # 重命名RV列为币种名称
    RV.columns = ['DT', f'RV_{coin_name}']
    return RV


# 计算每种加密货币的RV
rv_dfs = []
for coin, file_path in data_files.items():
    df = pd.read_csv(file_path)
    rv_df = calculate_rv(df, coin)
    rv_dfs.append(rv_df)

# 合并所有RV数据框，按DT对齐
rv_merged = rv_dfs[0]  # 以第一个RV数据框（BTC）为基础
for rv_df in rv_dfs[1:]:
    rv_merged = rv_merged.merge(rv_df, on='DT', how='outer')

# 将DT转换为datetime格式（可选）
rv_merged['DT'] = pd.to_datetime(rv_merged['DT'])

# 按日期排序
all_RV = rv_merged.sort_values('DT').reset_index(drop=True)

all_RV= all_RV.dropna()  # 删除包含NaN的行

# 数据准备
# 数据准备
all_RV.columns = ["DT","BTC", "DASH","ETH","LTC","XLM","XRP"]

all_RV = all_RV.set_index('DT')
all_RV_log = np.log(all_RV)



# 步骤1：创建 model_data，以 BTC RV 作为目标变量
model_data_plus = rs_plus_log[[ 'BTC', 'DASH', 'ETH', 'LTC', 'XLM', 'XRP']].copy()
model_data_plus = model_data_plus.rename(columns={'BTC': 'RV'})  # 重命名 BTC 为 RV

# 步骤2：计算 BTC RV 的分位数
rv_quantiles_plus = model_data_plus['RV'].quantile([0.05, 0.95])

# 步骤3：定义市场状态
model_data_plus['market_state'] = pd.cut(model_data_plus['RV'],
                                        bins=[-np.inf, rv_quantiles_plus[0.05], rv_quantiles_plus[0.95], np.inf],
                                        labels=['Bear', 'Normal', 'Bull'])

# 步骤4：根据市场状态动态选择 RV（滞后一期）
model_data_plus['Dynamic_plus_lag1'] = np.where(model_data_plus['market_state'] == 'Bear',
                                             model_data_plus['XRP'].shift(1),  # 熊市用 XRP
                                             np.where(model_data_plus['market_state'] == 'Bull',
                                                      model_data_plus['XRP'].shift(1),  # 牛市用 XRP
                                                      model_data_plus['ETH'].shift(1)))  # 正常市场用 ETH

# 步骤1：创建 model_data，以 BTC RV 作为目标变量
model_data_minus = rs_minus_log[[ 'BTC', 'DASH', 'ETH', 'LTC', 'XLM', 'XRP']].copy()
model_data_minus = model_data_minus.rename(columns={'BTC': 'RV'})  # 重命名 BTC 为 RV

# 步骤2：计算 BTC RV 的分位数
rv_quantiles_minus = model_data_minus['RV'].quantile([0.05, 0.95])

# 步骤3：定义市场状态
model_data_minus['market_state'] = pd.cut(model_data_minus['RV'],
                                         bins=[-np.inf, rv_quantiles_minus[0.05], rv_quantiles_minus[0.95], np.inf],
                                         labels=['Bear', 'Normal', 'Bull'])

# 步骤4：根据市场状态动态选择 RV（滞后一期）
model_data_minus['Dynamic_minus_lag1'] = np.where(model_data_minus['market_state'] == 'Bear',
                                              model_data_minus['XRP'].shift(1),  # 熊市用 XRP
                                              np.where(model_data_minus['market_state'] == 'Bull',
                                                       model_data_minus['XRP'].shift(1),  # 牛市用 XRP
                                                       model_data_minus['ETH'].shift(1)))  # 正常市场用 ETH


rs_p_lag1 =rs_plus_log['BTC'].shift(1)
rs_p_lag5 = rs_plus_log['BTC'].rolling(window=5).mean().shift(1)
rs_p_lag22 =rs_plus_log['BTC'].rolling(window=22).mean().shift(1)
rs_m_lag1 = rs_minus_log['BTC'].shift(1)
rs_m_lag5 =rs_minus_log['BTC'].rolling(window=5).mean().shift(1)
rs_m_lag22 =rs_minus_log['BTC'].rolling(window=22).mean().shift(1)
btc_lag1  = model_data_plus['Dynamic_plus_lag1']
btc_lag2 = model_data_minus['Dynamic_minus_lag1']


model_data = pd.DataFrame({
    'RV':all_RV_log['BTC'],
    'rs_p_lag1':rs_p_lag1,
    'rs_p_lag5':rs_p_lag5,
    'rs_m_lag1':rs_m_lag1,
    'rs_m_lag5':rs_m_lag5,
    'rs_p_lag22':rs_p_lag22,
    'rs_m_lag22':rs_m_lag22,
    'btc_lag1': btc_lag1,
    'btc_lag2': btc_lag2
})
model_data = model_data.dropna()
print(model_data)

# 定义测试集和窗口大小
test_size = 500
window_size = 1500

# 分割训练集和测试集
train_end = len(model_data) - test_size
X_train = model_data[['rs_p_lag1', 'rs_p_lag5', 'rs_p_lag22', 'rs_m_lag1', 'rs_m_lag5', 'rs_m_lag22','btc_lag1','btc_lag2']].iloc[:train_end]
X_test = model_data[['rs_p_lag1', 'rs_p_lag5', 'rs_p_lag22', 'rs_m_lag1', 'rs_m_lag5', 'rs_m_lag22','btc_lag1','btc_lag2']].iloc[train_end:]
y_train = model_data['RV'].iloc[:train_end]
y_test = model_data['RV'].iloc[train_end:]

# 初始化预测和实际值列表
predictions_lr1 = []
predictions_lr5 = []
predictions_lr22 = []
actuals_lr1 = []
actuals_lr5 = []
actuals_lr22 = []

# 初始化滚动窗口（初始训练集为最后window_size个观测值）
rolling_X = X_train.iloc[-window_size:].copy()
rolling_y = y_train.iloc[-window_size:].copy()

# 滚动时间窗预测主循环
for i in range(len(X_test)):
    # 准备训练数据（保持DataFrame格式）
    X_train_loop = rolling_X
    y_train_loop = rolling_y

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
            temp_X['rs_p_lag1'] = pred_5
            temp_X['rs_p_lag5'] = temp_X['rs_p_lag1'].shift(4).fillna(pred_1)
            temp_X['rs_p_lag22'] = temp_X['rs_p_lag1'].shift(21).fillna(pred_1)
            temp_X['rs_m_lag1'] = pred_5
            temp_X['rs_m_lag5'] = temp_X['rs_m_lag1'].shift(4).fillna(pred_1)
            temp_X['rs_m_lag22'] = temp_X['rs_m_lag1'].shift(21).fillna(pred_1)
            temp_X['btc_lag1'] = pred_5
            temp_X['btc_lag2'] = pred_5
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
            temp_X['rs_p_lag1'] = pred_22
            temp_X['rs_p_lag5'] = temp_X['rs_p_lag1'].shift(4).fillna(pred_1)
            temp_X['rs_p_lag22'] = temp_X['rs_p_lag1'].shift(21).fillna(pred_1)
            temp_X['rs_m_lag1'] = pred_22
            temp_X['rs_m_lag5'] = temp_X['rs_m_lag1'].shift(4).fillna(pred_1)
            temp_X['rs_m_lag22'] = temp_X['rs_m_lag1'].shift(21).fillna(pred_1)
            temp_X['btc_lag1'] = pred_22
            temp_X['btc_lag2'] = pred_22
            pred_22 = model.predict(temp_X)[0]
        predictions_lr22.append(pred_22)
        actuals_lr22.append(y_test.iloc[i + 21])
    else:
        predictions_lr22.append(np.nan)
        actuals_lr22.append(np.nan)

    # 更新滚动窗口（保持window_size大小）
    new_obs_X = X_test.iloc[i:i+1]
    new_obs_y = y_test.iloc[i:i+1]
    rolling_X = pd.concat([rolling_X.iloc[1:], new_obs_X], ignore_index=True)
    rolling_y = pd.concat([rolling_y.iloc[1:], new_obs_y], ignore_index=True)

# 创建结果DataFrame
df_predictions_lr = pd.DataFrame({
    'Prediction_1': predictions_lr1,
    'Actual_1': actuals_lr1,
    'Prediction_5': predictions_lr5,
    'Actual_5': actuals_lr5,
    'Prediction_22': predictions_lr22,
    'Actual_22': actuals_lr22
})

df_predictions_lr.to_csv('log+har-rs.csv', index=False)
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