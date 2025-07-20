
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


df = pd.read_csv("c:/Users/lenovo/Desktop/spillover/crypto_5min_data/BTCUSDT_5m.csv")

data_filtered = df[df['code'] == "BTC"].copy()

# 按组进行分类统计
group_summary = df.groupby('code').size().reset_index(name='NumObservations')

def get_re(data, alpha):
    # 将数据转换为DataFrame并确保是副本
    result = data.copy()

    # 转换时间列 - 使用更健壮的方式处理日期
    try:
        # 如果时间格式是 "YYYY/M/D H" 这种格式
        result['day'] = pd.to_datetime(result['time'], format='%Y/%m/%d %H')
    except:
        try:
            # 如果上面的格式不工作，尝试其他常见格式
            result['day'] = pd.to_datetime(result['time'])
        except:
            # 如果还是不行，尝试先分割时间字符串
            result['day'] = pd.to_datetime(result['time'].str.split().str[0])

    # 只保留日期部分
    result['day'] = result['day'].dt.date

    # 按天分组进行计算
    def calculate_daily_metrics(group):
        # 计算简单收益率
        group['Ret'] = (group['close'] / group['close'].shift(1) - 1) * 100
        group['Ret'].iloc[0] = 0  # First return is 0
        group['Ret'][group['close'].shift(1) == 0] = np.nan  # Handle division by zero

        # 删除缺失值
        group = group.dropna()

        if len(group) == 0:
            return None

        # 计算标准差
        sigma = group['Ret'].std()

        # 计算分位数阈值
        r_minus = norm.ppf(alpha) * sigma
        r_plus = norm.ppf(1 - alpha) * sigma

        # 计算超额收益
        REX_minus = np.sum(np.where(group['Ret'] <= r_minus, group['Ret'] ** 2, 0))
        REX_plus = np.sum(np.where(group['Ret'] >= r_plus, group['Ret'] ** 2, 0))
        REX_moderate = np.sum(np.where((group['Ret'] > r_minus) & (group['Ret'] < r_plus),
                                       group['Ret'] ** 2, 0))

        return pd.Series({
            'REX_minus': REX_minus,
            'REX_plus': REX_plus,
            'REX_moderate': REX_moderate
        })

    # 按天分组计算指标
    result = result.groupby('day').apply(calculate_daily_metrics).reset_index()

    return result

# 使用函数
har_re = get_re(data_filtered, alpha=0.05)
print(har_re)


# Create data_ret DataFrame with renamed columns first
data_ret = df[['time', 'code', 'close']].copy()
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
from datetime import date
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


# 假设 RV 和 har_re 已定义
har_pd_re = pd.merge(RV, har_re, left_index=True, right_index=True)
print(har_pd_re)


# 选择除了日期列以外的所有列
numeric_columns = har_pd_re.select_dtypes(include=['float64', 'int64']).columns

# 对数值列取自然对数
har_pd_re[numeric_columns] = har_pd_re[numeric_columns].apply(np.log)

# 查看结果
print(har_pd_re)

har_pd_re.to_csv('har-re-data.csv', index=False)
har_pd_re['rex_m_lag1'] = har_pd_re['REX_minus'].shift(1)
har_pd_re['rex_m_lag5'] = har_pd_re['REX_minus'].rolling(window=5).mean().shift(1)
har_pd_re['rex_m_lag22'] = har_pd_re['REX_minus'].rolling(window=22).mean().shift(1)

har_pd_re['rex_p_lag1'] = har_pd_re['REX_plus'].shift(1)
har_pd_re['rex_p_lag5'] = har_pd_re['REX_plus'].rolling(window=5,).mean().shift(1)
har_pd_re['rex_p_lag22'] = har_pd_re['REX_plus'].rolling(window=22).mean().shift(1)
# #
har_pd_re['rex_moderate_lag1'] = har_pd_re['REX_moderate'].shift(1)
har_pd_re['rex_moderate_lag5'] = har_pd_re['REX_moderate'].rolling(window=5).mean().shift(1)
har_pd_re['rex_moderate_lag22'] = har_pd_re['REX_moderate'].rolling(window=22).mean().shift(1)


model_data = har_pd_re[['RV', 'rex_m_lag1', 'rex_m_lag5', 'rex_m_lag22',
                       'rex_p_lag1', 'rex_p_lag5', 'rex_p_lag22',
                      'rex_moderate_lag1', 'rex_moderate_lag5', 'rex_moderate_lag22']]
# #
# # # 删除缺失值
model_data = model_data.dropna()


# 定义测试集和窗口大小
test_size = 300
window_size = 1800


# 分割训练集和测试集
train_end = len(model_data) - test_size
X_train = model_data[['rex_m_lag1', 'rex_m_lag5', 'rex_m_lag22',
                      'rex_p_lag1', 'rex_p_lag5', 'rex_p_lag22',
                      'rex_moderate_lag1', 'rex_moderate_lag5', 'rex_moderate_lag22']].iloc[:train_end]
X_test = model_data[['rex_m_lag1', 'rex_m_lag5', 'rex_m_lag22',
                     'rex_p_lag1', 'rex_p_lag5', 'rex_p_lag22',
                     'rex_moderate_lag1', 'rex_moderate_lag5', 'rex_moderate_lag22']].iloc[train_end:]
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

# 初始化滚动窗口（初始训练集为最后window_size个观测值）
rolling_X = X_train.iloc[-window_size:].copy()
rolling_y = y_train.iloc[-window_size:].copy()

# 滚动时间窗预测主循环
for i in range(len(X_test)):
    # 准备训练数据（限制为window_size大小）
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
            temp_X['rex_m_lag1'] = pred_5
            temp_X['rex_m_lag5'] = temp_X['rex_m_lag1'].shift(4).fillna(pred_1)
            temp_X['rex_m_lag22'] = temp_X['rex_m_lag1'].shift(21).fillna(pred_1)
            temp_X['rex_p_lag1'] = pred_5
            temp_X['rex_p_lag5'] = temp_X['rex_p_lag1'].shift(4).fillna(pred_1)
            temp_X['rex_p_lag22'] = temp_X['rex_p_lag1'].shift(21).fillna(pred_1)
            temp_X['rex_moderate_lag1'] = pred_5
            temp_X['rex_moderate_lag5'] = temp_X['rex_moderate_lag1'].shift(4).fillna(pred_1)
            temp_X['rex_moderate_lag22'] = temp_X['rex_moderate_lag1'].shift(21).fillna(pred_1)
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
            temp_X['rex_m_lag1'] = pred_22
            temp_X['rex_m_lag5'] = temp_X['rex_m_lag1'].shift(4).fillna(pred_1)
            temp_X['rex_m_lag22'] = temp_X['rex_m_lag1'].shift(21).fillna(pred_1)
            temp_X['rex_p_lag1'] = pred_22
            temp_X['rex_p_lag5'] = temp_X['rex_p_lag1'].shift(4).fillna(pred_1)
            temp_X['rex_p_lag22'] = temp_X['rex_p_lag1'].shift(21).fillna(pred_1)
            temp_X['rex_moderate_lag1'] = pred_22
            temp_X['rex_moderate_lag5'] = temp_X['rex_moderate_lag1'].shift(4).fillna(pred_1)
            temp_X['rex_moderate_lag22'] = temp_X['rex_moderate_lag1'].shift(21).fillna(pred_1)
            pred_22 = model.predict(temp_X)[0]
        predictions_lr22.append(pred_22)
        actuals_lr22.append(y_test.iloc[i + 21])
    else:
        predictions_lr22.append(np.nan)
        actuals_lr22.append(np.nan)

    # 更新滚动窗口（保持window_size大小）
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

df_predictions_lr.to_csv('har-re.csv', index=False)
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