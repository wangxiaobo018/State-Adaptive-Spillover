import pandas as pd
import numpy as np
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
# 定义文件路径字典
data_files = {
    'BTC': "c:/Users/lenovo/Desktop/spillover/crypto_5min_data/BTCUSDT_5m.csv",
    'DASH': "c:/Users/lenovo/Desktop/spillover/crypto_5min_data/DASHUSDT_5m.csv",
    'ETH': "c:/Users/lenovo/Desktop/spillover/crypto_5min_data/ETHUSDT_5m.csv",
    'LTC': "c:/Users/lenovo/Desktop/spillover/crypto_5min_data/LTCUSDT_5m.csv",
    'XLM': "c:/Users/lenovo/Desktop/spillover/crypto_5min_data/XLMUSDT_5m.csv",
    'XRP': "c:/Users/lenovo/Desktop/spillover/crypto_5min_data/XRPUSDT_5m.csv"
}
# 定义 get_RV_BV 函数
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

        # 计算 RV
        RV = np.sum(group['Ret'] ** 2)

        # 计算 BV
        abs_ret = np.abs(group['Ret'])
        BV = (np.pi / 2) * np.sum(abs_ret.shift(1) * abs_ret.shift(-1).dropna())

        # 计算 TQ
        TQ_coef = n * (2 ** (2 / 3) * gamma(7 / 6) / gamma(0.5)) ** (-3) * (n / (n - 4))

        term1 = abs_ret.iloc[4:].values
        term2 = abs_ret.iloc[2:-2].values
        term3 = abs_ret.iloc[:-4].values

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

        # 计算 JV 和 C_t
        q_alpha = norm.ppf(1 - alpha)
        JV = (RV - BV) * (Z_test > q_alpha)
        C_t = (Z_test <= q_alpha) * RV + (Z_test > q_alpha) * BV

        results.append({
            'day': day,
            'C_t': C_t,
            'JV': JV
        })

    result_df = pd.DataFrame(results)
    return result_df

# 创建字典来存储 C_t 和 JV 数据
c_t_dict = {}
jv_dict = {}

# 遍历每个数据文件，计算 HAR-CJ
for code, file_path in data_files.items():
    # 读取数据
    df = pd.read_csv(file_path)

    # 筛选数据
    data_filtered = df[df['code'] == code].copy()

    # 计算 HAR-CJ
    har_cj = get_RV_BV(data_filtered, alpha=0.05, times=True)

    # 将 day 转换为字符串，以便合并时使用
    har_cj['day'] = har_cj['day'].astype(str)

    # 存储 C_t 和 JV 数据
    c_t_dict[code] = har_cj[['day', 'C_t']].set_index('day')['C_t']
    jv_dict[code] = har_cj[['day', 'JV']].set_index('day')['JV']

# 转换为 DataFrame
ct = pd.DataFrame(c_t_dict)
jv = pd.DataFrame(jv_dict)

all_CT = ct.reset_index()  # Resets the 'day' index to a column
all_CT = all_CT.rename(columns={'day': 'DT'})  # Renames the 'day' column to 'DT'

all_JV = jv.reset_index()  # Resets the 'day' index to a column
all_JV = all_JV.rename(columns={'day': 'DT'})  # Renames the 'day' column to 'DT'

def simple_log_transform(df):
    """简洁版本的对数变换"""
    df_log = df.copy()

    # 获取除DT外的数值列
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    log_columns = [col for col in numeric_cols if col != 'DT']

    # 对所有数值列应用条件对数变换
    for col in log_columns:
        df_log[col] = np.where(df[col] > 0, np.log(df[col]), 0)

    return df_log


# 使用简洁版本
print("\n=== 使用简洁版本处理 ===")
all_CT_log_simple = simple_log_transform(all_CT)
all_JV_log_simple = simple_log_transform(all_JV)


# 步骤1：创建 model_data，以 BTC RV 作为目标变量
model_data = all_CT_log_simple[['DT', 'BTC', 'DASH', 'ETH', 'LTC', 'XLM', 'XRP']].copy()
model_data = model_data.rename(columns={'BTC': 'RV'})  # 重命名 BTC 为 RV


# 步骤2：计算 BTC RV 的分位数
rv_quantiles = model_data['RV'].quantile([0.05, 0.95])


# 步骤3：定义市场状态
model_data['market_state'] = pd.cut(model_data['RV'],
                                    bins=[-np.inf, rv_quantiles[0.05], rv_quantiles[0.95], np.inf],
                                    labels=['Bear', 'Normal', 'Bull'])

# 步骤4：根据市场状态动态选择 RV（滞后一期）
model_data['Dynamic_RV_lag1'] = np.where(model_data['market_state'] == 'Bear',
                                         model_data['ETH'],  # 熊市用 XLM
                                         np.where(model_data['market_state'] == 'Bull',
                                                  model_data['XRP'],  # 牛市用 XRP
                                                  model_data['ETH']))  #


# 步骤1：创建 model_data，以 BTC RV 作为目标变量
model_data_jv = all_JV_log_simple[['DT', 'BTC', 'DASH', 'ETH', 'LTC', 'XLM', 'XRP']].copy()
model_data_jv = model_data_jv.rename(columns={'BTC': 'RV'})  # 重命名 BTC 为 RV


# 步骤2：计算 BTC RV 的分位数
rv_quantiles = model_data_jv['RV'].quantile([0.05, 0.95])

# 步骤3：定义市场状态
model_data_jv['market_state'] = pd.cut(model_data_jv['RV'],
                                    bins=[-np.inf, rv_quantiles[0.05], rv_quantiles[0.95], np.inf],
                                    labels=['Bear', 'Normal', 'Bull'])

# 步骤4：根据市场状态动态选择 RV（滞后一期）
model_data_jv['Dynamic_jv_lag1'] = np.where(model_data_jv['market_state'] == 'Bear',
                                         model_data_jv['DASH'],  # 熊市用 XLM
                                         np.where(model_data_jv['market_state'] == 'Bull',
                                                  model_data_jv['LTC'],  # 牛市用 XRP
                                                  model_data_jv['ETH']))  #


JV_lag1 = all_JV_log_simple['BTC'].shift(1)
C_t_lag1 = all_CT_log_simple['BTC'].shift(1)
JV_lag5 = all_JV_log_simple['BTC'].rolling(window=5).mean().shift(1)
C_t_lag5 = all_CT_log_simple['BTC'].rolling(window=5).mean().shift(1)
JV_lag22 = all_JV_log_simple['BTC'].rolling(window=22).mean().shift(1)
C_t_lag22 = all_CT_log_simple['BTC'].rolling(window=22).mean().shift(1)

btc_lag1 = model_data['Dynamic_RV_lag1'].shift(1)
btc_lag2 = model_data_jv['Dynamic_jv_lag1'].shift(1)

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
all_RV.columns = ["DT","BTC", "DASH","ETH","LTC","XLM","XRP"]


all_RV['DT'] = pd.to_datetime(all_RV['DT'])
# # 对除开DT的数据进行对数转换
numeric_cols = [col for col in all_RV.columns if col != 'DT']
all_RV[numeric_cols] = np.log(all_RV[numeric_cols])

model_data= pd.DataFrame({
    'RV':all_RV['BTC'],
    'Jv_lag1': JV_lag1,
    'Jv_lag5': JV_lag5,
    'Jv_lag22': JV_lag22,
    'C_t_lag1': C_t_lag1,
    'C_t_lag5': C_t_lag5,
    'C_t_lag22': C_t_lag22,
    'btc_lag1': btc_lag1,
    'btc_lag2': btc_lag2

})
model_data = model_data.dropna()

#
# # 定义测试集和窗口大小
test_size = 500
window_size = 1500
#
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# 分割训练集和测试集
train_end = len(model_data) - test_size
X_train = model_data[['Jv_lag1', 'Jv_lag5', 'Jv_lag22', 'C_t_lag1', 'C_t_lag5', 'C_t_lag22','btc_lag1','btc_lag2']].iloc[:train_end]
X_test = model_data[['Jv_lag1', 'Jv_lag5', 'Jv_lag22', 'C_t_lag1', 'C_t_lag5', 'C_t_lag22','btc_lag1','btc_lag2']].iloc[train_end:]
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
            temp_X['Jv_lag1'] = pred_22
            temp_X['Jv_lag5'] = temp_X['Jv_lag1'].shift(4).fillna(pred_1)
            temp_X['Jv_lag22'] = temp_X['Jv_lag1'].shift(21).fillna(pred_1)
            temp_X['C_t_lag1'] = pred_22
            temp_X['C_t_lag5'] = temp_X['C_t_lag1'].shift(4).fillna(pred_1)
            temp_X['C_t_lag22'] = temp_X['C_t_lag1'].shift(21).fillna(pred_1)
            temp_X['btc_lag1'] = pred_22
            temp_X['btc_lag2'] = pred_22
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
df_predictions_lr.to_csv('log+har-cj.csv', index=False)
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