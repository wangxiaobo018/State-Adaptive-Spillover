import pandas as pd
import numpy as np
from scipy.stats import norm
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
# 定义数据文件路径
# 读取数据
data_files = {
    'BTC': "c:/Users/lenovo/Desktop/spillover/crypto_5min_data/BTCUSDT_5m.csv",
    'DASH': "c:/Users/lenovo/Desktop/spillover/crypto_5min_data/DASHUSDT_5m.csv",
    'ETH': "c:/Users/lenovo/Desktop/spillover/crypto_5min_data/ETHUSDT_5m.csv",
    'LTC': "c:/Users/lenovo/Desktop/spillover/crypto_5min_data/LTCUSDT_5m.csv",
    'XLM': "c:/Users/lenovo/Desktop/spillover/crypto_5min_data/XLMUSDT_5m.csv",
    'XRP': "c:/Users/lenovo/Desktop/spillover/crypto_5min_data/XRPUSDT_5m.csv"
}

def get_re(data, alpha):
    """
    计算REX指标的函数
    """
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
        group.loc[group.index[0], 'Ret'] = 0  # First return is 0
        group.loc[group['close'].shift(1) == 0, 'Ret'] = np.nan  # Handle division by zero

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
    result = result.groupby('day', group_keys=False).apply(calculate_daily_metrics, include_groups=False).reset_index()
    # 过滤掉None值
    result = result.dropna()

    return result


# 存储所有加密货币的REX数据
all_rex_data = {}

# 处理每个加密货币文件
for crypto_name, file_path in data_files.items():
    print(f"正在处理 {crypto_name}...")

    try:
        # 读取数据
        df = pd.read_csv(file_path)

        # 如果文件中有code列，过滤对应的数据；否则直接使用所有数据
        if 'code' in df.columns:
            data_filtered = df[df['code'] == crypto_name].copy()
        else:
            data_filtered = df.copy()

        # 计算REX指标
        har_re = get_re(data_filtered, alpha=0.05)

        # 添加加密货币标识列
        har_re['crypto'] = crypto_name

        # 存储结果
        all_rex_data[crypto_name] = har_re

        print(f"{crypto_name} 处理完成，共 {len(har_re)} 天数据")

    except Exception as e:
        print(f"处理 {crypto_name} 时出错: {e}")

# 合并所有数据
if all_rex_data:
    # 将所有数据合并成一个DataFrame
    combined_data = pd.concat(all_rex_data.values(), ignore_index=True)

    # 创建三个分别的数据集
    # 1. REX_minus 数据 (包含day, crypto, REX_minus)
    rex_minus_data = combined_data[['day', 'crypto', 'REX_minus']].copy()
    rex_minus_pivot = rex_minus_data.pivot(index='day', columns='crypto', values='REX_minus')
    rex_minus_pivot.index.name = 'DT'
    all_RD = rex_minus_pivot

    # 2. REX_plus 数据 (包含day, crypto, REX_plus)
    rex_plus_data = combined_data[['day', 'crypto', 'REX_plus']].copy()
    rex_plus_pivot = rex_plus_data.pivot(index='day', columns='crypto', values='REX_plus')
    rex_plus_pivot.index.name = 'DT'
    all_RP = rex_plus_pivot


    # 3. REX_moderate 数据 (包含day, crypto, REX_moderate)
    rex_moderate_data = combined_data[['day', 'crypto', 'REX_moderate']].copy()
    rex_moderate_pivot = rex_moderate_data.pivot(index='day', columns='crypto', values='REX_moderate')
    rex_moderate_pivot.index.name = 'DT'
    all_RM = rex_moderate_pivot

# 修正：将索引转换为 datetime，而不是替换整个 DataFrame
all_RM.index = pd.to_datetime(all_RM.index)
all_RD.index = pd.to_datetime(all_RD.index)
all_RP.index = pd.to_datetime(all_RP.index)

# 确保 all_RM, all_RD, all_RP 是 DataFrame
columns = ['BTC', 'DASH', 'ETH', 'LTC', 'XLM', 'XRP']
if isinstance(all_RM, np.ndarray):
    all_RM = pd.DataFrame(all_RM, columns=columns)
if isinstance(all_RD, np.ndarray):
    all_RD = pd.DataFrame(all_RD, columns=columns)
if isinstance(all_RP, np.ndarray):
    all_RP = pd.DataFrame(all_RP, columns=columns)

# 现在可以安全地应用对数变换
all_RM = np.log(all_RM)  # 对数变换数据
all_RD = np.log(all_RD)  # 对数变换数据
all_RP = np.log(all_RP)  # 对数变换数据

# 替换无限值和 NaN
all_RM = all_RM.where(~all_RM.isin([np.inf, -np.inf, np.nan]), 0)
all_RD = all_RD.where(~all_RD.isin([np.inf, -np.inf, np.nan]), 0)
all_RP = all_RP.where(~all_RP.isin([np.inf, -np.inf, np.nan]), 0)

print(all_RP)
print(all_RM)
print(all_RD)

# Step 1: 为 all_RM 创建 model_data
model_data_rm = all_RM[['BTC', 'DASH', 'ETH', 'LTC', 'XLM', 'XRP']].copy()
model_data_rm = model_data_rm.rename(columns={'BTC': 'RV'})  # 重命名 BTC 为 RV

# Step 2: 计算 RV 分位数
rm_quantiles = model_data_rm['RV'].quantile([0.05, 0.95])

# Step 3: 定义市场状态
model_data_rm['market_state'] = pd.cut(model_data_rm['RV'],
                                       bins=[-np.inf, rm_quantiles[0.05], rm_quantiles[0.95], np.inf],
                                       labels=['Bear', 'Normal', 'Bull'])

# Step 4: 动态 RV 选择（滞后）
model_data_rm['Dynamic_RM_lag1'] = np.where(model_data_rm['market_state'] == 'Bear',
                                            model_data_rm['XRP'],  # 熊市: ETH
                                            np.where(model_data_rm['market_state'] == 'Bull',
                                                     model_data_rm['XRP'],  # 牛市: XRP
                                                     model_data_rm['ETH']))  # 正常: ETH

# 为 all_RD 重复上述步骤
model_data_rd = all_RD[['BTC', 'DASH', 'ETH', 'LTC', 'XLM', 'XRP']].copy()
model_data_rd = model_data_rd.rename(columns={'BTC': 'RV'})

rd_quantiles = model_data_rd['RV'].quantile([0.05, 0.95])

model_data_rd['market_state'] = pd.cut(model_data_rd['RV'],
                                       bins=[-np.inf, rd_quantiles[0.05], rd_quantiles[0.95], np.inf],
                                       labels=['Bear', 'Normal', 'Bull'])

model_data_rd['Dynamic_RD_lag1'] = np.where(model_data_rd['market_state'] == 'Bear',
                                            model_data_rd['ETH'],  # 熊市: XRP
                                            np.where(model_data_rd['market_state'] == 'Bull',
                                                     model_data_rd['XRP'],  # 牛市: XRP
                                                     model_data_rd['ETH']))  # 正常: XLM

# 为 all_RP 重复上述步骤
model_data_rp = all_RP[['BTC', 'DASH', 'ETH', 'LTC', 'XLM', 'XRP']].copy()
model_data_rp = model_data_rp.rename(columns={'BTC': 'RV'})

rp_quantiles = model_data_rp['RV'].quantile([0.05, 0.95])

model_data_rp['market_state'] = pd.cut(model_data_rp['RV'],
                                       bins=[-np.inf, rp_quantiles[0.05], rp_quantiles[0.95], np.inf],
                                       labels=['Bear', 'Normal', 'Bull'])

model_data_rp['Dynamic_RP_lag1'] = np.where(model_data_rp['market_state'] == 'Bear',
                                            model_data_rp['ETH'],  # 熊市: XLM
                                            np.where(model_data_rp['market_state'] == 'Bull',
                                                     model_data_rp['XRP'],  # 牛市: XRP
                                                     model_data_rp['ETH']))  # 正常: XLM
print(model_data_rd)
print(model_data_rp)
print(model_data_rm)
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
all_RV = rv_merged.sort_values('DT')

all_RV = all_RV.dropna()  # 删除包含NaN的行

# --- 关键修正 2: 将 DT 列设为索引，而不是重置它 ---
all_RV.set_index('DT', inplace=True)

# 现在 all_RV 的索引是 DatetimeIndex

# 数据准备 (原来的列名 "RV_BTC" 等需要重命名)
all_RV.columns = ["BTC", "DASH","ETH","LTC","XLM","XRP"]

# data_df 现在将自动继承 DatetimeIndex
data_df = np.log(all_RV)



rex_minus_lag1 = all_RD['BTC'].shift(1)
rex_plus_lag1 = all_RP['BTC'].shift(1)
rex_moderate_lag1 = all_RM['BTC'].shift(1)
rex_minus_lag5 = all_RD['BTC'].rolling(window=5).mean().shift(1)
rex_plus_lag5 = all_RP['BTC'].rolling(window=5).mean().shift(1)
rex_moderate_lag5 = all_RM['BTC'].rolling(window=5).mean().shift(1)
rex_minus_lag22 = all_RD['BTC'].rolling(window=22).mean().shift(1)
rex_plus_lag22 = all_RP['BTC'].rolling(window=22).mean().shift(1)
rex_moderate_lag22 = all_RM['BTC'].rolling(window=22).mean().shift(1)
btc_lag1 = model_data_rm ['Dynamic_RM_lag1'].shift(1)
btc_lag2 = model_data_rd['Dynamic_RD_lag1'].shift(1)
btc_lag3 = model_data_rp['Dynamic_RP_lag1'].shift(1)

model_data = pd.DataFrame({
    'RV': data_df['BTC'],
    'rex_minus_lag1': rex_minus_lag1,
    'rex_plus_lag1': rex_plus_lag1,
    'rex_moderate_lag1': rex_moderate_lag1,
    'rex_minus_lag5': rex_minus_lag5,
    'rex_plus_lag5': rex_plus_lag5,
    'rex_moderate_lag5': rex_moderate_lag5,
    'rex_minus_lag22': rex_minus_lag22,
    'rex_plus_lag22': rex_plus_lag22,
    'rex_moderate_lag22': rex_moderate_lag22,
    'btc_lag1': btc_lag1,
    'btc_lag2': btc_lag2,
    'btc_lag3': btc_lag3
})
model_data = model_data.dropna()
print(model_data)

# 定义测试集和窗口大小
test_size = 500
window_size = 1500


# 分割训练集和测试集
train_end = len(model_data) - test_size
X_train = model_data[['rex_minus_lag1', 'rex_minus_lag5', 'rex_minus_lag22',
                        'rex_plus_lag1', 'rex_plus_lag5', 'rex_plus_lag22',
                      'rex_moderate_lag1', 'rex_moderate_lag5', 'rex_moderate_lag22','btc_lag1','btc_lag2','btc_lag3']].iloc[:train_end]
X_test = model_data[['rex_minus_lag1', 'rex_minus_lag5', 'rex_minus_lag22',
                        'rex_plus_lag1', 'rex_plus_lag5', 'rex_plus_lag22',
                     'rex_moderate_lag1', 'rex_moderate_lag5', 'rex_moderate_lag22','btc_lag1','btc_lag2','btc_lag3']].iloc[train_end:]
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
rolling_X = X_train.iloc[-window_size:].copy()
rolling_y = y_train.iloc[-window_size:].copy()

# 滚动时间窗预测主循环
for i in range(len(X_test)):
    # 准备训练数据
    X_train_loop = rolling_X.values
    y_train_loop = rolling_y.values

    # 训练线性回归模型
    model = LinearRegression()
    model.fit(X_train_loop, y_train_loop)

    # 1步预测
    pred_1 = model.predict(X_test.iloc[i:i+1].values)[0]
    predictions_lr1.append(pred_1)
    actuals_lr1.append(y_test.iloc[i])
    prediction_dates.append(X_test.index[i])

    # 5步预测
    if i + 5 <= len(X_test):
        pred_5 = pred_1
        temp_X = X_test.iloc[i:i+1].copy()
        for j in range(4):  # 预测额外4步以达到5步
            # 更新现有特征列
            temp_X['rex_minus_lag1'] = pred_5
            temp_X['rex_minus_lag5'] = temp_X['rex_minus_lag1'].shift(4).fillna(pred_1)
            temp_X['rex_minus_lag22'] = temp_X['rex_minus_lag1'].shift(21).fillna(pred_1)
            temp_X['rex_plus_lag1'] = pred_5
            temp_X['rex_plus_lag5'] = temp_X['rex_plus_lag1'].shift(4).fillna(pred_1)
            temp_X['rex_plus_lag22'] = temp_X['rex_plus_lag1'].shift(21).fillna(pred_1)
            temp_X['rex_moderate_lag1'] = pred_5
            temp_X['rex_moderate_lag5'] = temp_X['rex_moderate_lag1'].shift(4).fillna(pred_1)
            temp_X['rex_moderate_lag22'] = temp_X['rex_moderate_lag1'].shift(21).fillna(pred_1)
            temp_X['btc_lag1'] = pred_5
            temp_X['btc_lag2'] = pred_5
            temp_X['btc_lag3'] = pred_5
            pred_5 = model.predict(temp_X.values)[0]
        predictions_lr5.append(pred_5)
        actuals_lr5.append(y_test.iloc[i+4])
    else:
        predictions_lr5.append(np.nan)
        actuals_lr5.append(np.nan)

    # 22步预测
    if i + 22 <= len(X_test):
        pred_22 = pred_1
        temp_X = X_test.iloc[i:i+1].copy()
        for j in range(21):  # 预测额外21步以达到22步
            # 更新现有特征列
            temp_X['rex_minus_lag1'] = pred_22
            temp_X['rex_minus_lag5'] = temp_X['rex_minus_lag1'].shift(4).fillna(pred_1)
            temp_X['rex_minus_lag22'] = temp_X['rex_minus_lag1'].shift(21).fillna(pred_1)
            temp_X['rex_plus_lag1'] = pred_22
            temp_X['rex_plus_lag5'] = temp_X['rex_plus_lag1'].shift(4).fillna(pred_1)
            temp_X['rex_plus_lag22'] = temp_X['rex_plus_lag1'].shift(21).fillna(pred_1)
            temp_X['rex_moderate_lag1'] = pred_22
            temp_X['rex_moderate_lag5'] = temp_X['rex_moderate_lag1'].shift(4).fillna(pred_1)
            temp_X['rex_moderate_lag22'] = temp_X['rex_moderate_lag1'].shift(21).fillna(pred_1)
            temp_X['btc_lag1'] = pred_22
            temp_X['btc_lag2'] = pred_22
            temp_X['btc_lag3'] = pred_22
            pred_22 = model.predict(temp_X.values)[0]
        predictions_lr22.append(pred_22)
        actuals_lr22.append(y_test.iloc[i+21])
    else:
        predictions_lr22.append(np.nan)
        actuals_lr22.append(np.nan)

    # 更新滚动窗口
    new_obs_X = X_test.iloc[i:i+1]
    new_obs_y = y_test.iloc[i:i+1]
    rolling_X = pd.concat([rolling_X.iloc[1:], new_obs_X], ignore_index=True)
    rolling_y = pd.concat([rolling_y.iloc[1:], new_obs_y], ignore_index=True)

# 创建结果DataFrame
df_predictions_lr = pd.DataFrame({
    'Date': prediction_dates,
    'Prediction_1': predictions_lr1,
    'Actual_1': actuals_lr1,
    'Prediction_5': predictions_lr5,
    'Actual_5': actuals_lr5,
    'Prediction_22': predictions_lr22,
    'Actual_22': actuals_lr22
})

print(df_predictions_lr)

df_predictions_lr.to_csv('log+har-re.csv', index=False)
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