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
test_size = 300
window_size = 1800


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



from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
def multi_step_predict_improved(model_data_slice, model, target_step):

    # 创建历史序列的副本
    history_rv = model_data_slice['RV'].tolist()
    history_rex_minus_lag1 = model_data_slice['rex_minus_lag1'].tolist()
    history_rex_minus_lag5 = model_data_slice['rex_minus_lag5'].tolist()
    history_rex_minus_lag22 = model_data_slice['rex_minus_lag22'].tolist()
    history_rex_plus_lag1 = model_data_slice['rex_plus_lag1'].tolist()
    history_rex_plus_lag5 = model_data_slice['rex_plus_lag5'].tolist()
    history_rex_plus_lag22 = model_data_slice['rex_plus_lag22'].tolist()
    history_rex_moderate_lag1 = model_data_slice['rex_moderate_lag1'].tolist()
    history_rex_moderate_lag5 = model_data_slice['rex_moderate_lag5'].tolist()
    history_rex_moderate_lag22 = model_data_slice['rex_moderate_lag22'].tolist()
    history_btc_lag1 = model_data_slice['btc_lag1'].tolist()
    history_btc_lag2 = model_data_slice['btc_lag2'].tolist()
    history_btc_lag3 = model_data_slice['btc_lag3'].tolist()

    for step in range(target_step):
        # 计算当前步骤所需的滞后特征
        current_len = len(history_rv)

        # Lag-1 features (all require only 1 period of history)
        if current_len >= 1:
            lag1_rex_minus = history_rex_minus_lag1[-1]
            lag1_rex_plus = history_rex_plus_lag1[-1]
            lag1_rex_moderate = history_rex_moderate_lag1[-1]
            lag1_btc1 = history_btc_lag1[-1]  # 1-period lag of first BTC variable
            lag1_btc2 = history_btc_lag2[-1]  # 1-period lag of second BTC variable
            lag1_btc3 = history_btc_lag3[-1]  # 1-period lag of third BTC variable
        else:
            raise ValueError("历史数据不足")

        # Lag-5 features
        if current_len >= 5:
            lag5_rex_minus = history_rex_minus_lag5[-5]
            lag5_rex_plus = history_rex_plus_lag5[-5]
            lag5_rex_moderate = history_rex_moderate_lag5[-5]
        else:
            lag5_rex_minus = history_rex_minus_lag5[0]
            lag5_rex_plus = history_rex_plus_lag5[0]
            lag5_rex_moderate = history_rex_moderate_lag5[0]

        # Lag-22 features
        if current_len >= 22:
            lag22_rex_minus = history_rex_minus_lag22[-22]
            lag22_rex_plus = history_rex_plus_lag22[-22]
            lag22_rex_moderate = history_rex_moderate_lag22[-22]
        else:
            lag22_rex_minus = history_rex_minus_lag22[0]
            lag22_rex_plus = history_rex_plus_lag22[0]
            lag22_rex_moderate = history_rex_moderate_lag22[0]

        # 预测下一步，使用所有特征
        X_input = np.array([[lag1_rex_minus, lag5_rex_minus, lag22_rex_minus,
                             lag1_rex_plus, lag5_rex_plus, lag22_rex_plus,
                             lag1_rex_moderate, lag5_rex_moderate, lag22_rex_moderate,
                             lag1_btc1, lag1_btc2, lag1_btc3]])
        pred = model.predict(X_input)[0]

        # 将预测值添加到 RV 历史序列
        history_rv.append(pred)

        # 更新滞后特征序列（占位）
        # rex_minus, rex_plus, rex_moderate need decomposition logic
        history_rex_minus_lag1.append(lag1_rex_minus)  # Placeholder
        history_rex_plus_lag1.append(lag1_rex_plus)    # Placeholder
        history_rex_moderate_lag1.append(lag1_rex_moderate)  # Placeholder
        history_rex_minus_lag5.append(lag5_rex_minus)
        history_rex_plus_lag5.append(lag5_rex_plus)
        history_rex_moderate_lag5.append(lag5_rex_moderate)
        history_rex_minus_lag22.append(lag22_rex_minus)
        history_rex_plus_lag22.append(lag22_rex_plus)
        history_rex_moderate_lag22.append(lag22_rex_moderate)

        # Update btc lags: Placeholder for new values of the three BTC variables
        history_btc_lag1.append(lag1_btc1)  # Placeholder: Need forecast for first BTC variable
        history_btc_lag2.append(lag1_btc2)  # Placeholder: Need forecast for second BTC variable
        history_btc_lag3.append(lag1_btc3)  # Placeholder: Need forecast for third BTC variable

    return history_rv[-1]

def rolling_forecast_corrected(model_data, test_size=300, window_size=None):
    """
    修正后的滚动预测主循环，适配HAR-CJ模型

    Args:
        model_data: 包含RV, Jv_lag1, Jv_lag5, Jv_lag22, C_t_lag1, C_t_lag5, C_t_lag22的DataFrame
        test_size: 测试集大小
        window_size: 滚动窗口大小（若为None，则使用扩展窗口）

    Returns:
        包含预测和实际值的字典
    """
    predictions_lr1 = []
    predictions_lr5 = []
    predictions_lr20 = []
    actuals_lr1 = []
    actuals_lr5 = []
    actuals_lr20 = []

    # 分割数据
    train_end = len(model_data) - test_size
    train_data = model_data.iloc[:train_end].copy()
    test_data = model_data.iloc[train_end:].copy()

    # 滚动预测
    for i in range(len(test_data)):
        # 当前训练窗口
        current_train_end = train_end + i
        if window_size is not None:
            current_train_start = max(0, current_train_end - window_size)
            current_train_data = model_data.iloc[current_train_start:current_train_end]
        else:
            current_train_data = model_data.iloc[:current_train_end]

        # 准备训练特征和目标
        X_train_current = current_train_data[['rex_minus_lag1', 'rex_minus_lag5', 'rex_minus_lag22',
                        'rex_plus_lag1', 'rex_plus_lag5', 'rex_plus_lag22',
                      'rex_moderate_lag1', 'rex_moderate_lag5', 'rex_moderate_lag22','btc_lag1','btc_lag2','btc_lag3']].values
        y_train_current = current_train_data['RV'].values

        # 训练模型
        model = LinearRegression()
        model.fit(X_train_current, y_train_current)

        # === 1步预测 ===
        X_test_current = test_data.iloc[i][['rex_minus_lag1', 'rex_minus_lag5', 'rex_minus_lag22',
                        'rex_plus_lag1', 'rex_plus_lag5', 'rex_plus_lag22',
                      'rex_moderate_lag1', 'rex_moderate_lag5', 'rex_moderate_lag22','btc_lag1','btc_lag2','btc_lag3']].values.reshape(1, -1)
        pred_1 = model.predict(X_test_current)[0]
        predictions_lr1.append(pred_1)
        actuals_lr1.append(test_data.iloc[i]['RV'])

        # === 5步预测 ===
        if i + 4 < len(test_data):
            history_data = model_data.iloc[:current_train_end]
            pred_5 = multi_step_predict_improved(history_data, model, 5)
            predictions_lr5.append(pred_5)
            actuals_lr5.append(test_data.iloc[i + 4]['RV'])
        else:
            predictions_lr5.append(np.nan)
            actuals_lr5.append(np.nan)

        # === 20步预测 ===
        if i + 21 < len(test_data):
            history_data = model_data.iloc[:current_train_end]
            pred_20 = multi_step_predict_improved(history_data, model, 22)
            predictions_lr20.append(pred_20)
            actuals_lr20.append(test_data.iloc[i + 21]['RV'])
        else:
            predictions_lr20.append(np.nan)
            actuals_lr20.append(np.nan)

    return {
        'predictions_1': predictions_lr1,
        'predictions_5': predictions_lr5,
        'predictions_20': predictions_lr20,
        'actuals_1': actuals_lr1,
        'actuals_5': actuals_lr5,
        'actuals_20': actuals_lr20
    }

# 使用示例
if __name__ == "__main__":

    results = rolling_forecast_corrected(model_data, test_size=300)

    # 计算评估指标
    def calculate_metrics(predictions, actuals):
        # 移除NaN值
        valid_idx = ~(np.isnan(predictions) | np.isnan(actuals))
        pred_clean = np.array(predictions)[valid_idx]
        actual_clean = np.array(actuals)[valid_idx]

        if len(pred_clean) == 0:
            return {'MSE': np.nan, 'RMSE': np.nan, 'MAE': np.nan}

        mse = mean_squared_error(actual_clean, pred_clean)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual_clean, pred_clean)

        return {'MSE': mse, 'RMSE': rmse, 'MAE': mae}

    # 计算各步预测的评估指标
    metrics_1 = calculate_metrics(results['predictions_1'], results['actuals_1'])
    metrics_5 = calculate_metrics(results['predictions_5'], results['actuals_5'])
    metrics_22 = calculate_metrics(results['predictions_20'], results['actuals_20'])

    print("1步预测指标:", metrics_1)
    print("5步预测指标:", metrics_5)
    print("22步预测指标:", metrics_22)
# 将预测结果保存为CSV
    max_len = len(results['predictions_1'])  # 以1步预测的长度为基准
    df_predictions = pd.DataFrame({
        'Prediction_1': results['predictions_1'],
        'Actual_1': results['actuals_1'],
        'Prediction_5': results['predictions_5'] + [np.nan] * (max_len - len(results['predictions_5'])),
        'Actual_5': results['actuals_5'] + [np.nan] * (max_len - len(results['actuals_5'])),
        'Prediction_20': results['predictions_20'] + [np.nan] * (max_len - len(results['predictions_20'])),
        'Actual_20': results['actuals_20'] + [np.nan] * (max_len - len(results['actuals_20']))
    })
    df_predictions.to_csv('log+har-re.csv', index=False)
    print("预测结果已保存至 'forecast_results.csv'")