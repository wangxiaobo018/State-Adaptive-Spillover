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
test_size = 300
window_size = 1800
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




from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

def multi_step_predict_improved(model_data_slice, model, target_step):
    """
    改进的多步预测函数 for HAR-CJ model with btc_lag1 and btc_lag2

    Args:
        model_data_slice: 包含足够历史数据的DataFrame切片，包含 'RV', 'Jv_lag1', 'Jv_lag5', 'Jv_lag22',
                         'C_t_lag1', 'C_t_lag5', 'C_t_lag22', 'btc_lag1', 'btc_lag2'
        model: 训练好的模型 (LinearRegression trained on all eight features)
        target_step: 目标预测步数

    Returns:
        目标步数的预测值
    """
    # 创建历史序列的副本
    history_rv = model_data_slice['RV'].tolist()
    history_jv_lag1 = model_data_slice['Jv_lag1'].tolist()
    history_jv_lag5 = model_data_slice['Jv_lag5'].tolist()
    history_jv_lag22 = model_data_slice['Jv_lag22'].tolist()
    history_ct_lag1 = model_data_slice['C_t_lag1'].tolist()
    history_ct_lag5 = model_data_slice['C_t_lag5'].tolist()
    history_ct_lag22 = model_data_slice['C_t_lag22'].tolist()
    history_btc_lag1 = model_data_slice['btc_lag1'].tolist()
    history_btc_lag2 = model_data_slice['btc_lag2'].tolist()

    for step in range(target_step):
        current_len = len(history_rv)

        # Jv_lag1, C_t_lag1, btc_lag1, btc_lag2
        if current_len >= 1:
            lag1_jv = history_jv_lag1[-1]
            lag1_ct = history_ct_lag1[-1]
            lag1_btc = history_btc_lag1[-1]
            lag2_btc = history_btc_lag2[-1]
        else:
            raise ValueError("历史数据不足")


        # Jv_lag5 and C_t_lag5
        if current_len >= 5:
            lag5_jv = history_jv_lag5[-5]
            lag5_ct = history_ct_lag5[-5]
        else:
            lag5_jv = history_jv_lag5[0]
            lag5_ct = history_ct_lag5[0]

        # Jv_lag22 and C_t_lag22
        if current_len >= 22:
            lag22_jv = history_jv_lag22[-22]
            lag22_ct = history_ct_lag22[-22]
        else:
            lag22_jv = history_jv_lag22[0]
            lag22_ct = history_ct_lag22[0]

        # 预测下一步，使用所有特征
        X_input = np.array([[lag1_jv, lag5_jv, lag22_jv, lag1_ct, lag5_ct, lag22_ct, lag1_btc, lag2_btc]])
        pred = model.predict(X_input)[0]

        # 将预测值添加到 RV 历史序列
        history_rv.append(pred)


        history_jv_lag1.append(lag1_jv)
        history_ct_lag1.append(lag1_ct)
        history_jv_lag5.append(lag5_jv)
        history_ct_lag5.append(lag5_ct)
        history_jv_lag22.append(lag22_jv)
        history_ct_lag22.append(lag22_ct)
        history_btc_lag2.append(lag2_btc)
        history_btc_lag1.append(lag1_btc)

    return history_rv[-1]

def rolling_forecast_corrected(model_data, test_size=300, window_size=None):

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
        X_train_current = current_train_data[['Jv_lag1', 'Jv_lag5', 'Jv_lag22', 'C_t_lag1', 'C_t_lag5', 'C_t_lag22','btc_lag1','btc_lag2']].values
        y_train_current = current_train_data['RV'].values

        # 训练模型
        model = LinearRegression()
        model.fit(X_train_current, y_train_current)

        # === 1步预测 ===
        X_test_current = test_data.iloc[i][['Jv_lag1', 'Jv_lag5', 'Jv_lag22', 'C_t_lag1', 'C_t_lag5', 'C_t_lag22','btc_lag1','btc_lag2']].values.reshape(1, -1)
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
    df_predictions.to_csv('log+har-cj.csv', index=False)
    print("预测结果已保存至 'forecast_results.csv'")