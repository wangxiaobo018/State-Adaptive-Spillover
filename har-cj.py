
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

# 分割训练集和测试集
train_end = len(model_data) - test_size
X_train = model_data[['Jv_lag1', 'Jv_lag5', 'Jv_lag22', 'C_t_lag1', 'C_t_lag5', 'C_t_lag22']].iloc[:train_end]
X_test = model_data[['Jv_lag1', 'Jv_lag5', 'Jv_lag22', 'C_t_lag1', 'C_t_lag5', 'C_t_lag22']].iloc[train_end:]
y_train = model_data['RV'].iloc[:train_end]
y_test = model_data['RV'].iloc[train_end:]


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

def multi_step_predict_improved(model_data_slice, model, target_step):
    """
    改进的多步预测函数 for HAR-CJ model

    Args:
        model_data_slice: 包含足够历史数据的DataFrame切片，包含 'RV', 'Jv_lag1', 'Jv_lag5', 'Jv_lag22',
                         'C_t_lag1', 'C_t_lag5', 'C_t_lag22'
        model: 训练好的模型 (LinearRegression trained on all six lagged features)
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

    for step in range(target_step):
        # 计算当前步骤所需的滞后特征
        current_len = len(history_rv)

        # Jv_lag1
        if current_len >= 1:
            lag1_jv = history_jv_lag1[-1]
            lag1_ct = history_ct_lag1[-1]
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
        X_input = np.array([[lag1_jv, lag5_jv, lag22_jv, lag1_ct, lag5_ct, lag22_ct]])
        pred = model.predict(X_input)[0]

        # 将预测值添加到 RV 历史序列
        history_rv.append(pred)

        history_jv_lag1.append(lag1_jv)  # 占位，实际需根据 HAR-CJ 定义更新
        history_ct_lag1.append(lag1_ct)  # 占位，实际需根据 HAR-CJ 定义更新
        history_jv_lag5.append(lag5_jv)
        history_ct_lag5.append(lag5_ct)
        history_jv_lag22.append(lag22_jv)
        history_ct_lag22.append(lag22_ct)

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
        X_train_current = current_train_data[['Jv_lag1', 'Jv_lag5', 'Jv_lag22', 'C_t_lag1', 'C_t_lag5', 'C_t_lag22']].values
        y_train_current = current_train_data['RV'].values

        # 训练模型
        model = LinearRegression()
        model.fit(X_train_current, y_train_current)

        # === 1步预测 ===
        X_test_current = test_data.iloc[i][['Jv_lag1', 'Jv_lag5', 'Jv_lag22', 'C_t_lag1', 'C_t_lag5', 'C_t_lag22']].values.reshape(1, -1)
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
    df_predictions.to_csv('har-cj.csv', index=False)
    print("预测结果已保存至 'forecast_results.csv'")