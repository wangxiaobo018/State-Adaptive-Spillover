
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

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
def multi_step_predict_improved(model_data_slice, model, target_step):
    """
    改进的多步预测函数 for HAR-CJ model

    Args:
        model_data_slice: 包含足够历史数据的DataFrame切片，包含 'RV', 'rex_m_lag1', 'rex_m_lag5',
                         'rex_m_lag22', 'rex_p_lag1', 'rex_p_lag5', 'rex_p_lag22', 'rex_moderate_lag1',
                         'rex_moderate_lag5', 'rex_moderate_lag22'
        model: 训练好的模型 (LinearRegression trained on all nine lagged features)
        target_step: 目标预测步数

    Returns:
        目标步数的预测值
    """
    # 创建历史序列的副本
    history_rv = model_data_slice['RV'].tolist()
    history_rex_m_lag1 = model_data_slice['rex_m_lag1'].tolist()
    history_rex_m_lag5 = model_data_slice['rex_m_lag5'].tolist()
    history_rex_m_lag22 = model_data_slice['rex_m_lag22'].tolist()
    history_rex_p_lag1 = model_data_slice['rex_p_lag1'].tolist()
    history_rex_p_lag5 = model_data_slice['rex_p_lag5'].tolist()
    history_rex_p_lag22 = model_data_slice['rex_p_lag22'].tolist()
    history_rex_moderate_lag1 = model_data_slice['rex_moderate_lag1'].tolist()
    history_rex_moderate_lag5 = model_data_slice['rex_moderate_lag5'].tolist()
    history_rex_moderate_lag22 = model_data_slice['rex_moderate_lag22'].tolist()

    for step in range(target_step):
        # 计算当前步骤所需的滞后特征
        current_len = len(history_rv)

        # Lag-1 features
        if current_len >= 1:
            lag1_rex_m = history_rex_m_lag1[-1]
            lag1_rex_p = history_rex_p_lag1[-1]
            lag1_rex_moderate = history_rex_moderate_lag1[-1]
        else:
            raise ValueError("历史数据不足")

        # Lag-5 features
        if current_len >= 5:
            lag5_rex_m = history_rex_m_lag5[-5]
            lag5_rex_p = history_rex_p_lag5[-5]
            lag5_rex_moderate = history_rex_moderate_lag5[-5]
        else:
            lag5_rex_m = history_rex_m_lag5[0]
            lag5_rex_p = history_rex_p_lag5[0]
            lag5_rex_moderate = history_rex_moderate_lag5[0]

        # Lag-22 features
        if current_len >= 22:
            lag22_rex_m = history_rex_m_lag22[-22]
            lag22_rex_p = history_rex_p_lag22[-22]
            lag22_rex_moderate = history_rex_moderate_lag22[-22]
        else:
            lag22_rex_m = history_rex_m_lag22[0]
            lag22_rex_p = history_rex_p_lag22[0]
            lag22_rex_moderate = history_rex_moderate_lag22[0]

        # 预测下一步，使用所有特征
        X_input = np.array([[lag1_rex_m, lag5_rex_m, lag22_rex_m,
                             lag1_rex_p, lag5_rex_p, lag22_rex_p,
                             lag1_rex_moderate, lag5_rex_moderate, lag22_rex_moderate]])
        pred = model.predict(X_input)[0]

        # 将预测值添加到 RV 历史序列
        history_rv.append(pred)

        # 更新滞后特征序列（占位）
        # Note: rex_m, rex_p, rex_moderate need to be updated based on HAR-CJ decomposition
        history_rex_m_lag1.append(lag1_rex_m)  # Placeholder: Update with decomposition logic
        history_rex_p_lag1.append(lag1_rex_p)  # Placeholder: Update with decomposition logic
        history_rex_moderate_lag1.append(lag1_rex_moderate)  # Placeholder: Update with decomposition logic
        history_rex_m_lag5.append(lag5_rex_m)
        history_rex_p_lag5.append(lag5_rex_p)
        history_rex_moderate_lag5.append(lag5_rex_moderate)
        history_rex_m_lag22.append(lag22_rex_m)
        history_rex_p_lag22.append(lag22_rex_p)
        history_rex_moderate_lag22.append(lag22_rex_moderate)

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
        X_train_current = current_train_data[['rex_m_lag1', 'rex_m_lag5', 'rex_m_lag22',
                     'rex_p_lag1', 'rex_p_lag5', 'rex_p_lag22',
                     'rex_moderate_lag1', 'rex_moderate_lag5', 'rex_moderate_lag22']].values
        y_train_current = current_train_data['RV'].values

        # 训练模型
        model = LinearRegression()
        model.fit(X_train_current, y_train_current)

        # === 1步预测 ===
        X_test_current = test_data.iloc[i][['rex_m_lag1', 'rex_m_lag5', 'rex_m_lag22',
                     'rex_p_lag1', 'rex_p_lag5', 'rex_p_lag22',
                     'rex_moderate_lag1', 'rex_moderate_lag5', 'rex_moderate_lag22']].values.reshape(1, -1)
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
    df_predictions.to_csv('har-re.csv', index=False)
    print("预测结果已保存至 'forecast_results.csv'")