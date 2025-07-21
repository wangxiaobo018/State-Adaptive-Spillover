
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

# 修改为您的文件路径
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
all_RV.columns = ["DT","BTC", "DASH","ETH","LTC","XLM","XRP"]


all_RV['DT'] = pd.to_datetime(all_RV['DT'])
# # 对除开DT的数据进行对数转换
numeric_cols = [col for col in all_RV.columns if col != 'DT']
all_RV[numeric_cols] = np.log(all_RV[numeric_cols])

# 步骤1：创建 model_data，以 BTC RV 作为目标变量
model_data = all_RV[['DT', 'BTC', 'DASH', 'ETH', 'LTC', 'XLM', 'XRP']].copy()
model_data = model_data.rename(columns={'BTC': 'RV'})  # 重命名 BTC 为 RV


# 步骤2：计算 BTC RV 的分位数
rv_quantiles = model_data['RV'].quantile([0.05, 0.95])
print(f"τ=0.05 分位数: {rv_quantiles[0.05]:.6f}")
print(f"τ=0.95 分位数: {rv_quantiles[0.95]:.6f}")

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

print(model_data)
# 创建滞后变量
rv_lag1 = all_RV['BTC'].shift(1)
rv_lag5 =all_RV['BTC'].rolling(window=5).mean().shift(1)
rv_lag22 = all_RV['BTC'].rolling(window=22).mean().shift(1)
btc_lag1 = model_data['Dynamic_RV_lag1'].shift(1)

model1 = pd.DataFrame({
    'RV': all_RV['BTC'],
    'rv_lag1': rv_lag1,
    'rv_lag5': rv_lag5,
    'rv_lag22': rv_lag22,
    'BTC_lag1': btc_lag1
}).dropna()

model_data = model1  # 使用新的数据结构

window_size = 1800
test_size = 300



import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 更好的解决方案：维护完整的历史序列
def multi_step_predict_improved(model_data_slice, model, target_step):
    """
    改进的多步预测函数
    """
    # 创建历史序列的副本
    history_rv = model_data_slice['RV'].tolist()
    history_btc_lag1 = model_data_slice['BTC_lag1'].tolist()  # 初始化 BTC_lag1 历史序列

    for step in range(target_step):
        current_len = len(history_rv)

        if current_len >= 1:
            lag1_rv = history_rv[-1]
            lag1_btc = history_btc_lag1[-1]  # 使用历史序列中的最后一个 BTC_lag1
        else:
            raise ValueError("历史数据不足")

        if current_len >= 5:
            lag5_rv = history_rv[-5]
        else:
            lag5_rv = history_rv[0]

        if current_len >= 22:
            lag22_rv = history_rv[-22]
        else:
            lag22_rv = history_rv[0]

        # 预测下一步
        X_input = np.array([[lag1_rv, lag5_rv, lag22_rv, lag1_btc]])
        pred = model.predict(X_input)[0]
        # 更新历史序列
        history_rv.append(pred)
        history_btc_lag1.append(lag1_btc)

    return history_rv[-1]
# 修正后的主循环
def rolling_forecast_corrected(model_data, test_size=300):
    """
    修正后的滚动预测主循环
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
        current_train_data = model_data.iloc[:current_train_end]

        # 准备训练特征和目标，包含新变量BTC_lag1
        X_train_current = current_train_data[['rv_lag1', 'rv_lag5', 'rv_lag22', 'BTC_lag1']].values
        y_train_current = current_train_data['RV'].values

        # 训练模型
        model = LinearRegression()
        model.fit(X_train_current, y_train_current)

        # === 1步预测 ===
        X_test_current = test_data.iloc[i][['rv_lag1', 'rv_lag5', 'rv_lag22', 'BTC_lag1']].values.reshape(1, -4)
        pred_1 = model.predict(X_test_current)[0]
        predictions_lr1.append(pred_1)
        actuals_lr1.append(test_data.iloc[i]['RV'])

        # === 5步预测 ===
        if i + 4 < len(test_data):  # 确保有足够的实际值用于比较
            # 获取用于5步预测的历史数据
            history_data = model_data.iloc[:current_train_end]
            pred_5 = multi_step_predict_improved(history_data, model, 5)
            predictions_lr5.append(pred_5)
            actuals_lr5.append(test_data.iloc[i + 4]['RV'])

        # === 22步预测 ===
        if i + 21 < len(test_data):  # 确保有足够的实际值用于比较
            history_data = model_data.iloc[:current_train_end]
            pred_20 = multi_step_predict_improved(history_data, model, 22)
            predictions_lr20.append(pred_20)
            actuals_lr20.append(test_data.iloc[i + 21]['RV'])

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
    max_len = len(results['predictions_1'])  # 以1步预测的长度为基准
    df_predictions = pd.DataFrame({
        'Prediction_1': results['predictions_1'],
        'Actual_1': results['actuals_1'],
        'Prediction_5': results['predictions_5'] + [np.nan] * (max_len - len(results['predictions_5'])),
        'Actual_5': results['actuals_5'] + [np.nan] * (max_len - len(results['actuals_5'])),
        'Prediction_20': results['predictions_20'] + [np.nan] * (max_len - len(results['predictions_20'])),
        'Actual_20': results['actuals_20'] + [np.nan] * (max_len - len(results['actuals_20']))
    })
    df_predictions.to_csv('log+har-rv.csv', index=False)
    print("预测结果已保存至 'forecast_results.csv'")