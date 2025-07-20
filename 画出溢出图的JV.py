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


features = all_JV_log_simple.drop(columns=['DT'])
data = features.to_numpy()

columns = ["BTC", "DASH","ETH","LTC","XLM","XRP"]
data_df = pd.DataFrame(data, columns=columns)
data_df = (data_df)  # 对数变换数据


class TVP_QVAR_DY:
    """
    时变参数分位数向量自回归动态溢出模型
    Time-Varying Parameter Quantile VAR Dynamic Spillover Model
    """

    def __init__(self, data, var_names=None):
        """
        初始化模型
        """
        if isinstance(data, pd.DataFrame):
            self.data = data.values
            self.var_names = data.columns.tolist()
            self.dates = data.index
        else:
            self.data = data
            self.var_names = var_names if var_names else [f'Var{i + 1}' for i in range(data.shape[1])]
            self.dates = None

        self.n_vars = self.data.shape[1]
        self.n_obs = self.data.shape[0]

    def QVAR(self, p=1, tau=0.5):
        """
        分位数向量自回归模型估计
        """
        y = self.data
        k = self.n_vars
        coef_matrix = []
        residuals = []

        for i in range(k):
            # 构建滞后矩阵
            yx = self._embed(y, p + 1)
            y_dep = y[p:, i]
            x_indep = yx[:, k:]

            # 分位数回归
            try:
                qr_model = QuantReg(y_dep, x_indep)
                qr_result = qr_model.fit(q=tau, max_iter=10000, p_tol=1e-5)

                # 提取系数
                coef = qr_result.params
                coef_matrix.append(coef)

                # 计算残差
                res = qr_result.resid
                residuals.append(res)
            except Exception as e:
                print(f"Warning: 变量 {i} 的分位数回归失败: {e}")
                # 使用OLS作为备选
                from sklearn.linear_model import LinearRegression
                lr = LinearRegression(fit_intercept=False)
                lr.fit(x_indep, y_dep)
                coef = lr.coef_
                coef_matrix.append(coef)
                res = y_dep - lr.predict(x_indep)
                residuals.append(res)

        # 计算残差协方差矩阵
        residuals = np.column_stack(residuals)
        Q = np.dot(residuals.T, residuals) / len(residuals)
        B = np.array(coef_matrix)

        return {'B': B, 'Q': Q}

    def _embed(self, y, dimension):
        """
        创建嵌入矩阵（类似R的embed函数）
        """
        n = len(y)
        k = y.shape[1]
        m = n - dimension + 1

        result = np.zeros((m, dimension * k))
        for i in range(m):
            for j in range(dimension):
                result[i, j * k:(j + 1) * k] = y[i + dimension - j - 1]

        return result

    def GFEVD(self, Phi, Sigma, n_ahead=10, normalize=True, standardize=True):
        """
        广义预测误差方差分解（修正版）
        """
        # 从伴随矩阵中提取VAR系数
        k = Sigma.shape[0]  # 变量个数

        # 如果Phi是伴随矩阵形式，提取前k行
        if Phi.shape[0] > k:
            Phi_reduced = Phi[:k, :]
        else:
            Phi_reduced = Phi

        # 计算脉冲响应
        A = self._tvp_Phi(Phi_reduced, n_ahead - 1)
        gi = np.zeros_like(A)
        sigmas = np.sqrt(np.diag(Sigma))
        sigmas[sigmas == 0] = 1e-10  # 避免除零

        for j in range(A.shape[2]):
            gi[:, :, j] = np.dot(np.dot(A[:, :, j], Sigma),
                                 np.linalg.inv(np.diag(sigmas))).T

        # 标准化
        if standardize:
            girf = np.zeros_like(gi)
            diag_gi = np.diag(gi[:, :, 0]).copy()
            diag_gi[diag_gi == 0] = 1
            for i in range(gi.shape[2]):
                girf[:, :, i] = gi[:, :, i] / diag_gi[:, np.newaxis]
            gi = girf

        # 计算FEVD
        num = np.sum(gi ** 2, axis=2)
        den = np.sum(num, axis=1)
        den[den == 0] = 1
        fevd = (num.T / den).T

        if normalize:
            row_sums = np.sum(fevd, axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            fevd = fevd / row_sums

        return {'GFEVD': fevd, 'GIRF': gi}

    def _tvp_Phi(self, x, nstep=30):
        """
        计算时变VAR的脉冲响应函数（修正版）
        """
        K = x.shape[0]
        if x.shape[1] % K != 0:
            raise ValueError(f"系数矩阵维度不匹配: {x.shape}")

        p = x.shape[1] // K

        # 提取VAR系数矩阵
        A = np.zeros((K, K, p))
        for i in range(p):
            A[:, :, i] = x[:, i * K:(i + 1) * K]

        # 计算脉冲响应函数
        Phi = np.zeros((K, K, nstep + 1))
        Phi[:, :, 0] = np.eye(K)

        for i in range(1, nstep + 1):
            Phi[:, :, i] = np.zeros((K, K))
            for j in range(min(i, p)):
                Phi[:, :, i] += np.dot(Phi[:, :, i - j - 1], A[:, :, j])

        return Phi

    def DCA(self, CV, digit=2):
        """
        动态连通性分析（修正版）
        """
        k = CV.shape[0]
        CT = CV * 100  # 转换为百分比

        # 计算各项指标
        OWN = np.diag(np.diag(CT))
        TO = np.sum(CT - OWN, axis=0)
        FROM = np.sum(CT - OWN, axis=1)
        NET = TO - FROM
        TCI = np.mean(TO)
        NPSO = CT - CT.T
        NPDC = np.sum(NPSO > 0, axis=1)

        # 构建完整的结果表格（类似R代码的格式）
        # 首先是CT矩阵和FROM列
        table = np.column_stack([CT, FROM])

        # 添加TO行
        to_row = np.append(TO, np.sum(TO))

        # 添加Inc.Own行（包含自身影响的总影响）
        inc_own = np.append(np.sum(CT, axis=0), TCI)

        # 添加NET行
        net_row = np.append(NET, TCI)

        # 添加NPDC行
        npdc_row = np.append(NPDC, 0)

        # 组合所有行
        full_table = np.vstack([table, to_row, inc_own, net_row, npdc_row])

        return {
            'CT': CT,
            'TCI': TCI,
            'TCI_corrected': TCI * k / (k - 1),
            'TO': TO,
            'FROM': FROM,
            'NET': NET,
            'NPSO': NPSO,
            'NPDC': NPDC,
            'TABLE': full_table
        }

    def _create_RHS_NI(self, templag, r, nlag, t):
        """创建右侧变量矩阵（修正版）"""
        K = nlag * (r ** 2)
        x_t = np.zeros(((t - nlag) * r, K))

        for i in range(t - nlag):
            for eq in range(r):
                row_idx = i * r + eq
                col_start = 0

                for j in range(nlag):
                    xtemp = templag[i, j * r:(j + 1) * r]

                    for var in range(r):
                        col_idx = col_start + eq * r + var
                        x_t[row_idx, col_idx] = xtemp[var]

                    col_start += r * r

        Flag = np.vstack([np.zeros((nlag * r, K)), x_t])
        return Flag

    def KFS_parameters(self, Y, l, nlag, beta_0_mean, beta_0_var, Q_0):
        """
        卡尔曼滤波和平滑参数估计（修正版）
        """
        n = p = Y.shape[1]
        r = p
        m = nlag * (r ** 2)
        k = nlag * r
        t = Y.shape[0]

        # 初始化矩阵
        beta_pred = np.zeros((m, t))
        beta_update = np.zeros((m, t))
        Rb_t = np.zeros((m, m, t))
        Sb_t = np.zeros((m, m, t))
        beta_t = np.zeros((k, k, t))
        Q_t = np.zeros((r, r, t))

        # 衰减因子
        l_2, l_4 = l[1], l[3]

        # 构建滞后矩阵
        yy = Y[nlag:]
        templag = self._embed(Y, nlag + 1)[:, Y.shape[1]:]

        # 构建状态矩阵
        Flag = self._create_RHS_NI(templag, r, nlag, t)

        # 卡尔曼滤波
        for irep in range(t):
            if irep % 100 == 0:
                print(f"卡尔曼滤波进度: {irep}/{t}")

            # 更新Q矩阵
            if irep == 0:
                Q_t[:, :, irep] = Q_0
            elif irep > 0:
                if irep <= nlag:
                    Gf_t = 0.1 * np.outer(Y[irep], Y[irep])
                else:
                    idx = irep - nlag - 1
                    if idx < len(yy) and irep > 0:
                        B_prev = self._construct_B_matrix(beta_update[:, irep - 1], r, nlag)
                        y_pred = np.dot(templag[idx], B_prev[:r, :].T)
                        resid = yy[idx] - y_pred
                        Gf_t = np.outer(resid, resid)
                    else:
                        Gf_t = Q_t[:, :, irep - 1]

                Q_t[:, :, irep] = l_2 * Q_t[:, :, irep - 1] + (1 - l_2) * Gf_t[:r, :r]

            # 更新beta
            if irep <= nlag:
                beta_pred[:, irep] = beta_0_mean
                beta_update[:, irep] = beta_pred[:, irep]
                Rb_t[:, :, irep] = beta_0_var
            else:
                beta_pred[:, irep] = beta_update[:, irep - 1]
                Rb_t[:, :, irep] = (1 / l_4) * Sb_t[:, :, irep - 1]

            # 卡尔曼更新
            if irep >= nlag and (irep - 1) * r < Flag.shape[0]:
                try:
                    flag_slice = Flag[(irep - 1) * r:irep * r, :]
                    Rx = np.dot(Rb_t[:, :, irep], flag_slice.T)
                    KV_b = Q_t[:, :, irep] + np.dot(flag_slice, Rx)
                    KG = np.dot(Rx, np.linalg.pinv(KV_b))

                    if irep < t:
                        innovation = Y[irep] - np.dot(flag_slice, beta_pred[:, irep])
                        beta_update[:, irep] = beta_pred[:, irep] + np.dot(KG, innovation)
                        Sb_t[:, :, irep] = Rb_t[:, :, irep] - np.dot(KG, np.dot(flag_slice, Rb_t[:, :, irep]))
                except Exception as e:
                    print(f"Warning at time {irep}: {e}")
                    beta_update[:, irep] = beta_pred[:, irep]
                    Sb_t[:, :, irep] = Rb_t[:, :, irep]

            # 构建B矩阵
            B = self._construct_B_matrix(beta_update[:, irep], r, nlag)

            # 检查稳定性
            eigenvalues = np.linalg.eigvals(B)
            if np.max(np.abs(eigenvalues)) <= 1.1 or irep == 0:
                beta_t[:, :, irep] = B
            else:
                beta_t[:, :, irep] = beta_t[:, :, irep - 1] if irep > 0 else B
                beta_update[:, irep] = 0.99 * beta_update[:, irep - 1] if irep > 0 else beta_update[:, irep]

        return {'beta_t': beta_t, 'Q_t': Q_t}

    def _construct_B_matrix(self, beta_vec, r, nlag):
        """构建VAR的伴随矩阵"""
        k = nlag * r
        B = np.zeros((k, k))

        # 重塑beta向量
        beta_mat = beta_vec.reshape(r, -1)

        # 填充第一行块
        B[:r, :] = beta_mat

        # 添加单位矩阵部分（如果nlag > 1）
        if nlag > 1:
            B[r:, :r * (nlag - 1)] = np.eye(r * (nlag - 1))

        return B

    def run_analysis(self, nlag=1, nfore=10, tau=0.5,
                     l=[0.99, 0.99, 0.99, 0.96], window=None):
        """
        运行完整的TVP-QVAR-DY分析
        """
        results = {}

        # 1. 静态分析
        print("运行静态QVAR分析...")
        static_qvar = self.QVAR(p=nlag, tau=tau)
        static_gfevd = self.GFEVD(static_qvar['B'], static_qvar['Q'],
                                  n_ahead=nfore, normalize=True, standardize=True)
        static_dca = self.DCA(static_gfevd['GFEVD'])

        results['static'] = {
            'qvar': static_qvar,
            'gfevd': static_gfevd,
            'dca': static_dca
        }

        # 打印静态溢出矩阵
        print("\n静态溢出矩阵:")
        print(f"总连通性指数 (TCI): {static_dca['TCI']:.2f}%")

        # 2. 时变分析
        if window is None:
            print("\n运行时变参数估计...")
            # 初始化参数
            beta_0_mean = static_qvar['B'].flatten()
            beta_0_var = 0.05 * np.eye(len(beta_0_mean))
            Q_0 = static_qvar['Q']

            # 运行卡尔曼滤波
            kfs_results = self.KFS_parameters(self.data, l, nlag,
                                              beta_0_mean, beta_0_var, Q_0)

            # 计算动态溢出
            print("\n计算动态溢出指数...")
            t = self.n_obs
            total = np.zeros(t)
            gfevd = np.zeros((self.n_vars, self.n_vars, t))
            net = np.zeros((t, self.n_vars))
            to = np.zeros((t, self.n_vars))
            from_others = np.zeros((t, self.n_vars))
            npso = np.zeros((self.n_vars, self.n_vars, t))

            for i in range(t):
                if i % 100 == 0:
                    print(f"动态溢出计算进度: {100 * i / t:.2f}%")

                try:
                    # 计算GFEVD
                    gfevd_i = self.GFEVD(Phi=kfs_results['beta_t'][:, :, i],
                                         Sigma=kfs_results['Q_t'][:, :, i],
                                         n_ahead=nfore, standardize=True, normalize=True)
                    gfevd[:, :, i] = gfevd_i['GFEVD']

                    # 计算DCA
                    dca_i = self.DCA(gfevd[:, :, i])
                    to[i, :] = dca_i['TO']
                    from_others[i, :] = dca_i['FROM']
                    net[i, :] = dca_i['NET']
                    npso[:, :, i] = dca_i['NPSO']
                    total[i] = dca_i['TCI']
                except Exception as e:
                    if i % 100 == 0:  # 减少警告输出
                        print(f"Warning at time {i}: {e}")
                    # 使用前一期的值
                    if i > 0:
                        to[i, :] = to[i - 1, :]
                        from_others[i, :] = from_others[i - 1, :]
                        net[i, :] = net[i - 1, :]
                        npso[:, :, i] = npso[:, :, i - 1]
                        total[i] = total[i - 1]

            results['dynamic'] = {
                'total': total,
                'to': to,
                'from': from_others,
                'net': net,
                'npso': npso,
                'gfevd': gfevd
            }

        return results

    # 在你的 TVP_QVAR_DY 类中，用这个新版本替换原来的 plot_results 函数

    def plot_results(self, results, tau, dates=None):
        """
        绘制分析结果 (修改版)
        - 为每个变量生成独立的净溢出图
        - 在标题中显示分位数 tau
        - 去掉横轴标签
        - 将图表保存为PDF文件
        """
        if dates is None:
            dates = self.dates if self.dates is not None else np.arange(self.n_obs)

        # --- 1. 绘制总溢出指数图 ---
        if 'dynamic' in results and 'total' in results['dynamic']:
            plt.figure(figsize=(12, 6))
            # 提取动态结果
            dynamic_results = results['dynamic']
            total_spillover = dynamic_results['total']

            # 确保 dates 和 total_spillover 的长度匹配
            if len(dates) > len(total_spillover):
                dates = dates[len(dates) - len(total_spillover):]

            plt.plot(dates, total_spillover, 'k-', linewidth=1.5)
            plt.fill_between(dates, 0, total_spillover, color='red', alpha=0.4)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.ylabel('TSI(%)', fontsize=12)
            plt.xlabel('')  # 去掉横轴标签
            plt.tight_layout()

            # 保存为PDF
            pdf_filename_tci = f'total_spillover_tau_{str(tau).replace(".", "")}.pdf'
            plt.savefig(pdf_filename_tci, format='pdf', bbox_inches='tight')
            print(f"总溢出图已保存为: {pdf_filename_tci}")
            plt.show()

        # --- 2. 为每个变量绘制独立的净溢出图 ---
        if 'dynamic' in results and 'net' in results['dynamic']:
            net_spillover = results['dynamic']['net']

            # 确保 dates 和 net_spillover 的长度匹配
            if len(dates) > net_spillover.shape[0]:
                dates = dates[len(dates) - net_spillover.shape[0]:]

            for i in range(self.n_vars):
                var_name = self.var_names[i]

                plt.figure(figsize=(12, 6))

                net_series = net_spillover[:, i]

                plt.plot(dates, net_series, 'k-', linewidth=1.2)

                # 填充正负区域
                plt.fill_between(dates, net_series, 0, where=(net_series >= 0),
                                 color='green', alpha=0.4, interpolate=True, label='NT')
                plt.fill_between(dates, net_series, 0, where=(net_series < 0),
                                 color='red', alpha=0.4, interpolate=True, label='NR')

                plt.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
                plt.grid(True, linestyle='--', alpha=0.6)

                plt.ylabel('NET(%)', fontsize=12)
                plt.xlabel('')  # 去掉横轴标签
                plt.legend()
                plt.tight_layout()

                # 保存为PDF
                pdf_filename_net = f'net_spillover_{var_name}_tau_{str(tau).replace(".", "")}.pdf'
                plt.savefig(pdf_filename_net, format='pdf', bbox_inches='tight')
                print(f"{var_name} 的净溢出图已保存为: {pdf_filename_net}")
                plt.show()


# ===================================================================
# 核心分析函数 (已修改绘图部分)
# ===================================================================
def analyze_spillover_to_btc(model, quantiles, nlag=1, nfore=10, save_results=True):
    """
    分析其他加密货币对BTC在不同分位数下的溢出贡献
    """
    TARGET_VAR = 'BTC'

    if TARGET_VAR not in model.var_names:
        raise ValueError(f"目标变量 '{TARGET_VAR}' 不在数据中。可用变量: {model.var_names}")

    source_vars = [var for var in model.var_names if var != TARGET_VAR]
    spillover_to_btc = {var: [] for var in source_vars}

    btc_idx = model.var_names.index(TARGET_VAR)

    print(f"--- 开始分析其他加密货币对 {TARGET_VAR} 的溢出贡献 ---")

    for tau in tqdm(quantiles, desc="正在处理各分位数"):
        static_qvar = model.QVAR(p=nlag, tau=tau)

        if np.any(np.linalg.eigvals(static_qvar['Q']) <= 1e-9):
            static_qvar['Q'] += np.eye(model.n_vars) * 1e-9

        static_gfevd = model.GFEVD(static_qvar['B'], static_qvar['Q'],
                                   n_ahead=nfore, normalize=True)
        gfevd_matrix = static_gfevd['GFEVD']

        for var in source_vars:
            j = model.var_names.index(var)
            spillover_to_btc[var].append(gfevd_matrix[btc_idx, j] * 100)

    # ==================== 绘图代码修改区域 ====================
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(14, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(source_vars)))
    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'X']

    for i, (var, spillovers) in enumerate(spillover_to_btc.items()):
        plt.plot(quantiles, spillovers,
                 color=colors[i],
                 marker=markers[i % len(markers)],
                 markersize=8,
                 linewidth=2.5,
                 label=var)

    plt.xlabel('Quantile (τ)', fontsize=14)
    plt.ylabel(f'Spillover Contribution to {TARGET_VAR} (%)', fontsize=14)

    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xticks(quantiles, rotation=45)

    # 【修改1】将图例位置从 'best' 改为 'upper right'
    # 并且只调用一次 legend()
    plt.legend(fontsize=12, loc='upper right',frameon=True)

    # 【修改2】在 axvspan 中移除 label 参数，使其不出现在图例中
    ax = plt.gca()
    ax.axvspan(min(quantiles), 0.25, alpha=0.1, color='red')  # 移除 label
    ax.axvspan(0.75, max(quantiles), alpha=0.1, color='green')  # 移除 label

    plt.tight_layout()

    if save_results:
        filename = f'spillover_to_{TARGET_VAR}_by_quantile.pdf'
        plt.savefig(filename, format='pdf', dpi=300, bbox_inches='tight')
        print(f"\n溢出贡献曲线图已保存到: {filename}")
    plt.show()
    # ==================== 绘图代码修改结束 ====================

    summary_df = pd.DataFrame(spillover_to_btc, index=quantiles)
    summary_df.index.name = 'Quantile'

    if save_results:
        filename = f'spillover_to_{TARGET_VAR}_summary.csv'
        summary_df.to_csv(filename)
        print(f"溢出贡献原始数据已保存到: {filename}")

    return spillover_to_btc, summary_df


# ... (您所有的类和函数定义保持不变) ...

# ===================================================================
# 全新的 Main 函数，用于计算和绘制【每个时间点】的净溢出
# ===================================================================
def main_dynamic_analysis():
    """
    主执行函数，聚焦于计算和绘制【每个时间点】的净溢出。
    """

    model = TVP_QVAR_DY(data_df)

    # --- 2. 运行完整的时变分析 ---
    # 我们需要选择一个分位数 τ 来进行时变分析，通常选择中位数 τ=0.5
    # l 是衰减因子，通常使用默认值
    print("\n--- 开始运行完整的TVP-QVAR时变分析 (这可能需要一些时间) ---")
    analysis_results = model.run_analysis(
        nlag=1,
        nfore=10,
        tau=0.95,  # 在中位数上进行时变分析
        window=None  # 确保 window=None 来触发卡尔曼滤波
    )

    # --- 3. 提取并绘制结果 ---
    # analysis_results['dynamic'] 中包含了所有随时间变化的结果
    if 'dynamic' in analysis_results:
        print("\n--- 分析完成，正在绘制结果 ---")

        # 调用您类中已有的绘图函数
        # 传递 tau 参数以便在文件名中体现
        model.plot_results(analysis_results, tau=0.95)
    else:
        print("错误：动态分析未能生成结果。")


# --- 脚本执行入口 ---
if __name__ == "__main__":

    # 您可以保留原来的静态分位数分析
    print("\n========== 开始静态【分位数依赖】溢出分析 ==========")
    # main() # 这是您原来的函数

    # 运行新的动态时变分析
    print("\n\n========== 开始动态【时间依赖】溢出分析 ==========")
    main_dynamic_analysis()