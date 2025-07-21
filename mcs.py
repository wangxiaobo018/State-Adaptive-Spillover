import pandas as pd
import numpy as np
from arch.bootstrap import MCS
from IPython.display import display

# --- 1. 设置与准备 ---
data_path = 'D:/pycharm/work/滞后溢出/预测300/'
model_names = [
    'har-rv', 'har-cj', 'har-re', 'har-rs',
    'log+har-rv', 'log+har-re', 'log+har-rs', 'log+har-cj'
]
horizons = [1, 5, 20]
loss_functions_to_run = ['MSE', 'MAE', 'RMSE', 'QLIKE']
final_results_list = []

# --- 2. 主循环：遍历所有步长 ---
for step in horizons:
    print(f"\n{'=' * 25} 正在处理 Step {step} {'=' * 25}")

    # --- 2.1 数据加载与对齐 ---
    all_preds, all_actuals = {}, {}
    min_len = float('inf')
    valid_models = []
    base_actuals = None

    for model_name in model_names:
        try:
            df = pd.read_csv(f'{data_path}{model_name}.csv')
            pred_col, actual_col = f'Prediction_{step}', f'Actual_{step}'
            if pred_col not in df.columns or actual_col not in df.columns:
                continue

            # 应用指数化处理，针对 exp_log_har-* 模型
            predictions = np.exp(df[pred_col].dropna().values) if 'exp_log_har' in model_name else df[pred_col].dropna().values
            # 只加载第一个模型的 actuals 作为基准，并应用指数化处理
            if base_actuals is None:
                base_actuals = np.exp(df[actual_col].dropna().values) if 'exp_log_har' in model_name else df[actual_col].dropna().values

            if len(predictions) == 0:
                continue

            # 检查异常值
            if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)) or \
               np.any(np.isnan(base_actuals)) or np.any(np.isinf(base_actuals)):
                print(f"警告: 模型 {model_name} 的 Step {step} 数据包含inf/nan，跳过此模型")
                continue

            # 限制值范围，避免溢出
            predictions = np.clip(predictions, 1e-10, 1e10)
            base_actuals = np.clip(base_actuals, 1e-10, 1e10)

            all_preds[model_name] = predictions
            min_len = min(min_len, len(predictions))
            valid_models.append(model_name)

        except Exception:
            continue

    if len(valid_models) < 2 or base_actuals is None:
        print(f"Step {step}: 有效模型或数据不足，无法继续。")
        continue

    # 对齐数据长度
    final_len = min(min_len, len(base_actuals))
    print(f"Step {step}: 对齐数据长度为 {final_len}，有效模型数量 {len(valid_models)}")

    base_actuals_aligned = base_actuals[:final_len]
    aligned_preds = {name: all_preds[name][:final_len] for name in valid_models}

    # --- 2.2 计算每种损失函数的排名和MCS ---
    step_model_results = {name: {'Horizon': step, 'Model': name} for name in valid_models}

    for metric in loss_functions_to_run:
        print(f"--- 计算 Step {step}, 损失函数: {metric} ---")

        losses_ts, summary_losses = {}, {}
        current_valid_models_for_metric = []
        for name in valid_models:
            pred = aligned_preds[name]
            actual = base_actuals_aligned

            if metric == 'MSE':
                loss = (pred - actual) ** 2
            elif metric == 'MAE':
                loss = np.abs(pred - actual)
            elif metric == 'RMSE':
                loss = (pred - actual) ** 2
            elif metric == 'QLIKE':
                loss = np.log(pred + 1e-10) + actual / (pred + 1e-10)  # Add small constant to avoid division issues

            if np.any(np.isnan(loss)) or np.any(np.isinf(loss)):
                print(f"警告: 模型 {name} 的 {metric} 损失包含inf/nan，跳过此模型。")
                continue

            losses_ts[name] = loss
            current_valid_models_for_metric.append(name)
            if metric == 'RMSE':
                summary_losses[name] = np.sqrt(np.mean(loss))
            else:
                summary_losses[name] = np.mean(loss)

        if not summary_losses:
            continue

        losses_df = pd.DataFrame(losses_ts)

        # 计算平均损失并排名
        ranks = pd.Series(summary_losses).rank(method='min').astype(int).to_dict()

        # 执行两次MCS检验
        alpha, block_size = 0.25, int(np.ceil(final_len ** (1 / 3)))

        # 1. 对应 T_M (method='max')
        mcs_tm = MCS(losses_df, size=alpha, block_size=block_size, method='max')
        mcs_tm.compute()
        included_tm = set(mcs_tm.included)

        # 2. 对应 T_R (method='R')
        mcs_tr = MCS(losses_df, size=alpha, block_size=block_size, method='R')
        mcs_tr.compute()
        included_tr = set(mcs_tr.included)

        # 存储结果
        for name in current_valid_models_for_metric:
            step_model_results[name][f'{metric}_rank_TM'] = ranks.get(name)
            step_model_results[name][f'{metric}_rank_TR'] = ranks.get(name)
            step_model_results[name][f'{metric}_in_mcs_TM'] = (name in included_tm)
            step_model_results[name][f'{metric}_in_mcs_TR'] = (name in included_tr)

    final_results_list.extend(step_model_results.values())

# --- 3. 构建并格式化最终的DataFrame ---
if not final_results_list:
    print("\n没有足够的数据生成最终报告。")
else:
    report_df = pd.DataFrame(final_results_list).fillna(False)
    report_df.set_index(['Horizon', 'Model'], inplace=True)

    # 创建多级列
    col_tuples = []
    for metric in loss_functions_to_run:
        col_tuples.append((f'MCS_{metric}', 'T_M'))
        col_tuples.append((f'MCS_{metric}', 'T_R'))

    multi_columns = pd.MultiIndex.from_tuples(col_tuples)
    display_rank_df = pd.DataFrame(index=report_df.index, columns=multi_columns)

    for metric in loss_functions_to_run:
        if f'{metric}_rank_TM' in report_df.columns and f'{metric}_rank_TR' in report_df.columns:
            display_rank_df[(f'MCS_{metric}', 'T_M')] = report_df[f'{metric}_rank_TM']
            display_rank_df[(f'MCS_{metric}', 'T_R')] = report_df[f'{metric}_rank_TR']

    if display_rank_df.empty:
        print("\n没有有效的排名数据可以显示。")
    else:
        display_rank_df = display_rank_df.astype(int, errors='ignore')

        def style_mcs(val, row_idx, col_idx, source_df):
            try:
                model_name = row_idx[1]
                loss_func = col_idx[0].replace('MCS_', '')
                stat_type = col_idx[1]
                mcs_col_name = f'{loss_func}_in_mcs_{stat_type}'
                if source_df.loc[row_idx, mcs_col_name]:
                    return 'font-weight: bold'
            except KeyError:
                pass
            return ''

        styled_df = display_rank_df.style.apply(
            lambda x: [style_mcs(val, x.name, (x.index[i][0], x.index[i][1]), report_df) for i, val in enumerate(x)],
            axis=1
        )

        print("\n\n--- 最终模型评估表 (已格式化) ---")
        display(styled_df)

    styled_df.to_html('my_mcs_table.html')
    print("\n结果已保存到 my_mcs_table.html 文件中。")