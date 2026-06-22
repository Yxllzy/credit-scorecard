# ============================================================
# W3 补丁：用"手动分箱"重新计算三个逾期字段的 IV
# 原因：这三个字段大量为0，等频分箱(qcut)失效，导致IV假性为0
# ============================================================
import pandas as pd
import numpy as np

df = pd.read_csv('cs-clean.csv', index_col=0)
target = 'SeriousDlqin2yrs'

late_cols = [
    'NumberOfTime30-59DaysPastDueNotWorse',
    'NumberOfTimes90DaysLate',
    'NumberOfTime60-89DaysPastDueNotWorse'
]

def manual_woe_iv(data, feature, target):
    df_temp = data[[feature, target]].copy()
    # 手动分箱：0次 / 1次 / 2次 / 3次及以上
    df_temp['bin'] = pd.cut(
        df_temp[feature],
        bins=[-1, 0, 1, 2, np.inf],
        labels=['0次', '1次', '2次', '3次及以上']
    )
    grouped = df_temp.groupby('bin', observed=True)[target].agg(['count', 'sum'])
    grouped.columns = ['总人数', '坏人数']
    grouped['好人数'] = grouped['总人数'] - grouped['坏人数']

    total_bad = grouped['坏人数'].sum()
    total_good = grouped['好人数'].sum()
    grouped['坏人占比'] = (grouped['坏人数'] + 0.5) / total_bad
    grouped['好人占比'] = (grouped['好人数'] + 0.5) / total_good
    grouped['坏账率'] = (grouped['坏人数'] / grouped['总人数'] * 100).round(2)
    grouped['WOE'] = np.log(grouped['好人占比'] / grouped['坏人占比'])
    grouped['IV_每箱'] = (grouped['好人占比'] - grouped['坏人占比']) * grouped['WOE']
    iv = grouped['IV_每箱'].sum()
    return grouped, iv

print("========== 重新计算三个逾期字段（手动分箱）==========")
for col in late_cols:
    detail, iv = manual_woe_iv(df, col, target)
    print(f"\n【{col}】  IV = {iv:.4f}")
    print(detail[['总人数', '坏人数', '坏账率', 'WOE']])
