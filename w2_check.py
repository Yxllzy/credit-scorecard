# ============================================================
# W2 第一步：侦查脏数据（异常值）
# ============================================================
import pandas as pd

df = pd.read_csv('cs-training.csv', index_col=0)

# ---------- 1. 看年龄有没有不合理的值 ----------
print("年龄最小值：", df['age'].min())
print("年龄=0 的有几个：", (df['age'] == 0).sum())

# ---------- 2. 看三个"逾期次数"字段的取值分布 ----------
# 正常人逾期几次？看看是不是藏着 96/98 这种怪值
late_cols = [
    'NumberOfTime30-59DaysPastDueNotWorse',
    'NumberOfTimes90DaysLate',
    'NumberOfTime60-89DaysPastDueNotWorse'
]
for col in late_cols:
    print(f"\n【{col}】的取值分布：")
    print(df[col].value_counts().sort_index())
