# ============================================================
# W2：数据清洗 —— 把脏数据处理干净
# 处理：缺失值(月收入、抚养人数) + 异常值(年龄=0、逾期96/98)
# 最终输出干净文件：cs-clean.csv
# ============================================================
import pandas as pd
import numpy as np

# ---------- 0. 读数据 ----------
df = pd.read_csv('cs-training.csv', index_col=0)
print("清洗前数据形状：", df.shape)

# ============================================================
# 1. 处理三个逾期字段里的异常值 96 / 98
#    思路：96/98 是异常采集码，不是真实次数。
#    我们把它替换为"该列正常值(<90)里的最大值"，
#    含义=把它当成"逾期次数非常多的高风险人群"，保留其风险信号。
# ============================================================
late_cols = [
    'NumberOfTime30-59DaysPastDueNotWorse',
    'NumberOfTimes90DaysLate',
    'NumberOfTime60-89DaysPastDueNotWorse'
]
for col in late_cols:
    # 找出这一列里 <90 的正常值的最大值
    normal_max = df.loc[df[col] < 90, col].max()
    # 把 >=90 的（也就是96/98）替换成这个正常最大值
    n_bad = (df[col] >= 90).sum()
    df.loc[df[col] >= 90, col] = normal_max
    print(f"【{col}】替换了 {n_bad} 个异常值(96/98) → {normal_max}")

# ============================================================
# 2. 处理年龄 = 0 的异常值（用年龄中位数替换）
# ============================================================
age_median = df['age'].median()
n_age0 = (df['age'] == 0).sum()
df.loc[df['age'] == 0, 'age'] = age_median
print(f"\n年龄=0 替换了 {n_age0} 个 → 中位数 {age_median}")

# ============================================================
# 3. 处理月收入缺失 → 用中位数填补
# ============================================================
income_median = df['MonthlyIncome'].median()
n_income_na = df['MonthlyIncome'].isnull().sum()
df['MonthlyIncome'] = df['MonthlyIncome'].fillna(income_median)
print(f"月收入缺失填补了 {n_income_na} 个 → 中位数 {income_median}")

# ============================================================
# 4. 处理抚养人数缺失 → 用 0 填补（最常见值）
# ============================================================
n_dep_na = df['NumberOfDependents'].isnull().sum()
df['NumberOfDependents'] = df['NumberOfDependents'].fillna(0)
print(f"抚养人数缺失填补了 {n_dep_na} 个 → 0")

# ============================================================
# 5. 最终检查：确认没有缺失值了
# ============================================================
print("\n========== 清洗后检查 ==========")
print("还有没有缺失值（应该全是0）：")
print(df.isnull().sum())
print("\n年龄最小值（应该>0了）：", df['age'].min())
print("逾期字段最大值（应该没有96/98了）：")
for col in late_cols:
    print(f"  {col}: 最大值 = {df[col].max()}")

# ============================================================
# 6. 存成干净文件，供下周(W3)使用
# ============================================================
df.to_csv('cs-clean.csv')
print("\n✅ 干净数据已保存为 cs-clean.csv，形状：", df.shape)
