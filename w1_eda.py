# ============================================================
# W1：探索性数据分析（EDA）—— 先认识数据
# 目标：搞清楚数据有多少、长啥样、有没有缺失、好人坏人比例
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 让图能正常显示中文（如果报错可先忽略）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ---------- 1. 读数据 ----------
# 第一列是无意义的行号(index)，用 index_col=0 把它忽略掉
df = pd.read_csv('cs-training.csv', index_col=0)

print("数据有多少行、多少列：", df.shape)   # (行数, 列数)
print("\n前 5 行长这样：")
print(df.head())

# ---------- 2. 看每列的基本信息 ----------
print("\n每列的类型 + 有没有缺失值：")
print(df.info())

print("\n每列的统计概况（均值、最大最小值等）：")
print(df.describe())

# ---------- 3. 看缺失值情况 ----------
# 缺失值是数据里的"空白"，后面要处理它，先看看哪些列缺
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
print("\n各列缺失值数量 / 占比：")
print(pd.DataFrame({'缺失数量': missing, '缺失占比%': missing_pct}))

# ---------- 4. 看"好人/坏人"比例（最关键！）----------
# SeriousDlqin2yrs = 1 是坏人(会赖账), = 0 是好人
target = df['SeriousDlqin2yrs']
print("\n好人(0)/坏人(1) 数量：")
print(target.value_counts())
print("\n坏人占比：{:.2%}".format(target.mean()))

# 画个图直观看
target.value_counts().plot(kind='bar')
plt.title('好人(0) vs 坏人(1) 数量对比')
plt.xlabel('0=好人, 1=坏人')
plt.ylabel('人数')
plt.tight_layout()
plt.savefig('w1_好坏比例.png')   # 存成图片
plt.show()

# ---------- 5. 看几个关键字段的分布 ----------
# 看年龄分布
df['age'].plot(kind='hist', bins=50)
plt.title('年龄分布')
plt.xlabel('年龄')
plt.tight_layout()
plt.savefig('w1_年龄分布.png')
plt.show()

print("\n========== W1 完成！你现在应该知道： ==========")
print("1. 这份数据有多少条、多少个特征")
print("2. 哪些列有缺失值（后面要处理）")
print("3. 坏人占比大概多少（这叫样本不平衡）")
