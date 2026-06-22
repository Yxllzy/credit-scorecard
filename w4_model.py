# W4：训练逻辑回归模型 + 严防数据泄露；流程：切分 → (仅用训练集)算WOE → 套用到测试集 → 建模 → 评估
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

# 读清洗好的数据
df = pd.read_csv('cs-clean.csv', index_col=0)
target = 'SeriousDlqin2yrs'

# 保留 W3 选出的 8 个有用特征（舍弃2个弱变量）
features = [
    'RevolvingUtilizationOfUnsecuredLines',
    'age',
    'DebtRatio',
    'MonthlyIncome',
    'NumberOfOpenCreditLinesAndLoans',
    'NumberOfTime30-59DaysPastDueNotWorse',
    'NumberOfTimes90DaysLate',
    'NumberOfTime60-89DaysPastDueNotWorse',
]

X = df[features]
y = df[target]

# 1. 切分训练集 / 测试集
#    test_size=0.2 → 20%留作测试
#    stratify=y    → 保证训练/测试里坏人比例一致
#    random_state  → 固定随机种子，保证每次切分结果一样
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print("训练集大小：", X_train.shape, " 坏人占比：{:.2%}".format(y_train.mean()))
print("测试集大小：", X_test.shape, " 坏人占比：{:.2%}".format(y_test.mean()))

# 2. 定义"学WOE规则"的函数（只在训练集上学）；返回：每个箱的边界 + 每个箱的WOE值
def fit_woe(x_train, y_train, feature, manual_bins=None, q=5):
    """在训练集上学习分箱边界和WOE值"""
    df_t = pd.DataFrame({'x': x_train, 'y': y_train})

    if manual_bins is not None:
        # 手动分箱（给逾期字段用）
        df_t['bin'] = pd.cut(df_t['x'], bins=manual_bins)
        edges = manual_bins
    else:
        # 等频分箱（给连续字段用），并记录边界
        df_t['bin'], edges = pd.qcut(df_t['x'], q=q, duplicates='drop', retbins=True)

    # 算每箱WOE
    g = df_t.groupby('bin', observed=True)['y'].agg(['count', 'sum'])
    g.columns = ['cnt', 'bad']
    g['good'] = g['cnt'] - g['bad']
    total_bad, total_good = g['bad'].sum(), g['good'].sum()
    g['woe'] = np.log(((g['good'] + 0.5) / total_good) / ((g['bad'] + 0.5) / total_bad))

    woe_map = g['woe'].to_dict()        
    return edges, woe_map, g

def apply_woe(x, edges, woe_map):
    """把学到的规则套用到数据上（训练/测试都用同一套规则）"""
    bins = pd.cut(x, bins=edges, include_lowest=True)
    return bins.map(woe_map).astype(float)

# 3. 对每个特征：在训练集学WOE → 套用到训练集和测试集
# 三个逾期字段用手动分箱(0/1/2/3+)，其余用等频5箱
manual_late_bins = [-1, 0, 1, 2, np.inf]
late_cols = [
    'NumberOfTime30-59DaysPastDueNotWorse',
    'NumberOfTimes90DaysLate',
    'NumberOfTime60-89DaysPastDueNotWorse',
]

X_train_woe = pd.DataFrame(index=X_train.index)
X_test_woe = pd.DataFrame(index=X_test.index)

for f in features:
    if f in late_cols:
        edges, woe_map, _ = fit_woe(X_train[f], y_train, f, manual_bins=manual_late_bins)
    else:
        edges, woe_map, _ = fit_woe(X_train[f], y_train, f, q=5)

    # 训练集和测试集都用“从训练集学到的”edges和woe_map
    X_train_woe[f] = apply_woe(X_train[f], edges, woe_map)
    X_test_woe[f]  = apply_woe(X_test[f],  edges, woe_map)

# 套用后可能因边界问题产生极少数空值，用0填补(WOE=0表示中性)
X_train_woe = X_train_woe.fillna(0)
X_test_woe = X_test_woe.fillna(0)

print("\nWOE编码完成。训练集前3行：")
print(X_train_woe.head(3))

# 4. 训练逻辑回归模型
#    class_weight='balanced' ：自动给坏人(少数)更高权重，应对样本不平衡
model = LogisticRegression(class_weight='balanced', max_iter=1000)
model.fit(X_train_woe, y_train)
print("\n模型训练完成！")

# 5. 在训练集和测试集上分别评估 AUC（看有没有过拟合）
train_pred = model.predict_proba(X_train_woe)[:, 1]   # 预测为坏人的概率
test_pred  = model.predict_proba(X_test_woe)[:, 1]

train_auc = roc_auc_score(y_train, train_pred)
test_auc  = roc_auc_score(y_test, test_pred)

print(f"\n训练集 AUC：{train_auc:.4f}")
print(f"测试集 AUC：{test_auc:.4f}")

# 计算 KS 值
def calc_ks(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return max(tpr - fpr)

print(f"测试集 KS ：{calc_ks(y_test, test_pred):.4f}")

# 6. 保存预测结果
result = pd.DataFrame({'y_true': y_test, 'y_prob': test_pred})
result.to_csv('w4_test_predictions.csv', index=False)
print("\n测试集预测结果已保存为 w4_test_predictions.csv")
