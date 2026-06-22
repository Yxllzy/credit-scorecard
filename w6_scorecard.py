# W6：信用评分卡转换 + SHAP模型解释
# 包含：重新训练模型 → 转评分卡 → 解释单个客户
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 0~3. 重复W4的建模流程（数据→切分→WOE→训练）
df = pd.read_csv('cs-clean.csv', index_col=0)
target = 'SeriousDlqin2yrs'
features = [
    'RevolvingUtilizationOfUnsecuredLines', 'age', 'DebtRatio',
    'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans',
    'NumberOfTime30-59DaysPastDueNotWorse', 'NumberOfTimes90DaysLate',
    'NumberOfTime60-89DaysPastDueNotWorse',
]
X, y = df[features], df[target]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

manual_late_bins = [-1, 0, 1, 2, np.inf]
late_cols = ['NumberOfTime30-59DaysPastDueNotWorse', 'NumberOfTimes90DaysLate',
             'NumberOfTime60-89DaysPastDueNotWorse']

def fit_woe(x_train, y_train, manual_bins=None, q=5):
    df_t = pd.DataFrame({'x': x_train, 'y': y_train})
    if manual_bins is not None:
        df_t['bin'] = pd.cut(df_t['x'], bins=manual_bins); edges = manual_bins
    else:
        df_t['bin'], edges = pd.qcut(df_t['x'], q=q, duplicates='drop', retbins=True)
    g = df_t.groupby('bin', observed=True)['y'].agg(['count', 'sum'])
    g.columns = ['cnt', 'bad']; g['good'] = g['cnt'] - g['bad']
    tb, tg = g['bad'].sum(), g['good'].sum()
    g['woe'] = np.log(((g['good']+0.5)/tg) / ((g['bad']+0.5)/tb))
    return edges, g['woe'].to_dict()

def apply_woe(x, edges, woe_map):
    return pd.cut(x, bins=edges, include_lowest=True).map(woe_map).astype(float)

X_train_woe = pd.DataFrame(index=X_train.index)
X_test_woe = pd.DataFrame(index=X_test.index)
woe_rules = {}
for f in features:
    if f in late_cols:
        edges, wmap = fit_woe(X_train[f], y_train, manual_bins=manual_late_bins)
    else:
        edges, wmap = fit_woe(X_train[f], y_train, q=5)
    woe_rules[f] = (edges, wmap)
    X_train_woe[f] = apply_woe(X_train[f], edges, wmap)
    X_test_woe[f]  = apply_woe(X_test[f],  edges, wmap)
X_train_woe = X_train_woe.fillna(0); X_test_woe = X_test_woe.fillna(0)

model = LogisticRegression(class_weight='balanced', max_iter=1000)
model.fit(X_train_woe, y_train)
print("模型训练完成")

# 4. 评分卡转换
#    公式：Score = Base - PDO/ln(2) * (模型对数几率)
#    每个特征的每个箱 → 一个分数
# 评分卡两个参数
BASE_SCORE = 600
PDO = 50
ODDS = (1 - y_train.mean()) / y_train.mean()   # 用数据真实好坏比
B = PDO / np.log(2)
A = BASE_SCORE + B * np.log(ODDS)

coef = model.coef_[0]
intercept = model.intercept_[0]
total_base = A - B * intercept

print(f"\n========== 信用评分卡（最终版）==========")
print(f"配置: Base={BASE_SCORE}, PDO={PDO}, ODDS={ODDS:.1f}, 整体基础分={total_base:.1f}")

scorecard = []
for i, f in enumerate(features):
    edges, wmap = woe_rules[f]
    for bin_range, woe in wmap.items():
        points = -B * coef[i] * woe
        scorecard.append({'特征': f, '分箱': str(bin_range),
                          'WOE': round(woe, 3), '该箱得分': round(points, 1)})
scorecard_df = pd.DataFrame(scorecard)
print("\n评分卡明细(节选)：")
print(scorecard_df.head(15).to_string(index=False))
scorecard_df.to_csv('w6_评分卡.csv', index=False, encoding='utf-8-sig')

# 5. 算总信用分
def score_one(row_woe):
    s = total_base
    for i, f in enumerate(features):
        s += -B * coef[i] * row_woe[f]
    return s

test_scores = X_test_woe.apply(score_one, axis=1)
print(f"\n测试集信用分：最低{test_scores.min():.0f}, 最高{test_scores.max():.0f}, 平均{test_scores.mean():.0f}")

plt.figure(figsize=(8, 5))
plt.hist(test_scores[y_test==0], bins=50, alpha=0.6, label='好人', density=True)
plt.hist(test_scores[y_test==1], bins=50, alpha=0.6, label='坏人', density=True)
plt.xlabel('信用分'); plt.ylabel('密度'); plt.title('好人 vs 坏人 信用分分布')
plt.legend(); plt.tight_layout(); plt.savefig('w6_信用分分布.png', dpi=120)
plt.show()

# 6. SHAP 解释

try:
    import shap
    explainer = shap.LinearExplainer(model, X_train_woe)
    shap_values = explainer.shap_values(X_test_woe)

    # 整体特征重要性图
    shap.summary_plot(shap_values, X_test_woe, feature_names=features,
                      show=False)
    plt.tight_layout(); plt.savefig('w6_SHAP重要性.png', dpi=120, bbox_inches='tight')
    plt.show()
    print("\n SHAP 重要性图已生成 w6_SHAP重要性.png")
except ImportError:
    print("\n 未安装shap库，跳过SHAP部分。评分卡已完成。")
    print("   如需SHAP，请在终端运行: pip install shap")
