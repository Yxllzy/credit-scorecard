# W7：业务价值转化 
# 对比"不用模型"vs"用模型"的净收益，算出模型创造的价值
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 重建模型（同W4）
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

def fit_woe(xt, yt, manual_bins=None, q=5):
    d = pd.DataFrame({'x': xt, 'y': yt})
    if manual_bins is not None:
        d['bin'] = pd.cut(d['x'], bins=manual_bins); edges = manual_bins
    else:
        d['bin'], edges = pd.qcut(d['x'], q=q, duplicates='drop', retbins=True)
    g = d.groupby('bin', observed=True)['y'].agg(['count', 'sum'])
    g.columns = ['cnt', 'bad']; g['good'] = g['cnt'] - g['bad']
    tb, tg = g['bad'].sum(), g['good'].sum()
    g['woe'] = np.log(((g['good']+0.5)/tg) / ((g['bad']+0.5)/tb))
    return edges, g['woe'].to_dict()

def apply_woe(x, edges, wmap):
    return pd.cut(x, bins=edges, include_lowest=True).map(wmap).astype(float)

X_train_woe = pd.DataFrame(index=X_train.index)
X_test_woe = pd.DataFrame(index=X_test.index)
for f in features:
    if f in late_cols:
        edges, wmap = fit_woe(X_train[f], y_train, manual_bins=manual_late_bins)
    else:
        edges, wmap = fit_woe(X_train[f], y_train, q=5)
    X_train_woe[f] = apply_woe(X_train[f], edges, wmap)
    X_test_woe[f]  = apply_woe(X_test[f],  edges, wmap)
X_train_woe = X_train_woe.fillna(0); X_test_woe = X_test_woe.fillna(0)

model = LogisticRegression(class_weight='balanced', max_iter=1000)
model.fit(X_train_woe, y_train)
test_prob = model.predict_proba(X_test_woe)[:, 1]   # 预测为坏人的概率

# 业务假设（W7）
LOAN_AMOUNT = 10000      # 每笔贷款金额(元)
PROFIT_RATE = 0.20       # 好人收益率：借给好人赚20% = 2000元
LOSS_RATE   = 1.00       # 坏人损失率：借给坏人亏100% = 10000元

profit_per_good = LOAN_AMOUNT * PROFIT_RATE   # 每个好人赚2000
loss_per_bad    = LOAN_AMOUNT * LOSS_RATE     # 每个坏人亏10000

# 测试集真实情况
n_good = (y_test == 0).sum()
n_bad  = (y_test == 1).sum()
print(f"测试集：好人{n_good}个, 坏人{n_bad}个")

# 策略A：不用模型，全部放贷
profit_A = n_good * profit_per_good - n_bad * loss_per_bad
print(f"\n【策略A·全部放贷】净收益 = {profit_A:,.0f} 元")
print(f"  (好人赚 {n_good*profit_per_good:,.0f} - 坏人亏 {n_bad*loss_per_bad:,.0f})")

# 策略B：用模型，找出"最赚钱的分数线"；遍历各种概率阈值，算每个阈值下的净收益，挑最高的
thresholds = np.arange(0.05, 0.95, 0.01)
profits = []
for t in thresholds:
    approve = test_prob < t          # 预测坏人概率 < 阈值 → 放贷
    good_approved = ((y_test == 0) & approve).sum()  # 通过的好人
    bad_approved  = ((y_test == 1) & approve).sum()  # 漏过的坏人
    profit = good_approved * profit_per_good - bad_approved * loss_per_bad
    profits.append(profit)

profits = np.array(profits)
best_idx = profits.argmax()
best_threshold = thresholds[best_idx]
profit_B = profits[best_idx]

# 最优阈值下的详细情况
approve = test_prob < best_threshold
good_approved = ((y_test == 0) & approve).sum()
bad_approved  = ((y_test == 1) & approve).sum()
good_rejected = ((y_test == 0) & ~approve).sum()  # 误伤的好人
bad_rejected  = ((y_test == 1) & ~approve).sum()  # 拦截的坏人

print(f"\n【策略B·模型放贷】最优阈值 = {best_threshold:.2f}")
print(f"  通过放贷：好人{good_approved}个、坏人{bad_approved}个")
print(f"  拒绝放贷：拦截坏人{bad_rejected}个、误伤好人{good_rejected}个")
print(f"  净收益 = {profit_B:,.0f} 元")

# 模型价值 = B - A
value = profit_B - profit_A
print(f"\n{'='*50}")
print(f"模型创造的价值 = {value:,.0f} 元（测试集3万人）")

# 换算成更大规模（如100万客户/年）
scale = 1_000_000 / len(y_test)
print(f"按100万客户/年估算 ≈ {value*scale/10000:,.0f} 万元/年")

# 坏账率改善
bad_rate_A = n_bad / (n_good + n_bad)  # 不筛选的坏账率
bad_rate_B = bad_approved / (good_approved + bad_approved)
 # 筛选后坏账率
print(f"\n坏账率：不用模型 {bad_rate_A:.2%} → 用模型 {bad_rate_B:.2%}")
print(f"坏账率下降 {(bad_rate_A - bad_rate_B)/bad_rate_A:.1%}")
print(f"{'='*50}")

# 画图：不同阈值下的净收益曲线
plt.figure(figsize=(9, 5))
plt.plot(thresholds, profits/10000, linewidth=2)
plt.axvline(x=best_threshold, color='red', linestyle='--',
            label=f'最优阈值={best_threshold:.2f}')
plt.axhline(y=profit_A/10000, color='gray', linestyle=':',
            label=f'不用模型={profit_A/10000:.0f}万')
plt.xlabel('拒贷阈值(预测坏人概率超过此值就拒贷)')
plt.ylabel('净收益(万元)')
plt.title('不同决策阈值下的净收益')
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig('w7_收益曲线.png', dpi=120)
plt.show()
print("\n W7完成！收益曲线已保存 w7_收益曲线.png")
