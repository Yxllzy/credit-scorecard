# ============================================================
# W5：模型评估报告 —— ROC曲线 + KS曲线 + PSI稳定性
# 用 W4 保存的预测结果(w4_test_predictions.csv)
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ---------- 读预测结果 ----------
res = pd.read_csv('w4_test_predictions.csv')
y_true = res['y_true']
y_prob = res['y_prob']

# ============================================================
# 1. ROC 曲线 + AUC
# ============================================================
fpr, tpr, thresholds = roc_curve(y_true, y_prob)
auc = roc_auc_score(y_true, y_prob)

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, label=f'模型 (AUC = {auc:.4f})', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', label='随机猜测 (AUC = 0.5)')
plt.xlabel('误伤率 (把好人当坏人) FPR')
plt.ylabel('抓获率 (正确抓出坏人) TPR')
plt.title('ROC 曲线')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('w5_ROC曲线.png', dpi=120)
plt.show()
print(f"AUC = {auc:.4f}")

# ============================================================
# 2. KS 曲线
#    KS = 累计抓获率(TPR) 与 累计误伤率(FPR) 的最大差距
# ============================================================
ks_values = tpr - fpr
ks = ks_values.max()
ks_idx = ks_values.argmax()       # KS最大处的位置

plt.figure(figsize=(7, 6))
plt.plot(thresholds, tpr, label='坏人累计抓获率 TPR', linewidth=2)
plt.plot(thresholds, fpr, label='好人累计误伤率 FPR', linewidth=2)
# 在KS最大处画一条竖线
plt.axvline(x=thresholds[ks_idx], color='red', linestyle='--',
            label=f'KS = {ks:.4f}')
plt.xlabel('预测概率阈值')
plt.ylabel('累计比例')
plt.title('KS 曲线')
plt.legend()
plt.gca().invert_xaxis()          # 阈值从高到低看更直观
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('w5_KS曲线.png', dpi=120)
plt.show()
print(f"KS = {ks:.4f}")

# ============================================================
# 3. PSI 群体稳定性指标
#    做法：把预测概率分成10档，对比"两个群体"分布差异。
#    这里没有未来数据，我们把测试集随机分成两半模拟"两个时间段"，
#    演示PSI怎么算(真实工作中是用上线后的新数据对比训练数据)。
# ============================================================
def calc_psi(expected, actual, bins=10):
    """expected=基准群体, actual=对比群体, 返回PSI值"""
    # 用基准群体的分位数定边界
    breakpoints = np.quantile(expected, np.linspace(0, 1, bins + 1))
    breakpoints[0], breakpoints[-1] = -np.inf, np.inf

    exp_counts = np.histogram(expected, breakpoints)[0] / len(expected)
    act_counts = np.histogram(actual, breakpoints)[0] / len(actual)

    # 防止0导致除零/log错误
    exp_counts = np.where(exp_counts == 0, 0.0001, exp_counts)
    act_counts = np.where(act_counts == 0, 0.0001, act_counts)

    psi = np.sum((act_counts - exp_counts) * np.log(act_counts / exp_counts))
    return psi

# 把测试集随机分两半，模拟两个时间段的客户
half = len(y_prob) // 2
group_A = y_prob.iloc[:half]      # 假设"上线时"的客户
group_B = y_prob.iloc[half:]      # 假设"3个月后"的客户
psi = calc_psi(group_A.values, group_B.values)

print(f"\nPSI = {psi:.4f}")
print("PSI判定标准： <0.1 稳定 | 0.1~0.25 轻微波动 | >0.25 需重训模型")
if psi < 0.1:
    print("→ 结论：模型在两个群体上分布稳定 ✅")
elif psi < 0.25:
    print("→ 结论：轻微波动，可继续监控")
else:
    print("→ 结论：分布差异大，建议重新训练")

print("\n✅ W5 完成！生成了 ROC曲线、KS曲线 两张图")
