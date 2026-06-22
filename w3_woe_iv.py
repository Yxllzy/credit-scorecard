# ============================================================
# W3：特征工程 —— 分箱 + 计算 WOE / IV，筛选有用特征
# 这是评分卡的核心。产出：每个字段的 IV 排名 + WOE 编码后的数据
# ============================================================
import pandas as pd
import numpy as np

# ---------- 0. 读清洗好的数据 ----------
df = pd.read_csv('cs-clean.csv', index_col=0)

target = 'SeriousDlqin2yrs'   # 我们要预测的目标列（1=坏人）
features = [c for c in df.columns if c != target]   # 其余都是特征

# ============================================================
# 定义一个函数：对单个字段做分箱 + 算 WOE / IV
# ============================================================
def calc_woe_iv(data, feature, target, bins=5):
    """
    data    : 数据
    feature : 要分析的字段名
    target  : 目标列名(0/1)
    bins    : 分成几箱(默认5箱)
    返回该字段每一箱的 WOE 明细表 和 总 IV 值
    """
    df_temp = data[[feature, target]].copy()

    # 用 qcut 按分位数分箱（让每箱人数尽量均匀）；
    # duplicates='drop' 防止边界重复报错
    try:
        df_temp['bin'] = pd.qcut(df_temp[feature], q=bins, duplicates='drop')
    except Exception:
        # 如果某些字段取值太集中分不了箱，就用普通等宽分箱
        df_temp['bin'] = pd.cut(df_temp[feature], bins=bins, duplicates='drop')

    # 按每一箱统计：好人数、坏人数
    grouped = df_temp.groupby('bin', observed=True)[target].agg(['count', 'sum'])
    grouped.columns = ['总人数', '坏人数']
    grouped['好人数'] = grouped['总人数'] - grouped['坏人数']

    # 总的好人/坏人数量
    total_bad = grouped['坏人数'].sum()
    total_good = grouped['好人数'].sum()

    # 每箱中：坏人占总坏人比例、好人占总好人比例
    # +0.5 是为了防止某箱坏人数为0时除零出错（叫"拉普拉斯平滑"）
    grouped['坏人占比'] = (grouped['坏人数'] + 0.5) / total_bad
    grouped['好人占比'] = (grouped['好人数'] + 0.5) / total_good

    # WOE = ln(好人占比 / 坏人占比)
    grouped['WOE'] = np.log(grouped['好人占比'] / grouped['坏人占比'])

    # IV 每箱贡献 = (好人占比 - 坏人占比) * WOE
    grouped['IV_每箱'] = (grouped['好人占比'] - grouped['坏人占比']) * grouped['WOE']

    iv = grouped['IV_每箱'].sum()   # 该字段总 IV
    return grouped, iv


# ============================================================
# 对所有字段计算 IV，并排名
# ============================================================
iv_results = {}
woe_details = {}

for f in features:
    detail, iv = calc_woe_iv(df, f, target, bins=5)
    iv_results[f] = iv
    woe_details[f] = detail

# 按 IV 从高到低排名
iv_df = pd.DataFrame({
    '字段': list(iv_results.keys()),
    'IV值': list(iv_results.values())
}).sort_values('IV值', ascending=False).reset_index(drop=True)

# 加一列"评价"
def iv_level(iv):
    if iv < 0.02: return '几乎没用'
    elif iv < 0.1: return '有点用'
    elif iv < 0.3: return '不错'
    else: return '很强'
iv_df['评价'] = iv_df['IV值'].apply(iv_level)

print("========== 各字段 IV 排名（实力榜）==========")
print(iv_df.to_string(index=False))

# ============================================================
# 看一个具体字段的 WOE 明细（以 IV 最高的字段为例）
# ============================================================
top_feature = iv_df.iloc[0]['字段']
print(f"\n========== IV最高字段【{top_feature}】的分箱 WOE 明细 ==========")
print(woe_details[top_feature][['总人数', '坏人数', '好人数', 'WOE', 'IV_每箱']])

# ============================================================
# 筛选：保留 IV >= 0.02 的字段（扔掉几乎没用的）
# ============================================================
selected = iv_df[iv_df['IV值'] >= 0.02]['字段'].tolist()
print(f"\n✅ 筛选后保留 {len(selected)} 个有用字段：")
print(selected)

# 把 IV 排名表存下来，后面写报告用
iv_df.to_csv('w3_iv_ranking.csv', index=False, encoding='utf-8-sig')
print("\nIV排名已保存为 w3_iv_ranking.csv")
