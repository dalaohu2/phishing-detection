import pandas as pd

# 从网上直接读取钓鱼网址数据集
url = "https://raw.githubusercontent.com/GregaVrbancic/Phishing-Dataset/master/dataset_full.csv"
df = pd.read_csv(url)

# 看看数据长什么样
print("数据形状（行数, 列数）：", df.shape)
print()
print("前 5 行：")
print(df.head())
print()
print("标签分布（0=正常网址, 1=钓鱼网址）：")
print(df["phishing"].value_counts())

# ===== 训练第一个模型 =====
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. 把"特征"和"答案"分开
X = df.drop("phishing", axis=1)   # 前 111 列：特征
y = df["phishing"]                # 最后 1 列：答案

# 2. 分成训练集和测试集（80% 用来学，20% 留着考试）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. 训练模型
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# 4. 在"没见过的"测试集上预测，看成绩
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print()
print("===== 模型结果 =====")
print("准确率：", round(accuracy * 100, 2), "%")
print()
print(classification_report(y_test, predictions, target_names=["正常", "钓鱼"]))

# ===== 看模型主要靠哪些线索（特征重要性）=====
importances = pd.Series(model.feature_importances_, index=X.columns)
top10 = importances.sort_values(ascending=False).head(10)
print()
print("===== 模型最看重的 10 个特征 =====")
print(top10)

# ===== 规避测试：把钓鱼网址"伪装"成正常网址 =====

# 1. 从测试集里挑出真正的钓鱼网址
X_phish = X_test[y_test == 1]
print()
print("===== 规避测试 =====")
print("测试集里的真钓鱼网址：", len(X_phish), "个")

# 2. 伪装前：模型现在能抓住多少？
caught_before = (model.predict(X_phish) == 1).sum()
print("伪装前，模型抓住：", caught_before,
      f"（{round(caught_before/len(X_phish)*100, 1)}%）")

# 攻击者真正能改的特征：网址里那些"长度 / 数量"类
MANIPULABLE = ["directory_length", "length_url", "file_length",
               "qty_slash_directory", "qty_slash_url",
               "qty_dollar_directory", "qty_dot_file",
               "qty_underline_directory"]

def disguise(X_phish, X_train, y_train, manipulable=MANIPULABLE):
    """把钓鱼样本在可改特征上，伪装成正常网址的典型值。
    legit_typical 只用训练集的正常样本来算 —— 不许碰测试集，否则算作弊。
    """
    legit_typical = X_train[y_train == 0][manipulable].median()
    X_disguised = X_phish.copy()
    for col in manipulable:
        X_disguised[col] = legit_typical[col]
    return X_disguised

# 4. 伪装后：模型还能抓住多少？
# 真正执行伪装，得到伪装后的钓鱼样本
X_disguised = disguise(X_phish, X_train, y_train)
caught_after = (model.predict(X_disguised) == 1).sum()
print("伪装后，模型抓住：", caught_after,
      f"（{round(caught_after/len(X_phish)*100, 1)}%）")
print("多漏掉的钓鱼网址：", caught_before - caught_after, "个")

# ========================================================
#  项目二：对抗训练 —— 把模型"教聪明"，让它骗不动
# ========================================================

# 1. 造对抗教材：从【训练集】里挑钓鱼样本，伪装它们，但标签仍然标 1（还是钓鱼）
#    关键：只能用训练集的钓鱼，绝不能碰测试集 —— 否则等于拿考试题去训练，作弊
X_train_phish = X_train[y_train == 1]
X_train_adv = disguise(X_train_phish, X_train, y_train)
y_train_adv = pd.Series([1] * len(X_train_adv))


# 2. 把对抗教材拼到原始训练集后面，组成"加强版训练集"
X_train_aug = pd.concat([X_train, X_train_adv], ignore_index=True)
y_train_aug = pd.concat([y_train, y_train_adv], ignore_index=True)
print()
print("原始训练集：", len(X_train), "→ 加对抗样本后：", len(X_train_aug))

# 3. 用加强版训练集，训练一个新模型（加固版）
model_robust = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model_robust.fit(X_train_aug, y_train_aug)

# 4. 拿【同一批】伪装过的测试钓鱼，考新模型
#    注意：这批测试样本新模型从没见过，它只是"见过伪装这种套路" —— 所以对比公平
caught_robust = (model_robust.predict(X_disguised) == 1).sum()

print()
print("===== 加固前 vs 加固后（都用伪装过的测试钓鱼）=====")
print(f"旧模型抓住：{caught_after}（{round(caught_after/len(X_phish)*100, 1)}%）")
print(f"新模型抓住：{caught_robust}（{round(caught_robust/len(X_phish)*100, 1)}%）")

# ========================================================
#  第 4 步：诚实体检 —— 加固有没有代价？（都用干净测试集）
# ========================================================
from sklearn.metrics import recall_score

X_legit = X_test[y_test == 0]   # 测试集里的正常网址

def checkup(m, name):
    acc = accuracy_score(y_test, m.predict(X_test))      # 干净数据整体准确率
    fp = (m.predict(X_legit) == 1).sum()                 # 正常网址被误判成钓鱼的个数
    fp_rate = fp / len(X_legit) * 100
    rec = recall_score(y_test, m.predict(X_test))        # 对【没伪装的】普通钓鱼的召回
    print(f"{name}：准确率 {acc*100:.2f}% | 正常网址误报 {fp} 个（{fp_rate:.2f}%）| 普通钓鱼召回 {rec*100:.2f}%")

print()
print("===== 诚实体检 =====")
checkup(model, "旧模型")
checkup(model_robust, "新模型")

# ========================================================
#  第 5 步：为什么能防住？—— 看模型把注意力挪去了哪
# ========================================================
imp_old = pd.Series(model.feature_importances_, index=X.columns)
imp_new = pd.Series(model_robust.feature_importances_, index=X.columns)

print()
print("===== 旧模型 vs 新模型：在 8 个'可伪造特征'上的重要性 =====")
compare = pd.DataFrame({"旧模型": imp_old[MANIPULABLE],
                        "新模型": imp_new[MANIPULABLE]})
print(compare.round(4))
print()
print("这 8 个可伪造特征的重要性【之和】：")
print(f"  旧模型：{imp_old[MANIPULABLE].sum():.4f}")
print(f"  新模型：{imp_new[MANIPULABLE].sum():.4f}")

# ========================================================
#  压轴：换个更聪明的伪装，考验"鲁棒性能不能泛化"
# ========================================================
import numpy as np

def disguise_smart(X_phish, X_train, y_train, manipulable=MANIPULABLE, seed=42):
    """聪明伪装：每个钓鱼样本随机抄一个真实正常网址的这 8 个值，
    而不是全抄同一组固定中位数。参照值只从训练集的正常样本抽。"""
    rng = np.random.default_rng(seed)
    legit_pool = X_train[y_train == 0][manipulable]
    X_d = X_phish.copy()
    picks = rng.integers(0, len(legit_pool), size=len(X_phish))
    X_d[manipulable] = legit_pool.iloc[picks].values   # 整行随机模仿，不再是固定值
    return X_d

# 用聪明伪装处理同一批测试集钓鱼
X_smart = disguise_smart(X_phish, X_train, y_train)

caught_old_smart = (model.predict(X_smart) == 1).sum()
caught_new_smart = (model_robust.predict(X_smart) == 1).sum()
n = len(X_phish)

print()
print("===== 聪明伪装 vs 笨伪装：两个模型分别抓住多少 =====")
print(f"【笨伪装·固定中位数】 旧模型 {caught_after/n*100:.1f}%  |  新模型 {caught_robust/n*100:.1f}%")
print(f"【聪明伪装·随机模仿】 旧模型 {caught_old_smart/n*100:.1f}%  |  新模型 {caught_new_smart/n*100:.1f}%")