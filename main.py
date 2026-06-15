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

# 3. 攻击者能改的"文字类"特征 → 全部伪装成"正常网址的典型值"
manipulable = ["directory_length", "length_url", "file_length",
               "qty_slash_directory", "qty_slash_url",
               "qty_dollar_directory", "qty_dot_file",
               "qty_underline_directory"]
legit_typical = X_train[y_train == 0][manipulable].median()

X_disguised = X_phish.copy()
for col in manipulable:
    X_disguised[col] = legit_typical[col]

# 4. 伪装后：模型还能抓住多少？
caught_after = (model.predict(X_disguised) == 1).sum()
print("伪装后，模型抓住：", caught_after,
      f"（{round(caught_after/len(X_phish)*100, 1)}%）")
print("多漏掉的钓鱼网址：", caught_before - caught_after, "个")