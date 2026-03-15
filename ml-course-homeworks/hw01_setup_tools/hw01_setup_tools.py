import sys
import platform
import sklearn
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

np.random.seed(42)

# TODO: создайте матрицу 100x5
X = np.random.randn(100, 5)

print("X.shape:", X.shape)
print("X.dtype:", X.dtype)
print("First 5 rows:\n", X[:5])

# TODO: статистики по всей матрице
overall_mean = X.mean()
overall_std = X.std()

# TODO: статистики по столбцам
col_mean = X.mean(axis=0)
col_std = X.std(axis=0)
col_min = X.min(axis=0)
col_max = X.max(axis=0)

print("Overall mean:", overall_mean)
print("Overall std:", overall_std)
print("\nPer-column mean:", col_mean)
print("Per-column std:", col_std)
print("Per-column min:", col_min)
print("Per-column max:", col_max)

# TODO: веса и линейная комбинация
w = np.random.randn(5)
y = X @ w

print("w:", w)
print("y.shape:", y.shape)
print("y[:5]:", y[:5])

# TODO: "истинные" веса, шум, y_true
w_true = np.array([1.5, -2.0, 0.0, 0.7, 3.0])
noise = 0.1 * np.random.randn(X.shape[0])
y_true = X @ w_true + noise

# TODO: "предсказанные" веса и y_pred (можете взять w_true + небольшой шум)
w_guess = w_true + 0.5 * np.random.randn(5)
y_pred = X @ w_guess

# Метрики
mse = np.mean((y_true - y_pred) ** 2)
mae = np.mean(np.abs(y_true - y_pred))
rmse = np.sqrt(mse)

# TODO (bonus): R2
ss_res = np.sum((y_true - y_pred) ** 2)
ss_tot = np.sum((y_true - y_true.mean()) ** 2)
r2 = 1 - ss_res / ss_tot

print("MSE:", mse)
print("MAE:", mae)
print("RMSE:", rmse)
print("R2:", r2)

# TODO: DataFrame из X
col_names = [f"x{i}" for i in range(X.shape[1])]
df = pd.DataFrame(X, columns=col_names)

# TODO: добавьте target
df["target"] = y_true

print(df.head())
print("Shape:", df.shape)
print(df.describe())
print("\nMissing values:\n", df.isnull().sum())

# TODO: пример фильтрации и сортировки (поменяйте условие на своё)
filtered = df[df["x0"] > 0].sort_values("target", ascending=False)
print(filtered.head(10))

# TODO: создаём бины по x0 и делаем groupby
df["x0_bin"] = pd.cut(df["x0"], bins=4)

grouped = df.groupby("x0_bin")["target"].mean()
print(grouped)

plt.figure()
sns.histplot(df["target"], bins=30, kde=True)
plt.title("Distribution of target")
plt.show()

# TODO: корреляции
corr = df.corr(numeric_only=True)

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".2f")
plt.title("Correlation heatmap")
plt.show()

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

iris = load_iris(as_frame=True)
iris_df = iris.frame.copy()
print(iris_df.head())

print("Class distribution:")
print(iris_df["target"].value_counts())

plt.figure()
iris_df["target"].value_counts().sort_index().plot(kind="bar")
plt.title("Iris class distribution")
plt.xlabel("class id")
plt.ylabel("count")
plt.show()

# Train/test split
X_iris = iris_df.drop(columns=["target"])
y_iris = iris_df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X_iris, y_iris, test_size=0.2, random_state=42, stratify=y_iris
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification report:\n", classification_report(y_test, y_pred))
