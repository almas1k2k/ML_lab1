import pandas as pd
import matplotlib.pyplot as plt
from config import csv_path, ix, iy
from matplotlib.patches import Rectangle

df = pd.read_csv(csv_path)

columns = df.columns.tolist()
n = df.shape[0]
X = df[columns[ix]].to_numpy()
Y = df[columns[iy]].to_numpy()

# пункт 3
print(f"Число строк: {n}")
print(f"  | Min   | Max    | Mean")
print(f"X | {X.min()}   | {X.max()}   | {X.mean():.2f}")
print(f"Y | {Y.min()} | {Y.max()} | {Y.mean():.0f}")
print()

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

# пункт 4
ax1.scatter(X, Y)
ax1.set_title("Исходные данные")
ax1.grid(True)

# пункты 5,6 - алгоритм метода наименьших квадратов
a = (n*(X*Y).sum() - X.sum()*Y.sum()) / (n*(X*X).sum() - (X.sum())**2)
b = (Y.sum() - a*X.sum()) / n
print(f"f: {a}x + {b}")

x_line = pd.Series([X.min(), X.max()])
y_line = a * x_line + b

ax2.scatter(X, Y)
ax2.plot(x_line, y_line, color="red")
ax2.set_title(f"Метод наименьших квадратов")
ax2.grid(True)

# пункт 7
ax3.scatter(X, Y)
ax3.plot(x_line, y_line, color="red")

for xi, yi in zip(X, Y):
    y_hat = a * xi + b
    e = abs(yi - y_hat)

    x_proj = (yi - b) / a

    x_left = min(xi, x_proj)
    y_bottom = min(yi, y_hat)
    width = abs(x_proj - xi)
    height = e

    rect = Rectangle( (x_left, y_bottom), width, height, facecolor="orange", edgecolor="darkorange", alpha=0.3)
    ax3.add_patch(rect)

ax3.set_title("Квадраты ошибок")
ax3.grid(True)
ax3.autoscale_view()

plt.tight_layout()
plt.show()