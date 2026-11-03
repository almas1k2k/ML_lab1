import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from config import csv_path, ix, iy

df = pd.read_csv(csv_path)

columns = df.columns.tolist()
X = df[columns[ix]].to_numpy()
Y = df[columns[iy]].to_numpy()

# пункт 3
print(f"Число строк: {df.shape[0]}")
print(f"  | Min   | Max    | Mean")
print(f"X | {X.min()}   | {X.max()}   | {X.mean():.2f}")
print(f"Y | {Y.min()} | {Y.max()} | {Y.mean():.0f}")

# пункт 4
plt.figure(figsize=(8, 5))
plt.scatter(X, Y)
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.show()