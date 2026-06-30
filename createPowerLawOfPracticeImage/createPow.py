import numpy as np
import matplotlib.pyplot as plt

# Power Law of Practice
# T = A + B(N + E)^(-a)

A = 2.0
B = 18.0
E = 0.0
a = 0.45

# Nを1刻みで定義
N = np.arange(1, 101, 1)

# Tを計算
T = A + B * (N + E) ** (-a)

# 作図
fig, ax = plt.subplots(figsize=(6, 4))

# Nが1変化するごとに点をプロット
ax.plot(N, T, marker="o", markersize=2, linewidth=1.5)

# 軸ラベルのみ
ax.set_xlabel("Trial number")
ax.set_ylabel("Task completion time")

# 余計な要素を消す
ax.grid(False)
ax.legend().remove() if ax.get_legend() else None
ax.set_title("")

fig.tight_layout()

# 保存
fig.savefig("power_law_practice.png", dpi=300, bbox_inches="tight")

plt.show()