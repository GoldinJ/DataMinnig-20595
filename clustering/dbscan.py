import matplotlib.pyplot as plt
import numpy as np
import math
from collections import namedtuple
from sklearn.cluster import DBSCAN

# --- Define data ---
Point = namedtuple('Point', ['x', 'y'])

points = [
    ("A1", Point(x=2, y=10)),
    ("A2", Point(x=2, y=5)),
    ("A3", Point(x=8, y=4)),
    ("A4", Point(x=5, y=8)),
    ("A5", Point(x=7, y=5)),
    ("A6", Point(x=6, y=4)),
    ("A7", Point(x=1, y=2)),
    ("A8", Point(x=4, y=9)),
]

labels = [label for label, _ in points]
coords = np.array([[p.x, p.y] for _, p in points])

# --- DBSCAN parameters ---
eps = math.sqrt(10)      # Max distance for neighborhood
min_samples = 2  # Min points to form a core point

# --- Run DBSCAN ---
db = DBSCAN(eps=eps, min_samples=min_samples)
db.fit(coords)

# Extract results
core_sample_mask = np.zeros_like(db.labels_, dtype=bool)
core_sample_mask[db.core_sample_indices_] = True
cluster_labels = db.labels_

# --- Collect pseudo-iterations: mark core points, then add border points ---
history = []

# Step 1: Show only core points
history.append((np.full_like(cluster_labels, -1), core_sample_mask))

# Step 2: Show core + border points
intermediate_labels = np.full_like(cluster_labels, -1)
for idx, is_core in enumerate(core_sample_mask):
    if is_core:
        intermediate_labels[idx] = cluster_labels[idx]
history.append((intermediate_labels, core_sample_mask))

# Step 3: Full DBSCAN result
history.append((cluster_labels, core_sample_mask))

# --- Plotting ---
def plot_dbscan_history(coords, history, labels):
    cols = 3
    rows = 1
    fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axs = axs.flatten()

    colors = ['red', 'green', 'blue', 'purple', 'orange']
    for i, (step_labels, core_mask) in enumerate(history):
        ax = axs[i]

        for k in set(step_labels):
            if k == -1:
                # Noise points
                mask = (step_labels == -1)
                ax.scatter(coords[mask, 0], coords[mask, 1], c='gray', marker='x', label='Noise')
            else:
                mask = (step_labels == k)
                ax.scatter(coords[mask, 0], coords[mask, 1], color=colors[k % len(colors)], label=f'Cluster {k + 1}')

        # Highlight core points
        ax.scatter(coords[core_mask, 0], coords[core_mask, 1], s=250, facecolors='none', edgecolors='black', linewidths=2)

        for j, point_label in enumerate(labels):
            ax.text(coords[j, 0] + 0.2, coords[j, 1], point_label)

        ax.set_title(f'DBSCAN Step {i + 1}')
        ax.legend()
        ax.grid()

    plt.tight_layout()
    plt.show()

# --- Run plotting ---
plot_dbscan_history(coords, history, labels)
