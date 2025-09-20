import matplotlib.pyplot as plt
import numpy as np
from collections import namedtuple

# --- Define data ---
Point = namedtuple('Point', ['x', 'y'])

points = [
    ("A1", Point(x=2, y=10)),
    ("A2", Point(x=2, y=5)),
    ("A3", Point(x=8, y=4)),
    ("B1", Point(x=5, y=8)),
    ("B2", Point(x=7, y=5)),
    ("B3", Point(x=6, y=4)),
    ("C1", Point(x=1, y=2)),
    ("C2", Point(x=4, y=9)),
]

labels = [label for label, _ in points]
coords = np.array([[p.x, p.y] for _, p in points])

initial_centers = np.array([
    [2, 10],  # A1
    [5, 8],   # B1
    [1, 2]    # C1
])

# --- Collect data for plotting ---
def kmeans_collect(coords, initial_centers):
    centers = initial_centers.copy()
    history = []  # To store assignments and centers after each iteration

    for iteration in range(1, 10):
        # Assign points to the nearest cluster center
        distances = np.linalg.norm(coords[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)
        assignments = np.argmin(distances, axis=1)

        # Save current state for plotting
        history.append((centers.copy(), assignments.copy()))

        # Update centers
        new_centers = np.array([
            coords[assignments == i].mean(axis=0) if np.any(assignments == i) else centers[i]
            for i in range(len(centers))
        ])

        # Stop if converged
        if np.allclose(new_centers, centers):
            break

        centers = new_centers

    return history

# Run k-means and collect data
history = kmeans_collect(coords, initial_centers)

# --- Plot all iterations as subplots ---
cols = 3
rows = (len(history) + cols - 1) // cols
fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
axs = axs.flatten()

colors = ['red', 'green', 'blue']

for i, (centers, assignments) in enumerate(history):
    ax = axs[i]
    for j in range(len(centers)):
        cluster_points = coords[np.array(assignments) == j]
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], color=colors[j], label=f'Cluster {j + 1}')
        ax.scatter(*centers[j], color=colors[j], marker='X', s=200, edgecolor='black')
    for k, label in enumerate(labels):
        ax.text(coords[k, 0] + 0.2, coords[k, 1], label)
    ax.set_title(f'Iteration {i + 1}')
    ax.legend()
    ax.grid()

# Remove empty subplots
for j in range(len(history), len(axs)):
    fig.delaxes(axs[j])

plt.tight_layout()
plt.show()
