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

# Initialize medoids by indexes of A1, B1, C1
initial_medoid_indices = [0, 3, 6]

# --- Helper functions ---
def total_cost(coords, medoids_idx, assignments):
    return sum(np.linalg.norm(coords[i] - coords[medoids_idx[assignments[i]]]) for i in range(len(coords)))

def assign_points(coords, medoids_idx):
    distances = np.array([
        [np.linalg.norm(coords[i] - coords[medoid_idx]) for medoid_idx in medoids_idx]
        for i in range(len(coords))
    ])
    return np.argmin(distances, axis=1)

# --- Run K-Medoids and collect history ---
def k_medoids(coords, initial_medoid_indices, max_iter=10):
    medoids_idx = initial_medoid_indices.copy()
    history = []

    for iteration in range(max_iter):
        # Assign each point to the closest medoid
        assignments = assign_points(coords, medoids_idx)
        history.append((medoids_idx.copy(), assignments.copy()))

        updated = False

        # Try swapping each medoid with each non-medoid, check if it improves the total cost
        for m_idx in range(len(medoids_idx)):
            for candidate_idx in range(len(coords)):
                if candidate_idx in medoids_idx:
                    continue  # already a medoid

                # Swap and compute cost
                temp_medoids = medoids_idx.copy()
                temp_medoids[m_idx] = candidate_idx
                temp_assignments = assign_points(coords, temp_medoids)
                if total_cost(coords, temp_medoids, temp_assignments) < total_cost(coords, medoids_idx, assignments):
                    medoids_idx = temp_medoids
                    updated = True

        if not updated:
            break  # No improvement, stop

    return history

# --- Plotting ---
def plot_kmedoids_history(coords, history):
    cols = 3
    rows = (len(history) + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axs = axs.flatten()

    colors = ['red', 'green', 'blue']

    for i, (medoids_idx, assignments) in enumerate(history):
        ax = axs[i]
        for j, color in enumerate(colors):
            cluster_points = coords[np.array(assignments) == j]
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1], color=color, label=f'Cluster {j + 1}')
            ax.scatter(*coords[medoids_idx[j]], color=color, marker='D', s=200, edgecolor='black')  # medoid

        for k, label in enumerate(labels):
            ax.text(coords[k, 0] + 0.2, coords[k, 1], label)

        ax.set_title(f'K-Medoids Iteration {i + 1}')
        ax.legend()
        ax.grid()

    # Remove any empty subplots
    for j in range(len(history), len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    plt.show()

# --- Run ---
history = k_medoids(coords, initial_medoid_indices)
plot_kmedoids_history(coords, history)
