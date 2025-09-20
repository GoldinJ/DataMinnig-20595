import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from collections import namedtuple

# Define the Point data structure
Point = namedtuple('Point', ['x', 'y'])

# Define the points
points = [
    ("A1", Point(x=2, y=0)),
    ("A2", Point(x=4, y=0)),
    ("A3", Point(x=7, y=0)),
    ("A4", Point(x=8, y=0)),
    ("A5", Point(x=12, y=0)),
    ("A6", Point(x=14, y=0)),
    # ("A7", Point(x=1, y=2)),
    # ("A8", Point(x=4, y=9)),
]

# Extract labels and coordinates
labels = [label for label, _ in points]
coords = np.array([[p.x, p.y] for _, p in points])

# Define linkage methods to test
linkage_methods = ['ward', 'single', 'complete']
titles = ['Ward Linkage', 'Single Linkage', 'Complete Linkage']

# Create subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()

# Plot each dendrogram in its subplot
for ax, method, title in zip(axes, linkage_methods, titles):
    linked = linkage(coords, method=method)
    dendrogram(linked, labels=labels, distance_sort='ascending',
               show_leaf_counts=True, ax=ax)
    ax.set_title(f'AGNES - {title}')
    ax.set_xlabel('Points')
    ax.set_ylabel('Distance')
    ax.grid(True)

plt.tight_layout()
plt.show()
