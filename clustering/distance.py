import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import namedtuple
from math import sqrt

# Step 1: Define the Point namedtuple and point list
Point = namedtuple('Point', ['x', 'y'])

def distance(a: Point, b: Point) -> float:
    return sqrt((a.x - b.x)**2 + (a.y - b.y)**2)

def distance_map(points: list[Point]):
    labels = [name for name, _ in points]
    size = len(points)
    matrix = np.zeros((size, size))

    for i in range(size):
        for j in range(size):
            matrix[i][j] = round(distance(points[i][1], points[j][1]), 2)

    # Step 4: Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, xticklabels=labels, yticklabels=labels, annot=True, cmap="YlGnBu", fmt=".2f")
    plt.title("Distance Matrix Heatmap")
    plt.xlabel("To")
    plt.ylabel("From")
    plt.tight_layout()
    plt.show()
    


points = [
    ("A1", Point(2, 10)),
    ("A2", Point(2, 5)),
    ("A3", Point(8, 4)),
    ("A4", Point(5, 8)),
    ("A5", Point(7, 5)),
    ("A6", Point(6, 4)),
    ("A7", Point(1, 2)),
    ("A8", Point(4, 9)),
]



distance_map(points)