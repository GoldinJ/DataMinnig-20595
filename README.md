# DataMinnig-20595

A Python project containing data mining algorithms including clustering methods (K-means, K-medoids, DBSCAN, AGNES) and decision tree implementations.

## Prerequisites

- Python 3.13 or higher
- [uv](https://github.com/astral-sh/uv) package manager (recommended) or pip

## Installation

### Option 1: Using uv (Recommended)

1. Install uv if you haven't already:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Clone the repository:
   ```bash
   git clone https://github.com/GoldinJ/DataMinnig-20595.git
   cd DataMinnig-20595
   ```

3. Install dependencies:
   ```bash
   uv sync
   ```

4. Run the project:
   ```bash
   uv run python main.py
   ```

### Option 2: Using pip

1. Clone the repository:
   ```bash
   git clone https://github.com/GoldinJ/DataMinnig-20595.git
   cd DataMinnig-20595
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -e .
   ```

4. Run the project:
   ```bash
   python main.py
   ```

## Project Structure

- `clustering/` - Clustering algorithm implementations (K-means, K-medoids, DBSCAN, AGNES)
- `decision_tree/` - Decision tree implementations and examples
- `main.py` - Main entry point

## Dependencies

- pandas >= 2.3.1
- scikit-learn >= 1.7.0
- matplotlib >= 3.10.3
- seaborn >= 0.13.2
- scipy >= 1.16.0
- ipykernel >= 6.29.5
