# Adaptive AI Navigation in Hazardous and Evolving Mazes

## Overview

This project simulates four intelligent bots navigating a 40x40 grid maze in Python, attempting to reach a button while avoiding a dynamically spreading fire. The bots use variants of the A* search algorithm, and the environment simulates realistic risk using a probabilistic fire spread model.

This simulation is designed to analyze and compare the robustness of each bot across multiple environments and fire spread rates.

---

## Features

- **Maze Representation**  
  2D grid with randomly placed walls, start point, button, and fire.

- **Fire Progression**  
  Uses a fixed 3D NumPy array to simulate consistent fire spread across experiments.

- **Bot Algorithms**
  - **Bot 1**: A* pathfinder (ignores fire).
  - **Bot 2**: Re-plans at each step, avoids fire cells.
  - **Bot 3**: Avoids fire + adjacent fire cells when re-planning.
  - **Bot 4**: Risk-aware A* using custom heuristic (distance + fire risk).

- **Data Generation & Evaluation**
  - Tested on 20 mazes with fire probabilities: `q = 0.00, 0.05, 0.10, 0.25, 1.00`
  - Consistent comparisons using controlled random seeds.

- **Visualization**
  - Matplotlib used for plotting bot success rates and overall maze winnability.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/KriChau95/AI-Project-1.git
cd AI-Project-1

# Install required dependencies
pip install numpy matplotlib
