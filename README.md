# Laken

This repository contains code of Laken presented in the paper "Enhancing Active Learning through Latent Space Exploration: A K-Nearest Neighbors Approach"

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## How It Works

1. **VAE Training:**
   * Run `vae_train.py` to train the VAE model. This script will save the trained VAE for later use;
   * **Important:** Configure the VAE training parameters directly within the `vae_train.py` script, in the section designated for parameter settings.

2. **Active Learning Loop:**
   * Execute `main_active_learner.py`. This script loads the trained VAE and begins the AL process.
   * **Important:**  Set the AL loop parameters (e.g., AL technique, number of iterations) directly within the `main_active_learner.py` script, in the parameter settings section;


## Requirements

* Python (3.10.6)
* PyTorch (2.2.1)
* Torchvision (0.17.1)
* NumPy (1.26.4)
* SciPy (1.12.0)
* Scikit-learn (1.4.1.post1)
* Pandas (2.2.1)
* IPython (8.22.2)
* matplotlib (3.8.3)
* plotly (5.20.0)
* tqdm (4.66.2)
