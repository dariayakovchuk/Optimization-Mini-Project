Energy-Conserving Descent (ECD) for Optimization Project
====
This repository contains the code for the BBI optimizer, introduced in the paper _Born-Infeld (BI) for AI: Energy-Conserving Descent (ECD) for Optimization_. (http://arxiv.org/abs/2201.11137). And aims to produce the experiments with the BBI optimizer.
Project organization:
### Project structure
- The BBI optimizer is implemented in the file `inflation.py` is based on https://github.com/gbdl/BBI by G. Bruno De Luca and Eva Silverstein, available under the license.
- `comp_optimizers.ipynb` - comparison of different optimizers including modifications of BBI.
- `highly_convex_functions_experiment.ipynb` - experiments with highly-nonconvex function (Rastrigin Function).
- ``

Project Plan:
- Compare to other optimizers that didn’t appear in the article for noise and noise-free setting.
- Try other highly-nonconvex functions.
- More experiments with shallow valleys.
- Another datasets and NN architecture(increasing number of layers and width) => Draw to compare.
- Draw figures with trajectories.
- Work with learning rate(adaptive and/or Gradient Descent: The Ultimate Optimizer and/or weight averaging).
- Adversarial attacks based on batch ordering.
- Framework for initialization and hyperparameter scaling with network size.
