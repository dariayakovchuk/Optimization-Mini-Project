Energy-Conserving Descent (ECD) for Optimization Project
====
This project aims to expand knowledge about experimental behavior of the BBI optimizer, introduced in the paper _Born-Infeld (BI) for AI: Energy-Conserving Descent (ECD) for Optimization_. (http://arxiv.org/abs/2201.11137).  

### Project structure
- The BBI optimizer is implemented in the file `inflation.py` is based on https://github.com/gbdl/BBI by G. Bruno De Luca and Eva Silverstein, available under the license.
- `comp_optimizers.ipynb` - comparison of different optimizers including modifications of BBI.
- `highly_nonconvex_functions_experiment.ipynb` - experiments with highly-nonconvex function (Rastrigin Function).
- `valley_shaped_experiments.ipynb` - experiments with shallow valleys (Rosenbrock Function).
- `functions.py` - helper functions for experiments.
- `different_architectures_experiments.ipynb` - comparison optimizers increasing number of layers of network.
- `initialization_and_hyperparameters_scaling_experimets.ipynb` - experiments with initialization framework.
- `data_ordering_adversarial_attacks.ipynb`, `resnet.py`, `batch_samplers.py` - files performing experiments for adveraserials attacks based on batch ordering.
- `plotly` - directory for interactive plots.



