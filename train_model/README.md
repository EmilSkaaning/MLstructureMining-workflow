# XGBoost Classifier Trainer for Pair Distribution Function Data

This project trains an XGBoost classifier on Pair Distribution Function data, derived from X-ray scattering data simulated from CIFs obtained from the Crystallography Open Database (COD). The user has the flexibility to modify various parameters and utilize Bayesian optimization for hyperparameter tuning. The training results, models, and hyperparameters are systematically stored for subsequent analysis.

## How to use

1. Navigate to the directory containing the main script.
2. Adjust the internal parameters as needed (see **Parameters** section).
3. Use the command-line interface to provide the required and optional arguments for training:

```
python <script_name>.py <directory> -n <number_of_cpus> -s <simple_load_flag> -d <number_of_data> -b <do_bayesopt_flag>
```

## Command-line Arguments

| Argument        | Default Value | Optional | Description                                                                                      |
|-----------------|---------------|----------|--------------------------------------------------------------------------------------------------|
| `directory`     | None          | No       | The directory where the data is located.                                                         |
| `-n, --n_cpu`       | 1             | Yes      | The number of CPUs to use for training.                                                          |
| `-s, --simple_load` | False     | Yes      | If set, only one type of structure is loaded per class.                                         |
| `-d, --n_data`  | -1            | Yes      | Number of data points to be used for training.                                                   |
| `-b, --do_bayesopt` | False     | Yes      | If set, the model will be trained using Bayesian optimization.                                   |


## Parameters
The following parameters can be set inside of the `train_model.py` script. 

### Program Settings

| Parameter        | Default Value | Description                                                                                         |
|------------------|---------------|-----------------------------------------------------------------------------------------------------|
| `init_points`    | 3             | Initial number of points to begin the Bayesian optimization.                                        |
| `n_iter`         | 3             | Number of iterations for Bayesian optimization.                                                     |
| `iterative_train`| 3             | Number of iterations to train the model iteratively.                                                |

### Hyperparameters

| Parameter      | Default Value    | Description                                                                                   |
|----------------|------------------|---------------------------------------------------------------------------------------------|
| `objective`    | "multi:softprob" | Specifies the learning task and the corresponding objective function.                        |
| `eval_metric`  | "mlogloss"       | Evaluation metric to be used for training.                                                  |
| `verbosity`    | 1                | Verbosity of printing messages.                                                             |
| `max_depth`    | 6                | Maximum depth of a tree.                                                                    |
| `tree_method`  | "hist"           | The tree construction algorithm used in XGBoost.                                           |
| `max_bin`      | 256              | Maximum number of discrete bins to bucket continuous features.                              |

### XGBoost Training Parameters

| Parameter               | Default Value | Description                                                                        |
|-------------------------|---------------|------------------------------------------------------------------------------------|
| `early_stopping_rounds` | 5            | Activates early stopping. Validation metric needs to improve at least once in every `early_stopping_rounds` round(s) to continue training. |
| `num_boost_round`       | 10            | The number of boosting rounds or trees to build.                                   |
| `verbose_eval`          | 250           | Frequency of printing messages during training.                                    |
| `subsample`             | 0.5           | Subsample ratio of the training instances.                                         |

### Bayesian Optimization Search Space

| Parameter           | Range          | Description                                |
|---------------------|----------------|--------------------------------------------|
| `learning_rate`     | (0.05, 1.0)    | Step size shrinkage used in update.        |
| `min_child_weight`  | (0.1, 10)      | Minimum sum of instance weight (hessian) needed in a child. |
| `max_depth`         | (3, 6)         | Maximum depth of a tree.                    |
| `max_delta_step`    | (0, 20)        | Maximum delta step we allow each tree’s weight estimation to be. |
| `subsample`         | (0.01, 1.0)    | Subsample ratio of the training instances. |
| `colsample_bytree`  | (0.01, 1.0)    | Subsample ratio of columns when constructing each tree. |
| `colsample_bylevel` | (0.01, 1.0)    | Subsample ratio of columns for each level. |
| `reg_lambda`        | (0, 10.0)      | L2 regularization term on weights.         |
| `reg_alpha`         | (0, 10.0)      | L1 regularization term on weights.         |
| `gamma`             | (0, 10.0)      | Minimum loss reduction to make a further partition. |
