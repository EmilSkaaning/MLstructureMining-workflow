import os
import sys
import time
import datetime
import yaml
import xgboost
import numpy as np
from typing import Tuple
from sklearn.metrics import log_loss, accuracy_score
from bayes_opt import BayesianOptimization
sys.path.append("..")
from utils.data_loader_in_mem import get_data_splits_from_clean_data
from utils.tools import accuracy_top_x, save_dict_to_yaml, save_list_to_txt
from utils.plotting import plot_loss_curve

# Program settings
program_setting = {
    "init_points": 25,
    "n_iter": 50,
    "best_bayes_opt_models": 15,
    "iterative_train":3
}

# Define hyperparameters
hp = {
    "objective": "multi:softprob",
    "eval_metric": "mlogloss",
    "verbosity": 1,
    "max_depth": 6,
    "tree_method": "hist",
    "max_bin": 256,
}

# Define parameters for xgboost training
params = {
    'hp': hp,
    'early_stopping_rounds': 5,
    'num_boost_round': 3_000,
    "verbose_eval": 250,
}

# Search space for Bayesian optimization
search_space = {
    'learning_rate': (0.01, 1.0),
    'min_child_weight': (0, 10),
    'max_depth': (3, 12),
    'max_delta_step': (0, 20),
    'subsample': (0.01, 1.0),
    'colsample_bytree': (0.01, 1.0),
    'colsample_bylevel': (0.01, 1.0),
    'reg_lambda': (1e-9, 1000),
    'reg_alpha': (1e-9, 1.0),
    'gamma': (1e-9, 0.5),
}


def test_model(booster, tst_xy, eval_dict, project_name, img_title: str):
    pred = booster.predict(tst_xy)
    pred_label = np.argmax(pred, axis=1)
    true_label = np.array(tst_xy.get_label(), dtype=int)

    # Calculate accuracy scores
    acc = accuracy_score(true_label, pred_label)
    acc_3 = accuracy_top_x(true_label, pred, 3)
    acc_5 = accuracy_top_x(true_label, pred, 5)
    acc_7 = accuracy_top_x(true_label, pred, 7)

    print('\nModel test acc: {:.2f} %'.format(acc))
    print('Model test acc top 3: {:.2f} %'.format(acc_3))
    print('Model test acc top 5: {:.2f} %'.format(acc_5))
    print('Model test acc top 7: {:.2f} %'.format(acc_7))

    mlogloss_tst = log_loss(true_label, pred)

    # Plot loss curve
    plot_loss_curve(
        eval_dict,
        project_name,
        mlogloss_tst,
        f'Loss: {mlogloss_tst:.2f}, Acc: {acc:.2f}, top 3: {acc_3:.2f}, top5: {acc_5:.2f}, top7: {acc_7:.2f}',
        img_title
    )
    return None


def init_xgb_model(params, trn_xy, eval_set, booster=None):
    eval_dict = {}
    booster = xgboost.train(
        params['hp'],
        dtrain=trn_xy,
        evals=eval_set,
        evals_result=eval_dict,
        num_boost_round=params['num_boost_round'],
        early_stopping_rounds=params["early_stopping_rounds"],
        verbose_eval=params["verbose_eval"],
        xgb_model=booster,
    )

    return booster, eval_dict


def iterative_train(n_iterations: int, params: dict, trn_xy: Tuple[np.ndarray, np.ndarray], 
                    tst_xy: Tuple[np.ndarray, np.ndarray], eval_set: Tuple[np.ndarray, np.ndarray],
                    project_name: str, stem: str = None) -> xgboost.Booster:
    """
    Train a model iteratively for a given number of iterations.

    Args:
        n_iterations (int): The number of iterations to train the model.
        params (dict): The parameters for the XGBoost model.
        trn_xy (Tuple[np.ndarray, np.ndarray]): The training data and labels.
        tst_xy (Tuple[np.ndarray, np.ndarray]): The test data and labels.
        eval_set (Tuple[np.ndarray, np.ndarray]): The evaluation data and labels.
        project_name (str): The name of the project.

    Returns:
        Booster: The trained XGBoost model.
    """
    booster = None

    for i in range(n_iterations):
        if stem is None:
            stem = f"{i:09}"
        print(f"\nIteration {i+1} of {n_iterations}.\n")
        
        booster, eval_dict = init_xgb_model(params, trn_xy, eval_set, booster)

        booster.save_model(os.path.join(project_name, f"xgb_model_{stem}.bin"))
        save_dict_to_yaml(eval_dict, os.path.join(project_name, f'data_{stem}.yml'))

        test_model(booster, tst_xy, eval_dict, project_name, f'loss_curve_{stem}.png')
        print(f"\nNumber of trees: {len(booster.trees_to_dataframe().Tree.unique())}")

        stem = None
    return booster


def main_train(directory: str, n_cpu: int=1, simple_load: bool=False, n_data: int=100, do_bayesopt: bool=False) -> None:
    """Function to train the model and save the results.

    Parameters:
    directory (str): The directory where the data is located.
    n_cpu (int): The number of CPUs to use for training. Default is 1.
    """

    # Create a new project directory with the current timestamp
    ct = str(datetime.datetime.now()).replace(' ', '_').replace(':', '-').replace('.', '-')
    project_name = os.path.join(directory, f'results_{ct}')
    os.makedirs(project_name, exist_ok=True)

    # Get the data splits
    trn_xy, vld_xy, tst_xy, eval_set, n_classes = get_data_splits_from_clean_data(directory, project_name, simple_load=simple_load, n_data=n_data)
    params["hp"]["num_class"] = n_classes
    params["hp"]["nthread"] = n_cpu

    # Save parameters
    save_dict_to_yaml(params, os.path.join(project_name, '_model_parameters.yml'))
    save_dict_to_yaml(program_setting, os.path.join(project_name, f'_program_setting.yml'))


    # Train the xgboost model
    start_time = time.time()

    
    if do_bayesopt:
        print(f"\nPerforming Bayesian optimization.\n")
        save_dict_to_yaml(search_space, os.path.join(project_name, f'_search_space.yml'))

        params['hp']["verbosity"] = 0
        params["verbose_eval"] = 0

        def black_box_functioimport os
import sys
import time
import datetime
import yaml
import xgboost
import numpy as np
from typing import Tuple
from sklearn.metrics import log_loss, accuracy_score
from bayes_opt import BayesianOptimization
sys.path.append("..")
from utils.data_loader_in_mem import get_data_splits_from_clean_data
from utils.tools import accuracy_top_x, save_dict_to_yaml, save_list_to_txt
from utils.plotting import plot_loss_curve

# Program settings
program_setting = {
    "init_points": 5,
    "n_iter": 15,
    "iterative_train": 1
}

# Define hyperparameters
hp = {
    "objective": "multi:softprob",
    "eval_metric": "mlogloss",
    "verbosity": 1,
    "max_depth": 6,
    "tree_method": "hist",
    "max_bin": 256,
}

# Define parameters for xgboost training
params = {
    'hp': hp,
    'early_stopping_rounds': 5,
    'num_boost_round': 1_500,
    "verbose_eval": 250,
}

# Search space for Bayesian optimization
search_space = {
    'learning_rate': (0.05, 1.0),
    'min_child_weight': (0.1, 10),
    'max_depth': (3, 6),
    'max_delta_step': (0, 20),
    'subsample': (0.01, 1.0),
    'colsample_bytree': (0.01, 1.0),
    'colsample_bylevel': (0.01, 1.0),
    'reg_lambda': (0, 10.0),
    'reg_alpha': (0, 10.0),
    'gamma': (0, 10.0),
}


def test_model(booster, tst_xy, eval_dict, project_name, img_title: str):
    pred = booster.predict(tst_xy)
    pred_label = np.argmax(pred, axis=1)
    true_label = np.array(tst_xy.get_label(), dtype=int)

    # Calculate accuracy scores
    acc = accuracy_score(true_label, pred_label)
    acc_3 = accuracy_top_x(true_label, pred, 3)
    acc_5 = accuracy_top_x(true_label, pred, 5)
    acc_7 = accuracy_top_x(true_label, pred, 7)

    print('\nModel test acc: {:.2f} %'.format(acc))
    print('Model test acc top 3: {:.2f} %'.format(acc_3))
    print('Model test acc top 5: {:.2f} %'.format(acc_5))
    print('Model test acc top 7: {:.2f} %'.format(acc_7))

    mlogloss_tst = log_loss(true_label, pred)

    # Plot loss curve
    plot_loss_curve(
        eval_dict,
        project_name,
        mlogloss_tst,
        f'Loss: {mlogloss_tst:.2f}, Acc: {acc:.2f}, top 3: {acc_3:.2f}, top5: {acc_5:.2f}, top7: {acc_7:.2f}',
        img_title
    )
    return None


def init_xgb_model(params, trn_xy, eval_set, booster=None):
    eval_dict = {}
    booster = xgboost.train(
        params['hp'],
        dtrain=trn_xy,
        evals=eval_set,
        evals_result=eval_dict,
        num_boost_round=params['num_boost_round'],
        early_stopping_rounds=params["early_stopping_rounds"],
        verbose_eval=params["verbose_eval"],
        xgb_model=booster,
    )

    return booster, eval_dict


def iterative_train(n_iterations: int, params: dict, trn_xy: Tuple[np.ndarray, np.ndarray], 
                    tst_xy: Tuple[np.ndarray, np.ndarray], eval_set: Tuple[np.ndarray, np.ndarray],
                    project_name: str, stem: str = None) -> xgboost.Booster:
    """
    Train a model iteratively for a given number of iterations.

    Args:
        n_iterations (int): The number of iterations to train the model.
        params (dict): The parameters for the XGBoost model.
        trn_xy (Tuple[np.ndarray, np.ndarray]): The training data and labels.
        tst_xy (Tuple[np.ndarray, np.ndarray]): The test data and labels.
        eval_set (Tuple[np.ndarray, np.ndarray]): The evaluation data and labels.
        project_name (str): The name of the project.

    Returns:
        Booster: The trained XGBoost model.
    """
    booster = None

    for i in range(n_iterations):
        if stem is None:
            stem = f"{i:09}"
        print(f"\nIteration {i+1} of {n_iterations}.\n")
        
        booster, eval_dict = init_xgb_model(params, trn_xy, eval_set, booster)

        booster.save_model(os.path.join(project_name, f"xgb_model_{stem}.bin"))
        save_dict_to_yaml(eval_dict, os.path.join(project_name, f'data_{stem}.yml'))

        test_model(booster, tst_xy, eval_dict, project_name, f'loss_curve_{stem}.png')
        print(f"\nNumber of trees: {len(booster.trees_to_dataframe().Tree.unique())}")

        stem = None
    return booster


def main_train(directory: str, n_cpu: int=1, simple_load: bool=False, n_data: int=100, do_bayesopt: bool=False) -> None:
    """Function to train the model and save the results.

    Parameters:
    directory (str): The directory where the data is located.
    n_cpu (int): The number of CPUs to use for training. Default is 1.
    """

    # Create a new project directory with the current timestamp
    ct = str(datetime.datetime.now()).replace(' ', '_').replace(':', '-').replace('.', '-')
    project_name = os.path.join(directory, f'results_{ct}')
    os.makedirs(project_name, exist_ok=True)

    # Get the data splits
    trn_xy, vld_xy, tst_xy, eval_set, n_classes = get_data_splits_from_clean_data(directory, project_name, simple_load=simple_load, n_data=n_data)
    params["hp"]["num_class"] = n_classes
    params["hp"]["nthread"] = n_cpu

    # Save parameters
    save_dict_to_yaml(params, os.path.join(project_name, '_model_parameters.yml'))
    save_dict_to_yaml(program_setting, os.path.join(project_name, f'_program_setting.yml'))


    # Train the xgboost model
    start_time = time.time()

    
    if do_bayesopt:
        print(f"\nPerforming Bayesian optimization.\n")
        save_dict_to_yaml(search_space, os.path.join(project_name, f'_search_space.yml'))

        params['hp']["verbosity"] = 0
        params["verbose_eval"] = 0

        def black_box_function(**sample_paras):
            """Function with unknown internals we wish to maximize.

            This is just serving as an example, for all intents and
            purposes think of the internals of this function, i.e.: the process
            which generates its output values, as unknown.
            """
            for sample_para in sample_paras.keys():
                if sample_para in ["max_depth"]:
                    params["hp"][sample_para] = int(sample_paras[sample_para])
                else:
                    params["hp"][sample_para] = sample_paras[sample_para]

            booster, eval_dict = init_xgb_model(params, trn_xy, eval_set)
            stem = f"bayse_optimization_{len(optimizer.res):05d}"  # unique identifier based on number of iterations so far
            booster.save_model(os.path.join(project_name, f"xgb_model_{stem}.bin"))
            save_dict_to_yaml(eval_dict, os.path.join(project_name, f'data_{stem}.yml'))
            test_model(booster, tst_xy, eval_dict, project_name, f'loss_curve_{stem}.png')
            
            return -eval_dict["train"]["mlogloss"][-1]

        optimizer = BayesianOptimization(
            f=black_box_function,
            pbounds=search_space,
            random_state=1,
        )

        optimizer.maximize(init_points=program_setting["init_points"],n_iter=program_setting["n_iter"])
        results = optimizer.res
        results = sorted(results, key=lambda d: d['target'], reverse=True)

        save_list_to_txt(results, os.path.join(project_name, f'_bayse_optimization_results.txt'))
    else:
        _ = iterative_train(program_setting["iterative_train"], params, trn_xy, tst_xy, eval_set, project_name)

    total_time = time.time() - start_time
    print('\nTraining, took {:6.1f} h.'.format(total_time / 3600))
    return None


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train a model with xgboost.')
    parser.add_argument('directory', type=str, help='The directory where the data is located.')
    parser.add_argument('-n', '--n_cpu', type=int, default=1, help='The number of CPUs to use for training.')
    parser.add_argument('-s', '--simple_load', default=True, help='If true only one type of structure is loaded per class.', action='store_true')
    parser.add_argument('-d', '--n_data', type=int, default=-1)
    parser.add_argument('-b', '--do_bayesopt', default=False, help='If the model should be trained with Bayesian optimization.', action='store_true')

    args = parser.parse_args()

    main_train(args.directory, args.n_cpu, args.simple_load, args.n_data, args.do_bayesopt)
n(**sample_paras):
            """Function with unknown internals we wish to maximize.

            This is just serving as an example, for all intents and
            purposes think of the internals of this function, i.e.: the process
            which generates its output values, as unknown.
            """
            for sample_para in sample_paras.keys():
                if sample_para in ["max_depth"]:
                    params["hp"][sample_para] = int(sample_paras[sample_para])
                else:
                    params["hp"][sample_para] = sample_paras[sample_para]

            best, eval_dict = init_xgb_model(params, trn_xy, eval_set)
            
            return -eval_dict["train"]["mlogloss"][-1]

        optimizer = BayesianOptimization(
            f=black_box_function,
            pbounds=search_space,
            random_state=1,
        )

        optimizer.maximize(init_points=program_setting["init_points"],n_iter=program_setting["n_iter"])
        results = optimizer.res
        results = sorted(results, key=lambda d: d['target'], reverse=True)

        save_list_to_txt(results, os.path.join(project_name, f'_bayse_optimization_results.txt'))

        print(f"\nRetraining the top-{program_setting['best_bayes_opt_models']} models from Bayesian optimization.")
        for idx, hps in enumerate(results[:program_setting["best_bayes_opt_models"]]):
            for sample_para in hps["params"].keys():
                if sample_para in ["max_depth"]:
                    params["hp"][sample_para] = int(hps["params"][sample_para])
                else:
                    params["hp"][sample_para] = hps["params"][sample_para]
            iterative_train(1, params, trn_xy, tst_xy, eval_set, project_name, stem=f"bayse_optimization_{idx:05d}")
            
    else:
        _ = iterative_train(program_setting["iterative_train"], params, trn_xy, tst_xy, eval_set, project_name)

    total_time = time.time() - start_time
    print('\nTraining, took {:6.1f} h.'.format(total_time / 3600))
    return None


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train a model with xgboost.')
    parser.add_argument('directory', type=str, help='The directory where the data is located.')
    parser.add_argument('-n', '--n_cpu', type=int, default=1, help='The number of CPUs to use for training.')
    parser.add_argument('-s', '--simple_load', default=True, help='If true only one type of structure is loaded per class.', action='store_true')
    parser.add_argument('-d', '--n_data', type=int, default=-1)
    parser.add_argument('-b', '--do_bayesopt', default=False, help='If the model should be trained with Bayesian optimization.', action='store_true')

    args = parser.parse_args()

    main_train(args.directory, args.n_cpu, args.simple_load, args.n_data, args.do_bayesopt)
