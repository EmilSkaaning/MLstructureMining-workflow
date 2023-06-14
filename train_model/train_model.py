import os
import sys
import time
import datetime
import yaml
import xgboost
import numpy as np
from sklearn.metrics import log_loss, accuracy_score
sys.path.append("..")
from utils.data_loader_in_mem import get_data_splits_from_clean_data
from utils.tools import accuracy_top_x, save_dict_to_yaml
from utils.plotting import plot_loss_curve


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
        f'Acc: {acc:.2f}, top 3: {acc_3:.2f}, top5: {acc_5:.2f}, top7: {acc_7:.2f}',
        img_title
    )
    return None

def iterative_train(n_iterations, params, trn_xy, tst_xy, eval_set, project_name):
    booster = None

    for i in range(n_iterations):
        print(f"\nIteration {i+1} of {n_iterations}.\n")
        eval_dict = {}

        booster = xgboost.train(params['hp'],
                                dtrain=trn_xy,
                                num_boost_round=params['num_boost_round'],
                                evals=eval_set,
                                evals_result=eval_dict,
                                early_stopping_rounds=params["early_stopping_rounds"],
                                xgb_model=booster,
                                verbose_eval=250
                                )
        print(f"\nNumber of trees: {len(booster.trees_to_dataframe().Tree.unique())}")

        booster.save_model(os.path.join(project_name, f"xgb_model_{i:09}.bin"))
        save_dict_to_yaml(eval_dict, os.path.join(project_name, f'data_{i:09}.yml'))

        test_model(booster, tst_xy, eval_dict, project_name, f'loss_curve_{i:09}.png')

    return booster


def main_train(directory: str, n_cpu: int=1, simple_load: bool=False) -> None:
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
    print(simple_load)
    trn_xy, vld_xy, tst_xy, eval_set, n_classes = get_data_splits_from_clean_data(directory, project_name, simple_load=simple_load)

    # Define hyperparameters
    hp = {
        "objective": "multi:softprob",
        "eval_metric": "mlogloss",
        "num_class": n_classes,
        "verbosity": 1,
        "nthread": n_cpu,
        "subsample": 1,
        "max_depth": 6,
        "tree_method": "hist",
        "max_bin": 256,
    }

    # Define parameters for xgboost training
    params = {
        'hp': hp,
        'early_stopping_rounds': 5,
        'num_boost_round': 10_000
    }
    save_dict_to_yaml(params, os.path.join(project_name, 'model_parameters.yml'))

    # Train the xgboost model
    start_time = time.time()

    _ = iterative_train(5, params, trn_xy, tst_xy, eval_set, project_name)

    total_time = time.time() - start_time
    print('\nTraining, took {:6.1f} h.'.format(total_time / 3600))
    return None


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train a model with xgboost.')
    parser.add_argument('directory', type=str, help='The directory where the data is located.')
    parser.add_argument('-n', '--n_cpu', type=int, default=1, help='The number of CPUs to use for training.')
    parser.add_argument('-s', '--simple_load', default=False, help='If true only one type of structure is loaded per class.', action='store_true')
    args = parser.parse_args()

    main_train(args.directory, args.n_cpu, args.simple_load)
