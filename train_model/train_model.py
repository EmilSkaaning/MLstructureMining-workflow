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
from utils.tools import accuracy_top_x
from utils.plotting import plot_loss_curve

def main_train(directory: str, n_cpu: int=1) -> None:
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
    trn_xy, vld_xy, tst_xy, eval_set, n_classes = get_data_splits_from_clean_data(directory, project_name)

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
        'early_stopping_rounds': 1,
        'num_boost_round': 999_999_999
    }

    eval_dict = {}

    # Train the xgboost model
    start_time = time.time()
    booster = xgboost.train(params['hp'], dtrain=trn_xy, num_boost_round=params['num_boost_round'], evals=eval_set, evals_result=eval_dict, early_stopping_rounds=params["early_stopping_rounds"])

    total_time = time.time() - start_time
    print('\nTraining, took {:6.1f} h.'.format(total_time / 3600))

    # Save the model
    booster.save_model(os.path.join(project_name, "xgb_model.bin"))

    # Predict on the test set
    pred = booster.predict(tst_xy)
    pred_label = np.argmax(pred, axis=1)
    true_label = np.array(tst_xy.get_label(), dtype=int)

    # Calculate accuracy scores
    acc = accuracy_score(true_label, pred_label)
    acc_3 = accuracy_top_x(true_label, pred, 3)
    acc_5 = accuracy_top_x(true_label, pred, 5)
    acc_7 = accuracy_top_x(true_label, pred, 7)

    print('Model test acc: {:.2f} %'.format(acc))
    print('Model test acc top 3: {:.2f} %'.format(acc_3))
    print('Model test acc top 5: {:.2f} %'.format(acc_5))
    print('Model test acc top 7: {:.2f} %'.format(acc_7))

    mlogloss_tst = log_loss(true_label, pred)

    # Plot loss curve
    plot_loss_curve(
        eval_dict,
        project_name,
        mlogloss_tst,
        f'Acc: {acc:.2f}, top 3: {acc_3:.2f}, top5: {acc_5:.2f}, top7: {acc_7:.2f}'
    )

    # Save the evaluation dictionary to a yaml file
    with open(os.path.join(project_name, 'data.yml'), 'w') as outfile:
        yaml.dump(eval_dict, outfile, default_flow_style=False)

    return None


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train a model with xgboost.')
    parser.add_argument('directory', type=str, help='The directory where the data is located.')
    parser.add_argument('-n', '--n_cpu', type=int, default=1, help='The number of CPUs to use for training.')
    args = parser.parse_args()

    main_train(args.directory, args.n_cpu)
