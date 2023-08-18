import argparse
import datetime
import sys
import os
import time
from typing import Dict, List, Tuple, Union

import numpy as np
import xgboost
from bayes_opt import BayesianOptimization
from sklearn.metrics import accuracy_score, log_loss

sys.path.append("..")

from utils.data_loader_in_mem import get_data_splits_from_clean_data
from utils.plotting import plot_loss_curve
from utils.tools import accuracy_top_x, save_dict_to_yaml, save_list_to_txt
from utils.zoo_attack import zoo_attach_xgb

# Program settings
program_setting: Dict[str, int] = {
    "init_points": 3,
    "n_iter": 3,
    "iterative_train": 3,
}

# Define hyperparameters
hp: Dict[str, Union[str, int, float]] = {
    "objective": "multi:softprob",
    "eval_metric": "mlogloss",
    "verbosity": 1,
    "max_depth": 6,
    "tree_method": "hist",
    "max_bin": 256,
}

# Define parameters for xgboost training
params: Dict[str, Union[Dict, int, float]] = {
    "hp": hp,
    "early_stopping_rounds": 5,
    "num_boost_round": 10,
    "verbose_eval": 250,
    "subsample": 0.5,
}

# Search space for Bayesian optimization
search_space: Dict[str, Tuple[float, float]] = {
    "learning_rate": (0.05, 1.0),
    "min_child_weight": (0.1, 10),
    "max_depth": (3, 6),
    "max_delta_step": (0, 20),
    "subsample": (0.01, 1.0),
    "colsample_bytree": (0.01, 1.0),
    "colsample_bylevel": (0.01, 1.0),
    "reg_lambda": (0, 10.0),
    "reg_alpha": (0, 10.0),
    "gamma": (0, 10.0),
}


def test_model(
    booster: xgboost.Booster,
    tst_xy: Tuple[np.ndarray, np.ndarray],
) -> Tuple[float, List[float]]:
    """
    Test the trained XGBoost model and display results.

    Parameters
    ----------
    booster : xgboost.Booster
        Trained XGBoost model.
    tst_xy : Tuple[np.ndarray, np.ndarray]
        Testing dataset with data and labels.

    Returns
    -------
    Tuple[float, List[float]]
        The log loss of the model on the test dataset and a list containing
        the overall accuracy and top-k accuracies.
    """
    pred = booster.predict(tst_xy)
    pred_label = np.argmax(pred, axis=1)
    true_label = np.array(tst_xy.get_label(), dtype=int)

    # Calculate accuracy scores
    acc = accuracy_score(true_label, pred_label)
    acc_3 = accuracy_top_x(true_label, pred, 3)
    acc_5 = accuracy_top_x(true_label, pred, 5)
    acc_7 = accuracy_top_x(true_label, pred, 7)

    print(
        f"\nTest acc: {acc:.2f}%, top 3: {acc_3:.2f}%, top 5: {acc_5:.2f}%, top 7: {acc_7:.2f}%"
    )

    mlogloss_tst = log_loss(true_label, pred)

    return mlogloss_tst, [acc, acc_3, acc_5, acc_7]


def init_xgb_model(
    params: Dict[str, Union[Dict, int, float]],
    trn_xy: Tuple[np.ndarray, np.ndarray],
    eval_set: Tuple[np.ndarray, np.ndarray],
    booster: xgboost.Booster = None,
) -> Tuple[xgboost.Booster, Dict[str, List[float]]]:
    """
    Initialize and train the XGBoost model.

    Parameters
    ----------
    params : Dict[str, Union[Dict, int, float]]
        XGBoost training parameters.
    trn_xy : Tuple[np.ndarray, np.ndarray]
        Training dataset with data and labels.
    eval_set : Tuple[np.ndarray, np.ndarray]
        Evaluation dataset with data and labels.
    booster : xgboost.Booster, optional
        Existing booster model to continue training, by default None.

    Returns
    -------
    Tuple[xgboost.Booster, Dict[str, List[float]]]
        Trained XGBoost booster and evaluation dictionary.
    """
    eval_dict = {}
    booster = xgboost.train(
        params["hp"],
        dtrain=trn_xy,
        evals=eval_set,
        evals_result=eval_dict,
        num_boost_round=params["num_boost_round"],
        early_stopping_rounds=params["early_stopping_rounds"],
        verbose_eval=params["verbose_eval"],
        xgb_model=booster,
    )

    return booster, eval_dict


def iterative_train(
    n_iterations: int,
    params: dict,
    trn_xy: Tuple[np.ndarray, np.ndarray],
    tst_xy: Tuple[np.ndarray, np.ndarray],
    eval_set: Tuple[np.ndarray, np.ndarray],
    project_name: str,
    stem: str = None,
    tst_tuple: Tuple[np.ndarray, np.ndarray] = None,
    n_cpu: int = 1,
) -> xgboost.Booster:
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
    save_model_dir, save_data_dir = os.path.join(project_name, "models"), os.path.join(
        project_name, "data"
    )
    os.makedirs(save_model_dir, exist_ok=True), os.makedirs(
        save_data_dir, exist_ok=True
    )
    for i in range(n_iterations):
        if stem is None:
            stem = f"{i:09}"
        print(f"\nIteration {i+1} of {n_iterations}.\n")

        booster, eval_dict = init_xgb_model(params, trn_xy, eval_set, booster)
        booster.save_model(os.path.join(save_model_dir, f"xgb_model_{stem}.bin"))

        mlogloss_tst, tst_acc = test_model(booster, tst_xy)
        plot_title = f"Loss: {mlogloss_tst:.2f}\nAcc: {tst_acc[0]:.2f}, top 3: {tst_acc[1]:.2f}, top5: {tst_acc[2]:.2f}, top7: {tst_acc[3]:.2f}"

        if tst_tuple is not None:
            zoo_attack_tst_acc = zoo_attach_xgb(
                booster, tst_tuple[0], tst_tuple[1], n_cpu
            )
            plot_title += f"\nZOO attack acc: {zoo_attack_tst_acc[0]:.2f}, top 3: {zoo_attack_tst_acc[1]:.2f}, top5: {zoo_attack_tst_acc[2]:.2f}, top7: {zoo_attack_tst_acc[3]:.2f}"
        save_dict_to_yaml(eval_dict, os.path.join(save_data_dir, f"data_{stem}.yml"))

        plot_loss_curve(
            eval_dict, project_name, mlogloss_tst, plot_title, f"loss_curve_{stem}.png"
        )

        stem = None
    return booster


def main_train(
    directory: str,
    n_cpu: int = 1,
    simple_load: bool = False,
    n_data: int = 100,
    do_bayesopt: bool = False,
    do_zoo_attack: bool = False,
) -> None:
    """
    Function to train the model and save the results.

    Parameters
    ----------
    directory : str
        The directory where the data is located.
    n_cpu : int, optional
        The number of CPUs to use for training. Default is 1.
    simple_load : bool, optional
        If true, only one type of structure is loaded per class. Default is False.
    n_data : int, optional
        Number of data points to be used for training. Default is 100.
    do_bayesopt : bool, optional
        If the model should be trained with Bayesian optimization. Default is False.
    do_zoo_attack : bool, optional
        If the trained model should be attacked using Zeroth Order Optimisation (ZOO)

    Returns
    -------
    None
    """

    # Create a new project directory with the current timestamp
    ct = (
        str(datetime.datetime.now())
        .replace(" ", "_")
        .replace(":", "-")
        .replace(".", "-")
    )
    project_name = os.path.join(directory, f"results_{ct}")
    os.makedirs(project_name, exist_ok=True)

    # Get the data splits

    (
        trn_xy,
        vld_xy,
        tst_xy,
        eval_set,
        n_classes,
        tst_tuple,
    ) = get_data_splits_from_clean_data(
        directory, project_name, simple_load=simple_load, n_data=n_data
    )
    if do_zoo_attack == False:
        tst_tuple = None
    params["hp"]["num_class"] = n_classes
    params["hp"]["nthread"] = n_cpu

    # Save parameters
    save_dict_to_yaml(params, os.path.join(project_name, "_model_parameters.yml"))
    save_dict_to_yaml(
        program_setting, os.path.join(project_name, f"_program_setting.yml")
    )

    # Train the xgboost model
    start_time = time.time()

    if do_bayesopt:
        print(f"\nPerforming Bayesian optimization.\n")
        save_dict_to_yaml(
            search_space, os.path.join(project_name, f"_search_space.yml")
        )
        params["hp"]["verbosity"] = 0
        params["verbose_eval"] = 0

        def bayesian_optimization_function(**sample_paras: Union[int, float]) -> float:
            """
            Function to perform Bayesian optimization for XGBoost model training.

            Parameters
            ----------
            **sample_paras : Union[int, float]
                Sample parameters for Bayesian optimization.

            Returns
            -------
            float
                Negative value of the evaluation metric (mlogloss) for the trained model.
            """
            # Update the parameters based on the samples provided
            for sample_para, value in sample_paras.items():
                if sample_para == "max_depth":
                    params["hp"][sample_para] = int(value)
                else:
                    params["hp"][sample_para] = value

            booster, eval_dict = init_xgb_model(params, trn_xy, eval_set)
            stem = f"bayse_optimization_{len(optimizer.res):05d}"  # unique identifier based on number of iterations so far

            save_model_dir, save_data_dir, save_hp_dir = (
                os.path.join(project_name, "models"),
                os.path.join(project_name, "data"),
                os.path.join(project_name, "model_hp"),
            )
            os.makedirs(save_model_dir, exist_ok=True), os.makedirs(
                save_data_dir, exist_ok=True
            ), os.makedirs(save_hp_dir, exist_ok=True)
            booster.save_model(os.path.join(save_model_dir, f"xgb_model_{stem}.bin"))
            save_dict_to_yaml(
                eval_dict, os.path.join(save_data_dir, f"data_{stem}.yml")
            )
            save_dict_to_yaml(params["hp"], os.path.join(save_hp_dir, f"hp_{stem}.yml"))

            mlogloss_tst, tst_acc = test_model(booster, tst_xy)
            plot_title = f"Loss: {mlogloss_tst:.2f}\nAcc: {tst_acc[0]:.2f}, top 3: {tst_acc[1]:.2f}, top5: {tst_acc[2]:.2f}, top7: {tst_acc[3]:.2f}"

            if tst_tuple is not None:
                zoo_attack_tst_acc = zoo_attach_xgb(
                    booster, tst_tuple[0], tst_tuple[1], n_cpu
                )
                plot_title += f"\nZOO attack acc: {zoo_attack_tst_acc[0]:.2f}, top 3: {zoo_attack_tst_acc[1]:.2f}, top5: {zoo_attack_tst_acc[2]:.2f}, top7: {zoo_attack_tst_acc[3]:.2f}"
            save_dict_to_yaml(
                eval_dict, os.path.join(save_data_dir, f"data_{stem}.yml")
            )

            plot_loss_curve(
                eval_dict,
                project_name,
                mlogloss_tst,
                plot_title,
                f"loss_curve_{stem}.png",
            )

            return -eval_dict["train"]["mlogloss"][-1]

        optimizer = BayesianOptimization(
            f=bayesian_optimization_function,
            pbounds=search_space,
            random_state=1,
        )

        optimizer.maximize(
            init_points=program_setting["init_points"], n_iter=program_setting["n_iter"]
        )
        results = optimizer.res
        results = sorted(results, key=lambda d: d["target"], reverse=True)
        save_list_to_txt(
            results, os.path.join(project_name, f"_bayse_optimization_results.txt")
        )
    else:
        _ = iterative_train(
            program_setting["iterative_train"],
            params,
            trn_xy,
            tst_xy,
            eval_set,
            project_name,
            tst_tuple=tst_tuple,
            n_cpu=n_cpu,
        )

    total_time = time.time() - start_time
    print("\nTraining, took {:6.1f} h.".format(total_time / 3600))
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model with xgboost.")
    parser.add_argument(
        "directory", type=str, help="The directory where the data is located."
    )
    parser.add_argument(
        "-n",
        "--n_cpu",
        type=int,
        default=1,
        help="The number of CPUs to use for training.",
    )
    parser.add_argument("-d", "--n_data", type=int, default=-1)

    parser.add_argument(
        "-s",
        "--simple_load",
        default=False,
        action="store_true",
        help="If true only one type of structure is loaded per class.",
    )
    parser.add_argument(
        "-b",
        "--do_bayesopt",
        default=False,
        action="store_true",
        help="If the model should be trained with Bayesian optimization.",
    )
    parser.add_argument(
        "-z",
        "--do_zoo_attack",
        default=False,
        action="store_true",
        help="If the trained model should be attacked using Zeroth Order Optimisation (ZOO)",
    )

    args = parser.parse_args()
    main_train(
        args.directory,
        args.n_cpu,
        args.simple_load,
        args.n_data,
        args.do_bayesopt,
        args.do_zoo_attack,
    )
