import warnings

import numpy as np
import xgboost as xgb
from art.attacks.evasion import ZooAttack
from art.estimators.classification import XGBoostClassifier
from art.utils import load_mnist
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score

from utils.tools import accuracy_top_x

warnings.filterwarnings("ignore")


def get_adversarial_examples(
    model: xgb.XGBClassifier, x_tst: np.ndarray, n_cpu: int = 1
) -> np.ndarray:
    """
    Generate adversarial test examples using the ZOO attack.

    Parameters
    ----------
    model : xgb.XGBClassifier
        The trained XGBoost classifier.
    x_tst : np.ndarray
        The test samples.
    n_cpu : int, optional
        Number of CPUs to use for parallel processing, by default 1.

    Returns
    -------
    np.ndarray
        Adversarial test samples.
    """
    print("\nGenerating adversarial test examples for ZOO attack...")
    art_classifier = XGBoostClassifier(
        model=model, nb_features=x_tst.shape[1], nb_classes=10
    )

    # Create ART Zeroth Order Optimization attack
    zoo = ZooAttack(
        classifier=art_classifier,  # The trained classifier that you wish to attack.
        confidence=0.0,  # Confidence required for an adversarial example to be considered successful.
        targeted=False,  # Whether the attack is targeted. If `False`, the attack will be untargeted.
        learning_rate=1e-1,  # The learning rate for the attack optimization.
        max_iter=20,  # Maximum number of iterations for the attack optimization.
        binary_search_steps=10,  # The number of times to adjust the `initial_const` with binary search. Larger values result in more precise results.
        initial_const=1e-3,  # The initial trade-off constant `c` to use in the optimization. This constant determines the balance between minimizing the perturbation size and the classification loss.
        abort_early=True,  # If `True`, the attack stops if the objective does not improve for some time.
        use_resize=False,  # If `True`, the attack will use resizing on the input image. Resizing can help find adversarial examples faster.
        use_importance=False,  # If `True`, the attack uses the importance of features to guide the search (requires `initial_const` to be more than 0).
        nb_parallel=n_cpu,  # Number of perturbations that should be processed in parallel.
        batch_size=1,  # The size of the batch being fed to the model during a single optimization step.
        variable_h=0.2,  # The maximum perturbation of a feature during an iteration.
    )

    # Generate adversarial samples with ART Zeroth Order Optimization attack
    x_tst_adv = zoo.generate(x_tst)

    return x_tst_adv


def zoo_attach_xgb(
    model: xgb.XGBClassifier, x_tst: np.ndarray, y_tst: np.ndarray, n_cpu: int = 1
) -> list:
    """
    Attack an XGBoost model using ZOO and evaluate its performance.

    Parameters
    ----------
    model : xgb.XGBClassifier
        The trained XGBoost classifier.
    x_tst : np.ndarray
        Test samples.
    y_tst : np.ndarray
        True labels for test samples.
    n_cpu : int, optional
        Number of CPUs to use for parallel processing, by default 1.

    Returns
    -------
    list
        A list containing accuracy scores for the model on adversarial examples.
    """
    x_tst_adv = get_adversarial_examples(model, x_tst, n_cpu)

    pred = model.predict(xgb.DMatrix(x_tst_adv))
    pred_label = np.argmax(pred, axis=1)

    # Calculate accuracy scores
    acc = accuracy_score(y_tst, pred_label)
    acc_3 = accuracy_top_x(y_tst, pred, 3)
    acc_5 = accuracy_top_x(y_tst, pred, 5)
    acc_7 = accuracy_top_x(y_tst, pred, 7)

    print(
        f"ZOO attack test acc: {acc:.2f}%, top 3: {acc_3:.2f}%, top 5: {acc_5:.2f}%, top 7: {acc_7:.2f}%"
    )
    return [acc, acc_3, acc_5, acc_7]
