import os, sys, xgboost, shutil, datetime, yaml
sys.path.append("..")
from utils.data_loader import get_data_splits
from utils.tools import accuracy_top_x
from utils.plotting import plot_loss_curve
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import log_loss, accuracy_score


def main(directory: str) -> None:
    ct = str(datetime.datetime.now()).replace(' ', '_').replace(':', '-').replace('.', '-')
    project_name = f'{directory}/results_{ct}'
    os.mkdir(project_name)

    trn_xy, vld_xy, tst_xy, eval_set = get_data_splits(directory, project_name)

    para = {
        'hp': {
            "objective": "multi:softprob",  #'multi:softmax',
            "eval_metric": "mlogloss",  # auc, merror, mlogloss
            "num_class": len(pd.read_csv(os.path.join(directory, "structure_catalog.csv"))),
            "verbosity": 1,
            "nthread": 12,
            "subsample": 1,  # default: 1
            "max_depth": 1000,  # default: 6
            "tree_method": "hist",  # default: "auto",
            "max_bin": 256,  # default: 256
        },
        "early_stopping_rounds": 25,
        'num_boost_round': 999_999_999
    }
    eval_dict = {}

    # Other tree methods including ``hist`` and ``gpu_hist`` also work, but has some caveats
    # as noted in following sections.
    booster = xgboost.train(para['hp'],
        dtrain=trn_xy,
        num_boost_round=para['num_boost_round'],
        evals=eval_set,
        evals_result=eval_dict,
        early_stopping_rounds=para["early_stopping_rounds"]
    )

    booster.save_model(os.path.join(project_name, "xgb_model.bin"))
    pred = booster.predict(tst_xy)
    pred_label = np.argmax(pred, axis=1)
    true_label = np.array(tst_xy.get_label(), dtype=int)

    acc = accuracy_score(true_label, pred_label)
    acc_3 = accuracy_top_x(true_label, pred, 3)
    acc_5 = accuracy_top_x(true_label, pred, 5)
    acc_7 = accuracy_top_x(true_label, pred, 7)

    print('Model test acc: {:.2f} %'.format(acc))
    print('Model test acc top 3: {:.2f} %'.format(acc_3))
    print('Model test acc top 5: {:.2f} %'.format(acc_5))
    print('Model test acc top 7: {:.2f} %'.format(acc_7))

    mlogloss_tst = log_loss(true_label, pred)

    plot_loss_curve(
        eval_dict,
        project_name,
        mlogloss_tst,
        f'Acc: {acc:.2f}, top 3: {acc_3:.2f}, top5: {acc_5:.2f}, top7: {acc_7:.2f}'
    )

    with open(os.path.join(project_name, 'data.yml'), 'w') as outfile:
        yaml.dump(eval_dict, outfile, default_flow_style=False)

    # bst = xgboost.Booster({'nthread': 2})
    # bst.load_model(os.path.join(project_name, "xgb_model.bin"))  # io.BytesIO(model_path))
    #
    # pred = bst.predict(tst_xy)
    # pred_label = np.argmax(pred, axis=1)
    # true_label = np.array(tst_xy.get_label(), dtype=int)
    #
    # acc = accuracy_score(true_label, pred_label)
    # acc_3 = accuracy_top_x(true_label, pred, 3)
    # acc_5 = accuracy_top_x(true_label, pred, 5)
    # acc_7 = accuracy_top_x(true_label, pred, 7)
    #
    # print('Model test acc: {:.2f} %'.format(acc))
    # print('Model test acc top 3: {:.2f} %'.format(acc_3))
    # print('Model test acc top 5: {:.2f} %'.format(acc_5))
    # print('Model test acc top 7: {:.2f} %'.format(acc_7))
    return None


if __name__ == '__main__':
    main('/mnt/c/Users/ETSK/Desktop/XGBOOST_BIG_BOI/test_case/structure_finder_2022-11-09_08-07-32-863123')


    #main('/mnt/c/Users/WindowsVirus/Documents/my_projects/XGBoost/structure_finder_2022-10-26_13-01-32-356873')