import os, sys
#import lightgbm as lgb
import xgboost as xgb
import numpy as np
import time
from sklearn.metrics import accuracy_score
import shap
import matplotlib.pyplot as plt

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from tools import data_tools

def train_LightGBM(train_data, valid_data, param):
    print('\nTraining LightGBM')
    model = lgb.train(params=param,
                      train_set=train_data,
                      valid_sets=valid_data,
                      valid_names=['valid'],
                      verbose_eval=False,
                      early_stopping_rounds=10
                      )

    return model

def train_XGBoost(train_data, valid_data, param):
    print('\nTraining XGBoost')

    evallist = [(train_data, 'train'), (valid_data, 'valid')]
    eval_dict = {}
    model= xgb.train(param,
                     train_data,
                     early_stopping_rounds=25,
                     evals=evallist,
                     evals_result=eval_dict,
                     num_boost_round=20000,
                     #verbose_eval=False
                     )

    return model, eval_dict, model.get_score()


RANDOM_STATE = 42

if __name__ == '__main__':
    path = '/home/ekjaer/Projects/db/CIF_finder_data/data/ext_metal_oxides_S_H_1_11_AA/split_00100_w_abc_uiso'
    savePath = '/home/ekjaer/Projects/CIF_Finder/CIF_Finder/preds/ext_metal_oxides_S_H_1_11_AA'
    #'E:/ml_packages/CIF_finder/CIF_Finder/preds/split_00100_w_qmax_qmin_abc_ext'#split_00100_w_qmax_qmin_abc_ext'
    stem_filename = 'metal_S_H_1_11_AA_split_00100_w_abc_uiso'
    droplist = ['xyz', 'a', 'b', 'c', 'alpha', 'beta', 'gamma', 'Uiso', 'Psize', 'rmin', 'rmax', 'rstep', 'qmin', 'qmax','qdamp', 'delta2']

    do_LightGBM = False
    do_XGBoost = True
    do_load_check = False

    numFiles = None
    sims = None
    ncpus = 40

    try:
        ncpus = int(os.environ["SLURM_JOB_CPUS_PER_NODE"])
        print("\nSLURM, number of node CPUs: {}".format(ncpus))
    except KeyError:
        print("\nNumber of CPUs: {}".format(ncpus))

    param_LightGBM = {'verbose' : -1,
                      'random_state' : RANDOM_STATE,
                      'learning_rate': 0.01,  # 0.01
                      'application': 'multiclass',
                      'num_class': numFiles,
                      'n_jobs': ncpus,
                      }


    param_XGBoost = {'objective': 'multi:softprob',
                     'random_state' : RANDOM_STATE,
                     'learning_rate': 0.01,  # 0.01
                     'num_class': numFiles,
                     #'n_jobs': ncpus,
                     #'tree_method': 'gpu_hist',
                     'tree_method': 'hist',
                     'subsample': 0.5,
                    }


    xTot, yTot, num_classes, sims, feature_n = data_tools.get_csv(path, numFiles, sims, droplist, stem_filename, savePath)
    X_train, X_valid, X_test, y_train, y_valid, y_test = data_tools.data_split(xTot, yTot, sims)

    param_LightGBM['num_class'] = num_classes
    param_XGBoost['num_class'] = num_classes

    # # #
    #
    # Train LightGBM model
    #
    # # #
    if do_LightGBM:
        start_time = time.time()

        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)
        test_data = lgb.Dataset(X_test, label=y_test)

        model = train_LightGBM(train_data, valid_data, param_LightGBM)
        model.save_model('{}/LightGBM_Model_{}_{}.txt'.format(savePath,numFiles, sims))

        preds = model.predict(X_test)
        preds = np.argmax(preds, axis=1)

        acc = accuracy_score(y_test, preds)
        print('Model test acc: {:.2f} %'.format(acc))
        end_time = time.time()
        print('Took: {:.2f} m'.format((end_time-start_time)/60))


    # # #
    #
    # Train XGBoost model
    #
    # # #
    if do_XGBoost:
        start_time = time.time()
        train_data = xgb.DMatrix(X_train, label=y_train, feature_names=feature_n)
        valid_data = xgb.DMatrix(X_valid, label=y_valid, feature_names=feature_n)
        test_data = xgb.DMatrix(X_test, label=y_test, feature_names=feature_n)
        # specify parameters via map

        model, loss_dict, weights = train_XGBoost(train_data, valid_data, param_XGBoost)
        model.save_model('{}/XGBoost_model_{}.bin'.format(savePath, stem_filename))




        preds = model.predict(test_data)


        acc = accuracy_score(y_test, np.argmax(preds, axis=1))
        print('Model test acc: {:.2f} %'.format(acc))
        top3 = data_tools.correctInTopX(preds, y_test, topBest=3)
        print('Model test top3 acc: {:.2f} %'.format(top3))
        top5 = data_tools.correctInTopX(preds, y_test)
        print('Model test top5 acc: {:.2f} %'.format(top5))
        try:
            epochs = np.arange(len(loss_dict['train']['mlogloss']))+1
            plt.plot(epochs, loss_dict['train']['mlogloss'], label='training')
            plt.plot(epochs, loss_dict['valid']['mlogloss'], label='validation')
        except KeyError:
            epochs = np.arange(len(loss_dict['train']['merror']))+1
            plt.plot(epochs, loss_dict['train']['merror'], label='training')
            plt.plot(epochs, loss_dict['valid']['merror'], label='validation')
        plt.title('Acc: {:.2f}, top3: {:.2f}, top5 {:.2f}'.format(acc, top3, top5))
        plt.legend()
        plt.tight_layout()
        plt.savefig('{}/loss_{}.png'.format(savePath, stem_filename), dpi=300)

        """shap_values = shap.TreeExplainer(model).shap_values(X_test)
        import matplotlib.pyplot as plt
        f = plt.figure()
        shap.summary_plot(shap_values[1], X_test)
        plt.tight_layout()
        f.savefig("summary_plot1.png", dpi=300)"""

        """from sklearn.metrics import confusion_matrix # plot_confusion_matrix
        np.set_printoptions(precision=2)
        plt.figure()

        titles_options = [("Confusion matrix, without normalization", None),
                          ("Normalized confusion matrix", 'true')]
        disp = confusion_matrix(model, X_test, y_test,
                                     display_labels=['0', '1', '2','3','4','5'],
                                     cmap=plt.cm.Blues,
                                     normalize=normalize)


        plt.show()"""

        end_time = time.time()
        print('Took: {:.2f} m'.format((end_time-start_time)/60))



    # # #
    #
    # Model save/load check
    #
    # # #
    if do_load_check:
        print('\nLoad model check')
        model = lgb.Booster(model_file='{}/LightGBM_Model_{}_{}.txt'.format(savePath,numFiles,sims))
        preds = model.predict(X_test)
        preds = np.argmax(preds, axis=1)

        acc = accuracy_score(y_test, preds)
        print('LightGBM test acc: {:.2f} %'.format(acc))

        model = xgb.Booster()  # init model
        model.load_model('{}/XGBoost_model_{}_{}.bin'.format(savePath,numFiles, sims))  # load data
        preds = model.predict(test_data)

        acc = accuracy_score(y_test, preds)
        print('XGBoost test acc: {:.2f} %'.format(acc))

    print(stem_filename)



