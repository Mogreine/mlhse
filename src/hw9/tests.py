from sklearn.datasets import make_blobs, make_moons
import numpy as np
import pandas
import random
import matplotlib.pyplot as plt
import matplotlib
from typing import Callable, Union, NoReturn, Optional, Dict, Any, List
from catboost import CatBoostClassifier, Pool, EFstrType

from src.hw9.hw9 import *


def feature_importance(rfc: RandomForestClassifier, X: np.ndarray, y: np.ndarray):
    n, m = X.shape
    importance = np.zeros((rfc.n_estimators, m))
    for i in range(len(rfc.trees)):
        inds = rfc.trees[i].oob_ids_mask
        X_, y_ = X[inds], y[inds]
        err_oob = np.mean(rfc.predict(X_) != y_)
        for col in range(m):
            X_shuffled = X_.copy()
            np.random.shuffle(X_shuffled[:, col])
            err_oob_col = np.mean(rfc.predict(X_shuffled) != y_)
            importance[i, col] = err_oob_col - err_oob

    importance = np.mean(importance, axis=0)
    return importance


def most_important_features(importance, names, k=20):
    # Выводит названия k самых важных признаков
    indices = np.argsort(importance)[::-1][:k]
    return np.array(names)[indices]


def synthetic_dataset(size):
    X = [(np.random.randint(0, 2), np.random.randint(0, 2), i % 6 == 3,
          i % 6 == 0, i % 3 == 2, np.random.randint(0, 2)) for i in range(size)]
    y = [i % 3 for i in range(size)]
    return np.array(X), np.array(y)


def read_dataset(path):
    dataframe = pandas.read_csv(path, header=0)
    dataset = dataframe.values.tolist()
    random.shuffle(dataset)
    y_age = [row[0] for row in dataset]
    y_sex = [row[1] for row in dataset]
    X = [row[2:] for row in dataset]

    return np.array(X), np.array(y_age), np.array(y_sex), list(dataframe.columns)[2:]


def catboost_test():
    X, y_age, y_sex, features = read_dataset("vk.csv")
    X_train, X_test, y_age_train, y_age_test, y_sex_train, y_sex_test = train_test_split(X, y_age, y_sex,
                                                                                         train_size=0.9)
    X_train, X_eval, y_age_train, y_age_eval, y_sex_train, y_sex_eval = train_test_split(X_train, y_age_train,
                                                                                         y_sex_train, train_size=0.8)

    cat_features = np.arange(X.shape[1])
    cat_features = None
    model = CatBoostClassifier(
        iterations=10000,
        depth=6,
#         min_data_in_leaf=50,
        learning_rate=0.01,
        loss_function='MultiClass',
        cat_features=cat_features,
        eval_metric='Accuracy',
        thread_count=-1,
        verbose=0,
        random_state=41,
        use_best_model=True
    )

    model.fit(X_train, y_age_train,
              eval_set=(X_eval, y_age_eval),
              early_stopping_rounds=1000)

    print("Accuracy:", np.mean(model.predict(X_test).flatten() == y_age_test))

    print("Most important features:")
    imp = model.get_feature_importance(data=Pool(X_train, y_age_train),
                                       type=EFstrType.FeatureImportance,
                                       prettified=False,
                                       thread_count=-1,
                                       verbose=False)

    for i, name in enumerate(most_important_features(imp, features, 10)):
        print(str(i + 1) + ".", name)

    model.fit(X_train, y_sex_train,
              eval_set=(X_eval, y_sex_eval),
              early_stopping_rounds=1000)
    print("Accuracy:", np.mean(model.predict(X_test).flatten() == y_sex_test))
    print("Most important features:")
    imp = model.get_feature_importance(data=Pool(X_train, y_sex_train),
                                       type=EFstrType.FeatureImportance,
                                       prettified=False,
                                       thread_count=-1,
                                       verbose=False)
    for i, name in enumerate(most_important_features(imp, features, 10)):
        print(str(i + 1) + ".", name)


if __name__ == "__main__":
    catboost_test()
