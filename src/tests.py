import numpy as np
from src.hw1.hw1 import KDTree, KNearest, read_spam_dataset, train_test_split, get_precision_recall_accuracy
import matplotlib.pyplot as plt
import matplotlib
import time


def test_kd():
    def true_closest(X_train, X_test, k):
        result = []
        for x0 in X_test:
            bests = list(sorted([(i, np.linalg.norm(x - x0)) for i, x in enumerate(X_train)], key=lambda x: x[1]))
            bests = [i for i, d in bests]
            result.append(bests[:min(k, len(bests))])
        return result

    k = 10
    X_train = np.random.randn(10000, 3)
    # X_test = np.random.randn(10, 30)
    X_test = np.ones((1, 3)) * 4
    tree = KDTree(X_train, leaf_size=10)
    predicted = tree.query(X_test, k=k)
    true = true_closest(X_train, X_test, k=k)

    if np.sum(np.abs(np.array(np.array(predicted).shape) - np.array(np.array(true).shape))) != 0:
        print("Wrong shape")
    else:
        errors = sum([1 for row1, row2 in zip(predicted, true) for i1, i2 in zip(row1, row2) if i1 != i2])
        if errors > 0:
            print("Encounted", errors, "errors")


def plot_roc_curve(X_train, y_train, X_test, y_test, max_k=30):
    positive_samples = sum(1 for y in y_test if y == 0)
    ks = list(range(1, max_k + 1))
    curves_tpr = []
    curves_fpr = []
    colors = []
    for k in ks:
        colors.append([k / ks[-1], 0, 1 - k / ks[-1]])
        knearest = KNearest(k)
        knearest.fit(X_train, y_train)
        p_pred = [p[0] for p in knearest.predict_proba(X_test)]
        tpr = []
        fpr = []
        for w in np.arange(-0.01, 1.02, 0.01):
            y_pred = [(0 if p > w else 1) for p in p_pred]
            tpr.append(sum(1 for yp, yt in zip(y_pred, y_test) if yp == 0 and yt == 0) / positive_samples)
            fpr.append(sum(1 for yp, yt in zip(y_pred, y_test) if yp == 0 and yt != 0) / (len(y_test) - positive_samples))
        curves_tpr.append(tpr)
        curves_fpr.append(fpr)
    plt.figure(figsize = (7, 7))
    for tpr, fpr, c in zip(curves_tpr, curves_fpr, colors):
        plt.plot(fpr, tpr, color=c)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.xlim(-0.01, 1.01)
    plt.ylim(-0.01, 1.01)
    plt.tight_layout()
    plt.show()


def plot_precision_recall(X_train, y_train, X_test, y_test, max_k=30):
    ks = list(range(1, max_k + 1))
    classes = len(np.unique(list(y_train) + list(y_test)))
    precisions = [[] for _ in range(classes)]
    recalls = [[] for _ in range(classes)]
    accuracies = []
    for k in ks:
        classifier = KNearest(k)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        precision, recall, acc = get_precision_recall_accuracy(y_pred, y_test)
        for c in range(classes):
            precisions[c].append(precision[c])
            recalls[c].append(recall[c])
        accuracies.append(acc)

    def plot(x, ys, ylabel, legend=True):
        plt.figure(figsize=(12, 3))
        plt.xlabel("K")
        plt.ylabel(ylabel)
        plt.xlim(x[0], x[-1])
        plt.ylim(np.min(ys) - 0.01, np.max(ys) + 0.01)
        for cls, cls_y in enumerate(ys):
            plt.plot(x, cls_y, label="Class " + str(cls))
        if legend:
            plt.legend()
        plt.tight_layout()
        plt.show()

    plot(ks, recalls, "Recall")
    plot(ks, precisions, "Precision")
    plot(ks, [accuracies], "Accuracy", legend=False)


def test_sub():
    x = np.array([7, 2, 3, 3, 2])
    y = [1, 2]
    print(np.concatenate([x, y]))


def test_spam():
    X, y = read_spam_dataset('hw1/data/spam.csv')
    X_train, y_train, X_test, y_test = train_test_split(X, y, 0.9)
    start = time.time()

    plot_precision_recall(X_train, y_train, X_test, y_test, max_k=20)
    plot_roc_curve(X_train, y_train, X_test, y_test, max_k=20)

    print(f"Elapsed time: {time.time() - start}")


if __name__ == "__main__":
    test_spam()
    #test_sub()
    test_kd()
