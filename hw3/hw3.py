from argparse import ArgumentParser
from json.tool import main
from tkinter.messagebox import YES
from typing import Tuple, Union, List, Any
from xml.etree.ElementInclude import include
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression

def data_preprocessing(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Return the preprocessed data (X_train, X_test, y_train, y_test). 
    You will need to remove the "Email No." column since it is not a valid feature.
    """
    n = data.shape[0]
    thres = int(n * 0.8)

    X_train = data.drop(columns= ['Email No.', 'Prediction']).head(thres)
    Y_train = data[['Prediction']].head(thres)
    X_test = data.drop(columns= ['Email No.', 'Prediction']).tail(n - thres)
    Y_test = data[['Prediction']].tail(n - thres)


    #X_train.reset_index(inplace = True, drop = True)
    #Y_train.reset_index(inplace = True, drop = True)
    X_test.reset_index(inplace = True, drop = True)
    Y_test.reset_index(inplace = True, drop = True)

    # print(X_train)
    # print(X_test)
    # print(Y_train)
    # print(Y_test)

    return X_train, X_test, Y_train, Y_test

def confusion(a, b):
    return 1 - (a/(a+b))**2 - (b/(a+b))**2

def total_confusion(c, d, e, f):
    return (c + d) / (c + d + e + f) * confusion(c, d) + (e + f) / (c + d + e + f) * confusion(e, f)

class DecisionTree:
    "Add more of your code here if you want to"
    def __init__(self, X = None, y = None):
        self.left = None
        self.right = None
        self.key = None
        self.thres = None
        self.val = None
        if type(X) != type(None) and type(y) != type(None):
            self.fit(X, y)
    
    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        "Fit the model with training data"
        n = y.shape[0]

        for i in range(1, n):
            if y['Prediction'][i] != y['Prediction'][i-1]:
                break
        else:
            # All the predictions are identical, it's a leaf node
            self.val = y['Prediction'][0]
            return
        
        keys = X.keys()
        minConf = 1000

        for key in keys:

            subdf = list(zip(X[key], y['Prediction']))
            subdf.sort()

            Lpos, Lneg, Rpos, Rneg = 0, 0, 0, 0
            
            for i in range(n):
                if subdf[i][1] == 1:
                    Rpos += 1
                else:
                    Rneg += 1

            for idx in range(n):
                if idx > 0 and subdf[idx][0] != subdf[idx-1][0]:
                    conf = total_confusion(Lpos, Lneg, Rpos, Rneg)
                    if conf < minConf:
                        minConf = conf
                        self.key = key
                        self.thres = (subdf[idx][0] + subdf[idx-1][0]) / 2
                
                if subdf[idx][1] == 1:
                    Lpos += 1
                    Rpos -= 1
                else:
                    Lneg += 1
                    Rneg -= 1
                
        if minConf == 1000:
            # all the testcases in X are identical but with inconsistent y
            cnt = 0
            for i in range(n):
                cnt += y['Prediction'][i]
            self.val = 1 if cnt > 0 else -1
            return

        # maindf.sort_values(by = self.key)
        idx = []
        for i in range(n):
            if X[self.key][i] < self.thres:
                idx.append(i)

        lX = X.iloc[idx]
        rX = X.drop(index = idx)
        ly = y.iloc[idx]
        ry = y.drop(index = idx)
        lX.reset_index(inplace = True, drop = True)
        rX.reset_index(inplace = True, drop = True)
        ly.reset_index(inplace = True, drop = True)
        ry.reset_index(inplace = True, drop = True)
        self.left = DecisionTree(lX, ly)
        self.right = DecisionTree(rX, ry)


    def subpredict(self, X: pd.Series) -> Any:
        if self.val != None:
            return self.val
        if X[self.key] < self.thres:
            return self.left.subpredict(X)
        else:
            return self.right.subpredict(X)


    def predict(self, X: pd.DataFrame) -> Any:
        "Make predictions for the testing data"
        n = X.shape[0]
        y = []
        for i in range(n):
            df = X.iloc[i]
            y.append(self.subpredict(df))

        y = pd.DataFrame({'Prediction': y})
        return y

class RandomForest:
    "Add more of your code here if you want to"
    def __init__(self, seed: int = 42, num_trees: int = 5):
        self.num_trees = num_trees
        np.random.seed(seed)

    def bagging(self, X: pd.DataFrame, y: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        "DO NOT modify this function. This function is deliberately given to make your result reproducible."
        index = np.random.randint(0, X.shape[0], int(X.shape[0] / 2))
        return X.iloc[index, :], y.iloc[index]

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        self.trees = []
        for _ in range(self.num_trees):
            Xf, yf = self.bagging(X, y)
            Xf.reset_index(inplace = True, drop = True)
            yf.reset_index(inplace = True, drop = True)
            self.trees.append(DecisionTree(Xf, yf))
        return
    

    def predict(self, X) -> Any:
        y = []
        n = X.shape[0]

        for i in range(n):
            tmp = 0
            xf = X.iloc[i]
            for j in range(self.num_trees):
                tmp += self.trees[j].subpredict(xf)
            y.append(-1 if tmp < 0 else 1)
            
        return pd.DataFrame({'Prediction': y})
        

def accuracy_score(y_pred: Any, y_label: Any) -> float:
    """
    y_pred: (1d array-like) your prediction
    y_label: (1d array-like) the groundtruth label
    Return the accuracy score
    """
    n, ac = len(y_pred), 0

    for i in range (n):
        if y_pred['Prediction'][i] == y_label['Prediction'][i]:
            ac += 1
    return ac / n


def f1_score(y_pred: Any, y_label: Any) -> float:
    """
    y_pred: (1d array-like) your prediction
    y_label: (1d array-like) the groundtruth label
    Return the F1 score
    """
    TP, FP, TN, FN = 0, 0, 0, 0

    for i in range (len(y_pred)):
        if y_pred['Prediction'][i] == 1 and y_label['Prediction'][i] == 1:
            TP += 1
        elif y_pred['Prediction'][i] == 1 and y_label['Prediction'][i] == -1:
            FP += 1
        elif y_pred['Prediction'][i] == -1 and y_label['Prediction'][i] == 1:
            FN += 1
        elif y_pred['Prediction'][i] == -1 and y_label['Prediction'][i] == -1:
            TN += 1
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    return 2 * (Precision * Recall) / (Precision + Recall)


def cross_validation(model: Union[LogisticRegression, DecisionTree, RandomForest], X: pd.DataFrame, y: pd.DataFrame, folds: int = 5) -> Tuple[float, float]:
    """
    Test the generalizability of the model with 5-fold cross validation
    Return the mean accuracy and F1 score
    """
    n = len(X)

    acc, f1 = 0, 0 

    for fd in range(folds):
        lb = (n * fd) // folds
        ub = (n * (fd + 1)) // folds

        X_train = X.drop(index = range(lb, ub))
        Y_train = y.drop(index = range(lb, ub))

        X_test = X.iloc[lb: ub]
        Y_test = y.iloc[lb: ub]

        X_train.reset_index(inplace = True, drop = True)
        Y_train.reset_index(inplace = True, drop = True)
        X_test.reset_index(inplace = True, drop = True)
        Y_test.reset_index(inplace = True, drop = True)

        model.fit(X_train, Y_train)
        Y_res = model.predict(X_test)
        if type(Y_res) != type(Y_test):
            Y_res = pd.DataFrame({'Prediction': Y_res})
        acc += accuracy_score(Y_res, Y_test)
        f1 += f1_score(Y_res, Y_test)

        # For verify
        '''
        Y_res = model.predict(X_train)
        if type(Y_res) != type(Y_train):
            Y_res = pd.DataFrame({'Prediction': Y_res})
        print("Verifying... acc = %f, f1 = %f" % (accuracy_score(Y_res, Y_train), f1_score(Y_res, Y_train)))
        '''

    return (acc / folds, f1 / folds)


def tune_random_forest(choices: List[int], X: pd.DataFrame, y: pd.DataFrame) -> int:
    """
    choices: List of candidates for the number of decision trees in the random forest
    Return the best choice
    """
    bestf1 = -1
    bestnum = None
    for num in choices:
        tree = RandomForest(num_trees = num)
        acc, f1 = cross_validation(tree, X, y)
        print('RandomForest with %d trees: acc = %f, f1 = %f' % (num, acc, f1))
        if f1 > bestf1:
            bestnum = f1
            bestnum = num

    return bestnum

def main(args):
    """
    This function is provided as a head start
    TA will use his own main function at test time.
    """
    data = pd.read_csv(args.data_path)
    print(data.head())
    print(data['Prediction'].value_counts())
    X_train, X_test, y_train, y_test = data_preprocessing(data)

    logistic_regression = LogisticRegression(solver='liblinear', max_iter=500)
    decision_tree = DecisionTree()
    random_forest = RandomForest()
    models = [logistic_regression, decision_tree, random_forest]

    best_f1, best_model = -1, None
    for model in models:
        accuracy, f1 = cross_validation(model, X_train, y_train, 5)
        print(accuracy, f1)
        if f1 > best_f1:
            best_f1, best_model = f1, model
    
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    if type(y_pred) != type(y_test):
        y_pred = pd.DataFrame({'Prediction': y_pred})
    
    print(accuracy_score(y_pred, y_test), f1_score(y_pred, y_test))
    print(tune_random_forest([5, 11, 17], X_train, y_train))

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./emails.csv')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main(parse_arguments())