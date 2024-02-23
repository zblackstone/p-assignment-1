import numpy as np
import seaborn as sns
from numpy.typing import NDArray
from typing import Any
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, f1_score, make_scorer, precision_score, recall_score, top_k_accuracy_score
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, KFold, ShuffleSplit
from sklearn.svm import SVC
from sklearn.utils import compute_class_weight
import utils as u
import new_utils as nu
"""
   In the first two set of tasks, we will narrowly focus on accuracy - 
   what fraction of our predictions were correct. However, there are several 
   popular evaluation metrics. You will learn how (and when) to use these evaluation metrics.
"""


# ======================================================================
class Section3:
    def __init__(
        self,
        normalize: bool = True,
        frac_train=0.2,
        seed=42,
    ):
        self.seed = seed
        self.normalize = normalize

    def analyze_class_distribution(self, y: NDArray[np.int32]) -> dict[str, Any]:
        """
        Analyzes and prints the class distribution in the dataset.

        Parameters:
        - y (array-like): Labels dataset.

        Returns:
        - dict: A dictionary containing the count of elements in each class and the total number of classes.
        """
        # Your code here to analyze class distribution
        # Hint: Consider using collections.Counter or numpy.unique for counting

        uniq, counts = np.unique(y, return_counts=True)
        print(f"{uniq=}")
        print(f"{counts=}")
        print(f"{np.sum(counts)=}")

        return {
            "class_counts": {},  # Replace with actual class counts
            "num_classes": 0,  # Replace with the actual number of classes
        }

    # --------------------------------------------------------------------------
    """
    A. Using the same classifier and hyperparameters as the one used at the end of part 2.B. 
       Get the accuracies of the training/test set scores using the top_k_accuracy score for k=1,2,3,4,5. 
       Make a plot of k vs. score for both training and testing data and comment on the rate of accuracy change. 
       Do you think this metric is useful for this dataset?
    """
    def partA(
        self,
        Xtrain: NDArray[np.floating],
        ytrain: NDArray[np.int32],
        Xtest: NDArray[np.floating],
        ytest: NDArray[np.int32],
    ) -> tuple[
        dict[Any, Any],
        NDArray[np.floating],
        NDArray[np.int32],
        NDArray[np.floating],
        NDArray[np.int32],
    ]:
        """ """
        clf = LogisticRegression(max_iter=300, multi_class='ovr', n_jobs=-1, random_state=self.seed, tol = 0.01)
        clf.fit(Xtrain, ytrain)

        k_values = [1, 2, 3, 4, 5]
        score_train = []
        score_test = []

        for k in k_values:
            topk_train = top_k_accuracy_score(ytrain, clf.predict_proba(Xtrain), k=k)
            topk_test = top_k_accuracy_score(ytest, clf.predict_proba(Xtest), k=k)
            score_train.append(k, topk_train)
            score_test.append(k, topk_test)

        plt.figure(figsize=(8, 5))
        plt.plot(k_values, score_train, label='Training Score', marker='o')
        plt.plot(k_values, score_test, label='Testing Score', marker='o')
        plt.title('Top-k Accuracy vs. k')
        plt.xlabel('k')
        plt.ylabel('Top-k Accuracy')
        plt.legend()
        plt.grid(True)
        plt.show()
        answer = {
            'clf': clf,
            'plot_k_vs_score_train': score_train,
            'plot_k_vs_score_test': score_test,
            'text_rate_accuracy_change': 'The rate of accuracy change starts fast from 1 to 2 but slows down dramatically after 3',
            'text_is_topk_useful_and_why': 'Top K is useful in the sense of finding which K value is the best fit given the parameters and computational costs.',
        }
        """
        
        # `answer` is a dictionary with the following keys:
        - integers for each topk (1,2,3,4,5)
        - "clf" : the classifier
        - "plot_k_vs_score_train" : the plot of k vs. score for the training data, 
                                    a list of tuples (k, score) for k=1,2,3,4,5
        - "plot_k_vs_score_test" : the plot of k vs. score for the testing data
                                    a list of tuples (k, score) for k=1,2,3,4,5

        # Comment on the rate of accuracy change for testing data
        - "text_rate_accuracy_change" : the rate of accuracy change for the testing data

        # Comment on the rate of accuracy change
        - "text_is_topk_useful_and_why" : provide a description as a string

        answer[k] (k=1,2,3,4,5) is a dictionary with the following keys: 
        - "score_train" : the topk accuracy score for the training set
        - "score_test" : the topk accuracy score for the testing set
        """
    
        return answer, Xtrain, ytrain, Xtest, ytest
    """"
    # --------------------------------------------------------------------------
    
    B. Repeat part 1.B but return an imbalanced dataset consisting of 90% of all 9s removed.  Also convert the 7s to 0s and 9s to 1s.
    """

    def partB(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
        Xtest: NDArray[np.floating],
        ytest: NDArray[np.int32],
    ) -> tuple[
        dict[Any, Any],
        NDArray[np.floating],
        NDArray[np.int32],
        NDArray[np.floating],
        NDArray[np.int32],
    ]:
        """"""
        X, y, Xtest, ytest = u.prepare_data() 

        Xtrain, ytrain = u.filter_out_7_9s(X, y)
        Xtest, ytest = u.filter_out_7_9s(Xtest, ytest)

        nines_indices = np.where(ytrain == 9)[0]
        np.random.seed(self.seed) 
        nines_to_remove = np.random.choice(nines_indices, size=int(len(nines_indices) * 0.9), replace=False)

        Xtrain = np.delete(Xtrain, nines_to_remove, axis=0)
        ytrain = np.delete(ytrain, nines_to_remove)


        ytrain = np.where(ytrain == 7, 0, ytrain)
        ytrain = np.where(ytrain == 9, 1, ytrain)
        ytest = np.where(ytest == 7, 0, ytest)
        ytest = np.where(ytest == 9, 1, ytest)

        Xtrain = nu.scale_data(Xtrain)
        Xtest = nu.scale_data(Xtest)


        answer = {
            "length_Xtrain": len(Xtrain),
            "length_Xtest": len(Xtest),
            "length_ytrain": len(ytrain),
            "length_ytest": len(ytest),
            "max_Xtrain": Xtrain.max(),
            "max_Xtest": Xtest.max(),
        }

        # Answer is a dictionary with the same keys as part 1.B

        return answer, X, y, Xtest, ytest

    # --------------------------------------------------------------------------
    """
    C. Repeat part 1.C for this dataset but use a support vector machine (SVC in sklearn). 
        Make sure to use a stratified cross-validation strategy. In addition to regular accuracy 
        also print out the mean/std of the F1 score, precision, and recall. As usual, use 5 splits. 
        Is precision or recall higher? Explain. Finally, train the classifier on all the training data 
        and plot the confusion matrix.
        Hint: use the make_scorer function with the average='macro' argument for a multiclass dataset. 
    """

    def partC(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
        Xtest: NDArray[np.floating],
        ytest: NDArray[np.int32],
    ) -> dict[str, Any]:
        """"""

        # Enter your code and fill the `answer` dictionary
        answer = {}

        clf = SVC(random_state = self.seed)
        cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = self.seed)
        
        scoring = {
            'accuracy': 'accuracy',
            'precision': make_scorer(precision_score, average='macro'),
            'recall': make_scorer(recall_score, average='macro'),
            'f1': make_scorer(f1_score, average='macro')
        }

        scores = cross_validate(clf, X, y, cv=cv, scoring=scoring, return_train_score=False)

        scores_mean = {metric: np.mean(scores[f'test_{metric}']) for metric in scoring.keys()}
        scores_std = {metric: np.std(scores[f'test_{metric}']) for metric in scoring.keys()}

        clf.fit(X, y)
        y_pred_train = clf.predict(X)
        conf_matrix_train = confusion_matrix(y, y_pred_train)

        y_pred_test = clf.predict(Xtest)
        conf_matrix_test = confusion_matrix(ytest, y_pred_test)

        plt.figure(figsize=(10, 7))
        plt.heatmap(conf_matrix_train, annot=True, fmt='g')
        plt.title('confusion matrix for training set')
        plt.xlabel('pred. labels')
        plt.ylabel('true labels')
        plt.show()

        is_precision_higher_than_recall = scores_mean['precision'] > scores_mean['recall']

        answer = {
           'scores': {
        'mean_accuracy': scores_mean['accuracy'],
        'mean_recall': scores_mean['recall'],
        'mean_precision': scores_mean['precision'],
        'mean_f1': scores_mean['f1'],
        'std_accuracy': scores_std['accuracy'],
        'std_recall': scores_std['recall'],
        'std_precision': scores_std['precision'],
        'std_f1': scores_std['f1']},
        'cv': 'StratifiedKFold',
        'clf': 'SVC',
        'is_precision_higher_than_recall': is_precision_higher_than_recall,
        'explain_is_precision_higher_than_recall': 'Precision is higher than recall' if is_precision_higher_than_recall else 'Recall is higher than precision',
        'confusion_matrix_train': conf_matrix_train,
        'confusion_matrix_test': conf_matrix_test,

        }
        


        """
        Answer is a dictionary with the following keys: 
        - "scores" : a dictionary with the mean/std of the F1 score, precision, and recall
        - "cv" : the cross-validation strategy
        - "clf" : the classifier
        - "is_precision_higher_than_recall" : a boolean
        - "explain_is_precision_higher_than_recall" : a string
        - "confusion_matrix_train" : the confusion matrix for the training set
        - "confusion_matrix_test" : the confusion matrix for the testing set
        
        answer["scores"] is dictionary with the following keys, generated from the cross-validator:
        - "mean_accuracy" : the mean accuracy
        - "mean_recall" : the mean recall
        - "mean_precision" : the mean precision
        - "mean_f1" : the mean f1
        - "std_accuracy" : the std accuracy
        - "std_recall" : the std recall
        - "std_precision" : the std precision
        - "std_f1" : the std f1
        """

        return answer

    # --------------------------------------------------------------------------
    """
    D. Repeat the same steps as part 3.C but apply a weighted loss function (see the class_weights parameter).  Print out the class weights, and comment on the performance difference. Use the `compute_class_weight` argument of the estimator to compute the class weights. 
    """

    def partD(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
        Xtest: NDArray[np.floating],
        ytest: NDArray[np.int32],
    ) -> dict[str, Any]:
        """"""
        # Enter your code and fill the `answer` dictionary
        classes = np.unique(y)
        class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
        class_weights_dict = {classes[i]: class_weights[i] for i in range(len(classes))}

        clf = SVC(random_state=self.seed, class_weight=class_weights_dict)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed)

        scoring_metrics = {
            'accuracy': make_scorer(accuracy_score),
            'precision': make_scorer(precision_score, average='macro', zero_division=0),
            'recall': make_scorer(recall_score, average='macro', zero_division=0),
            'f1': make_scorer(f1_score, average='macro', zero_division=0)
        }

        scores = cross_validate(clf, X, y, cv=cv, scoring=scoring_metrics, return_train_score=False)

        scores_mean = {metric: np.mean(scores['test_' + metric]) for metric in scoring_metrics}
        scores_std = {metric: np.std(scores['test_' + metric]) for metric in scoring_metrics}

        clf.fit(X, y)
        y_pred_train = clf.predict(X)
        conf_matrix_train = confusion_matrix(y, y_pred_train)
        y_pred_test = clf.predict(Xtest)
        conf_matrix_test = confusion_matrix(ytest, y_pred_test)
        answer = {
            "scores": {
            "mean_accuracy": scores_mean['accuracy'],
            "mean_recall": scores_mean['recall'],
            "mean_precision": scores_mean['precision'],
            "mean_f1": scores_mean['f1'],
            "std_accuracy": scores_std['accuracy'],
            "std_recall": scores_std['recall'],
            "std_precision": scores_std['precision'],
            "std_f1": scores_std['f1']},
            "cv": "StratifiedKFold",
            "clf": "SVC",
            "class_weights": class_weights_dict,
            "confusion_matrix_train": conf_matrix_train.tolist(),
            "confusion_matrix_test": conf_matrix_test.tolist(),
            "explain_purpose_of_class_weights": "Class weights are used to account for repeating classes. Less frequent classes are assigned more weight so they dont get washed out in calculations",
            "explain_performance_difference": "Class weights include all classes more evenly at the cost of skewing the results"
    }

        """
        Answer is a dictionary with the following keys: 
        - "scores" : a dictionary with the mean/std of the F1 score, precision, and recall
        - "cv" : the cross-validation strategy
        - "clf" : the classifier
        - "class_weights" : the class weights
        - "confusion_matrix_train" : the confusion matrix for the training set
        - "confusion_matrix_test" : the confusion matrix for the testing set
        - "explain_purpose_of_class_weights" : explanatory string
        - "explain_performance_difference" : explanatory string

        answer["scores"] has the following keys: 
        - "mean_accuracy" : the mean accuracy
        - "mean_recall" : the mean recall
        - "mean_precision" : the mean precision
        - "mean_f1" : the mean f1
        - "std_accuracy" : the std accuracy
        - "std_recall" : the std recall
        - "std_precision" : the std precision
        - "std_f1" : the std f1

        Recall: The scores are based on the results of the cross-validation step
        """

        return answer
