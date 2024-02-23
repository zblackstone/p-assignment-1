# Add your imports here.
# Note: only sklearn, numpy, utils and new_utils are allowed.

import numpy as np
from numpy.typing import NDArray
from typing import Any
import utils as u
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, KFold, ShuffleSplit
from sklearn.tree import DecisionTreeClassifier
# ======================================================================

# I could make Section 2 a subclass of Section 1, which would facilitate code reuse.
# However, both classes have the same function names. Better to pass Section 1 instance
# as an argument to Section 2 class constructor.


class Section2:
    def __init__(
        self,
        normalize: bool = True,
        seed: int = None,
        frac_train: float = 0.2,
    ):
        """
        Initializes an instance of MyClass.

        Args:
            normalize (bool, optional): Indicates whether to normalize the data. Defaults to True.
            seed (int, optional): The seed value for randomization. If None, each call will be randomized.
                If an integer is provided, calls will be repeatable.

        Returns:
            None
        """
        self.normalize = normalize
        self.seed = seed
        self.frac_train = frac_train

    # ---------------------------------------------------------

    """
    A. Repeat part 1.B but make sure that your data matrix (and labels) consists of
        all 10 classes by also printing out the number of elements in each class y and 
        print out the number of classes for both training and testing datasets. 
    """

    def partA(
        self,
    ) -> tuple[
        dict[str, Any],
        NDArray[np.floating],
        NDArray[np.int32],
        NDArray[np.floating],
        NDArray[np.int32],
    ]:
        Xtrain, ytrain, Xtest, ytest = u.prepare_data()
        uniq_class_ytrain, counts_ytrain = np.unique(ytrain, return_counts= True)
        uniq_class_ytest, counts_ytest = np.unique(ytest, return_counts = True)


        answer = {
            'nb_classes_train': len(np.unique(ytrain)),
            'nb_classes_test': len(np.unique(ytest)),
            'class_count_train': counts_ytrain.tolist(),
            'class_count_test': counts_ytest.tolist(),
            'length_Xtrain': len(Xtrain),
            'length_Xtest': len(Xtest),
            'length_ytrain': len(ytrain),
            'length_ytest': len(ytest),
            'max_Xtrain': np.max(Xtrain),
            'max_Xtest': np.max(Xtest),
        }
        # Enter your code and fill the `answer`` dictionary

        # `answer` is a dictionary with the following keys:
        # - nb_classes_train: number of classes in the training set
        # - nb_classes_test: number of classes in the testing set
        # - class_count_train: number of elements in each class in the training set
        # - class_count_test: number of elements in each class in the testing set
        # - length_Xtrain: number of elements in the training set
        # - length_Xtest: number of elements in the testing set
        # - length_ytrain: number of labels in the training set
        # - length_ytest: number of labels in the testing set
        # - max_Xtrain: maximum value in the training set
        # - max_Xtest: maximum value in the testing set

        # return values:
        # Xtrain, ytrain, Xtest, ytest: the data used to fill the `answer`` dictionary

        Xtrain = Xtest = np.zeros([1, 1], dtype="float")
        ytrain = ytest = np.zeros([1], dtype="int")

        return answer, Xtrain, ytrain, Xtest, ytest

    """
    B.  Repeat part 1.C, 1.D, and 1.F, for the multiclass problem. 
        Use the Logistic Regression for part F with 300 iterations. 
        Explain how multi-class logistic regression works (inherent, 
        one-vs-one, one-vs-the-rest, etc.).
        Repeat the experiment for ntrain=1000, 5000, 10000, ntest = 200, 1000, 2000.
        Comment on the results. Is the accuracy higher for the training or testing set?
        What is the scores as a function of ntrain.

        Given X, y from mnist, use:
        Xtrain = X[0:ntrain, :]
        ytrain = y[0:ntrain]
        Xtest = X[ntrain:ntrain+test]
        ytest = y[ntrain:ntrain+test]
    """

    def partB(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
        Xtest: NDArray[np.floating],
        ytest: NDArray[np.int32],
        ntrain_list: list[int] = [100, 500, 1000],
        ntest_list: list[int] = [20, 100, 200],
    ) -> dict[int, dict[str, Any]]:
        """ """
        # Enter your code and fill the `answer`` dictionary
        def calc_class_counts(y):
            return [list(y).count(i) for i in range(np.min(y), np.max(y) + 1)]
        
        def do_cross_validate(classifier, X, y, cv):
            scores = cross_validate(classifier, X, y, cv=cv, scoring = 'accuracy', return_train_score= False)
            mean_acc = scores['test_score'].mean()
            std_acc = scores['test_score'].std()
            return mean_acc, std_acc
        
        Xtrain = u.prepare_data()
        ytrain = u.prepare_data()
        Xtest = u.prepare_data()
        ytest = u.prepare_data()
        answer = {}

        for ntrain in ntrain_list:
            for ntest in ntest_list:
                Xtrain_subset = Xtrain[:ntrain]
                ytrain_subset = ytrain[:ntrain]
                Xtest_subset = Xtest[:ntest]
                ytest_subset = ytest[:ntest]
                #1.C
                clf= DecisionTreeClassifier(random_state = self.seed)
                cv_c = KFold(n_splits = 5, shuffle = True, random_state= self.seed)
                

                mean_acc_c, std_acc_c = do_cross_validate(clf, Xtrain_subset, ytrain_subset, cv_c)

                #1.D
                cv_d = ShuffleSplit(n_splits = 5, test_size = 0.2, random_state = self.seed)
                
                mean_acc_d, std_acc_d = do_cross_validate(clf, Xtrain_subset, ytrain_subset, cv_d)

                #1.F
                clf_f = LogisticRegression(max_iter = 300, multi_class = 'ovr', n_jobs = -1)
                clf_f.fit(Xtrain_subset, ytrain_subset)
                acc_train_F = clf_f.score(Xtrain_subset, ytrain_subset)
                acc_test_F = clf_f.score(Xtest_subset, ytest_subset)


       
            answer[ntrain] = {
                'partC': {
                    'mean_accuracy': mean_acc_c,
                    'std_accuracy': std_acc_c,
                },
                'partD': {
                    'mean_accuracy': mean_acc_d,
                    'std_accuracy': std_acc_d,
                },
                'partF': {
                    'accuracy_train': acc_train_F,
                    'accuracy_test': acc_test_F,
                },
                'ntrain': ntrain,
                'ntest': ntest,
                'class_count_train': calc_class_counts(ytrain),
                'class_count_test': calc_class_counts(ytest),
            }

            """
            `answer` is a dictionary with the following keys:
            - 1000, 5000, 10000: each key is the number of training samples

            answer[k] is itself a dictionary with the following keys
                - "partC": dictionary returned by partC section 1
                - "partD": dictionary returned by partD section 1
                - "partF": dictionary returned by partF section 1
                - "ntrain": number of training samples
                - "ntest": number of test samples
                - "class_count_train": number of elements in each class in
                                the training set (a list, not a numpy array)
                - "class_count_test": number of elements in each class in
                                the training set (a list, not a numpy array)
            """

        return answer
