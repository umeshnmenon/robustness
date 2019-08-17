from sklearn.metrics import auc, precision_recall_curve, f1_score, roc_auc_score, roc_curve
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.metrics import f1_score, roc_auc_score, classification_report, confusion_matrix
import numpy as np
from scipy import stats

class ClassificationEvaluator:
    """
    Contains all the model evaluation generic functions
    """
    @staticmethod
    def get_stats(y, y_pred):
        """
        calculates the important evaluation stats
        :param y:
        :param y_pred:
        :return:
        """
        return roc_auc_score(y, y_pred), f1_score(y, y_pred), precision_score(y, y_pred), recall_score(y, y_pred), accuracy_score(y, y_pred)
