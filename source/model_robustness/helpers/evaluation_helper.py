import numpy as np
from scipy import stats


class EvaluationHelper:
    """
    Contains all the model evaluation generic functions
    """
    @staticmethod
    def get_mean_se_ci(data, confidence=0.95):
        # change to ratio if given as percentage
        if confidence > 1:
            confidence = confidence / 100
        a = 1.0 * np.array(data)
        n = len(a)
        m, se = np.mean(a), stats.sem(a)
        h = se * stats.t.ppf((1 + confidence) / 2., n - 1)
        return m, se, m - h, m + h
