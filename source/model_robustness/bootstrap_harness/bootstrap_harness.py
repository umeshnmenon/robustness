from __future__ import division
import pandas as pd
import h2o
from model_robustness.utils.common import *
from model_robustness.utils.log import *
from model_robustness.utils.timer import *
from sklearn.utils import resample
from model_robustness.helpers.evaluation_helper import EvaluationHelper


setup_logger(log_level="INFO") #TODO: Parameterize this


class BootstrapHarness:
    """
    # CHECKS THE ROBUSTNESS OF A GIVEN MODEL ON THE GIVEN DATA SET
    # Method:
    # 1. Using bootstraping, sample as many as possible
    # 2. Predict on these samples
    # 3. Calculate the evaluation metrics
    # 4. Calculate the mean, SE, and CI of these metrics
    # Any model that exposes a predict() function can be input to the robustness package
    """

    def __init__(self):
        """
        Any initialization code goes here
        """
        stats_df = pd.DataFrame()

    @Timer(logging.getLogger())
    def robustness_index(self, data, model, targetcol=None, labels=[], nsamples=10, nrecods=100, clevels=[95], random_state=None):
        """
        The function will create nsamples of size nrecords with repetition using bootstrapping.
        :param nsamples:
        :param nrecods:
        :return:
        """
        is_h2o_model = False
        cols = data.columns
        # cbind features and target
        nlen = len(labels)
        if targetcol is None and nlen == 0:
            assert False, "Either targetcol or labels must be specified"
        if nlen > 0:
            if nlen != data.shape[0]:
                assert False, "Number of observations and number of labels must match"
            else:
                if isinstance(cols, pd.RangeIndex):
                    targetcol = pd.Index([len(cols)])
                    cols = pd.RangeIndex(start=0, stop=len(cols) + 1, step=1)
                else:
                    targetcol = 'target'
                    cols.append(targetcol)

                if is_h2o_frame(data):
                    data = h2o.cbind(data, labels)
                else:
                    data = pd.concat([data.reset_index(drop=True), labels], axis=1)

        # check the type of data
        if is_h2o_frame(data):  # if h2o then convert it
            np_data = h2o.as_list(data).values
            col_types = [v for v in data.types.values()]
            is_h2o_model = True
        else:
            np_data = data.values

        for i in range(1, nsamples + 1):
            logging.info("Sampling " + str(i))
            data_boot = resample(np_data, replace=True, n_samples=nrecods, random_state=random_state)
            data_boot_df = pd.DataFrame(data=data_boot[0:, 0:], columns=cols)
            if is_h2o_model:
                data_boot_df = h2o.H2OFrame(data_boot_df, column_types=col_types)
                y_act = h2o.as_list(data_boot_df[targetcol]).values
            else:
                y_act = data_boot_df[len(cols) - 1]

            # remove the target column for predicting as model
            if nlen > 0:
                data_boot_df = data_boot_df.drop([len(cols) - 1], axis=1)

            preds = model.predict(data_boot_df)
            if preds.ndim == 1:
                y_preds = preds
            else:
                y_preds = preds[:, 0]

            # make necessary transformation for h20 frames
            if is_h2o_frame(y_preds):
                y_preds = convert_h2o_list(y_preds)

            if is_h2o_frame(y_act):
                y_act = convert_h2o_list(y_act)

            self.prepare_stats_sample(y_act, y_preds)

        # We have estimations from multiple sample. Now get the mean, se, and CI
        self.prepare_robustness_index(clevels)
        return self.stats_df

    def prepare_stats_sample(self, y_act, y_preds):
        """
        Returns the predefined set of evaluation metrics for a classification setting
        Overriden by the specific implementations. for e.g. classification and regression
        :param y_act:
        :param y_preds:
        :return:
        """
        pass

    def prepare_robustness_index(self, clevels=[95]):
        """
        Prepares the robustness index from the evaluation stats sample
        Overriden by the specific implementations. for e.g. classification and regression
        :return:
        """
        pass

    def get_evaluation_stats(self, metric, estimates, cls):
        """
        returns a list that contains mean, se and various confidence levels
        :param estimates:
        :param cls:
        :return:
        """
        cols = ['Statistic', 'Mean', 'SE', 'CL', 'LB', 'UB', 'CI', 'Robustness_Index']
        est_df = pd.DataFrame(columns=cols)
        # loop through the confidence interval
        for cl in cls:
            m, se, lb, ub = EvaluationHelper.get_mean_se_ci(estimates, cl)
            ci_lbl = str(round(lb, 4)) + " - " + str(round(ub, 4))
            est_df = est_df.append(pd.DataFrame(
                [[metric, round(m, 4), round(se, 4), cl, lb, ub, ci_lbl, round((ub - lb), 4)]],
                columns=cols))
        return est_df

    def get_evaluation_stats1(self, metric, estimates, cls):
        """
        returns a list that contains mean, se and various confidence levels
        :param estimates:
        :param cls:
        :return:
        """
        cols = ['Statistic', 'Mean', 'SE']
        est_df = pd.DataFrame(columns=cols)
        ci_lst, ci_lb, ci_ub = []
        # loop through the confidence interval
        for cl in cls:
            cols.append('CI@' + str(cl))
            # Accuracy
            m, se, lb, ub = EvaluationHelper.get_mean_se_ci(estimates, cl)
            ci_lst = ci_lst.append(str(round(lb, 4)) + " - " + str(round(ub, 4)))
            mn = m
            stde = se
            ci_lb = lb
            ci_ub = ub

        est_df = est_df.append(pd.DataFrame(
            [[metric, round(mn, 4), round(stde, 4), ci_lst]],
            columns=cols))
        return est_df
