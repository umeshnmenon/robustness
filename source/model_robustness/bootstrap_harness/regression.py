from __future__ import division
import pandas as pd
from ..bootstrap_harness.bootstrap_harness import BootstrapHarness
from ..helpers.evaluation.regression import RegressionEvaluator


class RegressionHarness(BootstrapHarness):
    """
    # CHECK THE ROBUSTNESS OF A REGRESSION MODEL ON THE GIVEN DATA SET
    # Method:
    # 1. Using bootstraping, sample as many as possible
    # 2. Predict on these samples
    # 3. Calculate the evaluation metrics
    # 4. Calculate the mean, SE, and CI of these metrics
    # Any model that exposes a predict() function can be input to the robustness package
    """

    def __init__(self, *args):
        """
        Any initialization code goes here
        """
        self.rse = []
        self.rmse = []
        self.rsq = []
        self.mse = []
        BootstrapHarness.__init__(self, *args)

    def prepare_stats_sample(self, y_act, y_preds):
        """
        Returns the predefined set of evaluation metrics for a classification setting
        :param y_act:
        :param y_preds:
        :return:
        """
        rmse, mse, rsq, rse = RegressionEvaluator.get_stats(y_act, y_preds)
        self.rmse.append(rmse)
        self.mse.append(mse)
        self.rsq.append(rsq)
        self.rse.append(rse)

    def prepare_robustness_index(self, clevels=[95]):
        """
        Prepares the robustness index from the evaluation stats sample
        :return:
        """
        # We have estimations from multiple sample. Now get the mean, se, and CI
        self.stats_df = self.get_evaluation_stats("RMSE", self.rmse, clevels)
        mse_df = self.get_evaluation_stats("MSE", self.mse, clevels)
        self.stats_df = self.stats_df.append(mse_df)
        rsq_df = self.get_evaluation_stats("RSQ", self.rsq, clevels)
        self.stats_df = self.stats_df.append(rsq_df)
        rse_df = self.get_evaluation_stats("RSE", self.rse, clevels)
        self.stats_df = self.stats_df.append(rse_df)
