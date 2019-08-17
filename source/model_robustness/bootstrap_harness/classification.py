from __future__ import division
import pandas as pd
from ..bootstrap_harness.bootstrap_harness import BootstrapHarness
from ..helpers.evaluation.classification import ClassificationEvaluator


class ClassificationHarness(BootstrapHarness):
    """
    # CHECK THE ROBUSTNESS OF A CLASSIFICATION MODEL ON THE GIVEN DATA SET
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
        self.est_auc = []
        self.est_f1 = []
        self.est_prec = []
        self.est_recall = []
        self.est_acc = []
        BootstrapHarness.__init__(self, *args)

    def prepare_stats_sample(self, y_act, y_preds):
        """
        Returns the predefined set of evaluation metrics for a classification setting
        :param y_act:
        :param y_preds:
        :return:
        """
        auc, f1, prec, recl, acc = ClassificationEvaluator.get_stats(y_act, y_preds)
        self.est_auc.append(auc)
        self.est_f1.append(f1)
        self.est_prec.append(prec)
        self.est_recall.append(recl)
        self.est_acc.append(acc)

    def prepare_robustness_index(self, clevels=[95]):
        """
        Prepares the robustness index from the evaluation stats sample
        :return:
        """
        # We have estimations from multiple sample. Now get the mean, se, and CI
        self.stats_df = self.get_evaluation_stats("Accuracy", self.est_acc, clevels)
        f1_df = self.get_evaluation_stats("F1", self.est_f1, clevels)
        self.stats_df = self.stats_df.append(f1_df)
        auc_df = self.get_evaluation_stats("AUC", self.est_auc, clevels)
        self.stats_df = self.stats_df.append(auc_df)

