
class RegressionEvaluator:
    """
    Contains the popular evaluation functions for regression model
    """
    @staticmethod
    def get_stats(y, y_pred):
        """
        calculates the important evaluation stats
        :param y:
        :param y_pred:
        :return:
        """
        # TODO: Fill in the metric calculation formula here
        rmse = 0
        mse = 0
        rsq = 0
        rse = 0
        return rmse, mse, rsq, rse
