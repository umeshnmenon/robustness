# Model Robustness Package

Accuracy metrics are measured by Enhanced Holdout Validation using Bootstraping to measure the Robustness.
The bootstrap samples will be used to derive robust estimates of standard errors and confidence intervals of a population parameter.
Robustness Index is calculated for each model using the confidence interval by making predictions on the bootstrapped samples. 
A Robustness index is the bandwidth of 95% interval of a performance metric (e.g. Accuracy, F1 Score, Lift etc.).
Lower the Robustness index, more robust the model is. 