# Model Robustness Harness

Robustness refers to how stable a model predicts on a new unseen data. This package helps to determine the robustness of
a Machine Learning model using an Enhanced Bootstrapping validation technique.

The package is available for both Classification and Regression problems. For each of these settings, the popular model
evaluation metrics are measured by Enhanced Holdout Validation using Bootstraping to measure the Robustness.                                                  The bootstrap samples will be used to derive robust estimates of standard errors and confidence intervals of a population parameter.
Robustness Index is calculated using the confidence interval by making predictions on the bootstrapped samples.
A Robustness index is the bandwidth of 95% interval of a performance metric (e.g. Accuracy, F1 Score, Lift etc.).
Lower the Robustness index, more robust the model is.

Any model that exposes a predict() method can be evaluated using this package. The input data must be either a pandas
dataframe or an h2o dataframe or an array like object.

## Classification Harness

Following popular evaluation metrics are calculated for a Classification setting.

## Run Book

```python
# creating a simple classification model
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# Load dataset
data = load_breast_cancer()

# Organize our data
label_names = data['target_names']
labels = data['target']
feature_names = data['feature_names']
features = data['data']

# Split our data
train, test, train_labels, test_labels = train_test_split(features,
                                                          labels,
                                                          test_size=0.33,
                                                          random_state=42)

# Initialize our classifier
gnb = GaussianNB()

# Train our classifier
model = gnb.fit(train, train_labels)
```

```python
# Load the packages
from model_robustness.bootstrap_harness.classification import ClassificationHarness
import pandas as pd
```

```python
# Convert the np arrays to data frame as harness accepts only pandas dataframe and h2o dataframe
test_df = pd.DataFrame(test)
test_labels_df = pd.DataFrame(test_labels)
harness = ClassificationHarness()
res = harness.robustness_index(test_df, gnb, labels=test_labels_df, nsamples=3, nrecods=100)
res
```

## Building the package

Go tho the source folder where you have the `setup.py`. Run below command to build the package.

```sh
python setup.py sdist bdist_wheel
```

Once the package is built, you will see a `dist` folder and within the folder a `.tar.gz` file and `.whl` file. Run the
below command to install the package

```sh
python -m pip install name_of_the_whl_file.whl
```

You can also download the pre-built pacakge from `bin` folder and run the above command if you do not want to build.