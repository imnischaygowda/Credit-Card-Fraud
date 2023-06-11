Credit Card Fraud Analytics
===========================

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)

This Machine Learning project is learning on how to deal with high variable, high values dataset and imbalanced that are for detecting Credit card Fraud case.
Below are the steps followed right from preparing dataset to building model with accuracte prediction.

Data Source - https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

**1. Data Preparation and Sampling**

- We have in this dataset V1-V28 ( 28 data variables; PCA applied data to protect customer identity) credit card transaction along with Amount and Time of transaction column, with Target column lablled Class=0/1(No Fraud/Fraud) repectively.
- Checking the total number of datapoints for Fraud/NotFraud, we found the data is highly imbalanced. We also observed that data is highly variable.
- So we perfomed [Robust scaling](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html) which is very effective against outliers.
- Next we use Stratified sampling, so we have euqal probability of all type of data samples.
- To address the main issue of imbalance data, we persom a sampling on stratified sampled output to get euqal number of fraud and NotFraud cases.

**2. Data Analysis**

- Given the data is PCA data, we plot a correlation matrix to check the influence of variables on Class label.
- We got the Higly Negative and Positive correlated variables. Plotting Box plot for both Positive and Negative variables, to identify the data distribution and outlier patterns. 
- These outliers to are to be removed as they are anamolies which would affect our model.

**3. Removing Outlier and Dimensionality reduction**

- To deal with outliers, we calculate the quartile ranges Q25 and Q75 values to calculate IQR.
- We calculate the cutoff range for each valaues, 1.5 * IQR. Using this we get Lower and Upper bound of variable.
- All values outside the lower and upper bound are treated as outliers and dropped form the data.
- Given we have 28 data variables, we perform a dimensionality reduction. I have used techniques such as,

  - t-SNE - t-distributed stochastic neighbor embedding
  - PCA - Principal Component Analysis
  - TSVD - Truncated Singluar Value Decompisition

**4. Classifier and Evaulation Metrics**
 
- X - predictor variables and Y - Target variables.
- Simple Train and Test split, since data is already scaled and sampled with equal probability of both classes.
- We used Lositic Regression, K-Nearest Neighbour, Support Vector and Decision tree classifier.
- We used GridsearchCV for hyperparameter tuning, to get best results for each model.
- Learning curve plots to observe the training and testing phase patterns of each model.
- Plot ROC(receiver operating characteristic curve) to performance of a classification model at all classification thresholds.

We observed the below ROC score,

- Logistic Regression 0.9510652185989668
- KNN 0.930923976445753
- SVC 0.9457946780734403
- Decision Tree 0.9217126826287428

One key metric to access our model performance is False Negative rate. The cost of False Negatives is much higher than the cost of False Positives.
To put in simple words, wrongly predict the transaction to be genuine instead of a fraud one then that would cause a massive problem for the credit card company.

1. Logistic Regression: 
- Specificity Score : 0.98
- Sensitivity Score: 0.94


2. K-Nearest Neighbour: 
- Specificity Score: 1.00
- Sensitivity Score: 0.90


3. Support Vector Classifier: 
- Specificity Score: 0.99
- Sensitivity Score: 0.94


4. Decision Tree Classifier: 
- Specificity Score: 0.99
- Sensitivity Score: 0.87

Conclusion, for this project suprisingly Logistic regression performed the best having low **False Negative rate.**
