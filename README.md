# CreditCardFraudDetection

This project focuses on the detection of fraudulent transactions using a credit card transactions dataset. The goal is to build and evaluate machine learning models capable of identifying fraudulent activities. Our approach includes preprocessing steps, exploratory data analysis, and experimenting with various machine learning models with an emphasis on handling the imbalanced nature of the dataset through oversampling and undersampling techniques.

The dataset contains transactions made by credit cards, where each transaction is labeled as fraudulent or genuine. Features V1 through V28 are the result of a PCA transformation for privacy reasons. The 'Time' feature represents the seconds elapsed between transactions, and 'Amount' is the transaction amount. The target variable 'Class' indicates whether a transaction is fraudulent (1) or not (0).

Note: The data is anonymized and contains numerical input variables which are the result of a PCA transformation.