## MACHINE LEARNING EXPLORATORY DATA ANLYSIS

This project buis a machine learning process to classify data as toxic or non-toxic using data in th CSV fife data.csv. The workflow includes EDA.

the Technology used are:

Python

Pandas

Numpy

Matplotlib

Seaborn

Project workflow

1.Load the dataset

2.Exploratory Data Analysis

3.Data Preprocessing

4.Feature selection

5.Model Training

6.Model Evaluation

Hyperparameter tuning significantly improved the Random Forest toxicity classifier compared to the baseline model with default settings. The baseline showed reasonable performance but misclassified several toxic and non‑toxic comments, suggesting that model complexity was not well matched to the dataset.

Using grid search, I systematically tested combinations of key hyperparameters (n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features, bootstrap) with cross‑validation. The best configuration increased test accuracy and F1‑score, mainly by limiting tree depth and increasing minimum samples, which reduced overfitting to noisy comments, while more trees and an appropriate max_features improved ensemble stability.

Random search explored a wider range of the same hyperparameters by sampling random combinations, reaching similar or slightly better performance with fewer model evaluations. Overall, both grid and random search show that tuning Random Forest hyperparameters is crucial for toxicity detection: it leads to more reliable identification of toxic comments and fewer incorrect flags than the untuned baseline.



