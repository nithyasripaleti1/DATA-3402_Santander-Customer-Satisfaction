# DATA-3402_Final_Project

# Santander Customer Satisfaction Prediction


* **One Sentence Summary:** This project uses the [Santander Customer Satisfaction Kaggle competition](https://www.kaggle.com/competitions/santander-customer-satisfaction/data) data to predict whether a customer is dissatisfied using anonymized banking features


## Overview


The goal of this project is to predict customer dissatisfaction using the Santander Customer Satisfaction dataset. The target variable is `TARGET`, where `1` represents a dissatisfied customer and `0` represents a satisfied customer. The dataset is challenging because the features are anonymized, highly sparse, and the target class is heavily imbalanced: only about **3.96%** of customers in the training set are dissatisfied.


This project formulates the task as a **binary classification problem**. The approach includes exploratory data analysis, feature cleanup, handling an abnormal encoded value, removing constant and duplicate columns, training multiple classification models, and tuning decision thresholds to improve performance on the minority class. The main models tested were **Logistic Regression**, **Random Forest**, and **Gradient Boosting**.


The best validation result came from a **Random Forest Classifier** with threshold tuning. At the selected threshold of **0.75**, the model achieved an **F1-score of 0.2823**, **precision of 0.2050**, **recall of 0.4535**, and **ROC-AUC of 0.8276** on the validation set. Although Gradient Boosting had the highest ROC-AUC, Random Forest was selected because it achieved the best F1-score, which was more appropriate for this imbalanced classification task.


## Summary of Work Done


### Data


* **Dataset:** Santander Customer Satisfaction Kaggle dataset
* **Task type:** Binary classification
* **Input:** CSV files containing anonymized numerical customer/banking features
* **Output:** `TARGET`
 * `0` = satisfied customer
 * `1` = dissatisfied customer
* **Original data size:**
 * Training set: **76,020 rows × 371 columns**
 * Test set: **75,818 rows × 370 columns**
* **Target distribution in training data:**
 * `TARGET = 0`: **73,012 customers**, about **96.04%**
 * `TARGET = 1`: **3,008 customers**, about **3.96%**
* **Train-validation split:**
 * Training split: **60,816 rows**
 * Validation split: **15,204 rows**
 * Split method: **80/20 stratified split** to preserve the target distribution


#### Preprocessing / Clean Up


The following preprocessing steps were performed:


* Verified that train and test feature columns match after excluding `TARGET`.
* Confirmed that all columns are numerical.
* Confirmed that there are **no missing values** in either train or test data.
* Checked for duplicate rows and found **0 duplicate rows**.
* Removed the `ID` column from model inputs because it is an identifier, not a predictive feature.
* Removed **34 constant features** because they contain only one unique value and do not provide predictive signal.
* Replaced the abnormal encoded value `-999999` in `var3` with the median valid value of `var3`, which was **2.0**.
* Detected and removed **29 duplicate feature columns**.
* After cleanup:
 * Cleaned training data shape: **76,020 rows × 308 columns** including `TARGET`
 * Cleaned test data shape: **75,818 rows × 307 columns**
 * Final model input shape after removing `ID` and `TARGET`: **306 features**


#### Data Visualization


Several visualizations were created in the notebook:


* **Target distribution bar chart:** showed that the dataset is highly imbalanced, with only about 3.96% dissatisfied customers.
* **Zero-ratio histogram:** showed that the dataset is highly sparse. The median feature had a zero ratio of about 99.6%.
* **Feature histograms for `var3`, `var15`, and `var38`:** showed that `var3` contained an abnormal value of `-999999`, `var15` had useful variation, and `var38` was highly right-skewed.
* **Feature distributions by target:** showed that `var15` has a noticeable shift between satisfied and dissatisfied customers.
* **Correlation analysis:** showed that no single feature strongly separates the two classes. The strongest absolute correlation with `TARGET` was around 0.15.
* **Model visualizations:** confusion matrices, ROC curves, precision-recall curves, threshold tuning plots, model comparison charts, and Random Forest feature importance plots were generated.


Important EDA observations:


* Accuracy alone is misleading because predicting every customer as satisfied gives about **96.04% accuracy** but completely fails to identify dissatisfied customers.
* Most features are sparse and anonymized, so the model must combine many weak signals rather than rely on a single interpretable feature.
* `var15` appeared to be one of the most useful individual features. In the Random Forest model, it was the most important feature.


### Problem Formulation


#### Input / Output


* **Input:** 306 cleaned numerical customer features after preprocessing
* **Output:** binary prediction for `TARGET`
 * `0`: satisfied
 * `1`: dissatisfied


Although the original Kaggle competition asks for probabilities of dissatisfaction, this project focused on binary classification and threshold tuning to improve F1-score for the dissatisfied class.


#### Models


The following models were tested:


1. **Baseline model**
  * Predicted every validation instance as `0`.
  * Used to show why accuracy is misleading for this dataset.


2. **Logistic Regression**
  * Used as a linear baseline model.
  * Included `StandardScaler` because Logistic Regression is sensitive to feature scale.
  * Used `class_weight="balanced"` to account for class imbalance.


3. **Random Forest Classifier**
  * Used as a nonlinear ensemble model.
  * Handles nonlinear relationships and feature interactions better than Logistic Regression.
  * Used class weighting to reduce the impact of class imbalance.
  * Threshold tuning was performed to improve F1-score.


4. **Gradient Boosting Classifier**
  * Used as another nonlinear ensemble method.
  * Trains trees sequentially, where each tree attempts to correct previous errors.
  * Compared against Random Forest to evaluate whether boosting improved performance.


#### Loss, Optimizer, and Hyperparameters


Since the models were implemented using scikit-learn classifiers, the optimization procedures are handled internally by each algorithm.


Key hyperparameters used:


* **Logistic Regression**
 * `max_iter=1000`
 * `class_weight="balanced"`
 * `random_state=42`


* **Random Forest**
 * `n_estimators=200`
 * `max_depth=10`
 * `min_samples_leaf=20` for the main validation experiment
 * `min_samples_leaf=10` for the near-constant feature experiment and final retraining
 * `class_weight="balanced"`
 * `random_state=42`
 * `n_jobs=-1`


* **Gradient Boosting**
 * `n_estimators=150`
 * `learning_rate=0.05`
 * `max_depth=3`
 * `random_state=42`


* **Threshold tuning**
 * Thresholds from **0.05 to 0.90** were tested in increments of **0.05**.
 * Best threshold was selected based on validation F1-score.


### Training


The models were trained in Python using scikit-learn inside a Jupyter Notebook.


The notebook does not record exact runtime or hardware specifications. The models are standard scikit-learn models and should be reproducible on a normal laptop or in Google Colab. Random Forest used `n_jobs=-1` to use all available CPU cores.


Training process:


1. Load `train.csv` and `test.csv`.
2. Perform exploratory data analysis.
3. Clean and preprocess the data.
4. Split the cleaned training data into stratified train and validation sets.
5. Train baseline, Logistic Regression, Random Forest, and Gradient Boosting models.
6. Tune classification thresholds using validation F1-score.
7. Compare models using accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrices.
8. Retrain the selected Random Forest model on the full cleaned training data.
9. Generate predictions on the Kaggle test set.
10. Save final predictions as `santander_submission.csv`.


Stopping criteria:


* Logistic Regression used `max_iter=1000`.
* Tree-based models used a fixed number of estimators.
* Model selection was based on validation metrics rather than iterative early stopping.


Difficulties encountered:


* The target variable was extremely imbalanced, making accuracy unreliable.
* The features were anonymized, limiting domain-specific interpretation.
* Many features were sparse, constant, near-constant, or duplicated.
* The default classification threshold of 0.50 was not always appropriate, especially for Gradient Boosting and Logistic Regression.


These were addressed by using class weighting, stratified splitting, feature cleanup, multiple evaluation metrics, and threshold tuning.


### Performance Comparison


The main evaluation metrics were:


* **Accuracy:** overall fraction of correct predictions
* **Precision:** among customers predicted as dissatisfied, how many were truly dissatisfied
* **Recall:** among truly dissatisfied customers, how many were detected
* **F1-score:** harmonic mean of precision and recall
* **ROC-AUC:** ability of the model to rank dissatisfied customers above satisfied customers


Because the dataset is highly imbalanced, **F1-score, recall, precision, and ROC-AUC** were more important than accuracy alone.


| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Baseline: Predict All 0 | 0.9604 | 0.0000 | 0.0000 | 0.0000 | 0.5000 |
| Logistic Regression, Default Threshold | 0.6846 | 0.0904 | 0.7691 | 0.1619 | 0.8031 |
| Logistic Regression, Best F1 Threshold | 0.8595 | 0.1498 | 0.5449 | 0.2350 | 0.8031 |
| Random Forest, Default Threshold | 0.7622 | 0.1098 | 0.7043 | 0.1900 | 0.8276 |
| Random Forest, Best F1 Threshold | **0.9087** | 0.2050 | **0.4535** | **0.2823** | 0.8276 |
| Gradient Boosting, Default Threshold | 0.9606 | **0.8000** | 0.0066 | 0.0132 | **0.8421** |
| Gradient Boosting, Best F1 Threshold | 0.9130 | **0.2054** | 0.4169 | 0.2752 | **0.8421** |


#### Best Model


The selected model was the **threshold-tuned Random Forest Classifier**.


Validation performance at threshold **0.75**:


* Accuracy: **0.9087**
* Precision: **0.2050**
* Recall: **0.4535**
* F1-score: **0.2823**
* ROC-AUC: **0.8276**


Confusion matrix:


| | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | 13,543 | 1,059 |
| Actual 1 | 329 | 273 |


The tuned Random Forest model identified **273 out of 602** dissatisfied customers in the validation set while maintaining a better precision-recall balance than the other tested models.


#### Cross-Validation


Stratified 5-fold cross-validation was also performed for Logistic Regression and Random Forest using the default threshold.


| Metric | Logistic Regression Mean CV Score | Random Forest Mean CV Score |
|---|---:|---:|
| Accuracy | 0.6910 | 0.7627 |
| Precision | 0.0898 | 0.1075 |
| Recall | 0.7450 | 0.6832 |
| F1-score | 0.1603 | 0.1858 |
| ROC-AUC | 0.7957 | 0.8127 |


The cross-validation results support the validation-set conclusion that Random Forest performs better than Logistic Regression overall.


### Conclusions


* The dataset is highly imbalanced, so accuracy is not a useful standalone metric.
* A baseline model can achieve about **96% accuracy** by predicting all customers as satisfied, but it has **0 recall** for dissatisfied customers.
* Logistic Regression achieved high recall but produced many false positives.
* Random Forest provided the best balance between precision and recall after threshold tuning.
* Gradient Boosting achieved the highest ROC-AUC but slightly lower F1-score than the tuned Random Forest.
* Removing near-constant features slightly increased ROC-AUC but slightly reduced F1-score, so near-constant features were kept in the final model.
* The selected final model was a **Random Forest Classifier with threshold 0.75**.


### Future Work


Possible next steps include:


* Submit predicted probabilities instead of hard binary labels to better align with the original Kaggle competition format.
* Try stronger gradient-boosted tree models such as XGBoost, LightGBM, or CatBoost.
* Perform more systematic hyperparameter tuning using grid search or randomized search.
* Use resampling methods such as SMOTE, undersampling, or hybrid sampling to address class imbalance.
* Calibrate predicted probabilities using Platt scaling or isotonic regression.
* Explore feature engineering based on sparse activity indicators.
* Try dimensionality reduction or feature selection methods for highly sparse features.
* Evaluate models using PR-AUC, which may be more informative for highly imbalanced classification.
* Compare threshold choices based on business costs, such as the cost of missing a dissatisfied customer versus the cost of incorrectly flagging a satisfied customer.


## How to Reproduce Results


### Overview of Files in Repository


Expected repository files:


* `DATA3402_Project (3)(1).ipynb`
 * Main notebook containing data loading, EDA, preprocessing, modeling, threshold tuning, evaluation, and final submission generation.
* `train.csv`
 * Kaggle training data with anonymized customer features and `TARGET` labels.
* `test.csv`
 * Kaggle test data with anonymized customer features but no `TARGET` column.
* `santander_submission.csv`
 * Output file created by the notebook containing final predictions for the test set.
* `UTA-DataScience-Logo.png`
 * Logo image referenced at the top of this README.
* `README.md`
 * Project description and reproduction instructions.


### Software Setup


Install Python 3 and the required packages:


```bash
pip install pandas numpy matplotlib scikit-learn jupyter
```


Required Python libraries:


* `pandas`
* `numpy`
* `matplotlib`
* `scikit-learn`
* `jupyter`


### Data


Download the data from the Kaggle competition page:


<https://www.kaggle.com/competitions/santander-customer-satisfaction/data>


Place the following files in the same directory as the notebook:


```text
train.csv
test.csv
```


The notebook expects these exact filenames:


```python
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
```


### Training


To train the models:


1. Open the notebook:


```bash
jupyter notebook "DATA3402_Project (3)(1).ipynb"
```


2. Run all cells from top to bottom.
3. The notebook will:
  * load the data,
  * clean the features,
  * split the training data,
  * train Logistic Regression, Random Forest, and Gradient Boosting models,
  * tune classification thresholds,
  * compare performance,
  * retrain the selected model on the full cleaned training data,
  * create `santander_submission.csv`.


### Performance Evaluation


Performance evaluation is included in the notebook. The main outputs are:


* baseline model metrics,
* Logistic Regression metrics,
* Random Forest metrics,
* Gradient Boosting metrics,
* threshold tuning tables,
* confusion matrices,
* ROC curves,
* precision-recall curves,
* model comparison table,
* cross-validation comparison,
* Random Forest feature importance chart.


The most important comparison table is the final model comparison table, where the tuned Random Forest model achieved the highest F1-score.

