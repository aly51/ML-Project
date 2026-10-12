# Credit Card Fraud Detection

I built this project to see which machine learning model catches credit card fraud best. I trained five models on the same data and compared them.

The data comes from the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) by MLG ULB on Kaggle. It has transactions made by European cardholders in September 2013. Each row is one transaction and the `Class` column says if it was fraud.

## The main problem

Fraud is really rare in this data. After removing duplicates, my test set had 85,118 transactions and only 142 of them were fraud. A model that says "no fraud" every time would still get over 99% accuracy. So I didn't trust accuracy much. I looked at precision, recall, F1 and PR-AUC instead.

## What I did

1. Explored the data in `EDA.ipynb`.
2. Dropped duplicate rows and the `Time` column.
3. Split the data 70/30 into train and test, keeping the fraud ratio the same in both.
4. Used SMOTE on the training set only, so the test set stays untouched.
5. Trained five models, one notebook each.
6. Compared them in `Model Comparisons.ipynb` and used SHAP to look at feature importance.

## Models

- Logistic Regression
- KNN (5 neighbors)
- Random Forest (100 trees)
- XGBoost (200 trees)
- MLP built with TensorFlow/Keras

Logistic Regression, KNN and the MLP use scaled features. The tree models don't need it.

## Results

All scores are for the fraud class on the test set.

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|-------|-----------|--------|----|---------|--------|
| Logistic Regression | 0.07 | 0.89 | 0.13 | 0.962 | 0.705 |
| KNN | 0.44 | 0.76 | 0.56 | 0.887 | 0.558 |
| MLP | 0.84 | 0.77 | 0.80 | 0.950 | 0.807 |
| Random Forest | 0.89 | 0.78 | 0.83 | 0.953 | 0.822 |
| XGBoost | 0.79 | 0.80 | 0.80 | 0.967 | 0.814 |

Random Forest had the best F1 and PR-AUC. XGBoost had the best ROC-AUC and a good balance between precision and recall.

Logistic Regression found the most fraud (89% recall) but it also flagged a lot of normal transactions as fraud. Its precision is only 7%, so it would annoy a lot of customers.

KNN was the weakest overall.

For the MLP I used a class weight of 3 for fraud and a decision threshold of 0.75 instead of 0.5. I picked these by hand, so there is probably room to tune them better.

## Project structure

- `EDA.ipynb`: cleaning, splitting and SMOTE
- `Logistic Regression.ipynb`
- `KNN.ipynb`
- `Random Forest.ipynb`
- `XGBoost.ipynb`
- `MLP.ipynb`
- `Model Comparisons.ipynb`: all models side by side
- `SHAP Comparison.ipynb`: which features drive predictions
- `KNN/`, `MLP/`, `XGBoost/`, `logistic_regression/`, `random_forest/`: metrics and reports for each model
- `models/`: saved models and scalers

## How to run it

Clone the repo:

```
git clone https://github.com/aly51/ML-Project.git
cd ML-Project
```

Install the libraries:

```
pip install numpy pandas scipy matplotlib seaborn scikit-learn imbalanced-learn xgboost tensorflow shap joblib jupyter
```

Download `creditcard.csv` from the Kaggle link above and put it in the project folder. It is not in the repo because it is too big.

Then run the notebooks in this order:

1. `EDA.ipynb`. This creates `X_train.csv`, `Y_train.csv`, `X_test.csv` and `Y_test.csv`.
2. The five model notebooks, in any order.
3. `Model Comparisons.ipynb` and `SHAP Comparison.ipynb`.

The comparison and SHAP notebooks need the model notebooks to run first, because they read the saved predictions and models.

## Notes

- Results can change a little if you run it again, since SMOTE and the models use random seeds. I set `random_state=42` where I could.
- The SHAP notebook only uses 100 sample rows to keep it fast.

## Author

aly51
