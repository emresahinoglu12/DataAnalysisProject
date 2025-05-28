#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cinema Hall Ticket Sales Analysis (Extended)
--------------------------------------------
Author: Emre Şahinoğlu
Date: 2025-05-19

Description:
    Extended pipeline for cinema ticket‐sales data:
    1. Data loading & cleaning
    2. Feature engineering
    3. Exploratory Data Analysis (EDA)
    4. Statistical hypothesis testing
    5. Predictive modeling & comparison of 13 classifiers
       - 5 from lecture: Logistic Regression, KNN, Naive Bayes,
         SVM, Decision Tree
       - 8 extras: Random Forest, Gradient Boosting, AdaBoost,
         Extra-Trees, XGBoost (optional), LDA, QDA, MLP
    6. Customer segmentation (K-Means clustering)
    7. Saving outputs & artifacts

Usage:
    python main.py

Outputs:
    - data/cleaned_cinema_data.csv
    - visuals/*.png
    - visuals/best_model.pkl
    - Console summary of EDA, stats, and model performance
"""

import os
import logging

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from sklearn.model_selection      import train_test_split
from sklearn.preprocessing        import StandardScaler, OneHotEncoder
from sklearn.compose              import ColumnTransformer
from sklearn.pipeline             import Pipeline
from sklearn.metrics              import (
    classification_report,
    roc_auc_score,
    roc_curve
)

# 5 in‐class classifiers
from sklearn.linear_model         import LogisticRegression
from sklearn.neighbors            import KNeighborsClassifier
from sklearn.naive_bayes          import GaussianNB
from sklearn.svm                  import SVC
from sklearn.tree                 import DecisionTreeClassifier

# 8 extra classifiers
from sklearn.ensemble             import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier
)

# XGBoost is optional—wrap in try/except
xgb_available = False
try:
    from xgboost import XGBClassifier
    xgb_available = True
except Exception as e:
    logging.getLogger(__name__).warning(
        "XGBoost import failed (%s); skipping XGBoost model.", e
    )

from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis
)
from sklearn.neural_network        import MLPClassifier

# Clustering
from sklearn.cluster import KMeans
from sklearn.metrics  import silhouette_score

import joblib

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
DATA_PATH      = 'data/cinema_hall_ticket_sales.csv'
CLEANED_PATH   = 'data/cleaned_cinema_data.csv'
ARTIFACT_DIR   = 'visuals'
BEST_MODEL     = os.path.join(ARTIFACT_DIR, 'best_model.pkl')
COMPARISON_PNG = os.path.join(ARTIFACT_DIR, 'model_comparison_auc.png')

os.makedirs(os.path.dirname(CLEANED_PATH) or '.', exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M'
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# STEP 1: DATA LOADING & CLEANING
# -----------------------------------------------------------------------------
def load_and_clean(path: str) -> pd.DataFrame:
    logger.info("Loading data from %s", path)
    df = pd.read_csv(path)
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
    df['number_of_person'] = df['number_of_person'] \
        .replace('Alone', '1').astype(int)
    df['purchase_again'] = df['purchase_again'].map({'Yes': 1, 'No': 0})
    logger.info("Missing values by column:\n%s", df.isnull().sum())
    df = df.dropna().reset_index(drop=True)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    return df

# -----------------------------------------------------------------------------
# STEP 2: FEATURE ENGINEERING
# -----------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df['revenue']          = df['number_of_person'] * df['ticket_price']
    df['price_per_person'] = df['ticket_price'] / df['number_of_person']
    bins  = [0, 17, 35, 50, 65, 120]
    labels= ['<18','18-35','36-50','51-65','65+']
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels)
    if 'date' in df.columns:
        df['is_weekend'] = df['date'].dt.dayofweek.isin([5,6]).astype(int)
    else:
        df['is_weekend'] = 0
    return df

# -----------------------------------------------------------------------------
# STEP 3: EXPLORATORY DATA ANALYSIS (EDA)
# -----------------------------------------------------------------------------
def run_eda(df: pd.DataFrame):
    sns.set(style='whitegrid')
    # Age distribution
    plt.figure(figsize=(8,5))
    sns.histplot(df['age'], bins=20, kde=True)
    plt.title('Age Distribution')
    plt.savefig(f'{ARTIFACT_DIR}/age_distribution.png'); plt.close()
    # Price by seat type
    plt.figure(figsize=(8,5))
    sns.boxplot(x='seat_type', y='ticket_price', data=df)
    plt.title('Ticket Price by Seat Type')
    plt.savefig(f'{ARTIFACT_DIR}/price_by_seat_type.png'); plt.close()
    # Genre counts
    plt.figure(figsize=(8,5))
    order = df['movie_genre'].value_counts().index
    sns.countplot(x='movie_genre', data=df, order=order)
    plt.title('Ticket Sales by Genre'); plt.xticks(rotation=45)
    plt.savefig(f'{ARTIFACT_DIR}/genre_counts.png'); plt.close()
    # Revenue by genre
    plt.figure(figsize=(8,5))
    rev = df.groupby('movie_genre')['revenue'].sum().sort_values(ascending=False)
    sns.barplot(x=rev.values, y=rev.index)
    plt.title('Revenue by Genre')
    plt.savefig(f'{ARTIFACT_DIR}/revenue_by_genre.png'); plt.close()
    # Repeat rate by age group
    plt.figure(figsize=(8,5))
    rate = df.groupby('age_group')['purchase_again'].mean()
    rate.plot(kind='bar'); plt.title('Repeat Purchase Rate by Age Group')
    plt.savefig(f'{ARTIFACT_DIR}/repeat_rate_by_age_group.png'); plt.close()
    # Correlation matrix
    plt.figure(figsize=(8,6))
    sns.heatmap(df.select_dtypes(include=[np.number]).corr(),
                annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix')
    plt.savefig(f'{ARTIFACT_DIR}/correlation_matrix.png'); plt.close()

# -----------------------------------------------------------------------------
# STEP 4: STATISTICAL HYPOTHESIS TESTING
# -----------------------------------------------------------------------------
def run_stat_tests(df: pd.DataFrame):
    logger.info("HYPOTHESIS TESTING RESULTS:")
    # H1
    chi2, p, *_ = stats.chi2_contingency(
        pd.crosstab((df['age']>40).astype(int),
                    df['seat_type']=='Premium')
    )
    logger.info("H1: χ²=%.2f, p=%.4f", chi2, p)
    # H2
    a = df[df['age_group']=='18-35']['purchase_again']
    b = df[df['age_group']!='18-35']['purchase_again']
    t, p2 = stats.ttest_ind(a, b, equal_var=False)
    logger.info("H2: t=%.2f, p=%.4f", t, p2)
    # H3
    groups = [g['ticket_price'].values
              for _, g in df.groupby('movie_genre')]
    f, p3 = stats.f_oneway(*groups)
    logger.info("H3: F=%.2f, p=%.4f", f, p3)

# -----------------------------------------------------------------------------
# STEP 5: MODEL TRAINING & COMPARISON
# -----------------------------------------------------------------------------
def compare_classifiers(df: pd.DataFrame):
    X = df.drop('purchase_again', axis=1)
    y = df['purchase_again']
    cat_cols = ['seat_type','movie_genre','age_group']
    num_cols = ['age','number_of_person','ticket_price',
                'revenue','price_per_person','is_weekend']
    pre = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), cat_cols)
    ])
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'KNN':                 KNeighborsClassifier(),
        'NaiveBayes':          GaussianNB(),
        'SVM':                 SVC(probability=True),
        'DecisionTree':        DecisionTreeClassifier(),
        'RandomForest':        RandomForestClassifier(n_estimators=100),
        'GradBoost':           GradientBoostingClassifier(),
        'AdaBoost':            AdaBoostClassifier(),
        'ExtraTrees':          ExtraTreesClassifier(),
    }
    if xgb_available:
        models['XGBoost'] = XGBClassifier(use_label_encoder=False,
                                          eval_metric='logloss')
    models.update({
        'LDA': LinearDiscriminantAnalysis(),
        'QDA': QuadraticDiscriminantAnalysis(),
        'MLP': MLPClassifier(max_iter=1000)
    })
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    results = []
    for name, clf in models.items():
        pipe = Pipeline([('prep', pre), ('clf', clf)])
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:,1]
        auc = roc_auc_score(y_test, proba)
        logger.info("%-15s AUC = %.3f", name, auc)
        results.append((name, auc, pipe))
    results.sort(key=lambda x: x[1], reverse=True)
    best_name, best_auc, best_pipe = results[0]
    joblib.dump(best_pipe, BEST_MODEL)
    logger.info("Best model: %s (AUC=%.3f)", best_name, best_auc)
    names, aucs, _ = zip(*results)
    plt.figure(figsize=(10,6))
    sns.barplot(x=list(aucs), y=list(names), palette='viridis')
    plt.xlabel('ROC AUC'); plt.title('Classifier Comparison')
    plt.savefig(COMPARISON_PNG); plt.close()
    pred = best_pipe.predict(X_test)
    logger.info("Classification report for %s:\n%s",
                best_name, classification_report(y_test, pred))

# -----------------------------------------------------------------------------
# STEP 6: CUSTOMER SEGMENTATION (K-Means)
# -----------------------------------------------------------------------------
def customer_segmentation(df: pd.DataFrame, max_k: int = 10):
    feats  = df[['age','ticket_price','number_of_person','revenue']]
    scaler = StandardScaler()
    Xs     = scaler.fit_transform(feats)
    ks     = list(range(2, max_k+1))
    inertias, sils = [], []
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42).fit(Xs)
        inertias.append(km.inertia_)
        sils.append(silhouette_score(Xs, km.labels_))
    plt.figure(); plt.plot(ks, inertias, 'o-'); plt.title('Elbow Method')
    plt.savefig(f'{ARTIFACT_DIR}/elbow.png'); plt.close()
    plt.figure(); plt.plot(ks, sils, 'o-'); plt.title('Silhouette Scores')
    plt.savefig(f'{ARTIFACT_DIR}/silhouette.png'); plt.close()
    best_k = ks[np.argmax(sils)]
    df['cluster'] = KMeans(n_clusters=best_k,
                           random_state=42).fit_predict(Xs)
    joblib.dump(scaler, os.path.join(ARTIFACT_DIR,'cluster_scaler.pkl'))
    logger.info("Cluster profiles:\n%s",
                df.groupby('cluster')
                  [['age','ticket_price','revenue','number_of_person']]
                  .mean())

# -----------------------------------------------------------------------------
# MAIN EXECUTION
# -----------------------------------------------------------------------------
def main():
    df = load_and_clean(DATA_PATH)
    df = engineer_features(df)
    df.to_csv(CLEANED_PATH, index=False)
    logger.info("Cleaned data saved to %s", CLEANED_PATH)

    run_eda(df)
    run_stat_tests(df)
    compare_classifiers(df)
    customer_segmentation(df)

    logger.info("✅ Pipeline complete. Artifacts in '%s'", ARTIFACT_DIR)

if __name__ == '__main__':
    main()
