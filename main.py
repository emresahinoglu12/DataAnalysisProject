#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cinema Hall Ticket Sales Analysis
---------------------------------
Author: Your Name
Date: 2025-05-19

Description:
    End-to-end pipeline for analyzing cinema ticket sales:
    1. Data loading & cleaning
    2. Feature engineering
    3. Exploratory Data Analysis (EDA)
    4. Statistical hypothesis testing
    5. Predictive modeling (logistic regression)
    6. Customer segmentation (K-Means clustering)
    7. Saving outputs & artifacts

Usage:
    python main.py

Outputs:
    - Cleaned data CSV
    - Model & scaler pickles
    - A suite of .png visuals under /visuals
    - Console summary of findings
"""

import os
import logging

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, roc_auc_score,
    roc_curve, confusion_matrix
)
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
DATA_PATH     = 'data/cinema_hall_ticket_sales.csv'
CLEANED_PATH  = 'data/cleaned_cinema_data.csv'
ARTIFACT_DIR  = 'visuals'
MODEL_PATH    = 'models/logistic_model.pkl'
SCALER_PATH   = 'models/scaler.pkl'

# ensure directories exist
os.makedirs(os.path.dirname(CLEANED_PATH) or '.', exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_PATH) or '.', exist_ok=True)

# set up logging
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
    """Load the raw CSV and perform initial cleaning."""
    logger.info("Loading data from %s", path)
    df = pd.read_csv(path)

    # Rename columns to snake_case
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]

    # Convert 'number_of_person' from 'Alone' or string to int
    df['number_of_person'] = df['number_of_person'] \
        .replace('Alone', '1').astype(int)

    # Map purchase_again
    df['purchase_again'] = df['purchase_again'].map({'Yes': 1, 'No': 0})

    # Log missing values, then drop any
    missing = df.isnull().sum()
    logger.info("Missing values by column:\n%s", missing)
    df = df.dropna().reset_index(drop=True)

    # Parse date if it exists
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])

    return df

# -----------------------------------------------------------------------------
# STEP 2: FEATURE ENGINEERING
# -----------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create additional features for deeper insights."""
    df['revenue'] = df['number_of_person'] * df['ticket_price']
    df['price_per_person'] = df['ticket_price'] / df['number_of_person']

    # Age groups
    bins = [0, 17, 35, 50, 65, 120]
    labels = ['<18','18-35','36-50','51-65','65+']
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels)

    if 'date' in df.columns:
        df['day_of_week'] = df['date'].dt.day_name()
        df['month']       = df['date'].dt.month_name()
        df['is_weekend']  = df['day_of_week'].isin(['Saturday','Sunday']).astype(int)

    return df

# -----------------------------------------------------------------------------
# STEP 3: EXPLORATORY DATA ANALYSIS (EDA)
# -----------------------------------------------------------------------------
def run_eda(df: pd.DataFrame):
    """Generate and save a suite of EDA visualizations."""
    sns.set(style='whitegrid')

    # 1. Age Distribution
    plt.figure(figsize=(8,5))
    sns.histplot(df['age'], bins=20, kde=True)
    plt.title('Age Distribution')
    plt.xlabel('Age'); plt.ylabel('Count')
    plt.savefig(f'{ARTIFACT_DIR}/age_distribution.png')
    plt.close()

    # 2. Ticket Price by Seat Type
    plt.figure(figsize=(8,5))
    sns.boxplot(x='seat_type', y='ticket_price', data=df)
    plt.title('Ticket Price by Seat Type')
    plt.savefig(f'{ARTIFACT_DIR}/price_by_seat_type.png')
    plt.close()

    # 3. Ticket Sales by Genre
    plt.figure(figsize=(8,5))
    order = df['movie_genre'].value_counts().index
    sns.countplot(x='movie_genre', data=df, order=order)
    plt.xticks(rotation=45)
    plt.title('Ticket Sales by Movie Genre')
    plt.savefig(f'{ARTIFACT_DIR}/genre_counts.png')
    plt.close()

    # 4. Total Revenue by Genre
    plt.figure(figsize=(8,5))
    rev_genre = df.groupby('movie_genre')['revenue'].sum().sort_values(ascending=False)
    sns.barplot(x=rev_genre.values, y=rev_genre.index)
    plt.title('Total Revenue by Genre')
    plt.xlabel('Revenue'); plt.ylabel('Genre')
    plt.savefig(f'{ARTIFACT_DIR}/revenue_by_genre.png')
    plt.close()

    # 5. Repeat Purchase Rate by Age Group
    plt.figure(figsize=(8,5))
    rate = df.groupby('age_group')['purchase_again'].mean()
    rate.plot(kind='bar')
    plt.title('Repeat Purchase Rate by Age Group')
    plt.ylabel('Repeat Purchase Rate')
    plt.savefig(f'{ARTIFACT_DIR}/repeat_rate_by_age_group.png')
    plt.close()

    # 6. Correlation Heatmap
    numeric = df.select_dtypes(include=[np.number])
    plt.figure(figsize=(10,8))
    sns.heatmap(numeric.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix')
    plt.savefig(f'{ARTIFACT_DIR}/correlation_matrix.png')
    plt.close()

    # 7. Weekly Sales Trend
    if 'date' in df.columns:
        plt.figure(figsize=(10,5))
        weekly = df.set_index('date').resample('W').size()
        weekly.plot()
        plt.title('Weekly Ticket Sales Trend')
        plt.ylabel('Tickets Sold')
        plt.savefig(f'{ARTIFACT_DIR}/weekly_sales_trend.png')
        plt.close()

# -----------------------------------------------------------------------------
# STEP 4: STATISTICAL HYPOTHESIS TESTING
# -----------------------------------------------------------------------------
def run_stat_tests(df: pd.DataFrame):
    """Perform and log results of key statistical tests."""
    logger.info("HYPOTHESIS TESTING RESULTS:")

    # H1: Older (>40) prefer Premium seats
    df['older'] = (df['age'] > 40).astype(int)
    cont = pd.crosstab(df['older'], df['seat_type']=='Premium')
    chi2, p, _, _ = stats.chi2_contingency(cont)
    logger.info("H1: Chi-square for Premium vs Age>40: χ²=%.2f, p=%.4f", chi2, p)

    # H2: Repeat purchase rate differs (18-35 vs others)
    young = df[df['age_group']=='18-35']['purchase_again']
    other = df[df['age_group']!='18-35']['purchase_again']
    t, p2 = stats.ttest_ind(young, other, equal_var=False)
    logger.info("H2: T-test for repeat rate (18-35 vs others): t=%.2f, p=%.4f", t, p2)

    # H3: Ticket price differs across genres (ANOVA)
    groups = [g['ticket_price'].values for _, g in df.groupby('movie_genre')]
    f, p3 = stats.f_oneway(*groups)
    logger.info("H3: ANOVA ticket price by genre: F=%.2f, p=%.4f", f, p3)

# -----------------------------------------------------------------------------
# STEP 5: PREDICTIVE MODELING
# -----------------------------------------------------------------------------
def build_and_evaluate_model(df: pd.DataFrame):
    """Train logistic regression to predict `purchase_again`, evaluate, and save artifacts."""
    X = df.drop(columns=['purchase_again'])
    y = df['purchase_again']

    cat_cols = ['seat_type','movie_genre','age_group']
    num_cols = ['age','number_of_person','ticket_price','revenue','price_per_person']
    if 'is_weekend' in df.columns:
        num_cols.append('is_weekend')

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), cat_cols)
    ])

    pipeline = Pipeline([
        ('prep', preprocessor),
        ('clf', LogisticRegression(max_iter=1000))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline.fit(X_train, y_train)
    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:,1]

    auc = roc_auc_score(y_test, y_proba)
    logger.info("Logistic Regression AUC: %.3f", auc)
    logger.info("Classification Report:\n%s", classification_report(y_test, y_pred))

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    plt.plot([0,1],[0,1],'--')
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.savefig(f'{ARTIFACT_DIR}/roc_curve.png')
    plt.close()

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted'); plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(f'{ARTIFACT_DIR}/confusion_matrix.png')
    plt.close()

    joblib.dump(pipeline, MODEL_PATH)
    logger.info("Saved model to %s", MODEL_PATH)

# -----------------------------------------------------------------------------
# STEP 6: CUSTOMER SEGMENTATION
# -----------------------------------------------------------------------------
def customer_segmentation(df: pd.DataFrame, max_k: int = 10):
    """Segment customers via KMeans and save elbow/silhouette plots & scaler."""
    feats = df[['age','ticket_price','number_of_person','revenue']]
    scaler = StandardScaler()
    Xs = scaler.fit_transform(feats)

    inertias = []
    silhouettes = []
    ks = list(range(2, max_k+1))
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42).fit(Xs)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(Xs, km.labels_))

    # Elbow plot
    plt.figure(figsize=(6,4))
    plt.plot(ks, inertias, marker='o')
    plt.title('Elbow Method')
    plt.xlabel('k'); plt.ylabel('Inertia')
    plt.savefig(f'{ARTIFACT_DIR}/elbow.png')
    plt.close()

    # Silhouette plot
    plt.figure(figsize=(6,4))
    plt.plot(ks, silhouettes, marker='o')
    plt.title('Silhouette Scores')
    plt.xlabel('k'); plt.ylabel('Score')
    plt.savefig(f'{ARTIFACT_DIR}/silhouette.png')
    plt.close()

    best_k = ks[np.argmax(silhouettes)]
    logger.info("Chosen k by silhouette: %d", best_k)

    km_final   = KMeans(n_clusters=best_k, random_state=42).fit(Xs)
    df['cluster'] = km_final.labels_

    joblib.dump(scaler, SCALER_PATH)
    logger.info("Saved scaler to %s", SCALER_PATH)

    profile = df.groupby('cluster')[['age','ticket_price','revenue','number_of_person']].mean()
    logger.info("Cluster profiles:\n%s", profile)

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
    build_and_evaluate_model(df)
    customer_segmentation(df)

    logger.info("✅ Analysis complete. Visuals in '%s'", ARTIFACT_DIR)

if __name__ == '__main__':
    main()
