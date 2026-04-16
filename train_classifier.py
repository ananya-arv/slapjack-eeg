"""
NeuroJack — SSVEP Classifier Training
=======================================
Trains an LDA + SVM ensemble classifier on the synthetic EEG sessions.
Saves the model as a pickle for the streaming server to load.

Run this once before starting the game server.
"""

import numpy as np
import pandas as pd
import json
import pickle
import os
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix

def main():
    print("=" * 60)
    print("NeuroJack — SSVEP Classifier Training")
    print("=" * 60)
    
    # ── Load data ─────────────────────────────────────────────────
    df = pd.read_csv('training_data/all_sessions_combined.csv')
    print(f"\nLoaded {len(df)} trials from {df['session_id'].nunique()} sessions")
    print(f"  Class balance: {df['label'].value_counts().to_dict()}")
    
    feat_cols = [c for c in df.columns if c.startswith('feat_')]
    X = df[feat_cols].values
    y = df['label_int'].values
    
    print(f"\nFeature matrix: {X.shape}")
    
    # ── Build classifiers ─────────────────────────────────────────
    # LDA — great for SSVEP (linear separation in frequency domain)
    lda = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LinearDiscriminantAnalysis(solver='svd', shrinkage=None))
    ])
    
    # SVM with RBF kernel — handles non-linearities
    svm = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42))
    ])
    
    # Random Forest — robust to noise
    rf = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1))
    ])
    
    # Ensemble voting — soft voting uses probabilities
    ensemble = VotingClassifier(
        estimators=[('lda', lda), ('svm', svm), ('rf', rf)],
        voting='soft',
        weights=[2, 2, 1]   # LDA and SVM weighted higher
    )
    
    # ── Cross-validation ──────────────────────────────────────────
    print("\nCross-validation (5-fold):")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, clf in [('LDA', lda), ('SVM', svm), ('RF', rf), ('Ensemble', ensemble)]:
        scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
        print(f"  {name:12s}: {scores.mean():.4f} ± {scores.std():.4f}  (min={scores.min():.4f})")
    
    # ── Final training on all data ────────────────────────────────
    print("\nTraining final model on all data...")
    ensemble.fit(X, y)
    
    # Evaluate on training data (sanity check)
    y_pred = ensemble.predict(X)
    print("\nTraining set performance:")
    print(classification_report(y, y_pred, target_names=['baseline', 'slap']))
    
    cm = confusion_matrix(y, y_pred)
    print(f"Confusion matrix:\n{cm}")
    
    # ── Save model ────────────────────────────────────────────────
    os.makedirs('model', exist_ok=True)
    
    model_bundle = {
        'classifier': ensemble,
        'feat_cols': feat_cols,
        'classes': ['baseline', 'slap'],
        'trained_on': len(df),
        'accuracy': float((y_pred == y).mean()),
    }
    
    with open('model/ssvep_classifier.pkl', 'wb') as f:
        pickle.dump(model_bundle, f, protocol=4)
    
    print(f"\n✓ Model saved → model/ssvep_classifier.pkl")
    print(f"  Accuracy on training data: {(y_pred==y).mean():.4f}")
    print(f"  Ready for real-time classification!")
    
    # Save feature importance (from RF)
    rf.fit(X, y)
    importances = rf.named_steps['clf'].feature_importances_
    feat_importance = dict(zip(feat_cols, importances.tolist()))
    feat_sorted = sorted(feat_importance.items(), key=lambda x: -x[1])
    
    print("\nTop features:")
    for feat, imp in feat_sorted[:5]:
        print(f"  {feat}: {imp:.4f}")
    
    with open('model/feature_importance.json', 'w') as f:
        json.dump({'features': feat_sorted}, f, indent=2)


if __name__ == '__main__':
    main()
