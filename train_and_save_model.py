"""
Train Credit Risk Model and Save as PKL
This script trains the CatBoost model and saves it as a pickle file
for use by the credit scoring API.
"""
import pickle
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

# Features used in the model
FEATURES = [
    "utilisation", "dpd_days", "cash_credit_ratio", "cash_debit_ratio",
    "inbound_cheque_bounce_count", "inbound_cheque_bounce_amt",
    "outbound_cheque_bounce_count", "outbound_cheque_bounce_amt",
    "total_amt_credit", "total_amt_debit", "no_of_banks"
]


def generate_dummy_data(n: int = 50_000) -> pd.DataFrame:
    """Generate dummy credit data for training."""
    rng = np.random.default_rng(42)
    cust = np.arange(1, n + 1)
    util = np.clip(rng.normal(5, 3, n).round(2), 0, 20)
    dpd = np.clip(rng.integers(0, 400, n), 0, 365)
    cash_c = np.clip(rng.normal(0.12, 0.06, n), 0, 0.35)
    cash_d = np.clip(rng.normal(0.12, 0.06, n), 0, 0.35)
    inb_cnt = rng.integers(0, 35, n)
    inb_amt = rng.integers(0, 500_000, n)
    out_cnt = rng.integers(0, 35, n)
    out_amt = rng.integers(0, 500_000, n)
    tot_cr = rng.integers(50_000, 3_000_000, n)
    tot_db = rng.integers(50_000, 3_000_000, n)
    banks = rng.integers(1, 16, n)
    
    # Create synthetic risky label
    prob = (
        0.25*(util/20) + 0.15*(dpd/365) + 0.15*cash_c + 0.15*cash_d
        + 0.10*(inb_cnt/30) + 0.10*(out_cnt/30)
        + 0.10*(np.maximum(inb_amt, out_amt)/5e5)
    )
    prob = np.clip(prob, 0, 1)
    thresh = np.quantile(prob, 0.70)  # ~30% positives
    y = (prob >= thresh).astype(np.int8)
    
    if np.unique(y).size < 2:
        order = np.argsort(prob)
        y[:] = 0
        y[order[-30:]] = 1
    
    df = pd.DataFrame({
        "customer_no": cust,
        "utilisation": util,
        "dpd_days": dpd,
        "cash_credit_ratio": cash_c,
        "cash_debit_ratio": cash_d,
        "inbound_cheque_bounce_count": inb_cnt,
        "inbound_cheque_bounce_amt": inb_amt,
        "outbound_cheque_bounce_count": out_cnt,
        "outbound_cheque_bounce_amt": out_amt,
        "total_amt_credit": tot_cr,
        "total_amt_debit": tot_db,
        "no_of_banks": banks,
        "risky": y
    })
    return df.astype("float32", errors="ignore")


def train_model(df: pd.DataFrame):
    """Train CatBoost model with robust parameters."""
    print("Preparing training data...")
    X = df[FEATURES].astype("float32")
    y = df["risky"].astype("int8")
    
    # Split data
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    # Further split training for validation
    Xtr_i, Xva, ytr_i, yva = train_test_split(
        Xtr, ytr, test_size=0.20, random_state=42, stratify=ytr
    )
    
    print(f"Training samples: {len(Xtr_i)}")
    print(f"Validation samples: {len(Xva)}")
    print(f"Test samples: {len(Xte)}")
    print(f"Class distribution (train): {ytr.value_counts().to_dict()}")
    
    # Create pools
    train_pool = Pool(Xtr_i, ytr_i)
    val_pool = Pool(Xva, yva)
    
    # Best parameters from grid search
    params = {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "iterations": 600,
        "early_stopping_rounds": 50,
        "random_seed": 42,
        "verbose": 100,
        "allow_writing_files": False,
        "subsample": 0.7,
        "colsample_bylevel": 0.7,
        "depth": 4,
        "learning_rate": 0.05,
        "l2_leaf_reg": 15.0,
        "min_data_in_leaf": 60
    }
    
    print("\nTraining CatBoost model...")
    model = CatBoostClassifier(**params)
    model.fit(train_pool, eval_set=val_pool)
    
    # Get best score
    best_score = model.get_best_score()
    val_auc = best_score.get("validation", {}).get("AUC", 0)
    print(f"\nValidation AUC: {val_auc:.4f}")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    yhat_test = model.predict(Xte).astype(int)
    proba_test = model.predict_proba(Xte)[:, 1]
    
    test_auc = roc_auc_score(yte, proba_test)
    print(f"Test AUC: {test_auc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(yte, yhat_test, target_names=["Safe", "Risky"]))
    
    # Feature importance
    print("\nTop 5 Important Features:")
    importance = model.get_feature_importance(Pool(Xte, yte), type="PredictionValuesChange")
    imp_df = pd.DataFrame({
        "feature": FEATURES,
        "importance": importance
    }).sort_values("importance", ascending=False)
    print(imp_df.head(5).to_string(index=False))
    
    return model


def save_model(model: CatBoostClassifier, filepath: str = "credit_risk_model.pkl"):
    """Save trained model to pickle file."""
    print(f"\nSaving model to {filepath}...")
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"✓ Model saved successfully!")
    
    # Verify saved model
    print("\nVerifying saved model...")
    with open(filepath, 'rb') as f:
        loaded_model = pickle.load(f)
    print("✓ Model loaded successfully from pickle file!")
    
    # Quick test
    test_data = pd.DataFrame([{
        "utilisation": 15.0,
        "dpd_days": 30,
        "cash_credit_ratio": 0.20,
        "cash_debit_ratio": 0.18,
        "inbound_cheque_bounce_count": 2,
        "inbound_cheque_bounce_amt": 5000,
        "outbound_cheque_bounce_count": 1,
        "outbound_cheque_bounce_amt": 2000,
        "total_amt_credit": 500000,
        "total_amt_debit": 450000,
        "no_of_banks": 3
    }])
    
    prob = loaded_model.predict_proba(test_data)[:, 1][0]
    print(f"\nTest prediction: {prob:.4f} ({prob*100:.2f}% risk)")
    print("✓ Model verification complete!")


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("Credit Risk Model Training Pipeline")
    print("=" * 60)
    
    # Generate or load data
    print("\nGenerating training data...")
    df = generate_dummy_data(n=50_000)
    print(f"Generated {len(df)} samples")
    print(f"Features: {FEATURES}")
    
    # Train model
    model = train_model(df)
    
    # Save model
    save_model(model)
    
    print("\n" + "=" * 60)
    print("Training complete! Model ready for API use.")
    print("=" * 60)


if __name__ == "__main__":
    main()
