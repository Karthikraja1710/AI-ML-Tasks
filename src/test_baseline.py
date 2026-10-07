import pandas as pd
import numpy as np

def run_analysis():
    # Load dataset
    data_path = r"g:\New folder\RabTech\AI-ML-Tasks\data\WA_Fn-UseC_-Telco-Customer-Churn.csv"
    df = pd.read_csv(data_path)
    
    # Preprocessing
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].astype(str).str.strip(), errors='coerce').fillna(0.0)
    
    y = (df['Churn'] == 'Yes').astype(int).values
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    cat_cols = [c for c in df.columns if c not in num_cols + ['customerID', 'Churn']]
    
    df_encoded = pd.get_dummies(df[cat_cols], drop_first=True, dtype=float)
    X_df = pd.concat([df[num_cols], df_encoded], axis=1)
    
    # Leak-free Stratified Split (80/20)
    np.random.seed(42)
    indices = np.arange(len(df))
    pos_idx = indices[y == 1]
    neg_idx = indices[y == 0]
    
    np.random.shuffle(pos_idx)
    np.random.shuffle(neg_idx)
    
    n_pos_test = int(len(pos_idx) * 0.2)
    n_neg_test = int(len(neg_idx) * 0.2)
    
    test_idx = np.concatenate([pos_idx[:n_pos_test], neg_idx[:n_neg_test]])
    train_idx = np.concatenate([pos_idx[n_pos_test:], neg_idx[n_neg_test:]])
    
    X_train, y_train = X_df.iloc[train_idx].values, y[train_idx]
    X_test, y_test = X_df.iloc[test_idx].values, y[test_idx]
    
    # Feature scaling (fit on train, transform test)
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0
    
    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std
    
    X_train_b = np.hstack([np.ones((len(X_train_scaled), 1)), X_train_scaled])
    X_test_b = np.hstack([np.ones((len(X_test_scaled), 1)), X_test_scaled])
    
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -25, 25)))
    
    # Train Logistic Regression
    weights = np.zeros(X_train_b.shape[1])
    lr = 0.1
    for epoch in range(1000):
        preds = sigmoid(X_train_b @ weights)
        weights -= lr * ((X_train_b.T @ (preds - y_train)) / len(y_train))
        
    probs = sigmoid(X_test_b @ weights)
    
    # Financial parameters
    C_FP = 20    # $15 discount + $5 admin outreach
    C_FN = 500   # Lost LTV ($780) minus acquisition recovery ($280) = $500
    
    print("=== MODEL EVALUATION & FINANCIAL COST SUMMARY ===")
    
    # 1. Majority Class Baseline
    maj_preds = np.zeros(len(y_test))
    maj_tp = np.sum((maj_preds == 1) & (y_test == 1))
    maj_fp = np.sum((maj_preds == 1) & (y_test == 0))
    maj_tn = np.sum((maj_preds == 0) & (y_test == 0))
    maj_fn = np.sum((maj_preds == 0) & (y_test == 1))
    maj_acc = (maj_tp + maj_tn) / len(y_test)
    maj_cost = maj_fp * C_FP + maj_fn * C_FN
    print(f"Baseline 0 (Majority Class): Acc={maj_acc:.4f}, FP={maj_fp}, FN={maj_fn}, Total Loss=${maj_cost:,.2f}")
    
    # 2. Heuristic Rule Baseline
    df_test = df.iloc[test_idx]
    h_mask = (df_test['Contract'] == 'Month-to-month') & (df_test['InternetService'] == 'Fiber optic')
    h_preds = h_mask.astype(int).values
    h_tp = np.sum((h_preds == 1) & (y_test == 1))
    h_fp = np.sum((h_preds == 1) & (y_test == 0))
    h_tn = np.sum((h_preds == 0) & (y_test == 0))
    h_fn = np.sum((h_preds == 0) & (y_test == 1))
    h_acc = (h_tp + h_tn) / len(y_test)
    h_prec = h_tp / (h_tp + h_fp) if (h_tp + h_fp) > 0 else 0
    h_rec = h_tp / (h_tp + h_fn) if (h_tp + h_fn) > 0 else 0
    h_f1 = 2 * h_prec * h_rec / (h_prec + h_rec) if (h_prec + h_rec) > 0 else 0
    h_cost = h_fp * C_FP + h_fn * C_FN
    print(f"Baseline 1 (Heuristic Rule): Acc={h_acc:.4f}, Prec={h_prec:.4f}, Rec={h_rec:.4f}, F1={h_f1:.4f}, FP={h_fp}, FN={h_fn}, Total Loss=${h_cost:,.2f}")
    
    # 3. Logistic Regression Threshold Sweep
    print("\nBaseline 2 (Logistic Regression - Threshold Tuning):")
    best_thresh = 0.5
    min_cost = float('inf')
    
    for thresh in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
        p = (probs >= thresh).astype(int)
        tp = np.sum((p == 1) & (y_test == 1))
        fp = np.sum((p == 1) & (y_test == 0))
        tn = np.sum((p == 0) & (y_test == 0))
        fn = np.sum((p == 0) & (y_test == 1))
        acc = (tp + tn) / len(y_test)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        cost = fp * C_FP + fn * C_FN
        if cost < min_cost:
            min_cost = cost
            best_thresh = thresh
        print(f"  Thresh={thresh:.2f} | Acc={acc:.4f} | Prec={prec:.4f} | Rec={rec:.4f} | F1={f1:.4f} | FP={fp:3d} | FN={fn:3d} | Financial Loss=${cost:,.2f}")
        
    print(f"\nCost-Optimal Threshold: {best_thresh:.2f} with Minimum Financial Loss: ${min_cost:,.2f}")

if __name__ == '__main__':
    run_analysis()
