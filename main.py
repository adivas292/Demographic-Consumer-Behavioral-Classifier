from src.preprocessing import preprocess_transaction_data
from src.train import train_behavioral_classifier

if __name__ == "__main__":
    print("🚀 Initializing Demographic Consumer Behavioral Pipeline...")
    
    # Route dataset target
    data_path = "data/credit_card_transactions.csv"
    
    # Execute modular pipeline stages
    X, y = preprocess_transaction_data(data_path)
    model, accuracy, cnf_matrix = train_behavioral_classifier(X, y)
    
    # Output performance indicators to system log
    print("\n--- MODEL PERFORMANCE DIAGNOSTICS ---")
    print(f"✅ Cleaned Feature Matrix Shape: {X.shape}")
    print(f"✅ Extracted Feature Set: {list(X.columns)}")
    print(f"✅ Classification Accuracy Score: {accuracy * 100:.2f}%\n")
