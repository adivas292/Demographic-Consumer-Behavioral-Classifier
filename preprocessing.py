import pandas as pd
import numpy as np

def preprocess_transaction_data(file_path):
    """
    Cleans missing rows, isolates continuous variables, executes proper 
    One-Hot Encoding on strings, and engineers the binary 'is_weekend' feature.
    """
    # Load dataset
    df = pd.read_csv(file_path)
    
    # Drop missing rows
    df = df.dropna()
    
    # Chronological Feature Extraction
    df['Card Usage Date'] = pd.to_datetime(df['Date'])
    df['is_weekend'] = (df['Card Usage Date'].dt.weekday >= 5).astype(int)
    
    # Correct One-Hot Encoding: Encode ONLY structural string categories
    categorical_cols = ['Card Type', 'Exp Type']
    encoded_cats = pd.get_dummies(df[categorical_cols], drop_first=True).astype(int)
    
    # Map the Target Vector (Female = 1, Male = 0)
    y = df['Gender'].map({'F': 1, 'M': 0}).astype(int)
    
    # Construct unified feature matrix X (Keep Amount as a single continuous feature)
    X = pd.concat([df[['Amount']], encoded_cats, df['is_weekend']], axis=1)
    
    return X, y