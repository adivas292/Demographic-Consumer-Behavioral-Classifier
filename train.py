from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

def train_behavioral_classifier(X, y):
    """
    Executes an 80/20 data partition and trains a deterministic 
    Logistic Regression classifier with zero data leakage.
    """
    # 80/20 train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and fit classification model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    # Extract structural predictions
    y_pred = model.predict(X_test)
    
    # Compute accuracy metric
    accuracy = accuracy_score(y_test, y_pred)
    cnf_matrix = confusion_matrix(y_test, y_pred)
    
    return model, accuracy, cnf_matrix
