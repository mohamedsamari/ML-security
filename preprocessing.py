import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess(filepath):
    # Load dataset
    df = pd.read_csv(filepath)
    
    # Identify categorical columns
    categorical_cols = ['protocol_type', 'service', 'flag']
    
    # Encode categorical columns
    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])
    
    # Prepare X and y
    X = df.drop('labels', axis=1)
    y = df['labels']
    
    # Convert y to binary if necessary, but code.py uses accuracy_score
    # Let's keep it as is (multiclass or binary depending on labels)
    
    return X, y
