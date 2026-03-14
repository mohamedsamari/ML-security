import pandas as pd
from sklearn.preprocessing import LabelEncoder

<<<<<<< HEAD
def load_and_preprocess(path):

    data = pd.read_csv(path)

    # Convertir labels
    data["labels"] = data["labels"].apply(lambda x: 0 if x == "normal" else 1)

    # Encoder les colonnes texte
    encoder = LabelEncoder()
    data["protocol_type"] = encoder.fit_transform(data["protocol_type"])
    data["service"] = encoder.fit_transform(data["service"])
    data["flag"] = encoder.fit_transform(data["flag"])

    # Séparer X et y
    X = data.drop("labels", axis=1)
    y = data["labels"]

    return X, y
=======
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
>>>>>>> 6e3a2a09634f8f8a92c540cd44e21c494b9ab5f8
