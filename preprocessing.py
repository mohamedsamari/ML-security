import pandas as pd
from sklearn.preprocessing import LabelEncoder

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