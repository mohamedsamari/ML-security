from preprocessing import load_and_preprocess
from sklearn.ensemble import RandomForestClassifier
import joblib

# charger les données
X, y = load_and_preprocess("dataset/kdd_train.csv")

# entraîner le modèle
model = RandomForestClassifier()

model.fit(X, y)

# sauvegarder le modèle
joblib.dump(model, "ids_model.pkl")

print("Model saved successfully")