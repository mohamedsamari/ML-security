from preprocessing import load_and_preprocess
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# charger le modèle sauvegardé
model = joblib.load("ids_model.pkl")

# charger les données de test
X_test, y_test = load_and_preprocess("dataset/kdd_test.csv")

# faire les prédictions
y_pred = model.predict(X_test)

# accuracy
print("Test Accuracy:", accuracy_score(y_test, y_pred))

# confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))