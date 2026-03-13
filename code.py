from preprocessing import load_and_preprocess
from train_model import train_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Charger et préparer les données
X, y = load_and_preprocess("dataset/kdd_train.csv")

# Séparer train et test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Entrainer modèle
model = train_model(X_train, y_train)

# Tester modèle
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))