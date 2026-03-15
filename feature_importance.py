from preprocessing import load_and_preprocess
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# charger données
X, y = load_and_preprocess("dataset/kdd_train.csv")

# entraîner modèle
model = RandomForestClassifier()

model.fit(X, y)

# importance des features
importances = model.feature_importances_

# afficher résultats
feature_importance = pd.DataFrame({
    "feature": X.columns,
    "importance": importances
})

print(feature_importance.sort_values(by="importance", ascending=False))