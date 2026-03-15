from preprocessing import load_and_preprocess
from imblearn.over_sampling import SMOTE
import pandas as pd

# charger données prétraitées
X, y = load_and_preprocess("dataset/kdd_train.csv")

# afficher distribution avant SMOTE
print("Before SMOTE:")
print(pd.Series(y).value_counts())

# appliquer SMOTE
smote = SMOTE(random_state=42, k_neighbors=1)
X_resampled, y_resampled = smote.fit_resample(X, y)

# afficher distribution après SMOTE
print("\nAfter SMOTE:")
print(pd.Series(y_resampled).value_counts())