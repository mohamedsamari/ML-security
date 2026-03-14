from sklearn.ensemble import RandomForestClassifier

<<<<<<< HEAD
#Entraîner le modèle
def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model
=======
def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model
>>>>>>> 6e3a2a09634f8f8a92c540cd44e21c494b9ab5f8
