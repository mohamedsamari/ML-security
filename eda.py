import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Charger le dataset
df = pd.read_csv("dataset/kdd_train.csv")

# Voir les premières lignes
print("First rows:")
print(df.head())

# Informations sur les données
print("\nDataset info:")
print(df.info())

# Statistiques
print("\nStatistics:")
print(df.describe())

# Distribution des classes
print("\nLabels distribution:")
print(df['labels'].value_counts())

# distribution des classes
sns.countplot(x=df['labels'])
plt.title("Attack distribution")
plt.xticks(rotation=90)
plt.show()