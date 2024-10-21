import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('healthcare-dataset-stroke-data.csv')

print("Dados faltantes antes da limpeza:")
print(df.isnull().sum())

df['bmi'] = df['bmi'].fillna(df['bmi'].mean())

df['smoking_status'] = df['smoking_status'].fillna('Desconhecido')

df['gender'] = df['gender'].map({'Male': 1, 'Female': 0, 'Other': 2}) 
df['ever_married'] = df['ever_married'].map({'Yes': 1, 'No': 0})
df['Residence_type'] = df['Residence_type'].map({'Urban': 1, 'Rural': 0})

df = pd.get_dummies(df, columns=['smoking_status'], drop_first=True)

print("Primeiras linhas do DataFrame após limpeza:")
print(df.head())
#----------------------------------------------------------------------------------------------------#
plt.figure(figsize=(8, 6))
sns.histplot(df[df['stroke'] == 1]['age'], bins=20, color='purple', kde=True, stat='density', alpha=0.7)
plt.title('Distribuição de Idade entre Pacientes com AVC')
plt.xlabel('Idade')
plt.ylabel('Densidade')
plt.show()
#----------------------------------------------------------------------------------------------------#
plt.figure(figsize=(10, 6))
sns.boxplot(x='hypertension', y='age', data=df)
plt.title('Distribuição de Idades por Hipertensão')
plt.xlabel('Hipertensão')
plt.ylabel('Idade')
plt.show()
#----------------------------------------------------------------------------------------------------#
plt.figure(figsize=(10, 6))
sns.countplot(x='stroke', hue='hypertension', data=df)
plt.title('Contagem de AVCs por Hipertensão')
plt.xlabel('Sofreu AVC (1=Sim, 0=Não)')
plt.ylabel('Contagem')
plt.legend(title='Hipertensão', loc='upper right', labels=['Não', 'Sim'])
plt.show()
#----------------------------------------------------------------------------------------------------#

X = df[['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi', 
        'gender', 'ever_married', 'Residence_type', 
        'smoking_status_formerly smoked', 'smoking_status_never smoked', 'smoking_status_smokes']]
y = df['stroke']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000, class_weight='balanced') 
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

print(f"Acurácia: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))

#----------------------------------------------------------------------------------------------------#
from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_estimator(model, X_test_scaled, y_test, cmap='Blues')
plt.title('Matriz de Confusão')
plt.show()
