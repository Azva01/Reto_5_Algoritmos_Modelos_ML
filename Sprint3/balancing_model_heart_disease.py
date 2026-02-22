
# %%
#Importar librerías
import os
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd

#librerias de Smote
from imblearn.over_sampling import SMOTE

# %%
def load_data(file_path):
    """
    Importa el dataset y muestra informacion.
    """
    df_heart = pd.read_csv(file_path)

    return df_heart
df_heart = load_data('heart_2020_cleaned.csv')

# %%
#Convertir variables a minusculas
df_heart.columns = [col.lower() for col in df_heart.columns]
#Transformar variable objetivo a binario (1 Yes y 0 No)
df_heart['heartdisease'] = df_heart['heartdisease'].map({'Yes': 1, 'No': 0})

#Dividir variables por tipo. numericas o categoricas
num_columns = [
    'bmi', 'physicalhealth', 'mentalhealth', 'sleeptime'
    ]

cat_columns = [
    col for col in df_heart.columns
    if col not in num_columns and col != 'heartdisease' #omitir esta
    ]

# %%
#Procesamiento de variables numericas
#Utilizar StandardScaler en variables numericas
s_scaler = StandardScaler()
df_heart[num_columns] = s_scaler.fit_transform(df_heart[num_columns])

# %%
#Procesamiento de variables categoricas

#Convertir en enteros binario para categoricas con solo 2
binaria = [
    'smoking', 'alcoholdrinking', 'stroke','diffwalking', 'sex', 
    'physicalactivity', 'asthma', 'kidneydisease', 'skincancer'
    ]

def transform_binaria(df_heart, type_columns):
    for col in type_columns:
        df_heart[col] = pd.factorize(df_heart[col])[0]
    
    return df_heart

# %%
#Usar onehotencoder de pd para categoricas de 3 o mas
multi_cat = [
    'agecategory', 'race', 'diabetic', 'genhealth'
    ]

def transform_one_hot_encoding(df_heart, type_columns):
    return pd.get_dummies(df_heart, columns=type_columns, dtype=int)#add dtype

# %%
#Aplicar funciones a ambos tipos de variable
df_heart = transform_binaria(df_heart, binaria)
df_heart = transform_one_hot_encoding(df_heart, multi_cat)

print(df_heart.info())
print (f"Forma del dataset: {df_heart.shape}")

print(df_heart.head(4))



# %%
#DIVISION DE DATOS 80 y 20 (no tocar ese 20)
X = df_heart.drop('heartdisease', axis=1)#Busca en columnas no filas
y = df_heart['heartdisease']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# %%
# Modelo 1: Tecnica de balanceo con Weighted Class Logistic Regression
model = LogisticRegression(
    max_iter=1000,
    C=1, #Valor ganador en GridSearch
    class_weight= 'balanced' #balancear weight
    )

# %%
#ENTRENAMIENTO de modelo
model.fit(X_train, y_train)

# %%
#EVALUACION de Modelo con balanceo
# precision, recall, accuracy y f1-score

y_train_predict_w = model.predict(X_train)
y_test_predict_w = model.predict(X_test)

print('---Evaluación conjunto de entrenamiento balanceado---')
print(classification_report(y_train, y_train_predict_w))

print('---Evaluación conjunto de prueba balanceado---')
print(classification_report(y_test, y_test_predict_w))

# %%
print('---Accuracy score---')
print(accuracy_score(y_test, y_test_predict_w))
# %%
#Modelo 2: Tecnica SMOTE
sm = SMOTE(random_state=42)
#aplicamos a los datos sin procesar por WCLR
X_res, y_res = sm.fit_resample(X_train, y_train) #Balance 

model_smote = LogisticRegression(
    max_iter=1000,
    C=1, #Valor ganador en GridSearch
    )
model_smote.fit(X_res, y_res)#usamos datos balanceados por smote

# %%
#EVALUACION de Modelo con balanceo SMOTE
# precision, recall, accuracy y f1-score

y_train_predict_smote = model_smote.predict(X_train)
y_test_predict_smote = model_smote.predict(X_test)

print('---Evaluación conjunto de entrenamiento balanceado---')
print(classification_report(y_train, y_train_predict_smote))

print('---Evaluación conjunto de prueba balanceado---')
print(classification_report(y_test, y_test_predict_smote))