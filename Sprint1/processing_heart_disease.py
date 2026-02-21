
# %%
#Importar librerías
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd

# %%
#Importación de datos
def load_data(file_path):
    """
    Importa el dataset y muestra informacion.
    """
    df_heart = pd.read_csv(file_path)
    print(
        f"Dataset 'heart_2020_cleaned.csv' cargado. "
        f"{df_heart.shape[0]} filas, {df_heart.shape[1]} columnas."
        )
    return df_heart
    

df_heart = load_data('heart_2020_cleaned.csv')
df_heart.head(6)

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
print(cat_columns)

# %%
#Procesamiento de variables numericas
#Utilizar StandardScaler en variables numericas
s_scaler = StandardScaler()
df_heart[num_columns] = s_scaler.fit_transform(df_heart[num_columns])

print(df_heart[num_columns].head())

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

# %%
#Guardar archivo .csv
df_heart.to_csv('processing_heart_disease.csv', index=False) #Siempre False
print('El archivo:"processing_heart_disease.csv" se ha creado correctamente.')

# %%
#Verificacion de correcto procesamiento de datos
print(df_heart.info())
print(f"Existe el archivo: {os.path.exists('processing_heart_disease.csv')}")
print (f"Forma del dataset: {df_heart.shape}")
