import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import urllib.request
import gzip
import os

# Define la ruta del archivo sell-in
file_path = 'data/raw/sell-in.txt.gz'
os.makedirs(os.path.dirname(file_path), exist_ok=True)

# Define la ruta del archivo de productos a predecir
products_to_predict_path = 'data/raw/product_id_apredecir201912.txt'

# Paso 1: Descargar archivos si no existen
if not os.path.exists(file_path):
    print("Descargando sell-in...")
    urllib.request.urlretrieve(
        'https://storage.googleapis.com/open-courses/austral2025-af91/labo3v/sell-in.txt.gz',
        file_path
    )

if not os.path.exists(products_to_predict_path):
    print("Descargando product_id a predecir...")
    urllib.request.urlretrieve(
        'https://storage.googleapis.com/open-courses/austral2025-af91/labo3v/product_id_apredecir201912.txt',
        products_to_predict_path
    )

# Cargar la lista de productos a predecir
productos_a_predecir_df = pd.read_csv(products_to_predict_path)
productos_a_predecir_lista = productos_a_predecir_df['product_id'].tolist()

# Paso 2: Leer y agregar por producto-mes
df = pd.read_csv(file_path, sep='\t', compression='gzip')
df_agg = df.groupby(['product_id', 'periodo'])['tn'].sum().reset_index()

# Paso 3: Crear lags de 12 meses
df_agg['periodo'] = pd.to_datetime(df_agg['periodo'], format='%Y%m')
df_agg = df_agg.sort_values(['product_id', 'periodo'])

for i in range(1, 12):
    df_agg[f'tn_{i}'] = df_agg.groupby('product_id')['tn'].shift(i)

# Paso 4: ESTRATEGIA DEL PROFESOR - Asignación directa para dic-2018
# Seleccionar registros de diciembre 2018
train_data = df_agg[df_agg['periodo'] == '2018-12-01'].copy()

# Buscar las toneladas de febrero 2019 para cada producto
feb_2019 = df_agg[df_agg['periodo'] == '2019-02-01'][['product_id', 'tn']]
feb_2019 = feb_2019.rename(columns={'tn': 'clase'})

# Asignar la clase (feb-2019) a los registros de dic-2018
train_data = train_data.merge(feb_2019, on='product_id', how='left')

# Paso 5: Filtrar solo los 33 productos mágicos
magicos = [20002, 20003, 20006, 20010, 20011, 20018, 20019, 20021,
           20026, 20028, 20035, 20039, 20042, 20044, 20045, 20046, 20049,
           20051, 20052, 20053, 20055, 20008, 20001, 20017, 20086, 20180,
           20193, 20320, 20532, 20612, 20637, 20807, 20838]

train_subset = train_data[train_data['product_id'].isin(magicos)]

# Verificar que los 33 productos tienen datos completos
features = ['tn'] + [f'tn_{i}' for i in range(1, 12)]
print(f"Productos mágicos con datos completos: {train_subset[features].notna().all(axis=1).sum()}")

# Paso 6: Entrenar regresión lineal
X_train = train_subset[features]
y_train = train_subset['clase']

modelo = LinearRegression()
modelo.fit(X_train, y_train)

print("\nCoeficientes encontrados:")
print(f"Intercept: {modelo.intercept_:.6f}")
for i, coef in enumerate(modelo.coef_):
    if i == 0:
        print(f"tn: {coef:.6f}")
    else:
        print(f"tn_{i}: {coef:.6f}")

# Paso 7: Predecir para diciembre 2019
predict_data_full = df_agg[df_agg['periodo'] == '2019-12-01'].copy()

# Filtrar solo los productos de la lista
predict_data = predict_data_full[predict_data_full['product_id'].isin(productos_a_predecir_lista)].copy()

# Separar completos e incompletos
mask_completos = predict_data[features].notna().all(axis=1)
completos = predict_data[mask_completos]
incompletos = predict_data[~mask_completos]

# Predecir para completos
if len(completos) > 0:
    X_pred = completos[features]
    completos['prediccion'] = modelo.predict(X_pred)
    completos['prediccion'] = completos['prediccion'].clip(lower=0)

# Para incompletos usar promedio últimos meses disponibles
if len(incompletos) > 0:
    incompletos['prediccion'] = incompletos[features].mean(axis=1)

# Paso 8: Combinar resultados
resultados = pd.concat([
    completos[['product_id', 'prediccion']],
    incompletos[['product_id', 'prediccion']]
])

# Paso 9: Guardar
resultados['product_id'] = resultados['product_id'].astype(int)
resultados = resultados.rename(columns={'prediccion': 'tn'})
resultados = resultados.sort_values('product_id')
resultados.to_csv('predicciones_simple.csv', index=False)

print(f"\nCompletos: {len(completos)} productos ({completos['prediccion'].sum():.2f} tn)")
print(f"Incompletos: {len(incompletos)} productos ({incompletos['prediccion'].sum():.2f} tn)")
print(f"Total predicho: {resultados['tn'].sum():.2f} toneladas")