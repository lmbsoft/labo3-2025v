# %%
# DATOS_DIR = '../../data/'
# SALIDAS_DIR = '../../salidas/'

# DIR VM
DATOS_DIR = '../datasets/'
SALIDAS_DIR = './salidas/'

# 1. Configuración Global
MODEL_CHOICE = "xgb"  # Opciones: "linear", "elasticnet", "randomforest", "xgb"
SAVE_MODELS = True  # Guardar modelos para análisis posterior
OUTPUT_DIR = "model_results"

nombre_archivo_export = 'forecast_v1tabularAutogloun_350PRODLES.csv'
nombre_fi_export= 'feature_importance_v1tabularAutogloun_350PRODLES.csv'
nombre_lb_export = 'leaderboard_v1tabularAutogloun_350PRODLES.csv' 

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from autogluon.tabular import TabularPredictor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
from autogluon.common.space import Int, Real, Categorical  # Importar espacios de búsqueda desde autogluon.core.space
warnings.filterwarnings("ignore")

# %%
# FILE_BYPRODUCT_SELLIN = DATOS_DIR + 'features_por_producto_20001_59col.csv'
FILE_BYPRODUCT_SELLIN = DATOS_DIR + 'dataset60col_byproducto350productosmenores.csv'
df = pd.read_csv(FILE_BYPRODUCT_SELLIN, parse_dates=["periodo_dt"]) 

# %%
# ------------------------
# 2) Separar en entrenamiento y test
#     👉 Train: periodos <= 201910
#     👉 Test:  periodo = 201912  (input para predecir mes+2)
# ------------------------
df['clase_log'] = np.log1p(df['clase'])
train_set = df[(df['periodo_dt'] <= '2019-10-01') & df['clase_log'].notnull()].copy()
test_set = df[(df['periodo_dt'] == '2019-12-01')].copy()

print(f"Train shape: {train_set.shape} | Test shape: {test_set.shape}")

# %%
features = [col for col in df.columns if col not in [
    'periodo_dt', 'clase_log','clase'
]]

print(f"Total features: {len(features)}")
print(features[:10])  # Algunos de ejemplo


# %%
# -------------------------------
# 4) Entrenar Autogluon Tabular
# -------------------------------
predictor = TabularPredictor(
    label='clase_log',
    problem_type='regression',
    eval_metric='mae',
    path='gcs_model_dir_con_prophet'
)

predictor.fit(
    train_data=train_set[features + ['clase_log']],
    time_limit=120,
     ag_args_fit={'sample_weight': 'sample_weight'},
    presets='best_quality',
    num_bag_folds=5,
    num_stack_levels=4
)

print("✅ Entrenamiento finalizado.")

# ---------------------------
# 4) Predicciones sobre el test set
# ---------------------------

test_set['tn_pred_log'] = predictor.predict(test_set[features])
test_set['tn_pred'] = np.expm1(test_set['tn_pred_log']).clip(lower=0)  # Vuelve a escala original y fuerza >= 0

# ---------------------------
# 5) Agregar predicciones por producto
# ---------------------------

df_final = (
    test_set.groupby('product_id', as_index=False)['tn_pred']
    .sum()
    .rename(columns={'tn_pred': 'tn'})
)

print(df_final.head())

# ---------------------------
# 6) Exportar CSV final
# ---------------------------

df_final.to_csv(SALIDAS_DIR + nombre_archivo_export, index=False)

print("✅ Archivo generado: " + nombre_archivo_export)
print(f"Productos únicos: {df_final['product_id'].nunique()}")
print(f"Total TN predichas (sum): {df_final['tn'].sum():,.2f}")

# %%
# -------------------------------
# 9) Leaderboard (performance interna)
# -------------------------------
print("\n🔍 Leaderboard:")
lb = predictor.leaderboard(silent=True)
print(lb)
lb.to_csv(SALIDAS_DIR + nombre_lb_export, index=False)
# -------------------------------
# 10) Importancia de features
# -------------------------------
print("\n🔍 Importancia de Features:")
fi = predictor.feature_importance(train_set[features + ['clase']])
fi = fi.reset_index().rename(columns={'index': 'feature'})
# print(fi.head(100))
fi.to_csv(SALIDAS_DIR + nombre_fi_export, index=False)
print("\n🔍 PROCESO FINALIZADO:")



