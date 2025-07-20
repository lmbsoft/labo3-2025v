# -*- coding: utf-8 -*-
"""
Pipeline de Ensamblado por Semillerío
=====================================

Este script implementa una técnica de ensamblado robusta conocida como "semillerío".
Su propósito es mejorar la estabilidad y generalización del modelo final entrenando
múltiples modelos de LightGBM con diferentes semillas aleatorias y promediando 
sus predicciones.

El flujo de trabajo es el siguiente:
1. Carga las features avanzadas y los mejores hiperparámetros generados por el 
   script `workaround_11_ratio_log.py`.
2. Genera una secuencia de semillas (números primos) para asegurar diversidad.
3. Entrena un modelo por cada semilla, guardando su predicción individual.
4. Ensambla todas las predicciones individuales (usando el promedio) para crear
   un forecast final más robusto.
5. Genera un archivo de submission para el ensamble y un reporte de resumen.

Parámetros Principales de Configuración (dentro de la clase Config):
--------------------------------------------------------------------
- GRANULARITY: 'product' o 'customer'. Debe ser consistente con el script de features.
- FEATURE_VERSION: Versión de las features a utilizar.
- INITIAL_SEED: Semilla inicial para generar la secuencia de modelos.
- NUM_MODELS: Cantidad de modelos a entrenar en el semillerío.

"""

import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import logging
import warnings
import optuna
import glob
from typing import List, Tuple, Optional

warnings.filterwarnings('ignore')

class Config:
    GRANULARITY     = 'product'       # 'product' | 'customer'
    FEATURE_VERSION = 'v6_log_ratio_advanced'  # Debe coincidir con workaround_11_ratio_log.py
    CACHE_PATH      = 'cache_log_ratio_advanced'
    ARTIFACTS_PATH  = 'artifacts_log_ratio_advanced'
    
    # URLs para descargar datos
    SELLIN_URL = 'https://storage.googleapis.com/open-courses/austral2025-af91/labo3v/sell-in.txt.gz'
    PRODUCTOS_A_PREDECIR_URL = 'https://storage.googleapis.com/open-courses/austral2025-af91/labo3v/product_id_apredecir201912.txt'
    
    # Configuración del semillerío
    INITIAL_SEED = 1019
    NUM_MODELS = 125  # Cantidad de modelos a entrenar
    MAX_RETRIES = 3  # Intentos máximos por modelo
    
    FINAL_TRAIN_END = '2019-12-01'
    
    # Features categóricas alineadas
    CATEGORICAL_FEATURES = [
        'product_id', 'customer_id', 'cat1', 'cat2', 'cat3', 'brand', 'quarter',
        *[f'bin_decil_lag_{i}' for i in range(0,13)],   
        *[f'bin_octil_lag_{i}' for i in range(0,13)],   
        *[f'bin_cuart_lag_{i}' for i in range(0,13)],   
        *[f'bin_decil_str_{k}' for k in range(4,9)],
        *[f'bin_octil_str_{k}' for k in range(4,9)],
        *[f'bin_cuart_str_{k}' for k in range(4,9)],
        'is_holiday_season', 'is_back_to_school', 'is_winter_peak', 'is_summer_low'
    ]
    
    # Columnas a excluir
    COLUMNS_TO_EXCLUDE = [
        'periodo', 'y', 'scaler', 'descripcion', 'row_id', 'cust_any_purchase',
        'historical_max_tn', 'tn', 'tn_future_unscaled'
    ]
    
    # LGBM Fixed Params ALINEADOS
    LGBM_FIXED_PARAMS = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt', 
        'n_estimators': 2000, 
        'n_jobs': -1,
        'verbose': -1,
        'bagging_freq': 1,
        'linear_tree': False, 
        'max_bin': 2800,
    }

def get_logger():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    optuna.logging.enable_propagation()
    optuna.logging.disable_default_handler()
    return logging.getLogger()

logger = get_logger()

def is_prime(n: int) -> bool:
    """Verifica si un número es primo."""
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

def next_prime(n: int) -> int:
    """Encuentra el siguiente número primo después de n."""
    candidate = n + 1
    while not is_prime(candidate):
        candidate += 1
    return candidate

def calculate_birth_dates(cfg):
    """Calcula las fechas de nacimiento de productos y clientes."""
    logger.info("Calculando fechas de nacimiento...")
    
    # Cargar sellin para calcular fechas de nacimiento
    sellin = pd.read_csv(
        cfg.SELLIN_URL,
        sep="\t", 
        compression='gzip',
        dtype={'periodo': str, 'customer_id': str, 'product_id': str}
    )
    sellin['periodo'] = pd.to_datetime(sellin['periodo'], format='%Y%m')
    
    birth_dates = {}
    
    # Fecha de nacimiento de productos
    birth_dates['product'] = sellin.groupby('product_id')['periodo'].min().reset_index()
    birth_dates['product'].columns = ['product_id', 'birth_date_product']
    
    # Fecha de nacimiento de clientes
    birth_dates['customer'] = sellin.groupby('customer_id')['periodo'].min().reset_index()
    birth_dates['customer'].columns = ['customer_id', 'birth_date_customer']
    
    return birth_dates

def load_features_with_birth_filter(cfg, birth_dates):
    """Carga features avanzadas (cache nivel 3) y aplica filtro de fechas de nacimiento."""
    # Buscar cache nivel 3 (features avanzadas)
    path = os.path.join(cfg.CACHE_PATH, f'features_{cfg.GRANULARITY}_{cfg.FEATURE_VERSION}_ii.parquet')
    
    if not os.path.exists(path):
        logger.error(f"No se encontró {path}")
        logger.error("Ejecuta primero workaround_11_ratio_log.py para crear features avanzadas")
        exit(1)
    
    logger.info(f"Cargando features avanzadas desde {path}")
    df = pd.read_parquet(path)
    
    # Aplicar filtro de fechas de nacimiento
    df = df.merge(birth_dates['product'], on='product_id', how='left')
    
    if cfg.GRANULARITY == 'customer':
        df = df.merge(birth_dates['customer'], on='customer_id', how='left')
    
    # Filtrar registros anteriores a fecha de nacimiento
    initial_rows = len(df)
    df = df[df['periodo'] >= df['birth_date_product']]
    
    if cfg.GRANULARITY == 'customer':
        df = df[df['periodo'] >= df['birth_date_customer']]
    
    filtered_rows = initial_rows - len(df)
    logger.info(f"Filtrados {filtered_rows:,} registros no nacidos de {initial_rows:,} totales")
    
    # Eliminar columnas de fecha de nacimiento
    cols_to_drop = ['birth_date_product']
    if cfg.GRANULARITY == 'customer':
        cols_to_drop.append('birth_date_customer')
    df = df.drop(columns=cols_to_drop)
    
    # Restaurar tipos categóricos después del merge
    for col in cfg.CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype('category')
    
    # Verificar features clave
    key_features = ['tn_lag_0']
    advanced_features = [f for f in df.columns if any(x in f for x in ['cat_avg_', 'fourier_', 'acceleration_', 'skewness_'])]
    
    logger.info(f"Features clave: {[f for f in key_features if f in df.columns]}")
    logger.info(f"Features avanzadas detectadas: {len(advanced_features)}")
    
    if 'tn_lag_0' in df.columns:
        logger.info("Framework consistente confirmado - tn_lag_0 presente")
    else:
        logger.error("tn_lag_0 NO encontrado - verificar cache de features")
        exit(1)
    
    return df

def load_best_params(cfg):
    """Carga los mejores hiperparámetros desde el estudio de Optuna."""
    db_name = f"lgbm-study-{cfg.GRANULARITY}-{cfg.FEATURE_VERSION}.db"
    db_path = os.path.join(cfg.ARTIFACTS_PATH, db_name)
    
    if not os.path.exists(db_path):
        logger.error(f"No se encontró {db_path}")
        logger.error("Ejecuta primero workaround_11_ratio_log.py para optimizar hiperparámetros")
        exit(1)
    
    storage = f"sqlite:///{db_path}"
    study_name = f"lgbm-study-{cfg.GRANULARITY}-{cfg.FEATURE_VERSION}"
    
    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
        if len(study.trials) == 0:
            logger.error(f"El estudio {study_name} no tiene trials completados")
            exit(1)
        
        logger.info(f"Estudio cargado: {len(study.trials)} trials, mejor WAPE: {study.best_value:.4f}")
        
        # Log de parámetros para verificación
        logger.info("Parámetros optimizados que se usarán:")
        for k, v in study.best_params.items():
            logger.info(f"  {k}: {v}")
        
        return study.best_params
    except Exception as e:
        logger.error(f"Error al cargar el estudio de Optuna: {e}")
        exit(1)

def load_productos_a_predecir(cfg):
    """Carga la lista de productos a predecir."""
    logger.info("Cargando lista de productos a predecir...")
    productos_a_predecir = pd.read_csv(
        cfg.PRODUCTOS_A_PREDECIR_URL,
        sep='\t', 
        dtype={'product_id': str}
    )
    return productos_a_predecir

def train_single_model(df, best_params, cfg, seed: int, retry_count: int = 0) -> Optional[pd.DataFrame]:
    """Entrena un modelo con una semilla específica y retorna las predicciones."""
    logger.info(f"Entrenando modelo con semilla {seed}... (intento {retry_count + 1})")
    
    try:
        # Usar base params alineada + best_params optimizados
        final_params = cfg.LGBM_FIXED_PARAMS.copy()
        final_params.update(best_params)
        final_params['seed'] = seed
        
        # Si es un reintento, solo cambiar la semilla
        if retry_count > 0:
            logger.info(f"Reintento {retry_count + 1}: solo cambiando semilla a {seed + 1000 * retry_count}")
            final_params['seed'] = seed + 1000 * retry_count
        
        # Log de verificación de parámetros clave
        key_params = ['learning_rate', 'num_leaves', 'lambda_l1', 'lambda_l2']
        params_str = ", ".join([f"{k}:{final_params.get(k, 'N/A')}" for k in key_params])
        logger.debug(f"Parámetros clave: [{params_str}]")
        
        df_clean = df.dropna(subset=['y'])
        train_final = df_clean[df_clean['periodo'] < cfg.FINAL_TRAIN_END]
        predict_data = df[df['periodo'] == pd.to_datetime(cfg.FINAL_TRAIN_END)].copy()
        
        features = [c for c in train_final.columns if c not in cfg.COLUMNS_TO_EXCLUDE]
        cat_features = [c for c in cfg.CATEGORICAL_FEATURES if c in features]
        
        # Verificar que tn_lag_0 esté en las features para entrenamiento
        if 'tn_lag_0' in features:
            logger.debug(f"tn_lag_0 incluido en entrenamiento (semilla {seed})")
        else:
            logger.error(f"tn_lag_0 NO incluido en entrenamiento (semilla {seed})")
            return None
        
        X_train, y_train = train_final[features], train_final['y']
        X_pred = predict_data[features]
        
        model = lgb.LGBMRegressor(**final_params)
        model.fit(X_train, y_train, categorical_feature=cat_features)
        
        # Post-procesamiento: misma lógica que workaround_11
        preds_log_ratio = model.predict(X_pred)
        
        # Usar tn sin escalar para la transformación inversa
        log1p_tn_current = np.log1p(predict_data['tn'])
        predict_data['tn_pred'] = np.expm1(preds_log_ratio + log1p_tn_current)
        
        # Cap predictions to 2x historical max
        upper_bound = predict_data['historical_max_tn'] * 2.0
        predict_data['tn_pred'] = predict_data['tn_pred'].clip(upper=upper_bound.values)
        
        if cfg.GRANULARITY == 'customer':
            final_pred = predict_data.groupby('product_id')['tn_pred'].sum().reset_index()
        else:
            final_pred = predict_data[['product_id', 'tn_pred']]
        
        productos_a_predecir = load_productos_a_predecir(cfg)
        submission = pd.merge(
            productos_a_predecir, final_pred,
            on='product_id', how='left'
        ).fillna(0)
        submission.rename(columns={'tn_pred': 'tn'}, inplace=True)
        
        # Guardar submission individual
        sub_path = os.path.join(
            cfg.ARTIFACTS_PATH,
            f'submission_202002_{cfg.GRANULARITY}_{cfg.FEATURE_VERSION}_seed{seed}.csv'
        )
        submission.to_csv(sub_path, index=False)
        logger.info(f"Submission guardado: {os.path.basename(sub_path)} | Total: {submission['tn'].sum():.1f} tn")
        
        return submission
        
    except lgb.basic.LightGBMError as e:
        logger.error(f"Error LightGBM con semilla {seed}: {e}")
        
        if retry_count < cfg.MAX_RETRIES - 1:
            logger.info(f"Reintentando con nueva semilla...")
            return train_single_model(df, best_params, cfg, seed + 1000, retry_count + 1)
        else:
            logger.error(f"Falló después de {cfg.MAX_RETRIES} intentos")
            return None
    
    except Exception as e:
        logger.error(f"Error inesperado con semilla {seed}: {e}")
        return None

def find_submission_files(cfg) -> List[str]:
    """Encuentra todos los archivos de submission para la granularidad actual, excluyendo semillerios."""
    pattern = os.path.join(
        cfg.ARTIFACTS_PATH, 
        f'submission_202002_{cfg.GRANULARITY}_{cfg.FEATURE_VERSION}_seed*.csv'
    )
    all_files = glob.glob(pattern)
    
    # Excluir archivos de semillerío previos
    submission_files = [f for f in all_files if 'semillerio' not in f]
    
    logger.info(f"Encontrados {len(submission_files)} archivos de submission para ensamblar")
    return sorted(submission_files)

def ensemble_predictions(cfg, submission_files: List[str]) -> pd.DataFrame:
    """Crea un ensemble promediando las predicciones de múltiples archivos."""
    if not submission_files:
        logger.error("No hay archivos para ensamblar")
        return None
    
    logger.info(f"Ensamblando {len(submission_files)} predicciones...")
    
    # Cargar primera submission como base
    ensemble_df = pd.read_csv(submission_files[0])
    ensemble_df = ensemble_df.rename(columns={'tn': 'tn_sum'})
    
    # Sumar el resto
    for file in submission_files[1:]:
        df = pd.read_csv(file)
        ensemble_df['tn_sum'] += df['tn']
    
    # Promediar
    ensemble_df['tn'] = ensemble_df['tn_sum'] / len(submission_files)
    ensemble_df = ensemble_df[['product_id', 'tn']]
    
    # Guardar ensemble
    ensemble_path = os.path.join(
        cfg.ARTIFACTS_PATH,
        f'{cfg.GRANULARITY}_{cfg.FEATURE_VERSION}_semillerio_{len(submission_files)}.csv'
    )
    ensemble_df.to_csv(ensemble_path, index=False)
    
    total_pred = ensemble_df['tn'].sum()
    logger.info(f"Ensemble guardado en: {os.path.basename(ensemble_path)}")
    logger.info(f"Total predicho en ensemble: {total_pred:,.2f} toneladas")
    
    # Generar resumen del ensemble
    summary_path = os.path.join(
        cfg.ARTIFACTS_PATH,
        f'{cfg.GRANULARITY}_{cfg.FEATURE_VERSION}_semillerio_{len(submission_files)}_summary.txt'
    )
    with open(summary_path, 'w') as f:
        f.write(f"Resumen del Ensemble - FEATURES AVANZADAS v11\n")
        f.write("="*80 + "\n")
        f.write(f"Granularidad: {cfg.GRANULARITY}\n")
        f.write(f"Feature Version: {cfg.FEATURE_VERSION}\n")
        f.write(f"Cache Level: Nivel 3 (_ii) - Features Avanzadas\n")
        f.write(f"Estrategia: y = log1p(tn_future) - log1p(tn_current)\n")
        f.write(f"Parámetros: Alineados con optimización Optuna\n")
        f.write(f"Framework: Lags 0-24 consistentemente escalados\n")
        f.write(f"Feature principal: tn_lag_0 (valor actual escalado)\n")
        f.write(f"Número de modelos ensamblados: {len(submission_files)}\n")
        f.write(f"Total toneladas predichas: {total_pred:,.2f}\n")
        f.write("\nFeatures Avanzadas Implementadas:\n")
        f.write("- A. Features de Productos Relacionados (contexto categorías 3 niveles)\n")
        f.write("- B. Features de Estacionalidad Avanzada (fourier + eventos argentinos)\n")
        f.write("- C. Features de Momentum y Aceleración (slopes + volatilidad)\n")
        f.write("- D. Features de Patrones de Consumo (skewness + regularidad)\n")
        f.write("\nMejoras de Alineación:\n")
        f.write("- Usa cfg.LGBM_FIXED_PARAMS.copy() + best_params\n")
        f.write("- Incluye bagging_freq=1 como main\n")
        f.write("- Reintentos solo cambian semilla, no parámetros\n")
        f.write("\nArchivos incluidos en el ensemble:\n")
        f.write("-"*80 + "\n")
        for file in submission_files:
            individual_total = pd.read_csv(file)['tn'].sum()
            f.write(f"  - {os.path.basename(file)}: {individual_total:,.1f} tn\n")
        
        f.write(f"\nObjetivo: Superar meseta con features avanzadas\n")
    
    return ensemble_df

def generate_seed_sequence(initial_seed: int, num_seeds: int) -> List[int]:
    """Genera una secuencia de semillas primas."""
    seeds = []
    current = initial_seed
    
    for _ in range(num_seeds):
        seeds.append(current)
        current = next_prime(current)
    
    return seeds

def main():
    logger.info("INICIANDO SEMILLERÍO v11 - FEATURES AVANZADAS")
    
    cfg = Config()
    
    # Crear directorios
    for p in [cfg.CACHE_PATH, cfg.ARTIFACTS_PATH]:
        os.makedirs(p, exist_ok=True)
    
    # Calcular fechas de nacimiento
    birth_dates = calculate_birth_dates(cfg)
    
    # Cargar datos con filtro de nacimientos (FEATURES AVANZADAS)
    df = load_features_with_birth_filter(cfg, birth_dates)
    best_params = load_best_params(cfg)
    
    # Generar secuencia de semillas primas
    seeds = generate_seed_sequence(cfg.INITIAL_SEED, cfg.NUM_MODELS)
    logger.info(f"Semillas a utilizar: {seeds[:5]}...{seeds[-3:]} ({len(seeds)} total)")
    
    # Verificar columnas clave del framework avanzado
    expected_base = ['tn_lag_0', 'bin_decil_lag_0', 'bin_octil_lag_0', 'bin_cuart_lag_0']
    expected_advanced = ['fourier_sin_1', 'is_holiday_season']
    
    missing_base = [col for col in expected_base if col not in df.columns]
    missing_advanced = [col for col in expected_advanced if col not in df.columns]
    
    if missing_base:
        logger.error(f"Columnas base faltantes: {missing_base}")
        exit(1)
    if missing_advanced:
        logger.warning(f"Columnas avanzadas faltantes: {missing_advanced}")
    
    logger.info("Framework avanzado verificado")
    
    # Entrenar modelos con cada semilla
    successful_models = 0
    failed_models = 0
    
    logger.info(f"\nIniciando entrenamiento de {cfg.NUM_MODELS} modelos...")
    
    for i, seed in enumerate(seeds, 1):
        logger.info(f"\n--- Modelo {i:2d}/{cfg.NUM_MODELS} ---")
        result = train_single_model(df, best_params, cfg, seed)
        
        if result is not None:
            successful_models += 1
        else:
            failed_models += 1
            logger.warning(f"Modelo {i} falló completamente")
    
    logger.info(f"\nRESULTADOS DEL SEMILLERÍO:")
    logger.info(f"Modelos exitosos: {successful_models}/{cfg.NUM_MODELS}")
    if failed_models > 0:
        logger.warning(f"Modelos fallidos: {failed_models}/{cfg.NUM_MODELS}")
    
    # Ensamblar predicciones solo si hay modelos exitosos
    if successful_models > 0:
        logger.info("\n--- CREANDO ENSEMBLE FINAL ---")
        submission_files = find_submission_files(cfg)
        
        if submission_files:
            ensemble_df = ensemble_predictions(cfg, submission_files)
            
            # Mostrar estadísticas del ensemble
            individual_totals = []
            for file in submission_files:
                df_temp = pd.read_csv(file)
                individual_totals.append(df_temp['tn'].sum())
            
            logger.info("\nEstadísticas del ensemble:")
            logger.info(f"  Media individual: {np.mean(individual_totals):,.2f}")
            logger.info(f"  Desv. estándar: {np.std(individual_totals):,.2f}")
            logger.info(f"  Mínimo: {np.min(individual_totals):,.2f}")
            logger.info(f"  Máximo: {np.max(individual_totals):,.2f}")
            logger.info(f"  Estrategia: FEATURES AVANZADAS")
            logger.info(f"  Feature principal: tn_lag_0 (valor actual escalado)")
    else:
        logger.error("No se pudo entrenar ningún modelo exitosamente")
    
    logger.info("\nSEMILLERÍO v11 FEATURES AVANZADAS COMPLETADO")
    logger.info(f"Revisa los archivos generados en '{cfg.ARTIFACTS_PATH}'")
    logger.info("Objetivo: Superar meseta con features A+B+C+D")

if __name__ == '__main__':
    main()
