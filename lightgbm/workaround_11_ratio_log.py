# -*- coding: utf-8 -*-
"""
Pipeline de Forecasting con Features Avanzadas y Optimización
==============================================================

Este script constituye el núcleo del pipeline de modelado para la predicción de ventas.
Implementa un flujo completo que incluye:
1. Carga y preprocesamiento de datos.
2. Creación de un set de features avanzado (v6_log_ratio_advanced), que incluye:
   - Features de contexto de productos relacionados (categorías, marcas).
   - Features de estacionalidad avanzada (Fourier, eventos).
   - Features de momentum y aceleración (tendencias, volatilidad).
   - Features de patrones de consumo (skewness, kurtosis, regularidad).
3. Optimización de hiperparámetros de LightGBM usando Optuna con pruner Hyperband.
4. Entrenamiento de un modelo final con los mejores parámetros encontrados.
5. Generación de predicciones para el horizonte definido.
6. Validación del modelo en una fecha conocida (backtesting) para medir confianza.

Parámetros Principales de Configuración (dentro de la clase Config):
--------------------------------------------------------------------
- GRANULARITY: 'product' o 'customer'. Define el nivel de agregación de los datos.
- FEATURE_VERSION: Versión de las features a utilizar. Clave para la gestión de caché.
- N_TRIALS: Número de iteraciones para la optimización con Optuna.
- TIMEOUT: Límite de tiempo en segundos para la optimización.
- FECHAS (OPT_TRAIN_END, FINAL_TRAIN_END, etc.): Definen los períodos de entrenamiento,
  validación y test.

"""

import os
import time
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import warnings
import logging
import optuna
from optuna.integration import LightGBMPruningCallback

warnings.filterwarnings('ignore')

class Config:
    GRANULARITY     = 'product'       # 'product' | 'customer'
    PREDICT_HORIZON = 2               # meses
    FEATURE_VERSION = 'v6_log_ratio_advanced'  # Nueva versión con features avanzadas
    CACHE_PATH      = 'cache_log_ratio_advanced'
    ARTIFACTS_PATH  = 'artifacts_log_ratio_advanced'
    FILES_CONFIG   = {
        'sellin': {
            'url': 'https://storage.googleapis.com/open-courses/austral2025-af91/labo3v/sell-in.txt.gz',
            'read_args': {
                'sep': "\t", 'compression': 'gzip',
                'dtype': {'periodo': str, 'customer_id': str, 'product_id': str}
            }
        },
        'productos': {
            'url': 'https://storage.googleapis.com/open-courses/austral2025-af91/labo3v/tb_productos.txt',
            'read_args': {'sep': "\t", 'dtype': {'product_id': str}}
        },
        'productos_a_predecir': {
            'url': 'https://storage.googleapis.com/open-courses/austral2025-af91/labo3v/product_id_apredecir201912.txt',
            'read_args': {'sep': "\t", 'dtype': {'product_id': str}}
        }
    }
    OPT_TRAIN_END  = '2019-09-01'
    OPT_VALID_END  = '2019-10-01'
    OPT_TEST_END   = '2019-11-01'
    FINAL_TRAIN_END = '2019-12-01'
    N_TRIALS = 5000
    TIMEOUT  = 3600*10   # 10h para más exploración
    LGBM_FIXED_PARAMS = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt', 
        'n_estimators': 8000, 
        'n_jobs': -1,
        'seed': 42, 
        'verbose': -1,
        'bagging_freq': 1,
        'linear_tree': False, 
        'max_bin': 2800,
    }
    GRANULARITY_CONFIGS = {
        'product': {
            'EARLY_STOPPING_ROUNDS': 2000,
            'N_WARMUP_STEPS': 20,
            'OPTUNA_SEARCH_SPACE': {
                'learning_rate':     ('float', 0.003, 0.3, {'log': True}),
                'num_leaves':        ('int',   15, 200),
                'feature_fraction':  ('float', 0.4, 1.0),
                'bagging_fraction':  ('float', 0.4, 1.0),
                'min_child_samples': ('int', 10, 150),
                'lambda_l1':         ('float', 1e-4, 20.0, {'log': True}),
                'lambda_l2':         ('float', 1e-4, 20.0, {'log': True}),
                'max_depth':         ('int', 3, 15),
                'min_gain_to_split': ('float', 0.0, 2.0),
                'subsample_for_bin': ('int', 50000, 300000),
            }
        },
        'customer': {
            'EARLY_STOPPING_ROUNDS': 1200,
            'N_WARMUP_STEPS': 5,
            'OPTUNA_SEARCH_SPACE': {
                'learning_rate':     ('float', 0.005, 0.15, {'log': True}),
                'num_leaves':        ('int',   20, 250),
                'feature_fraction':  ('float', 0.5, 1.0),
                'bagging_fraction':  ('float', 0.5, 1.0),
                'min_child_samples': ('int', 15, 120),
                'lambda_l1':         ('float', 1e-4, 15.0, {'log': True}),
                'lambda_l2':         ('float', 1e-4, 15.0, {'log': True}),
                'max_depth':         ('int', 4, 12),
                'min_gain_to_split': ('float', 0.0, 1.5),
                'subsample_for_bin': ('int', 80000, 400000),
            }
        }
    }
    # Features categóricas
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
    # Columnas a excluir del entrenamiento
    COLUMNS_TO_EXCLUDE = [
        'periodo', 'y', 'scaler', 'descripcion', 'row_id', 'cust_any_purchase',
        'historical_max_tn', 'tn', 'tn_future_unscaled'
    ]

def get_logger():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    optuna.logging.enable_propagation()
    optuna.logging.disable_default_handler()
    return logging.getLogger()

logger = get_logger()

class Timer:
    def __enter__(self):
        self.start = time.time(); return self
    def __exit__(self, *exc):
        logger.info(f'Bloque ejecutado en {time.time()-self.start:.2f} s.')

def load_data(cfg):
    dfs = {}
    birth_dates = {}
    
    for name, details in cfg.FILES_CONFIG.items():
        logger.info(f"Cargando {name} …")
        dfs[name] = pd.read_csv(details['url'], **details['read_args'])
    
    # Calcular fechas de nacimiento cuando cargamos sellin
    sellin = dfs['sellin']
    sellin['periodo'] = pd.to_datetime(sellin['periodo'], format='%Y%m')
    
    # Fecha de nacimiento de productos
    birth_dates['product'] = sellin.groupby('product_id')['periodo'].min().reset_index()
    birth_dates['product'].columns = ['product_id', 'birth_date_product']
    
    # Fecha de nacimiento de clientes
    birth_dates['customer'] = sellin.groupby('customer_id')['periodo'].min().reset_index()
    birth_dates['customer'].columns = ['customer_id', 'birth_date_customer']
    
    # Guardar las fechas de nacimiento en el diccionario de dataframes
    dfs['birth_dates'] = birth_dates
    
    return dfs

def preprocess_data(dfs, cfg):
    cache = os.path.join(cfg.CACHE_PATH,
                         f'processed_{cfg.GRANULARITY}.parquet')
    if os.path.exists(cache):
        logger.info(f"Cargando preprocesado desde {cache}")
        df = pd.read_parquet(cache)
        
        # Aplicar filtro de fechas de nacimiento al cargar desde cache
        birth_dates = dfs['birth_dates']
        df = df.merge(birth_dates['product'], on='product_id', how='left')
        
        # Solo merge con customer si la granularidad es customer
        if cfg.GRANULARITY == 'customer':
            df = df.merge(birth_dates['customer'], on='customer_id', how='left')
        
        # Filtrar registros anteriores a fecha de nacimiento
        df = df[df['periodo'] >= df['birth_date_product']]
        if cfg.GRANULARITY == 'customer':
            df = df[df['periodo'] >= df['birth_date_customer']]
        
        # Eliminar columnas de fecha de nacimiento ya que no las necesitamos más
        cols_to_drop = ['birth_date_product']
        if cfg.GRANULARITY == 'customer':
            cols_to_drop.append('birth_date_customer')
        df = df.drop(columns=cols_to_drop)
        
        return df

    with Timer():
        sellin, productos = dfs['sellin'], dfs['productos']
        sellin['periodo'] = pd.to_datetime(sellin['periodo'], format='%Y%m')

        gcols = ['periodo', 'product_id']
        if cfg.GRANULARITY == 'customer':
            gcols.append('customer_id')

        df = sellin.groupby(gcols)['tn'].sum().reset_index()
        periods   = pd.date_range(sellin['periodo'].min(), sellin['periodo'].max(), freq='MS')
        products  = sellin['product_id'].unique()

        if cfg.GRANULARITY == 'customer':
            customers = sellin['customer_id'].unique()
            grid = pd.MultiIndex.from_product([periods, products, customers],
                                              names=['periodo','product_id','customer_id'])
        else:
            grid = pd.MultiIndex.from_product([periods, products],
                                              names=['periodo','product_id'])

        df = df.set_index(gcols).reindex(grid, fill_value=0).reset_index()
        df = df.merge(productos, on='product_id', how='left')
        
        # Aplicar filtro de fechas de nacimiento
        birth_dates = dfs['birth_dates']
        df = df.merge(birth_dates['product'], on='product_id', how='left')
        
        # Solo merge con customer si la granularidad es customer
        if cfg.GRANULARITY == 'customer':
            df = df.merge(birth_dates['customer'], on='customer_id', how='left')
        
        # Filtrar registros anteriores a fecha de nacimiento
        df = df[df['periodo'] >= df['birth_date_product']]
        if cfg.GRANULARITY == 'customer':
            df = df[df['periodo'] >= df['birth_date_customer']]
        
        # Eliminar columnas de fecha de nacimiento
        cols_to_drop = ['birth_date_product']
        if cfg.GRANULARITY == 'customer':
            cols_to_drop.append('birth_date_customer')
        df = df.drop(columns=cols_to_drop)
        
        for col in cfg.CATEGORICAL_FEATURES:
            if col in df.columns:
                df[col] = df[col].astype('category')

        os.makedirs(cfg.CACHE_PATH, exist_ok=True)
        df.to_parquet(cache)
        logger.info("Preprocesamiento completado.")
    return df

def create_features_and_target(df, cfg):
    # CACHE NIVEL 2: Features base (original)
    cache_base = os.path.join(cfg.CACHE_PATH,
                              f'features_{cfg.GRANULARITY}_v5_log_ratio_consistent.parquet')
    
    # CACHE NIVEL 3: Features avanzadas (NUEVO)
    cache_advanced = os.path.join(cfg.CACHE_PATH,
                                  f'features_{cfg.GRANULARITY}_{cfg.FEATURE_VERSION}_ii.parquet')
    
    if os.path.exists(cache_advanced):
        logger.info(f"Cargando features avanzadas desde {cache_advanced}")
        return pd.read_parquet(cache_advanced)
    
    # Si no existe cache avanzado, verificar cache base
    if os.path.exists(cache_base):
        logger.info(f"Cargando features base desde {cache_base}")
        df = pd.read_parquet(cache_base)
        logger.info("Creando features avanzadas nivel 3...")
    else:
        logger.info("Creando features desde cero...")
        df = create_base_features(df, cfg)

    # CREAR FEATURES AVANZADAS (Nivel 3)
    df = create_advanced_features(df, cfg)
    
    # Guardar cache nivel 3
    os.makedirs(cfg.CACHE_PATH, exist_ok=True)
    df.to_parquet(cache_advanced)
    logger.info(f"Features avanzadas guardadas en {cache_advanced}")
    
    return df

def create_base_features(df, cfg):
    """Crear features base (misma lógica que v5_log_ratio_consistent)"""
    with Timer():
        gcols = ['product_id']
        if cfg.GRANULARITY == 'customer':
            gcols.append('customer_id')

        historical_max = df.groupby(gcols)['tn'].max().rename('historical_max_tn')
        df = df.merge(historical_max, on=gcols, how='left')

        df['periodo'] = pd.to_datetime(df['periodo'])
        df['month']  = df['periodo'].dt.month
        df['year']   = df['periodo'].dt.year
        df['quarter'] = df['periodo'].dt.quarter.astype('category')
        df['is_end_of_year'] = (df['month'] == 12).astype(int)
        df['sin_month']      = np.sin(2*np.pi*df['month']/12)
        df['cos_month']      = np.cos(2*np.pi*df['month']/12)

        # Flags de evento/catástrofe
        df['FLAG_EVENTO_AGO2019'] = (df['periodo'].dt.to_period('M') == '2019-08').astype(int)
        event_period  = pd.Period('2019-08', 'M')
        affected_per  = (event_period - cfg.PREDICT_HORIZON)
        df['FLAG_PREDICCION_AFECTADA_AGO2019'] = (df['periodo'].dt.to_period('M') == affected_per).astype(int)
        df['FLAG_RECESO_ESCOLAR'] = df['month'].isin([1,2,7]).astype(int)
        df['FLAG_VERANO']    = df['month'].isin([12,1,2,3]).astype(int)
        df['FLAG_OTONIO']    = df['month'].isin([4,5,6]).astype(int)
        df['FLAG_INVIERNO']  = df['month'].isin([7,8,9]).astype(int)
        df['FLAG_PRIMAVERA'] = df['month'].isin([10,11]).astype(int)

        # Lags desde 0 hasta 24 (ANTES del escalado)
        logger.info("Creando lags desde 0 hasta 24 (framework consistente)...")
        max_lag = 24
        for lag in range(0, max_lag+1):  
            df[f'tn_lag_{lag}'] = df.groupby(gcols)['tn'].shift(lag)

        # BINS desde lag 0 hasta lag 12 (ANTES del escalado)
        logger.info("Creando bins para lags 0-12...")
        for lag in range(0, 13):  
            col = f'tn_lag_{lag}'
            df[f'bin_decil_lag_{lag}'] = df.groupby(['product_id'])[col].transform(
                lambda x: pd.qcut(x, 10, labels=False, duplicates='drop')).astype('category')
            df[f'bin_octil_lag_{lag}'] = df.groupby(['product_id'])[col].transform(
                lambda x: pd.qcut(x, 8, labels=False, duplicates='drop')).astype('category')
            df[f'bin_cuart_lag_{lag}'] = df.groupby(['product_id'])[col].transform(
                lambda x: pd.qcut(x, 4, labels=False, duplicates='drop')).astype('category')

        # Cadenas de bins
        logger.info("Creando cadenas de bins...")
        BIN_LETTERS = {
            'decil': [chr(ord('A')+i) for i in range(10)],
            'octil': [chr(ord('A')+i) for i in range(8)],
            'cuartil': [chr(ord('A')+i) for i in range(4)]
        }
        def bin_str(row, prefix, k, letters):
            vals = [row[f'{prefix}_{i}'] for i in range(0, k)]
            vals = [0 if pd.isna(v) else int(v) for v in vals]
            return ''.join(letters[v] for v in reversed(vals))
        
        for k in range(4,9):
            df[f'bin_decil_str_{k}'] = df.apply(
                lambda r: bin_str(r, 'bin_decil_lag', k, BIN_LETTERS['decil']), axis=1).astype('category')
            df[f'bin_octil_str_{k}'] = df.apply(
                lambda r: bin_str(r, 'bin_octil_lag', k, BIN_LETTERS['octil']), axis=1).astype('category')
            df[f'bin_cuart_str_{k}'] = df.apply(
                lambda r: bin_str(r, 'bin_cuart_lag', k, BIN_LETTERS['cuartil']), axis=1).astype('category')

        # Rolling stats
        logger.info("Creando rolling statistics...")
        h = cfg.PREDICT_HORIZON
        tn_h = df.groupby(gcols)['tn'].shift(h)  
        for w in [3,6,9,12]:
            df[f'tn_roll_mean_{w}'] = tn_h.rolling(w).mean()
            df[f'tn_roll_std_{w}']  = tn_h.rolling(w).std()
            df[f'tn_roll_min_{w}']  = tn_h.rolling(w).min()
            df[f'tn_roll_max_{w}']  = tn_h.rolling(w).max()

        # Calcular scaler ANTES de crear el target
        logger.info("Calculando scaler...")
        scaler = df.groupby(gcols)['tn'].std().replace(0,1)
        df = df.join(scaler.rename('scaler'), on=gcols)
        
        # ESCALAR TODOS LOS LAGS Y ROLLING STATS
        logger.info("Escalando todos los lags y rolling stats...")
        for c in df.columns:
            if c.startswith('tn_lag_') or c.startswith('tn_roll_'):
                df[c] = df[c] / df['scaler']

        # TARGET: Log-ratio usando lags escalados
        logger.info("Creando target con log-ratio...")
        df['tn_future_unscaled'] = df.groupby(gcols)['tn'].shift(-h)
        df['y'] = np.log1p(df['tn_future_unscaled']) - np.log1p(df['tn'])
        df['tn_current_scaled'] = df['tn_lag_0']  

        # Deltas, ratios, tendencias, slopes
        eps = 1e-6
        logger.info("Creando features derivadas...")
        for w in [3,6,9,12,18,24]:
            if h+w<=max_lag:
                df[f'tn_delta_lag_{w}'] = (df[f'tn_lag_{h}'] - df[f'tn_lag_{h+w}'])
        for w in [3,6,9,12]:
            df[f'tn_delta_roll_mean_{w}'] = df[f'tn_lag_{h}'] - df[f'tn_roll_mean_{w}']
            df[f'tn_ratio_roll_mean_{w}'] = df[f'tn_lag_{h}']/(df[f'tn_roll_mean_{w}']+eps)
            df[f'tn_cv_{w}'] = df[f'tn_roll_std_{w}']/(df[f'tn_roll_mean_{w}']+eps)
            prev_mean = df.groupby(gcols)[f'tn_roll_mean_{w}'].shift(w)
            df[f'tn_trend_{w}'] = df[f'tn_roll_mean_{w}']/(prev_mean+eps)-1
            df[f'tn_slope_{w}'] = (df[f'tn_lag_{h}']-df[f'tn_lag_{h+w}'])/w

        # Edad producto y ventas categoría
        first_prod = df.groupby('product_id')['periodo'].transform('min')
        df['edad_producto_meses'] = df['periodo'].dt.to_period('M').astype(int) - first_prod.dt.to_period('M').astype(int)
        df['cat_total_tn'] = df.groupby(['periodo','cat1'])['tn'].transform('sum')

        # Meses sin comprar consecutivos (solo customer)
        if cfg.GRANULARITY == 'customer':
            cust_any = (df.groupby(['periodo','customer_id'])['tn'].transform('sum') > 0)
            df['cust_any_purchase'] = cust_any.astype(int)
            df = df.sort_values(['customer_id','periodo'])
            grp_id = df['cust_any_purchase'].groupby(df['customer_id']).cumsum()
            df['meses_sin_compra_consec'] = (1-df['cust_any_purchase']).groupby([df['customer_id'],grp_id]).cumsum()
            df['cust_prod'] = (df['customer_id'].astype(str)+'_'+df['product_id'].astype(str)).astype('category')
            if 'cust_prod' not in cfg.CATEGORICAL_FEATURES:
                cfg.CATEGORICAL_FEATURES.append('cust_prod')
        else:
            df['meses_sin_compra_consec'] = 0

        df['row_id'] = np.arange(len(df))
        
    return df

def create_advanced_features(df, cfg):
    """Crear features avanzadas - A, B, C, D"""
    logger.info("Creando Features Avanzadas...")
    
    gcols = ['product_id']
    if cfg.GRANULARITY == 'customer':
        gcols.append('customer_id')
    
    h = cfg.PREDICT_HORIZON
    eps = 1e-6

    # A. FEATURES DE PRODUCTOS RELACIONADOS
    logger.info("A. Features de Productos Relacionados...")
    
    # Contexto de categorías (3 niveles)
    for cat_level in ['cat1', 'cat2', 'cat3']:
        if cat_level in df.columns:
            for lag in [h, h+3, h+6, h+12]:
                if f'tn_lag_{lag}' in df.columns:
                    # Promedio de categoría
                    cat_col = f'cat_avg_tn_{cat_level}_lag_{lag}'
                    df[cat_col] = df.groupby(['periodo', cat_level])[f'tn_lag_{lag}'].transform('mean')
                    
                    # Ranking en categoría
                    rank_col = f'rank_in_{cat_level}_lag_{lag}'
                    df[rank_col] = df.groupby(['periodo', cat_level])[f'tn_lag_{lag}'].rank(ascending=False)
                    
                    # Share relativo
                    total_cat = df.groupby(['periodo', cat_level])[f'tn_lag_{lag}'].transform('sum')
                    share_col = f'share_in_{cat_level}_lag_{lag}'
                    df[share_col] = df[f'tn_lag_{lag}'] / (total_cat + eps)
                    
                    # Percentil en categoría
                    percentile_col = f'percentile_in_{cat_level}_lag_{lag}'
                    df[percentile_col] = df.groupby(['periodo', cat_level])[f'tn_lag_{lag}'].rank(pct=True)

    # Context features por brand
    if 'brand' in df.columns:
        for lag in [h, h+6, h+12]:
            if f'tn_lag_{lag}' in df.columns:
                df[f'brand_avg_tn_lag_{lag}'] = df.groupby(['periodo', 'brand'])[f'tn_lag_{lag}'].transform('mean')
                df[f'rank_in_brand_lag_{lag}'] = df.groupby(['periodo', 'brand'])[f'tn_lag_{lag}'].rank(ascending=False)
                total_brand = df.groupby(['periodo', 'brand'])[f'tn_lag_{lag}'].transform('sum')
                df[f'share_in_brand_lag_{lag}'] = df[f'tn_lag_{lag}'] / (total_brand + eps)

    # B. FEATURES DE ESTACIONALIDAD AVANZADA
    logger.info("B. Features de Estacionalidad Avanzada...")
    
    # Fourier components
    for k in [1, 2, 3, 4, 6]:
        df[f'fourier_sin_{k}'] = np.sin(2 * np.pi * k * df['month'] / 12)
        df[f'fourier_cos_{k}'] = np.cos(2 * np.pi * k * df['month'] / 12)

    # Eventos específicos del mercado argentino
    df['is_holiday_season'] = df['month'].isin([11, 12]).astype(int)
    df['is_back_to_school'] = df['month'].isin([2, 3]).astype(int)
    df['is_winter_peak'] = df['month'].isin([6, 7, 8]).astype(int)
    df['is_summer_low'] = df['month'].isin([1, 2]).astype(int)

    # Distancia a eventos clave
    df['months_to_december'] = ((12 - df['month']) % 12)
    df['months_to_march'] = ((3 - df['month']) % 12)

    # C. FEATURES DE MOMENTUM Y ACELERACIÓN
    logger.info("C. Features de Momentum y Aceleración...")
    
    # Aceleración de tendencias
    for w in [3, 6, 12]:
        if f'tn_slope_{w}' in df.columns:
            prev_slope = df.groupby(gcols)[f'tn_slope_{w}'].shift(1)
            df[f'tn_acceleration_{w}'] = df[f'tn_slope_{w}'] - prev_slope
            
            # Momentum de rolling means
            if f'tn_roll_mean_{w}' in df.columns:
                prev_roll_mean = df.groupby(gcols)[f'tn_roll_mean_{w}'].shift(3)
                df[f'tn_momentum_mean_{w}'] = df[f'tn_roll_mean_{w}'] - prev_roll_mean

    # Cambios en volatilidad
    for w in [3, 6, 12]:
        if f'tn_cv_{w}' in df.columns:
            prev_cv = df.groupby(gcols)[f'tn_cv_{w}'].shift(1)
            df[f'tn_volatility_change_{w}'] = df[f'tn_cv_{w}'] - prev_cv
            
            # Aceleración de volatilidad
            prev_vol_change = df.groupby(gcols)[f'tn_volatility_change_{w}'].shift(1)
            df[f'tn_volatility_acceleration_{w}'] = df[f'tn_volatility_change_{w}'] - prev_vol_change

    # D. FEATURES DE PATRONES DE CONSUMO
    logger.info("D. Features de Patrones de Consumo...")
    
    # Skewness y Kurtosis en ventanas rolling
    for w in [6, 12, 18]:
        if f'tn_lag_{h}' in df.columns:
            tn_values = df.groupby(gcols)[f'tn_lag_{h}'].rolling(w, min_periods=3)
            df[f'tn_skewness_{w}m'] = tn_values.skew().values
            df[f'tn_kurtosis_{w}m'] = tn_values.apply(lambda x: x.kurtosis()).values

    # Regularidad del producto
    for w in [6, 12, 18]:
        if f'tn_lag_{h}' in df.columns:
            threshold = 0.01  # Umbral mínimo escalado
            active_months = (df.groupby(gcols)[f'tn_lag_{h}'].rolling(w, min_periods=1)
                            .apply(lambda x: (x > threshold).sum()))
            df[f'months_active_{w}m'] = active_months.values
            df[f'consistency_ratio_{w}m'] = df[f'months_active_{w}m'] / w

    # Features específicas de sku_size
    if 'sku_size' in df.columns:
        if f'tn_lag_{h}' in df.columns:
            df['size_avg_in_cat1'] = df.groupby(['periodo', 'cat1', 'sku_size'])[f'tn_lag_{h}'].transform('mean')
            df['size_rank_in_cat1'] = df.groupby(['periodo', 'cat1'])[f'tn_lag_{h}'].rank(ascending=False)

    # Aplicar estandarización a nuevas features
    logger.info("Aplicando estandarización a nuevas features...")
    for col in df.columns:
        if (col.startswith('cat_avg_tn_') or 
            col.startswith('brand_avg_tn_') or
            col.endswith('_momentum_mean_') or
            col.startswith('size_avg_')):
            if 'scaler' in df.columns:
                df[col] = df[col] / df['scaler']

    # Configurar tipos categóricos para nuevas features
    categorical_cols = ['is_holiday_season', 'is_back_to_school', 'is_winter_peak', 'is_summer_low']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')

    logger.info(f"Features avanzadas creadas. Total columnas: {len(df.columns)}")
    return df

def run_hyperparameter_optimization(df, cfg):
    study_name   = f'lgbm-study-{cfg.GRANULARITY}-{cfg.FEATURE_VERSION}'
    storage_path = f"sqlite:///{os.path.join(cfg.ARTIFACTS_PATH, study_name+'.db')}"
    from optuna.pruners import HyperbandPruner
    gran_cfg = cfg.GRANULARITY_CONFIGS[cfg.GRANULARITY]
    # Mejorar pruning para búsqueda
    pruner = HyperbandPruner(
        min_resource=50,
        max_resource=2000,
        reduction_factor=3
    )
    with Timer():
        logger.info(f"Iniciando optimización para granularidad: {cfg.GRANULARITY}")
        study = optuna.create_study(
            direction='minimize', study_name=study_name,
            storage=storage_path, load_if_exists=True, pruner=pruner
        )
        try:
            study.optimize(lambda t: objective(t, df, cfg),
                           n_trials=cfg.N_TRIALS, timeout=cfg.TIMEOUT)
        except KeyboardInterrupt:
            logger.warning("Optimización interrumpida por el usuario.")
    
    logger.info(f"Optimización completada. Trials finalizados: {len(study.trials)}")
    if study.best_trial:
        logger.info(f"Mejor trial (WAPE): {study.best_value:.4f}")
        for k, v in study.best_params.items():
            logger.info(f"  {k}: {v}")
        return study.best_params
    else:
        logger.error("Optuna no produjo ningún trial válido.")
        return None

def objective(trial, df, cfg):
    import lightgbm
    granularity_cfg      = cfg.GRANULARITY_CONFIGS[cfg.GRANULARITY]
    search_space         = granularity_cfg['OPTUNA_SEARCH_SPACE']
    early_stopping_rounds= granularity_cfg['EARLY_STOPPING_ROUNDS']
    params = {}
    for param_name, config_tuple in search_space.items():
        param_type = config_tuple[0]
        args   = list(config_tuple[1:])
        kwargs = {}
        if args and isinstance(args[-1], dict):
            kwargs = args.pop()
        if param_type == 'float':
            params[param_name] = trial.suggest_float(param_name, *args, **kwargs)
        elif param_type == 'int':
            params[param_name] = trial.suggest_int(param_name, *args, **kwargs)
    params.update(cfg.LGBM_FIXED_PARAMS)
    
    df_cleaned = df.dropna(subset=['y'])
    train = df_cleaned[df_cleaned['periodo'] <  cfg.OPT_TRAIN_END]
    valid = df_cleaned[df_cleaned['periodo'] == pd.to_datetime(cfg.OPT_TRAIN_END)]
    test  = df_cleaned[df_cleaned['periodo'] == pd.to_datetime(cfg.OPT_VALID_END)]
    features = [c for c in train.columns if c not in cfg.COLUMNS_TO_EXCLUDE]
    cat_features = [c for c in cfg.CATEGORICAL_FEATURES if c in features]
    X_train, y_train = train[features], train['y']
    X_valid, y_valid = valid[features], valid['y']
    X_test,  y_test  = test[features],  test['y']
    try:
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric='rmse',
            callbacks=[
                LightGBMPruningCallback(trial, 'rmse'),
                lgb.early_stopping(early_stopping_rounds, verbose=False)
            ]
        )
        y_pred_log_ratio = model.predict(X_test)
        res = test[['product_id', 'y', 'tn', 'tn_future_unscaled', 'historical_max_tn']].copy()
        if cfg.GRANULARITY == 'customer':
            res['customer_id'] = test['customer_id']
        
        # POST-PROCESAMIENTO
        log1p_tn_current = np.log1p(res['tn'])
        res['y_pred_absolute'] = np.expm1(y_pred_log_ratio + log1p_tn_current)
        
        upper_bound = res['historical_max_tn'] * 2.0
        res['y_pred_absolute'] = res['y_pred_absolute'].clip(upper=upper_bound.values)
        
        sum_act = np.sum(np.abs(res['tn_future_unscaled']))
        wape = (np.sum(np.abs(res['tn_future_unscaled'] - res['y_pred_absolute'])) /
                sum_act) if sum_act != 0 else np.nan
        return wape
    except lightgbm.basic.LightGBMError as e:
        logger.warning(f"Trial failed due to LightGBMError: {e}")
        return np.nan

def train_final_model_and_predict(df, best_params, raw_data_frames, cfg):
    if best_params is None:
        logger.error("No hay mejores parámetros para entrenar el modelo final.")
        return
    with Timer():
        logger.info("Entrenando modelo final con FEATURES AVANZADAS...")
        final_params = cfg.LGBM_FIXED_PARAMS.copy()
        final_params.update(best_params)
        df_clean = df.dropna(subset=['y'])
        train_final = df_clean[df_clean['periodo'] < cfg.FINAL_TRAIN_END]
        predict_data = df[df['periodo'] == pd.to_datetime(cfg.FINAL_TRAIN_END)].copy()
        features = [c for c in train_final.columns if c not in cfg.COLUMNS_TO_EXCLUDE]
        cat_features = [c for c in cfg.CATEGORICAL_FEATURES if c in features]
        X_train, y_train = train_final[features], train_final['y']
        X_pred          = predict_data[features]
        
        # Verificar features clave
        key_features = ['tn_lag_0']
        advanced_features = [f for f in features if any(x in f for x in ['cat_avg_', 'fourier_', 'acceleration_', 'skewness_'])]
        logger.info(f"Features clave presentes: {[f for f in key_features if f in features]}")
        logger.info(f"Features avanzadas: {len(advanced_features)} creadas")
        
        model = lgb.LGBMRegressor(**final_params)
        model.fit(X_train, y_train, categorical_feature=cat_features)
        
        # POST-PROCESAMIENTO FINAL
        y_pred_log_ratio = model.predict(X_pred)
        log1p_tn_current = np.log1p(predict_data['tn'])
        predict_data['tn_pred'] = np.expm1(y_pred_log_ratio + log1p_tn_current)

        upper_bound = predict_data['historical_max_tn'] * 2.0
        predict_data['tn_pred'] = predict_data['tn_pred'].clip(upper=upper_bound.values)

        if cfg.GRANULARITY == 'customer':
            final_pred = (predict_data.groupby('product_id')['tn_pred'].sum().reset_index())
        else:
            final_pred = predict_data[['product_id', 'tn_pred']]
        submission = pd.merge(
            raw_data_frames['productos_a_predecir'], final_pred,
            on='product_id', how='left').fillna(0)
        submission.rename(columns={'tn_pred': 'tn'}, inplace=True)
        
        os.makedirs(cfg.ARTIFACTS_PATH, exist_ok=True)
        sub_path = os.path.join(cfg.ARTIFACTS_PATH,
                                f'submission_202002_{cfg.GRANULARITY}_{cfg.FEATURE_VERSION}.csv')
        submission.to_csv(sub_path, index=False)
        logger.info(f"Archivo de submission guardado en: {sub_path}")
        
        # Feature importance
        imp_df = pd.DataFrame({
            'feature': model.feature_name_,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Log de las top features
        logger.info("Top 15 features más importantes:")
        for i, (_, row) in enumerate(imp_df.head(15).iterrows()):
            logger.info(f"  {i+1:2d}. {row['feature']}: {row['importance']:.1f}")
        
        plt.figure(figsize=(12, 14))
        lgb.plot_importance(model, importance_type='split',
                            max_num_features=40, ax=plt.gca())
        plt.title(f'Feature Importance ({cfg.GRANULARITY})')
        plt.tight_layout()
        plot_path = os.path.join(cfg.ARTIFACTS_PATH,
                                 f'feature_importance_{cfg.GRANULARITY}_{cfg.FEATURE_VERSION}.png')
        plt.savefig(plot_path); plt.close()
        imp_df.to_csv(os.path.join(cfg.ARTIFACTS_PATH,
                                   f'feature_importance_{cfg.GRANULARITY}_{cfg.FEATURE_VERSION}.csv'),
                      index=False)
        
        total_pred = submission['tn'].sum()
        # Análisis de features avanzadas
        advanced_in_top20 = len([f for f in imp_df.head(20)['feature'] if any(x in f for x in ['cat_avg_', 'fourier_', 'acceleration_', 'skewness_'])])
        
        with open(os.path.join(cfg.ARTIFACTS_PATH,
                               f'prediction_summary_{cfg.GRANULARITY}_{cfg.FEATURE_VERSION}.txt'), 'w') as f:
            f.write(f"Resumen Predicción Final ({cfg.GRANULARITY})\n")
            f.write("="*80 + "\n")
            f.write(f"Total toneladas predichas para 2020-02: {total_pred:,.2f}\n")
            f.write(f"Estrategia: y = log1p(tn_future) - log1p(tn_current)\n")
            f.write(f"Cache Level 3: {cfg.FEATURE_VERSION}_ii\n")
            f.write(f"Total features: {len(features)}\n")
            f.write(f"Features avanzadas creadas: {len(advanced_features)}\n")
            f.write(f"Features avanzadas en Top-20: {advanced_in_top20}/20\n\n")
            
            f.write("Mejoras implementadas:\n")
            f.write("- A. Features de Productos Relacionados (contexto categorías)\n")
            f.write("- B. Features de Estacionalidad Avanzada (fourier, eventos)\n")
            f.write("- C. Features de Momentum y Aceleración (slopes, volatilidad)\n")
            f.write("- D. Features de Patrones de Consumo (skewness, regularidad)\n")
            f.write("- Search space expandido en Optuna\n")
            f.write("- HyperbandPruner para exploración\n")
            
        logger.info("Artefactos finales generados.")

def validate_on_known_date(df, best_params, raw_data_frames, cfg):
    """Validación en 201912 para medir confianza del modelo en fecha conocida."""
    if best_params is None:
        logger.error("No hay mejores parámetros para validación.")
        return
    
    with Timer():
        logger.info("VALIDACIÓN EN 201912 - Midiendo confianza del modelo...")
        
        # Configuración para validación en fecha conocida
        VALIDATION_TRAIN_END = '2019-10-01'  # 2 meses antes de 201912
        VALIDATION_TARGET = '2019-12-01'     # Predecir 201912
        
        final_params = cfg.LGBM_FIXED_PARAMS.copy()
        final_params.update(best_params)
        
        df_clean = df.dropna(subset=['y'])
        
        # Entrenar hasta octubre 2019
        train_validation = df_clean[df_clean['periodo'] < VALIDATION_TRAIN_END]
        predict_validation = df[df['periodo'] == pd.to_datetime(VALIDATION_TARGET)].copy()
        
        features = [c for c in train_validation.columns if c not in cfg.COLUMNS_TO_EXCLUDE]
        cat_features = [c for c in cfg.CATEGORICAL_FEATURES if c in features]
        
        X_train_val, y_train_val = train_validation[features], train_validation['y']
        X_pred_val = predict_validation[features]
        
        logger.info(f"Entrenando para validación: {len(train_validation):,} registros hasta {VALIDATION_TRAIN_END}")
        logger.info(f"Prediciendo: {VALIDATION_TARGET} ({len(predict_validation):,} productos)")
        
        model = lgb.LGBMRegressor(**final_params)
        model.fit(X_train_val, y_train_val, categorical_feature=cat_features)
        
        # POST-PROCESAMIENTO: Misma lógica que modelo principal
        y_pred_log_ratio = model.predict(X_pred_val)
        log1p_tn_current = np.log1p(predict_validation['tn'])
        predict_validation['tn_pred'] = np.expm1(y_pred_log_ratio + log1p_tn_current)

        # Cap predictions
        upper_bound = predict_validation['historical_max_tn'] * 2.0
        predict_validation['tn_pred'] = predict_validation['tn_pred'].clip(upper=upper_bound.values)

        if cfg.GRANULARITY == 'customer':
            final_pred_val = predict_validation.groupby('product_id')['tn_pred'].sum().reset_index()
        else:
            final_pred_val = predict_validation[['product_id', 'tn_pred']]
        
        # Crear submission de validación
        submission_val = pd.merge(
            raw_data_frames['productos_a_predecir'], final_pred_val,
            on='product_id', how='left'
        ).fillna(0)
        submission_val.rename(columns={'tn_pred': 'tn'}, inplace=True)
        
        # Guardar archivo de validación
        val_path = os.path.join(cfg.ARTIFACTS_PATH,
                               f'validation_201912_{cfg.GRANULARITY}_{cfg.FEATURE_VERSION}.csv')
        submission_val.to_csv(val_path, index=False)
        
        total_pred_val = submission_val['tn'].sum()
        logger.info(f"Validación 201912 guardada en: {val_path}")
        logger.info(f"Total predicho para 201912: {total_pred_val:,.2f} toneladas")
        
        # Generar resumen de validación
        val_summary_path = os.path.join(cfg.ARTIFACTS_PATH,
                                       f'validation_summary_201912_{cfg.GRANULARITY}_{cfg.FEATURE_VERSION}.txt')
        with open(val_summary_path, 'w') as f:
            f.write(f"Resumen Validación 201912 - {cfg.GRANULARITY}\n")
            f.write("="*60 + "\n")
            f.write(f"Entrenamiento hasta: {VALIDATION_TRAIN_END}\n")
            f.write(f"Predicción objetivo: {VALIDATION_TARGET}\n")
            f.write(f"Hiperparámetros: Optimizados con {cfg.FEATURE_VERSION}\n")
            f.write(f"Total productos: {len(submission_val)}\n")
            f.write(f"Total toneladas predichas: {total_pred_val:,.2f}\n")
            f.write(f"Estrategia: y = log1p(tn_future) - log1p(tn_current)\n")
            f.write(f"Features: {len(features)} (incluye avanzadas)\n")
            f.write(f"\nPropósito:\n")
            f.write(f"- Medir confianza del modelo en fecha conocida\n")
            f.write(f"- Comparar vs datos reales 201912 para Total Error Rate\n")
            f.write(f"- Usar como métrica para ensemble inteligente vs otras granularidades\n")
            f.write(f"- Selección producto por producto del mejor modelo\n")
        
        logger.info("Validación 201912 completada - Lista para comparación con datos reales")

if __name__ == '__main__':
    with Timer():
        logger.info("INICIANDO FORECASTING - FEATURES AVANZADAS v11")
        cfg = Config()
        for p in [cfg.CACHE_PATH, cfg.ARTIFACTS_PATH]:
            os.makedirs(p, exist_ok=True)
        logger.info("--- FASE 1: Preparación de Datos ---")
        raw_dfs     = load_data(cfg)
        processed   = preprocess_data(raw_dfs, cfg)
        final_df    = create_features_and_target(processed, cfg)
        logger.info("--- FASE 2: Optimización de Hiperparámetros ---")
        best_params = run_hyperparameter_optimization(final_df, cfg)
        logger.info("--- FASE 3: Entrenamiento Final y Predicción ---")
        train_final_model_and_predict(final_df, best_params, raw_dfs, cfg)
        logger.info("--- FASE 4: Validación en Fecha Conocida ---")
        validate_on_known_date(final_df, best_params, raw_dfs, cfg)
        logger.info("PROCESO FEATURES AVANZADAS v11 COMPLETADO")
        logger.info(f"Objetivo: Superar meseta con features A+B+C+D de alto impacto")
        logger.info(f"Validación 201912: Lista para ensemble inteligente")
        logger.info(f"Revisa los artefactos en '{cfg.ARTIFACTS_PATH}'")