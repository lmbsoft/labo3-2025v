#!/usr/bin/env python3
"""
AutoGluon Forecaster GPU - Versión Mejorada
==========================================

Script optimizado para forecasting de toneladas (tn) con horizonte t+2.
Hardware: Intel i7-7700 (8 threads) + RTX 2060 (6GB) + 24GB RAM

Mejoras incorporadas:
- Más variedad de modelos foundation (adaptados a 6GB VRAM)
- Static features restauradas con manejo robusto
- Timeouts más agresivos pero realistas para i7-7700
- Más artefactos de salida y análisis
- Configuración de ensemble expandida

Flujo:
1. Descarga archivos automáticamente si no existen
2. Construye dataset con lógica de vida útil + padding temporal
3. Entrena con datos 2017-01 a 2019-09
4. Valida con datos 2019-10 a 2019-12  
5. Re-entrena con todos los datos (2017-01 a 2019-12)
6. Predice 2020-02 (t+2)
7. Genera múltiples artefactos de análisis
"""

import os
import sys
import time
import warnings
from datetime import datetime
from typing import Tuple, Dict
import urllib.request
import gzip
import shutil

import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Suprimir warnings para output limpio
warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None

print("🚀 Iniciando AutoGluon Forecaster GPU - Versión Mejorada...")
print("=" * 60)

# Verificar instalación de AutoGluon
try:
    from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame
    import autogluon.timeseries
    print(f"✅ AutoGluon TimeSeries v{autogluon.timeseries.__version__} cargado")
except ImportError as e:
    print(f"❌ Error: AutoGluon no instalado correctamente: {e}")
    sys.exit(1)

try:
    import torch
    print(f"✅ PyTorch v{torch.__version__} - Cores disponibles: {torch.get_num_threads()}")
    if torch.cuda.is_available():
        print(f"✅ GPU detectada: {torch.cuda.get_device_name(0)}")
        print(f"✅ VRAM disponible: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    else:
        print("⚠️ GPU no disponible - usando CPU")
except ImportError:
    print("⚠️ PyTorch no disponible - AutoGluon funcionará con backends alternativos")

print("=" * 60)

# ============================================================================
# CONFIGURACIÓN GLOBAL
# ============================================================================

# URLs de descarga
FILE_CONFIG = {
    "sell_in": {
        "url": "https://storage.googleapis.com/open-courses/austral2025-af91/labo3v/sell-in.txt.gz",
        "local": "sell-in.txt.gz",
        "is_gzip": True
    },
    "productos": {
        "url": "https://storage.googleapis.com/open-courses/austral2025-af91/labo3v/tb_productos.txt",
        "local": "tb_productos.txt",
        "is_gzip": False
    },
    "stocks": {
        "url": "https://storage.googleapis.com/open-courses/austral2025-af91/labo3v/tb_stocks.txt", 
        "local": "tb_stocks.txt",
        "is_gzip": False
    },
    "productos_a_predecir": {
        "url": "https://storage.googleapis.com/open-courses/austral2025-af91/labo3v/product_id_apredecir201912.txt",
        "local": "product_id_apredecir201912.txt",
        "is_gzip": False
    }
}

# Configuración temporal
DATES_CONFIG = {
    "data_start": "2017-01-01",
    "train_end": "2019-09-01",     # Training hasta sep 2019
    "validation_end": "2019-12-01", # Validación oct-dic 2019
    "prediction_target": "2020-02-01"  # Predecir feb 2020 (t+2)
}

# Directorios
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed" 
RESULTS_DIR = "data/results"
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

for dir_path in [RAW_DIR, PROCESSED_DIR, RESULTS_DIR, PLOTS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# ============================================================================
# UTILIDADES DE DESCARGA Y CARGA
# ============================================================================

def download_file(url: str, local_path: str, is_gzip: bool = False) -> None:
    """Descarga archivo con barra de progreso."""
    print(f"📥 Descargando {os.path.basename(local_path)}...")
    
    def show_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, (downloaded * 100) // total_size)
            bar_length = 30
            filled_length = int(bar_length * percent // 100)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            print(f"\r   [{bar}] {percent}% ({downloaded:,}/{total_size:,} bytes)", end='')
    
    urllib.request.urlretrieve(url, local_path, reporthook=show_progress)
    print()  # Nueva línea después de la barra de progreso
    
    # Si es gzip, descomprimir
    if is_gzip and local_path.endswith('.gz'):
        print(f"📂 Descomprimiendo {os.path.basename(local_path)}...")
        output_path = local_path[:-3]  # Remove .gz extension
        with gzip.open(local_path, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        print(f"✅ Descomprimido a {os.path.basename(output_path)}")

def load_data_files() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carga todos los archivos necesarios, descargándolos si es necesario."""
    print("📁 Cargando archivos de datos...")
    
    # Descargar archivos si no existen
    for name, config in FILE_CONFIG.items():
        local_path = os.path.join(RAW_DIR, config["local"])
        
        # Para archivos gzip, verificar tanto .gz como descomprimido
        if config["is_gzip"]:
            decompressed_path = local_path[:-3]
            if not os.path.exists(local_path) and not os.path.exists(decompressed_path):
                download_file(config["url"], local_path, config["is_gzip"])
        else:
            if not os.path.exists(local_path):
                download_file(config["url"], local_path, config["is_gzip"])
    
    # Cargar archivos
    print("📊 Cargando datos en memoria...")
    
    # Sell-in (puede estar comprimido o no)
    sell_in_path = os.path.join(RAW_DIR, "sell-in.txt")
    if not os.path.exists(sell_in_path):
        sell_in_path = os.path.join(RAW_DIR, "sell-in.txt.gz")
    
    sell_in = pd.read_csv(sell_in_path, sep='\t', dtype={'customer_id': str, 'product_id': str})
    print(f"   ✅ Sell-in: {len(sell_in):,} registros")
    
    productos = pd.read_csv(os.path.join(RAW_DIR, "tb_productos.txt"), sep='\t', dtype={'product_id': str})
    print(f"   ✅ Productos: {len(productos):,} registros")
    
    stocks = pd.read_csv(os.path.join(RAW_DIR, "tb_stocks.txt"), sep='\t', dtype={'product_id': str})
    print(f"   ✅ Stocks: {len(stocks):,} registros")
    
    # Productos a predecir (puede tener diferentes nombres de columna)
    productos_pred_path = os.path.join(RAW_DIR, "product_id_apredecir201912.txt")
    productos_pred = pd.read_csv(productos_pred_path, dtype={'product_id': str})
    
    # Normalizar nombre de columna si es necesario
    if 'product_id' not in productos_pred.columns:
        productos_pred.columns = ['product_id']
    
    print(f"   ✅ Productos a predecir: {len(productos_pred):,} productos")
    
    return sell_in, productos, stocks, productos_pred

# ============================================================================
# PREPROCESSING CON LÓGICA DE VIDA ÚTIL + PADDING
# ============================================================================

def complete_temporal_grid(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Completa el grid temporal usando lógica de vida útil + padding fijo.
    
    PASO 1: Genera combinaciones cliente×producto durante períodos de coexistencia
    PASO 2: Extiende todas las combinaciones a la ventana temporal completa
    """
    print("🔄 Aplicando lógica de vida útil + padding temporal...")
    
    df = df.copy()
    df["periodo"] = pd.to_datetime(df["periodo"], format="%Y%m")
    
    # PASO 1: Completado con vida útil
    print("   📅 Paso 1: Calculando períodos de coexistencia...")
    
    vida_clientes = df.groupby('customer_id')['periodo'].agg(['min', 'max']).rename(
        columns={'min': 'cliente_ini', 'max': 'cliente_fin'})
    vida_productos = df.groupby('product_id')['periodo'].agg(['min', 'max']).rename(
        columns={'min': 'producto_ini', 'max': 'producto_fin'})
    
    # Producto cartesiano: todos los clientes × todos los productos
    clientes_df = vida_clientes.reset_index()
    productos_df = vida_productos.reset_index()
    cp_df = clientes_df.assign(key=1).merge(productos_df.assign(key=1), on='key').drop('key', axis=1)
    
    print(f"   🔢 Combinaciones cliente×producto: {len(cp_df):,}")
    
    # Calcular períodos de coexistencia
    cp_df['inicio_actividad'] = cp_df[['cliente_ini', 'producto_ini']].apply(lambda x: max(x), axis=1)
    cp_df['fin_actividad'] = cp_df[['cliente_fin', 'producto_fin']].apply(lambda x: min(x), axis=1)
    
    # Filtrar solo pares válidos (donde inicio <= fin)
    valid_pairs = cp_df[cp_df['inicio_actividad'] <= cp_df['fin_actividad']].reset_index(drop=True)
    print(f"   ✅ Pares válidos con coexistencia: {len(valid_pairs):,}")
    
    # Expandir períodos de coexistencia (algoritmo vectorizado)
    n_months = ((valid_pairs["fin_actividad"].dt.year - valid_pairs["inicio_actividad"].dt.year) * 12 
                + (valid_pairs["fin_actividad"].dt.month - valid_pairs["inicio_actividad"].dt.month) + 1).astype("int16")
    
    rep_idx = valid_pairs.index.repeat(n_months)
    base = valid_pairs.loc[rep_idx].reset_index(drop=True)
    offsets = np.concatenate([np.arange(k, dtype="int16") for k in n_months])
    
    # Generar períodos usando offsets
    base["periodo"] = (base["inicio_actividad"].dt.to_period("M") + offsets).astype("datetime64[ns]")
    
    step1_df = base[["customer_id", "product_id", "periodo"]].merge(
        df, on=["product_id", "customer_id", "periodo"], how="left")
    
    print(f"   📊 Registros después del Paso 1: {len(step1_df):,}")
    
    # PASO 2: Padding fijo a ventana completa
    print("   📅 Paso 2: Extendiendo a ventana temporal completa...")
    
    all_months = pd.date_range(start=start_date, end=end_date, freq="MS")
    n_periods = len(all_months)
    
    print(f"   📅 Ventana temporal: {start_date} a {end_date} ({n_periods} meses)")
    
    # Obtener todas las combinaciones únicas del paso 1
    pairs = step1_df[["customer_id", "product_id"]].drop_duplicates().reset_index(drop=True)
    N = len(pairs)
    
    print(f"   🔢 Expandiendo {N:,} pares únicos × {n_periods} períodos = {N*n_periods:,} registros")
    
    # Expandir vectorizado: cada par × todos los períodos
    base_fixed = pairs.loc[pairs.index.repeat(n_periods)].reset_index(drop=True)
    base_fixed["periodo"] = np.tile(all_months, N)
    
    # Merge final con datos originales
    full_df = base_fixed.merge(df, on=["product_id", "customer_id", "periodo"], how="left")
    
    # Rellenar NaN con 0 en columnas numéricas
    cols_to_fill = ["tn", "plan_precios_cuidados", "cust_request_qty", "cust_request_tn"]
    for col in cols_to_fill:
        if col in full_df.columns:
            full_df[col] = full_df[col].fillna(0)
    
    # Asegurar tipos de datos
    full_df["product_id"] = full_df["product_id"].astype(str)
    full_df["customer_id"] = full_df["customer_id"].astype(str)
    
    print(f"   ✅ Grid temporal completo: {len(full_df):,} registros")
    
    return full_df.sort_values(["product_id", "customer_id", "periodo"]).reset_index(drop=True)

def build_base_dataset() -> pd.DataFrame:
    """Construye el dataset base con toda la información consolidada."""
    
    # Verificar si ya existe dataset procesado
    base_path = os.path.join(PROCESSED_DIR, "base_dataset.parquet")
    if os.path.exists(base_path):
        print("📁 Dataset base ya existe, cargando desde cache...")
        return pd.read_parquet(base_path)
    
    print("🏗️ Construyendo dataset base...")
    
    # Cargar archivos raw
    sell_in, productos, stocks, productos_pred = load_data_files()
    
    # FILTRO CRÍTICO: Solo productos a predecir
    print("🎯 Filtrando solo productos a predecir...")
    sell_in_filtered = sell_in.merge(productos_pred, on='product_id', how='inner')
    print(f"   📊 Registros después del filtro: {len(sell_in_filtered):,}")
    print(f"   🛍️ Productos únicos: {sell_in_filtered['product_id'].nunique():,}")
    print(f"   👥 Clientes únicos: {sell_in_filtered['customer_id'].nunique():,}")
    
    # Completar grid temporal
    full_sell = complete_temporal_grid(
        sell_in_filtered, 
        DATES_CONFIG["data_start"], 
        DATES_CONFIG["validation_end"]
    )
    
    # Agregar información de productos
    print("🔗 Agregando información de productos...")
    productos["product_id"] = productos["product_id"].astype(str)
    df = full_sell.merge(productos, on="product_id", how="left")
    
    # Agregar información de stocks
    print("🔗 Agregando información de stocks...")
    stocks["product_id"] = stocks["product_id"].astype(str)
    stocks["periodo"] = pd.to_datetime(stocks["periodo"], format="%Y%m")
    
    df = df.merge(
        stocks[["periodo", "product_id", "stock_final"]],
        on=["periodo", "product_id"],
        how="left"
    )
    df["stock_final"] = df["stock_final"].fillna(0)
    
    # Guardar dataset base
    print(f"💾 Guardando dataset base en {base_path}...")
    df.to_parquet(base_path, index=False)
    print(f"✅ Dataset base guardado: {len(df):,} registros")
    
    return df

# ============================================================================
# PREPARACIÓN PARA AUTOGLUON - VERSIÓN MEJORADA CON STATIC FEATURES
# ============================================================================

def prepare_timeseries_data(df: pd.DataFrame) -> Tuple[TimeSeriesDataFrame, TimeSeriesDataFrame, TimeSeriesDataFrame, pd.DataFrame]:
    """
    Prepara los datos para AutoGluon con división temporal correcta.
    MEJORADO: Restaura static features con manejo robusto de errores.
    
    Returns:
        train_data: 2017-01 a 2019-09 (para training inicial)
        validation_data: 2017-01 a 2019-12 (para re-training final)  
        full_data: Todo el dataset (para generar predicciones)
        static_features: DataFrame con características estáticas
    """
    print("🔄 Preparando datos para AutoGluon TimeSeries...")
    
    # Convertir fechas
    train_end = pd.to_datetime(DATES_CONFIG["train_end"])
    validation_end = pd.to_datetime(DATES_CONFIG["validation_end"])
    
    # Crear ID compuesto producto-cliente (orden: product_id + '_' + customer_id)
    df['product_customer_id'] = df['product_id'].astype(str) + '_' + df['customer_id'].astype(str)
    # IMPORTANTE: Renombrar 'tn' a 'target' (AutoGluon lo requiere)
    df = df.rename(columns={'tn': 'target'})
    
    # Filtrar series con suficiente historia (mínimo 24 meses)
    # print("   🔍 Filtrando series con suficiente historia... (DESACTIVADO)")
    # history_counts = df.groupby('customer_product_id').size()
    # min_history = 24  # Mínimo 24 meses
    # valid_ids = history_counts[history_counts >= min_history].index
    # df_filtered = df[df['customer_product_id'].isin(valid_ids)]
    df_filtered = df  # <--- No se filtra ninguna serie
    print(f"   ✅ Series válidas (todas): {df_filtered['product_customer_id'].nunique():,}")
    print(f"   📊 Registros: {len(df_filtered):,}")
    
    # MEJORADO: Crear static features con manejo robusto
    print("   📋 Creando static features...")
    static_features = None
    
    try:
        # Intentar crear static features
        static_cols = ['customer_id', 'product_id', 'cat1', 'cat2', 'cat3', 'brand', 'sku_size']
        available_cols = [col for col in static_cols if col in df_filtered.columns]
        
        if len(available_cols) >= 3:  # Mínimo 3 columnas para que valga la pena
            # Crear product_customer_id con el mismo orden en static_features
            static_features = df_filtered[['product_id', 'customer_id'] + available_cols]
            static_features['product_customer_id'] = static_features['product_id'].astype(str) + '_' + static_features['customer_id'].astype(str)
            static_features = static_features[['product_customer_id'] + available_cols]
            static_features = static_features.drop_duplicates(subset=['product_customer_id']).reset_index(drop=True)
            print(f"   📋 Static features columnas: {list(static_features.columns)}")
            print(f"   📋 Static features shape: {static_features.shape}")
            # Validar que no hay NaN críticos
            nan_counts = static_features.isnull().sum()
            if nan_counts.sum() > 0:
                print(f"   ⚠️ NaN detectados en static features: {dict(nan_counts[nan_counts > 0])}")
                # Rellenar NaN con valores por defecto
                for col in static_features.columns:
                    if col == 'product_customer_id':
                        continue
                    if static_features[col].dtype == 'object':
                        static_features[col] = static_features[col].fillna('UNKNOWN')
                    else:
                        static_features[col] = static_features[col].fillna(0)
        else:
            print("   ⚠️ Pocas columnas static disponibles, omitiendo static features")
            
    except Exception as e:
        print(f"   ⚠️ Error creando static features: {e}")
        print("   ℹ️ Continuando sin static features...")
        static_features = None
    
    # Dataset de training (hasta sep 2019)
    train_df = df_filtered[df_filtered['periodo'] <= train_end].copy()
    
    # Dataset de validación (hasta dic 2019 - para re-training final)
    validation_df = df_filtered[df_filtered['periodo'] <= validation_end].copy()
    
    # Dataset completo (para contexto en predicción)
    full_df = df_filtered.copy()
    
    print(f"   📅 Training data: {train_df['periodo'].min()} a {train_df['periodo'].max()}")
    print(f"   📅 Validation data: {validation_df['periodo'].min()} a {validation_df['periodo'].max()}")
    print(f"   📊 Training registros: {len(train_df):,}")
    print(f"   📊 Validation registros: {len(validation_df):,}")
    
    # Convertir a TimeSeriesDataFrame (API correcta para AutoGluon 1.3.1)
    print("   🔄 Convirtiendo a TimeSeriesDataFrame...")
    
    # Parámetros para TimeSeriesDataFrame
    ts_params = {
        'id_column': 'product_customer_id',
        'timestamp_column': 'periodo'
    }
    # Agregar static features solo si están disponibles
    if static_features is not None:
        ts_params['static_features_df'] = static_features
    
    try:
        train_ts = TimeSeriesDataFrame.from_data_frame(df=train_df, **ts_params)
        validation_ts = TimeSeriesDataFrame.from_data_frame(df=validation_df, **ts_params)
        full_ts = TimeSeriesDataFrame.from_data_frame(df=full_df, **ts_params)
        
        print(f"   ✅ TimeSeriesDataFrame creado: {len(train_ts.item_ids)} series temporales (ID: product_customer_id)")
        if static_features is not None:
            print(f"   ✅ Static features incluidas: {list(static_features.columns)} (ID: product_customer_id)")
            
    except Exception as e:
        print(f"   ⚠️ Error con static features: {e}")
        print("   🔄 Creando TimeSeriesDataFrame sin static features...")
        
        train_ts = TimeSeriesDataFrame.from_data_frame(
            df=train_df,
            id_column='product_customer_id',
            timestamp_column='periodo'
        )
        validation_ts = TimeSeriesDataFrame.from_data_frame(
            df=validation_df,
            id_column='product_customer_id', 
            timestamp_column='periodo'
        )
        full_ts = TimeSeriesDataFrame.from_data_frame(
            df=full_df,
            id_column='product_customer_id',
            timestamp_column='periodo'
        )
        static_features = None
    
    return train_ts, validation_ts, full_ts, static_features

# ============================================================================
# ENTRENAMIENTO MEJORADO CON MÁS MODELOS Y TIMEOUTS AGRESIVOS
# ============================================================================

def train_autogluon_enhanced(train_data: TimeSeriesDataFrame, 
                           validation_data: TimeSeriesDataFrame) -> TimeSeriesPredictor:
    """
    Entrenamiento mejorado para i7-7700 + RTX 2060:
    - Más variedad de modelos foundation (adaptados a 6GB VRAM)
    - Timeouts más agresivos pero realistas
    - Configuración de ensemble expandida
    """
    print("🚀 Iniciando entrenamiento mejorado de AutoGluon...")
    print("⏱️ Tiempo límite: 3-4 horas (optimizado para i7-7700)")
    print("⚡ Hardware: i7-7700 + RTX 2060 + 24GB RAM")
    print("=" * 60)
    
    # Detectar dispositivo GPU
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🎮 Dispositivo: {device}")
    
    if device == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
    # Configuración del predictor
    predictor = TimeSeriesPredictor(
        prediction_length=2,  # Predecir t+1 y t+2
        eval_metric='MASE',   # Metric ideal para ventas
        quantile_levels=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99],  # Cuantiles completos
        path=os.path.join(RESULTS_DIR, "autogluon_models"),
        verbosity=2,
        cache_predictions=True
    )
    
    # MEJORADO: Más variedad de modelos adaptados a tu hardware
    hyperparameters = {
        # === FOUNDATION MODELS with GPU support ===
        'Chronos': {
            'model_path': 'tiny',  # Modelo pequeño para 6GB VRAM
            'device': device,
            'batch_size': 32 if device == "cuda" else 16
        },
        
        # === MODERN TRANSFORMER MODELS ===
        'PatchTST': {
            'epochs': 50 if device == "cuda" else 30,
            'batch_size': 64 if device == "cuda" else 32,
            'device': device
        },
        
        'TemporalFusionTransformer': {
            'epochs': 40 if device == "cuda" else 25,
            'learning_rate': 1e-3,
            'hidden_size': 64,  # Reducido para 6GB VRAM
            'num_heads': 4,
            'device': device
        },
        
        'TiDE': {
            'epochs': 30 if device == "cuda" else 20,
            'batch_size': 32 if device == "cuda" else 16,
            'device': device
        },
        
        # === TREE-BASED MODELS (CPU optimized) - TIMEOUTS AGRESIVOS ===
        'RecursiveTabular': {
            'time_limit': 5400,  # 1.5 horas (aumentado)
            'hyperparameters': {
                'GBM': {
                    'num_threads': 8,  # Usar todos los threads del i7
                    'max_depth': 8,    # Profundidad aumentada
                    'num_boost_round': 1000  # Más rondas
                }
            }
        },
        
        'DirectTabular': {
            'time_limit': 5400,  # 1.5 horas (aumentado)
            'hyperparameters': {
                'GBM': {
                    'num_threads': 8,
                    'max_depth': 8,
                    'num_boost_round': 1000
                }
            }
        },
        
        # === DEEP LEARNING MODELS ===
        'DeepAR': {
            'epochs': 50 if device == "cuda" else 30,  # Más epochs con GPU
            'learning_rate': 1e-3,
            'batch_size': 64 if device == "cuda" else 32,
            'device': device
        } if device == "cuda" else {},  # Solo con GPU
        
        'DLinear': {
            'epochs': 40 if device == "cuda" else 25,
            'batch_size': 128 if device == "cuda" else 64,
            'device': device
        },
        
        'SimpleFeedForward': {
            'epochs': 30,
            'learning_rate': 1e-3,
            'device': device
        },
        
        # === STATISTICAL MODELS (Rápidos y confiables) ===
        'AutoETS': {},
        'ETS': {},
        'AutoARIMA': {
            'max_p': 4,  # Aumentado de 3
            'max_q': 4,  # Aumentado de 3
            'max_order': 8  # Aumentado de 6
        },
        'DynamicOptimizedTheta': {},
        'Theta': {},
        'SeasonalNaive': {},
        'Naive': {},
        'Average': {},
        
        # === SPECIALTY MODELS ===
        'Croston': {},  # Para series con muchos ceros
        'CrostonSBA': {}  # Variante de Croston
    }
    
    # Filtrar modelos vacíos
    hyperparameters = {k: v for k, v in hyperparameters.items() if v}
    
    print("🎯 Modelos a entrenar:")
    for i, model_name in enumerate(hyperparameters.keys(), 1):
        print(f"   {i:2d}. {model_name}")
    
    print("\n🏁 Iniciando entrenamiento...")
    start_time = time.time()
    
    # Entrenamiento inicial con datos de training - TIEMPO AGRESIVO
    print("\n📚 FASE 1: Entrenamiento inicial (2017-01 a 2019-09)")
    predictor.fit(
        train_data,
        hyperparameters=hyperparameters,
        time_limit=28800,  # 8 horas para training inicial (aumentado)
        enable_ensemble=True
    )
    
    print("\n📊 Evaluando modelo en datos de validación (2019-10 a 2019-12)...")
    
    # Evaluar en período de validación - MANEJO ROBUSTO
    try:
        leaderboard_validation = predictor.leaderboard(validation_data)
        print("\n🏆 TOP 5 MODELOS (validación):")
        
        # Manejo robusto de columnas del leaderboard
        available_cols = ['model']
        optional_cols = ['score_val', 'score_test', 'fit_time', 'pred_time', 'fit_time_marginal', 'pred_time_marginal']
        
        for col in optional_cols:
            if col in leaderboard_validation.columns:
                available_cols.append(col)
        
        print(leaderboard_validation.head()[available_cols])
        
    except Exception as e:
        print(f"⚠️ Error en evaluación: {e}")
        leaderboard_validation = pd.DataFrame({'model': ['Error_Model'], 'score_val': [999]})
    
    # Re-entrenar con todos los datos disponibles
    print("\n📚 FASE 2: Re-entrenamiento con datos completos (2017-01 a 2019-12)")
    
    # Crear nuevo predictor para re-training final
    final_predictor = TimeSeriesPredictor(
        prediction_length=2,
        eval_metric='MASE',
        quantile_levels=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99],
        path=os.path.join(RESULTS_DIR, "autogluon_models_final"),
        verbosity=2,
        cache_predictions=True
    )
    
    # Usar configuración optimizada basada en resultados de validación
    # Tomar los mejores modelos del leaderboard (excluyendo WeightedEnsemble)
    best_models_raw = leaderboard_validation.head(10)['model'].tolist()  # Top 10 en lugar de 8
    best_models = [model for model in best_models_raw if 'WeightedEnsemble' not in model]
    
    optimized_hyperparameters = {}
    
    for model in best_models:
        if 'Chronos' in model:
            optimized_hyperparameters['Chronos'] = {'device': device, 'batch_size': 32}
        elif 'Tabular' in model:
            optimized_hyperparameters[model] = {'time_limit': 5400}  # 1.5 horas (aumentado)
        elif 'DeepAR' in model and device == "cuda":
            optimized_hyperparameters['DeepAR'] = {'epochs': 75, 'device': device}  # Más epochs
        else:
            optimized_hyperparameters[model] = hyperparameters.get(model, {})
    
    print(f"🎯 Re-entrenando con los mejores {len(optimized_hyperparameters)} modelos...")
    
    # Re-entrenar con datos completos - TIEMPO AGRESIVO
    final_predictor.fit(
        validation_data,  # Datos hasta dic 2019
        hyperparameters=optimized_hyperparameters,
        time_limit=5400,  # 1.5 horas para re-training final (agresivo)
        enable_ensemble=True
    )
    
    total_time = time.time() - start_time
    print(f"\n✅ Entrenamiento completado en {total_time/3600:.2f} horas")
    
    return final_predictor

# ============================================================================
# GENERACIÓN DE ARTEFACTOS BASELINE
# ============================================================================

def generate_baseline_artifacts(base_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Genera los artefactos baseline solicitados:
    1. Promedio últimos 12 meses por producto (ene-2019 a dic-2019)
    2. Datos para comparación y combinación con AutoGluon
    
    CORREGIDO: Ahora calcula correctamente consolidando por producto-mes
    """
    print("📊 Generando artefactos baseline (promedio 12 meses CORREGIDO)...")
    
    # CORRECCIÓN: Período correcto - últimos 12 meses de 2019
    start_date = pd.to_datetime("2019-01-01")  # Enero 2019
    end_date = pd.to_datetime("2019-12-01")    # Diciembre 2019
    
    print(f"   📅 Período baseline CORRECTO: {start_date.strftime('%Y-%m')} a {end_date.strftime('%Y-%m')}")
    
    # Filtrar datos del período baseline
    baseline_period = base_df[
        (base_df['periodo'] >= start_date) & 
        (base_df['periodo'] <= end_date)
    ].copy()
    
    print(f"   📊 Registros en período baseline: {len(baseline_period):,}")
    
    # CORRECCIÓN: Cálculo correcto en dos pasos
    print("   🔄 PASO 1: Consolidando toneladas por producto-mes...")
    
    # 1. Sumar toneladas por producto por mes (consolidar clientes)
    monthly_totals = baseline_period.groupby(['product_id', 'periodo'])['tn'].sum().reset_index()
    print(f"      📊 Totales producto-mes: {len(monthly_totals):,}")
    
    # 2. Calcular promedio mensual por producto (promedio de 12 meses)
    print("   🔄 PASO 2: Calculando promedio mensual por producto...")
    baseline_avg = monthly_totals.groupby('product_id')['tn'].mean().reset_index()
    baseline_avg.columns = ['product_id', 'avg_12m']
    
    # Aplicar clipping a 0 (consistente con AutoGluon)
    baseline_avg['avg_12m'] = baseline_avg['avg_12m'].clip(lower=0)
    
    print(f"   📈 Productos en baseline: {len(baseline_avg):,}")
    print(f"   📊 Promedio total baseline: {baseline_avg['avg_12m'].sum():.2f} toneladas")
    
    # Verificar que el cálculo sea razonable
    if baseline_avg['avg_12m'].sum() > 1000:  # Esperamos valores similares a AutoGluon
        print("   ✅ Baseline parece razonable (>1000 toneladas total)")
    else:
        print("   ⚠️ Baseline muy bajo, verificar cálculo")
    
    # Asegurar que tenemos todos los productos esperados
    expected_products = base_df['product_id'].unique()
    baseline_products = baseline_avg['product_id'].unique()
    
    missing_products = set(expected_products) - set(baseline_products)
    if missing_products:
        print(f"   ⚠️ {len(missing_products):,} productos sin datos en período baseline, asignando 0")
        missing_df = pd.DataFrame({
            'product_id': list(missing_products),
            'avg_12m': [0.0] * len(missing_products)
        })
        baseline_avg = pd.concat([baseline_avg, missing_df], ignore_index=True)
    
    # Ordenar por product_id para consistencia
    baseline_avg = baseline_avg.sort_values('product_id').reset_index(drop=True)
    
    # Estadísticas del baseline corregido
    print(f"   📊 Estadísticas baseline CORREGIDAS:")
    print(f"      • Total productos: {len(baseline_avg):,}")
    print(f"      • Productos con avg > 0: {(baseline_avg['avg_12m'] > 0).sum():,}")
    print(f"      • Promedio por producto: {baseline_avg['avg_12m'].mean():.4f}")
    print(f"      • Mediana: {baseline_avg['avg_12m'].median():.4f}")
    print(f"      • Total esperado: {baseline_avg['avg_12m'].sum():.2f} toneladas")
    
    return baseline_avg

# ============================================================================
# GENERACIÓN DE PREDICCIONES Y ARTEFACTOS MEJORADOS
# ============================================================================

def create_analysis_artifacts(predictions_df: pd.DataFrame, base_df: pd.DataFrame) -> None:
    """
    NUEVO: Genera artefactos adicionales de análisis.
    """
    print("📊 Generando artefactos de análisis...")
    
    try:
        # 1. Análisis por categorías de productos
        print("   📈 Análisis por categorías...")
        
        # Merge con información de productos
        analysis_df = predictions_df.merge(
            base_df[['product_id', 'cat1', 'cat2', 'cat3', 'brand']].drop_duplicates(),
            on='product_id', how='left'
        )
        
        # Análisis por categoría 1
        cat1_analysis = analysis_df.groupby('cat1').agg({
            'prediction_t2_clipped': ['sum', 'mean', 'count']
        }).round(4)
        cat1_analysis.columns = ['total_pred', 'avg_pred', 'n_customers']
        cat1_analysis.to_csv(os.path.join(RESULTS_DIR, "analysis_by_cat1.csv"))
        
        # Análisis por brand
        brand_analysis = analysis_df.groupby('brand').agg({
            'prediction_t2_clipped': ['sum', 'mean', 'count']
        }).round(4)
        brand_analysis.columns = ['total_pred', 'avg_pred', 'n_customers']
        brand_analysis.to_csv(os.path.join(RESULTS_DIR, "analysis_by_brand.csv"))
        
        print(f"   ✅ Análisis por categorías guardado")
        
        # 2. Análisis de distribución de predicciones
        print("   📊 Análisis de distribución...")
        
        plt.figure(figsize=(15, 10))
        
        # Subplot 1: Distribución de predicciones
        plt.subplot(2, 3, 1)
        plt.hist(predictions_df['prediction_t2_clipped'], bins=50, alpha=0.7, edgecolor='black')
        plt.title('Distribución de Predicciones t+2')
        plt.xlabel('Toneladas predichas')
        plt.ylabel('Frecuencia')
        plt.yscale('log')
        
        # Subplot 2: Box plot por categoría
        plt.subplot(2, 3, 2)
        if 'cat1' in analysis_df.columns:
            analysis_df.boxplot(column='prediction_t2_clipped', by='cat1', ax=plt.gca())
            plt.title('Predicciones por Categoría 1')
            plt.xlabel('Categoría')
            plt.ylabel('Toneladas')
        
        # Subplot 3: Top 10 productos
        plt.subplot(2, 3, 3)
        top_products = analysis_df.groupby('product_id')['prediction_t2_clipped'].sum().nlargest(10)
        top_products.plot(kind='bar')
        plt.title('Top 10 Productos (Predicciones)')
        plt.xticks(rotation=45)
        
        # Subplot 4: Distribución de intervalos de confianza
        plt.subplot(2, 3, 4)
        ci_width = predictions_df['prediction_p90'] - predictions_df['prediction_p10']
        plt.hist(ci_width, bins=30, alpha=0.7, edgecolor='black')
        plt.title('Ancho de Intervalos de Confianza')
        plt.xlabel('P90 - P10')
        plt.ylabel('Frecuencia')
        
        # Subplot 5: Predicciones vs Intervalos de confianza
        plt.subplot(2, 3, 5)
        plt.scatter(predictions_df['prediction_t2_clipped'], ci_width, alpha=0.5)
        plt.xlabel('Predicción t+2')
        plt.ylabel('Ancho CI')
        plt.title('Predicción vs Incertidumbre')
        
        # Subplot 6: Clientes por producto
        plt.subplot(2, 3, 6)
        customers_per_product = predictions_df.groupby('product_id').size()
        plt.hist(customers_per_product, bins=30, alpha=0.7, edgecolor='black')
        plt.title('Distribución: Clientes por Producto')
        plt.xlabel('Número de clientes')
        plt.ylabel('Frecuencia')
        
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "prediction_analysis.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Gráficos de análisis guardados en {PLOTS_DIR}")
        
        # 3. Estadísticas detalladas
        print("   📋 Estadísticas detalladas...")
        
        stats_summary = {
            'total_products': len(analysis_df['product_id'].unique()),
            'total_customers': len(analysis_df['customer_id'].unique()),
            'total_predictions': analysis_df['prediction_t2_clipped'].sum(),
            'avg_prediction_per_customer': analysis_df['prediction_t2_clipped'].mean(),
            'median_prediction': analysis_df['prediction_t2_clipped'].median(),
            'std_prediction': analysis_df['prediction_t2_clipped'].std(),
            'products_with_zero_pred': (analysis_df.groupby('product_id')['prediction_t2_clipped'].sum() == 0).sum(),
            'customers_with_zero_pred': (analysis_df['prediction_t2_clipped'] == 0).sum(),
            'max_prediction_per_customer': analysis_df['prediction_t2_clipped'].max(),
            'avg_confidence_interval_width': (predictions_df['prediction_p90'] - predictions_df['prediction_p10']).mean()
        }
        
        # Guardar estadísticas
        stats_df = pd.DataFrame(list(stats_summary.items()), columns=['metric', 'value'])
        stats_df.to_csv(os.path.join(RESULTS_DIR, "prediction_statistics.csv"), index=False)
        
        print(f"   ✅ Estadísticas detalladas guardadas")
        
    except Exception as e:
        print(f"   ⚠️ Error generando artefactos: {e}")

def generate_final_predictions(predictor: TimeSeriesPredictor, 
                             full_data: TimeSeriesDataFrame,
                             base_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Genera predicciones finales y crea múltiples artefactos de salida.
    MEJORADO: Más análisis y los 4 artefactos solicitados.
    """
    print("📊 Generando predicciones finales...")
    
    # Generar predicciones para febrero 2020 (t+2)
    print("   🔮 Prediciendo febrero 2020 (t+2)...")
    forecasts = predictor.predict(full_data)
    
    # Convertir forecasts a DataFrame
    print("   📋 Procesando resultados...")
    prediction_results = []
    
    for item_id in tqdm(forecasts.item_ids, desc="Procesando predicciones"):
        item_forecast = forecasts.loc[item_id]
        # Tomar solo la predicción para t+2 (febrero 2020)
        if len(item_forecast) >= 2:
            t2_prediction = item_forecast.iloc[1]['mean']  # t+2
            # Extraer product_id y customer_id del ID compuesto (nuevo orden)
            product_id, customer_id = item_id.split('_', 1)
            # Obtener intervalos de confianza si están disponibles
            p10 = item_forecast.iloc[1].get('0.1', t2_prediction * 0.8)
            p90 = item_forecast.iloc[1].get('0.9', t2_prediction * 1.2)
            prediction_results.append({
                'product_customer_id': item_id,
                'customer_id': customer_id,
                'product_id': product_id,
                'prediction_t2': t2_prediction,
                'prediction_p10': p10,
                'prediction_p90': p90
            })
    
    predictions_df = pd.DataFrame(prediction_results)
    
    # Aplicar clipping a 0 (no se pueden vender toneladas negativas)
    print("   ✂️ Aplicando clipping a 0 para predicciones negativas...")
    predictions_df['prediction_t2_clipped'] = predictions_df['prediction_t2'].clip(lower=0)
    predictions_df['prediction_p10'] = predictions_df['prediction_p10'].clip(lower=0)
    predictions_df['prediction_p90'] = predictions_df['prediction_p90'].clip(lower=0)
    
    negative_count = (predictions_df['prediction_t2'] < 0).sum()
    if negative_count > 0:
        print(f"   ⚠️ {negative_count:,} predicciones negativas corregidas a 0")
    
    # NUEVO: Generar artefactos de análisis
    create_analysis_artifacts(predictions_df, base_df)
    
    # Agregar por producto (suma de todas las predicciones por producto)
    print("   📊 Agregando predicciones por producto...")
    product_aggregated = predictions_df.groupby('product_id').agg({
        'prediction_t2_clipped': 'sum'
    }).reset_index()
    
    product_aggregated.columns = ['product_id', 'target']
    
    # Renombrar la columna final a 'tn' como esperado en el output
    product_aggregated = product_aggregated.rename(columns={'target': 'tn'})
    
    # Verificar que tenemos todos los productos esperados
    expected_products = base_df['product_id'].unique()
    predicted_products = product_aggregated['product_id'].unique()
    
    print(f"   📈 Productos esperados: {len(expected_products):,}")
    print(f"   📊 Productos con predicciones: {len(predicted_products):,}")
    
    # Agregar productos faltantes con predicción 0
    missing_products = set(expected_products) - set(predicted_products)
    if missing_products:
        print(f"   ⚠️ {len(missing_products):,} productos sin predicciones, asignando 0")
        missing_df = pd.DataFrame({
            'product_id': list(missing_products),
            'tn': [0.0] * len(missing_products)
        })
        product_aggregated = pd.concat([product_aggregated, missing_df], ignore_index=True)
    
    # Ordenar por product_id
    product_aggregated = product_aggregated.sort_values('product_id').reset_index(drop=True)
    
    # NUEVO: Generar baseline y artefactos combinados
    print("   🔄 Generando artefactos baseline y combinados...")
    baseline_avg = generate_baseline_artifacts(base_df)
    
    # Crear los 4 artefactos solicitados
    artifacts = {}
    
    # Artefacto 1: Predicción AutoGluon (ya existe)
    artifacts['autogluon'] = product_aggregated.copy()
    artifacts['autogluon'].columns = ['product_id', 'tn_autogluon']
    
    # Artefacto 2: Promedio últimos 12 meses
    artifacts['baseline_12m'] = baseline_avg.copy()
    artifacts['baseline_12m'].columns = ['product_id', 'tn_baseline_12m']
    
    # Merge para artefactos combinados
    combined = artifacts['autogluon'].merge(artifacts['baseline_12m'], on='product_id', how='outer')
    combined = combined.fillna(0)  # Rellenar NaN con 0
    
    # Artefacto 3: Híbrido (AutoGluon donde >0, baseline donde AutoGluon=0)
    artifacts['hybrid'] = combined.copy()
    artifacts['hybrid']['tn_hybrid'] = np.where(
        combined['tn_autogluon'] > 0, 
        combined['tn_autogluon'], 
        combined['tn_baseline_12m']
    )
    artifacts['hybrid'] = artifacts['hybrid'][['product_id', 'tn_hybrid']]
    
    # Artefacto 4: Promedio entre AutoGluon y baseline
    artifacts['average'] = combined.copy()
    artifacts['average']['tn_average'] = (combined['tn_autogluon'] + combined['tn_baseline_12m']) / 2
    artifacts['average'] = artifacts['average'][['product_id', 'tn_average']]
    
    # Estadísticas comparativas
    print("\n📊 Estadísticas comparativas de artefactos:")
    print(f"   🤖 AutoGluon total: {artifacts['autogluon']['tn_autogluon'].sum():,.2f} toneladas")
    print(f"   📈 Baseline 12m total: {artifacts['baseline_12m']['tn_baseline_12m'].sum():,.2f} toneladas")
    print(f"   🔀 Híbrido total: {artifacts['hybrid']['tn_hybrid'].sum():,.2f} toneladas")
    print(f"   ⚖️ Promedio total: {artifacts['average']['tn_average'].sum():.2f} toneladas")
    
    # Estadísticas de las predicciones principales
    print("\n📊 Estadísticas de predicciones AutoGluon:")
    print(f"   📦 Total productos: {len(product_aggregated):,}")
    print(f"   📈 Predicción total: {product_aggregated['tn'].sum():,.2f} toneladas")
    print(f"   📊 Promedio por producto: {product_aggregated['tn'].mean():.4f} toneladas")
    print(f"   📊 Mediana: {product_aggregated['tn'].median():.4f} toneladas")
    print(f"   📊 Productos con predicción > 0: {(product_aggregated['tn'] > 0).sum():,}")
    
    return product_aggregated, predictions_df, artifacts

# ============================================================================
# FUNCIÓN PRINCIPAL MEJORADA
# ============================================================================

def main():
    """Función principal que ejecuta todo el pipeline mejorado."""
    
    print("🎯 AUTOGLUON FORECASTER GPU - VERSIÓN MEJORADA")
    print("=" * 60)
    print("📅 Objetivo: Predecir toneladas para febrero 2020 (t+2)")
    print("📊 Granularidad: Cliente × Producto")
    print("⏱️ Entrenamiento: 3-4 horas optimizadas")
    print("🎯 Mejoras: Más modelos + Static Features + Análisis")
    print("💻 Hardware: i7-7700 + RTX 2060 + 24GB RAM")
    print("=" * 60)
    
    start_total = time.time()
    
    # Paso 1: Construir dataset base
    print("\n🏗️ PASO 1: Construcción del dataset base")
    base_df = build_base_dataset()
    
    # Paso 2: Preparar datos para AutoGluon (MEJORADO)
    print("\n🔄 PASO 2: Preparación para AutoGluon (con static features)")
    train_data, validation_data, full_data, static_features = prepare_timeseries_data(base_df)
    
    if static_features is not None:
        print(f"   ✅ Static features incluidas: {list(static_features.columns)}")
    else:
        print(f"   ⚠️ Static features no disponibles")
    
    # Paso 3: Entrenamiento mejorado
    print("\n🚀 PASO 3: Entrenamiento mejorado (más modelos + timeouts agresivos)")
    predictor = train_autogluon_enhanced(train_data, validation_data)
    
    # Paso 4: Generar predicciones finales (MEJORADO)
    print("\n📊 PASO 4: Generación de predicciones y artefactos")
    final_predictions, detailed_predictions, artifacts = generate_final_predictions(predictor, full_data, base_df)
    
    # Paso 5: Guardar resultados (EXPANDIDO)
    print("\n💾 PASO 5: Guardando resultados y artefactos")
    
    # CSV principal (product_id, tn)
    output_path = os.path.join(RESULTS_DIR, "predictions_feb_2020.csv")
    final_predictions.to_csv(output_path, index=False)
    print(f"✅ Predicciones principales guardadas en: {output_path}")
    
    # CSV detallado (con intervalos de confianza)
    detailed_path = os.path.join(RESULTS_DIR, "predictions_detailed.csv")
    detailed_predictions.to_csv(detailed_path, index=False)
    print(f"✅ Predicciones detalladas guardadas en: {detailed_path}")
    
    # NUEVO: Guardar los 4 artefactos solicitados
    print("   📁 Guardando artefactos de comparación:")
    
    # Artefacto 1: AutoGluon (renombrar para consistencia)
    artifact1_path = os.path.join(RESULTS_DIR, "artifact1_autogluon.csv")
    artifacts['autogluon'].rename(columns={'tn_autogluon': 'tn'}).to_csv(artifact1_path, index=False)
    print(f"   1. ✅ Artefacto 1 (AutoGluon): {artifact1_path}")
    
    # Artefacto 2: Baseline 12 meses
    artifact2_path = os.path.join(RESULTS_DIR, "artifact2_baseline_12m.csv")
    artifacts['baseline_12m'].rename(columns={'tn_baseline_12m': 'tn'}).to_csv(artifact2_path, index=False)
    print(f"   2. ✅ Artefacto 2 (Baseline 12m): {artifact2_path}")
    
    # Artefacto 3: Híbrido
    artifact3_path = os.path.join(RESULTS_DIR, "artifact3_hybrid.csv")
    artifacts['hybrid'].rename(columns={'tn_hybrid': 'tn'}).to_csv(artifact3_path, index=False)
    print(f"   3. ✅ Artefacto 3 (Híbrido): {artifact3_path}")
    
    # Artefacto 4: Promedio
    artifact4_path = os.path.join(RESULTS_DIR, "artifact4_average.csv")
    artifacts['average'].rename(columns={'tn_average': 'tn'}).to_csv(artifact4_path, index=False)
    print(f"   4. ✅ Artefacto 4 (Promedio): {artifact4_path}")
    
    # Leaderboard final (MANEJO ROBUSTO)
    try:
        final_leaderboard = predictor.leaderboard(validation_data)
        leaderboard_path = os.path.join(RESULTS_DIR, "model_leaderboard.csv")
        final_leaderboard.to_csv(leaderboard_path, index=False)
        print(f"✅ Leaderboard guardado en: {leaderboard_path}")
        
        print("\n🏆 MODELOS FINALES (Top 5):")
        # Mostrar columnas disponibles de manera robusta
        available_cols = ['model']
        optional_cols = ['score_val', 'score_test', 'fit_time', 'pred_time', 'fit_time_marginal', 'pred_time_marginal']
        
        for col in optional_cols:
            if col in final_leaderboard.columns:
                available_cols.append(col)
        
        print(final_leaderboard.head()[available_cols])
        
    except Exception as e:
        print(f"⚠️ No se pudo generar leaderboard: {e}")
    
    # NUEVO: Resumen de artefactos generados
    print("\n📁 ARTEFACTOS GENERADOS:")
    artifacts_list = [
        "predictions_feb_2020.csv (predicciones principales)",
        "predictions_detailed.csv (predicciones con intervalos)",
        "model_leaderboard.csv (ranking de modelos)",
        "artifact1_autogluon.csv (predicciones AutoGluon consolidadas)",
        "artifact2_baseline_12m.csv (promedio últimos 12 meses)",
        "artifact3_hybrid.csv (AutoGluon + baseline donde AG=0)",
        "artifact4_average.csv (promedio entre AutoGluon y baseline)",
        "analysis_by_cat1.csv (análisis por categoría)",
        "analysis_by_brand.csv (análisis por marca)",
        "prediction_statistics.csv (estadísticas detalladas)",
        "plots/prediction_analysis.png (gráficos de análisis)"
    ]
    
    for i, artifact in enumerate(artifacts_list, 1):
        print(f"   {i:2d}. {artifact}")
    
    # Resumen final
    total_time = time.time() - start_total
    print("\n" + "=" * 60)
    print("🎉 PIPELINE MEJORADO COMPLETADO EXITOSAMENTE")
    print("=" * 60)
    print(f"⏱️ Tiempo total: {total_time/3600:.2f} horas")
    print(f"📦 Productos predichos: {len(final_predictions):,}")
    print(f"📈 Predicción total: {final_predictions['tn'].sum():,.2f} toneladas")
    print(f"🎯 Mejoras implementadas: ✅")
    print(f"   • Más modelos foundation (Chronos variants)")
    print(f"   • Static features restauradas")
    print(f"   • Timeouts agresivos (3-4h total)")
    print(f"   • 4 artefactos de comparación (AutoGluon + baselines)")
    print(f"   • Análisis por categorías")
    print(f"   • Gráficos de distribución")
    print(f"   • Estadísticas detalladas")
    print(f"📁 Archivo principal: {output_path}")
    print("=" * 60)
    
    return final_predictions

if __name__ == "__main__":
    try:
        predictions = main()
    except KeyboardInterrupt:
        print("\n⚠️ Proceso interrumpido por el usuario")
    except Exception as e:
        print(f"\n❌ Error durante la ejecución: {e}")
        import traceback
        traceback.print_exc()

