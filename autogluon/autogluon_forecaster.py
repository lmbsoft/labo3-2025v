#!/usr/bin/env python3
"""
AutoGluon Forecaster - Predicción de Ventas Multiproducto
=========================================================

Script completo optimizado para forecasting de toneladas (tn) con horizonte t+2.
VERSIÓN REFACTORIZADA CON VALIDACIÓN AUTOMÁTICA (BACKTESTING).

Flujo Simplificado:
1. Descarga archivos automáticamente si no existen.
2. Construye dataset con lógica de vida útil + padding temporal.
3. Entrena y valida con backtesting automático (num_val_windows=3) 
   usando todos los datos históricos (2017-01 a 2019-12).
4. Predice 2020-02 (t+2).
5. Genera 4 CSV finales para experimentación en Kaggle.

Optimizado para 48 cores y 512GB RAM.
"""

import os
import sys
import time
import gzip
import shutil
import warnings
import urllib.request
from datetime import datetime
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# Suprimir warnings para output limpio
warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None

print("🚀 Iniciando AutoGluon Forecaster (Versión con Backtesting)...")
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

# Configuración temporal simplificada
DATES_CONFIG = {
    "data_start": "2017-01-01",
    "historical_data_end": "2019-12-01", # Fin de los datos para entrenamiento/validación
    "prediction_target": "2020-02-01"    # Predecir feb 2020 (t+2)
}

# Directorios
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
RESULTS_DIR = "data/results"

for dir_path in [RAW_DIR, PROCESSED_DIR, RESULTS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# ============================================================================
# UTILIDADES DE DESCARGA Y CARGA (Sin cambios)
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
    print()
    
    if is_gzip and local_path.endswith('.gz'):
        print(f"📂 Descomprimiendo {os.path.basename(local_path)}...")
        output_path = local_path[:-3]
        with gzip.open(local_path, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        print(f"✅ Descomprimido a {os.path.basename(output_path)}")

def load_data_files() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carga todos los archivos necesarios, descargándolos si es necesario."""
    print("📁 Cargando archivos de datos...")
    
    for name, config in FILE_CONFIG.items():
        local_path = os.path.join(RAW_DIR, config["local"])
        
        if config["is_gzip"]:
            decompressed_path = local_path[:-3]
            if not os.path.exists(local_path) and not os.path.exists(decompressed_path):
                download_file(config["url"], local_path, config["is_gzip"])
        else:
            if not os.path.exists(local_path):
                download_file(config["url"], local_path, config["is_gzip"])
    
    print("📊 Cargando datos en memoria...")
    
    sell_in_path = os.path.join(RAW_DIR, "sell-in.txt")
    if not os.path.exists(sell_in_path):
        sell_in_path = os.path.join(RAW_DIR, "sell-in.txt.gz")
    
    sell_in = pd.read_csv(sell_in_path, sep='\t', dtype={'customer_id': str, 'product_id': str})
    print(f"   ✅ Sell-in: {len(sell_in):,} registros")
    
    productos = pd.read_csv(os.path.join(RAW_DIR, "tb_productos.txt"), sep='\t', dtype={'product_id': str})
    print(f"   ✅ Productos: {len(productos):,} registros")
    
    stocks = pd.read_csv(os.path.join(RAW_DIR, "tb_stocks.txt"), sep='\t', dtype={'product_id': str})
    print(f"   ✅ Stocks: {len(stocks):,} registros")
    
    productos_pred_path = os.path.join(RAW_DIR, "product_id_apredecir201912.txt")
    productos_pred = pd.read_csv(productos_pred_path, dtype={'product_id': str})
    
    if 'product_id' not in productos_pred.columns:
        productos_pred.columns = ['product_id']
    
    print(f"   ✅ Productos a predecir: {len(productos_pred):,} productos")
    
    return sell_in, productos, stocks, productos_pred

# ============================================================================
# PREPROCESSING (Sin cambios)
# ============================================================================

def complete_temporal_grid(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    """Completa el grid temporal usando lógica de vida útil + padding fijo."""
    print("🔄 Aplicando lógica de vida útil + padding temporal...")
    
    df = df.copy()
    df["periodo"] = pd.to_datetime(df["periodo"], format="%Y%m")
    
    print("   📅 Paso 1: Calculando períodos de coexistencia...")
    vida_clientes = df.groupby('customer_id')['periodo'].agg(['min', 'max']).rename(columns={'min': 'cliente_ini', 'max': 'cliente_fin'})
    vida_productos = df.groupby('product_id')['periodo'].agg(['min', 'max']).rename(columns={'min': 'producto_ini', 'max': 'producto_fin'})
    
    clientes_df = vida_clientes.reset_index()
    productos_df = vida_productos.reset_index()
    cp_df = clientes_df.assign(key=1).merge(productos_df.assign(key=1), on='key').drop('key', axis=1)
    
    cp_df['inicio_actividad'] = cp_df[['cliente_ini', 'producto_ini']].apply(lambda x: max(x), axis=1)
    cp_df['fin_actividad'] = cp_df[['cliente_fin', 'producto_fin']].apply(lambda x: min(x), axis=1)
    
    valid_pairs = cp_df[cp_df['inicio_actividad'] <= cp_df['fin_actividad']].reset_index(drop=True)
    
    n_months = ((valid_pairs["fin_actividad"].dt.year - valid_pairs["inicio_actividad"].dt.year) * 12 + (valid_pairs["fin_actividad"].dt.month - valid_pairs["inicio_actividad"].dt.month) + 1).astype("int16")
    rep_idx = valid_pairs.index.repeat(n_months)
    base = valid_pairs.loc[rep_idx].reset_index(drop=True)
    offsets = np.concatenate([np.arange(k, dtype="int16") for k in n_months])
    base["periodo"] = (base["inicio_actividad"].dt.to_period("M") + offsets).astype("datetime64[ns]")
    step1_df = base[["customer_id", "product_id", "periodo"]].merge(df, on=["product_id", "customer_id", "periodo"], how="left")
    
    print("   📅 Paso 2: Extendiendo a ventana temporal completa...")
    all_months = pd.date_range(start=start_date, end=end_date, freq="MS")
    n_periods = len(all_months)
    pairs = step1_df[["customer_id", "product_id"]].drop_duplicates().reset_index(drop=True)
    N = len(pairs)
    
    base_fixed = pairs.loc[pairs.index.repeat(n_periods)].reset_index(drop=True)
    base_fixed["periodo"] = np.tile(all_months, N)
    
    full_df = base_fixed.merge(df, on=["product_id", "customer_id", "periodo"], how="left")
    
    cols_to_fill = ["tn", "plan_precios_cuidados", "cust_request_qty", "cust_request_tn"]
    for col in cols_to_fill:
        if col in full_df.columns:
            full_df[col] = full_df[col].fillna(0)
            
    full_df["product_id"] = full_df["product_id"].astype(str)
    full_df["customer_id"] = full_df["customer_id"].astype(str)
    
    print(f"   ✅ Grid temporal completo: {len(full_df):,} registros")
    return full_df.sort_values(["product_id", "customer_id", "periodo"]).reset_index(drop=True)

def build_base_dataset() -> pd.DataFrame:
    """Construye el dataset base con toda la información consolidada."""
    base_path = os.path.join(PROCESSED_DIR, "base_dataset.parquet")
    if os.path.exists(base_path):
        print("📁 Dataset base ya existe, cargando desde cache...")
        return pd.read_parquet(base_path)
    
    print("🏗️ Construyendo dataset base...")
    sell_in, productos, stocks, productos_pred = load_data_files()
    
    print("🎯 Filtrando solo productos a predecir...")
    sell_in_filtered = sell_in.merge(productos_pred, on='product_id', how='inner')
    
    full_sell = complete_temporal_grid(
        sell_in_filtered,
        DATES_CONFIG["data_start"],
        DATES_CONFIG["historical_data_end"]
    )
    
    print("🔗 Agregando información de productos y stocks...")
    productos["product_id"] = productos["product_id"].astype(str)
    df = full_sell.merge(productos, on="product_id", how="left")
    
    stocks["product_id"] = stocks["product_id"].astype(str)
    stocks["periodo"] = pd.to_datetime(stocks["periodo"], format="%Y%m")
    df = df.merge(stocks[["periodo", "product_id", "stock_final"]], on=["periodo", "product_id"], how="left")
    df["stock_final"] = df["stock_final"].fillna(0)
    
    print(f"💾 Guardando dataset base en {base_path}...")
    df.to_parquet(base_path, index=False)
    print(f"✅ Dataset base guardado: {len(df):,} registros")
    return df

# ============================================================================
# PREPARACIÓN PARA AUTOGLUON (Simplificado)
# ============================================================================

def prepare_timeseries_data(df: pd.DataFrame) -> Tuple[TimeSeriesDataFrame, pd.DataFrame]:
    """
    Prepara los datos para AutoGluon.
    Devuelve un único TimeSeriesDataFrame con toda la historia.
    """
    print("🔄 Preparando datos para AutoGluon TimeSeries...")
    
    df['item_id'] = df['product_id'].astype(str) + '_' + df['customer_id'].astype(str)
    df = df.rename(columns={'tn': 'target'})
    
    print(f"   ✅ Series válidas (todas): {df['item_id'].nunique():,}")
    print(f"   📊 Registros totales: {len(df):,}")
    
    print("   📋 Creando static features...")
    static_features = None
    try:
        static_cols = ['customer_id', 'product_id', 'cat1', 'cat2', 'cat3', 'brand', 'sku_size']
        available_cols = [col for col in static_cols if col in df.columns]
        
        if len(available_cols) >= 3:
            static_features = df[['item_id'] + available_cols].drop_duplicates(subset=['item_id']).reset_index(drop=True)
            
            nan_counts = static_features.isnull().sum()
            if nan_counts.sum() > 0:
                for col in static_features.columns:
                    if static_features[col].dtype == 'object':
                        static_features[col] = static_features[col].fillna('UNKNOWN')
                    else:
                        static_features[col] = static_features[col].fillna(0)
            static_features = static_features.set_index('item_id')
    except Exception as e:
        print(f"   ⚠️ Error creando static features: {e}. Continuando sin ellas.")
        static_features = None

    print("   🔄 Convirtiendo a TimeSeriesDataFrame...")
    ts_params = {'id_column': 'item_id', 'timestamp_column': 'periodo'}
    if static_features is not None:
        ts_params['static_features_df'] = static_features
        print(f"   ✅ Static features incluidas: {list(static_features.columns)}")

    try:
        historical_data = TimeSeriesDataFrame.from_data_frame(df=df, **ts_params)
    except Exception as e:
        print(f"   ⚠️ Error con static features: {e}. Creando sin ellas...")
        ts_params.pop('static_features_df', None)
        historical_data = TimeSeriesDataFrame.from_data_frame(df=df, **ts_params)

    print(f"   ✅ TimeSeriesDataFrame creado con {len(historical_data.item_ids)} series temporales.")
    return historical_data, static_features

# ============================================================================
# ENTRENAMIENTO CON BACKTESTING AUTOMÁTICO (Refactorizado)
# ============================================================================

def train_autogluon_with_backtesting(historical_data: TimeSeriesDataFrame) -> TimeSeriesPredictor:
    """
    Entrenamiento simplificado con backtesting automático y re-entrenamiento implícito.
    """
    print("🚀 Iniciando entrenamiento con backtesting automático...")
    print("⏱️ Tiempo límite: 24 horas")
    print("🔢 Ventanas de validación (backtesting): 3")
    print("=" * 60)
    
    predictor = TimeSeriesPredictor(
        prediction_length=2,
        path=os.path.join(RESULTS_DIR, "autogluon_models_backtesting"),
        target='target',
        eval_metric='MASE',
        quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        cache_predictions=True,
        verbosity=2
    )
    
    hyperparameters = {
        'Chronos': {'device': 'cpu'},
        'RecursiveTabular': {'time_limit': 14400}, # 4 horas
        'DirectTabular': {'time_limit': 14400},   # 4 horas
        'AutoETS': {}, 'AutoARIMA': {}, 'ETS': {}, 'ARIMA': {},
        'Theta': {}, 'DynamicOptimizedTheta': {},
        'SeasonalNaive': {}, 'Naive': {}, 'SeasonalAverage': {}, 'Average': {}
    }
    
    print("🎯 Modelos a entrenar (con tiempo extendido):")
    for i, (model_name, params) in enumerate(hyperparameters.items(), 1):
        print(f"   {i:2d}. {model_name} {params if params else ''}")
    
    print("\n🏁 Iniciando .fit() unificado...")
    start_time = time.time()
    
    predictor.fit(
        historical_data,
        hyperparameters=hyperparameters,
        time_limit=86400,  # 24 horas en total
        num_val_windows=3, # 3 ventanas para validación robusta
        enable_ensemble=True,
        # refit_full=True es el comportamiento por defecto después de la validación
    )
    
    total_time = time.time() - start_time
    print(f"\n✅ Entrenamiento completado en {total_time/3600:.2f} horas")
    
    print("\n🏆 LEADERBOARD FINAL (basado en backtesting):")
    print(predictor.leaderboard())
    
    return predictor

# ============================================================================
# GENERACIÓN DE ARTEFACTOS (Sin cambios)
# ============================================================================

def generate_baseline_artifacts(base_df: pd.DataFrame) -> pd.DataFrame:
    """Genera el artefacto baseline (promedio últimos 12 meses)."""
    print("📊 Generando artefacto baseline (promedio 12 meses)...")

    start_date = pd.to_datetime("2019-01-01")
    end_date = pd.to_datetime("2019-12-01")

    baseline_period = base_df[
        (base_df['periodo'] >= start_date) &
        (base_df['periodo'] <= end_date)
    ].copy()

    # ✅ CORRECCIÓN: Usar 'tn' en lugar de 'target'
    monthly_totals = baseline_period.groupby(['product_id', 'periodo'])['tn'].sum().reset_index()
    baseline_avg = monthly_totals.groupby('product_id')['tn'].mean().reset_index()
    baseline_avg.columns = ['product_id', 'avg_12m']
    baseline_avg['avg_12m'] = baseline_avg['avg_12m'].clip(lower=0)

    expected_products = base_df['product_id'].unique()
    missing_products = set(expected_products) - set(baseline_avg['product_id'].unique())
    if missing_products:
        missing_df = pd.DataFrame({'product_id': list(missing_products), 'avg_12m': 0.0})
        baseline_avg = pd.concat([baseline_avg, missing_df], ignore_index=True)

    return baseline_avg.sort_values('product_id').reset_index(drop=True)


def create_analysis_artifacts(predictions_df: pd.DataFrame, base_df: pd.DataFrame):
    """Genera artefactos adicionales de análisis (gráficos y CSVs)."""
    print("📊 Generando artefactos de análisis...")
    PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
    os.makedirs(PLOTS_DIR, exist_ok=True)

    try:
        # Renombramos 'target' a 'tn' en base_df para el merge y baseline
        base_df_renamed = base_df.rename(columns={'target': 'tn'})
        analysis_df = predictions_df.merge(
            base_df_renamed[['product_id', 'cat1', 'cat2', 'cat3', 'brand']].drop_duplicates(),
            on='product_id', how='left'
        )
        
        for col in ['cat1', 'brand']:
            if col in analysis_df.columns:
                agg_df = analysis_df.groupby(col).agg(
                    total_pred=('prediction_t2_clipped', 'sum'),
                    avg_pred=('prediction_t2_clipped', 'mean'),
                    n_customers=('prediction_t2_clipped', 'count')
                ).round(4)
                agg_df.to_csv(os.path.join(RESULTS_DIR, f"analysis_by_{col}.csv"))
        
        # ... (código de gráficos omitido por brevedad) ...
        print(f"   ✅ Artefactos de análisis (CSV) guardados en {RESULTS_DIR}")

    except Exception as e:
        print(f"   ⚠️ Error generando artefactos de análisis: {e}")

def generate_final_predictions(predictor: TimeSeriesPredictor, 
                             historical_data: TimeSeriesDataFrame,
                             base_df: pd.DataFrame):
    """Genera predicciones finales y crea múltiples artefactos de salida."""
    print("📊 Generando predicciones finales...")
    # Usamos historical_data para que el predictor tenga el contexto completo
    forecasts = predictor.predict(historical_data)
    
    print("   📋 Procesando resultados...")
    prediction_results = []
    for item_id in tqdm(forecasts.item_ids, desc="Procesando predicciones"):
        item_forecast = forecasts.loc[item_id]
        if len(item_forecast) >= 2:
            product_id, customer_id = item_id.split('_', 1)
            prediction_results.append({
                'item_id': item_id,
                'customer_id': customer_id,
                'product_id': product_id,
                'prediction_t2': item_forecast.iloc[1]['mean'],
                'prediction_p10': item_forecast.iloc[1].get('0.1', 0),
                'prediction_p90': item_forecast.iloc[1].get('0.9', 0)
            })
    
    predictions_df = pd.DataFrame(prediction_results)
    predictions_df['prediction_t2_clipped'] = predictions_df['prediction_t2'].clip(lower=0)
    
    create_analysis_artifacts(predictions_df, base_df)
    
    autogluon_agg = predictions_df.groupby('product_id')['prediction_t2_clipped'].sum().reset_index()
    autogluon_agg.columns = ['product_id', 'tn']
    
    baseline_df = generate_baseline_artifacts(base_df)
    baseline_df.columns = ['product_id', 'tn']

    combined = autogluon_agg.merge(baseline_df, on='product_id', how='outer', suffixes=['_ag', '_base']).fillna(0)
    
    hybrid_df = combined.copy()
    hybrid_df['tn'] = np.where(combined['tn_ag'] > 0, combined['tn_ag'], combined['tn_base'])
    
    average_df = combined.copy()
    average_df['tn'] = (combined['tn_ag'] + combined['tn_base']) / 2

    artifacts_to_save = {
        "prediccion_autogluon.csv": autogluon_agg,
        "prediccion_baseline_12m.csv": baseline_df,
        "prediccion_hibrida.csv": hybrid_df[['product_id', 'tn']],
        "prediccion_promedio.csv": average_df[['product_id', 'tn']]
    }
    
    print("\n💾 Guardando artefactos finales...")
    for filename, df_to_save in artifacts_to_save.items():
        path = os.path.join(RESULTS_DIR, filename)
        df_to_save.sort_values('product_id').to_csv(path, index=False)
        print(f"   ✅ Guardado: {path}")

# ============================================================================
# FUNCIÓN PRINCIPAL (Refactorizada)
# ============================================================================

def main():
    """Función principal que ejecuta el pipeline simplificado."""
    
    print("🎯 AUTOGLUON FORECASTER - PIPELINE CON BACKTESTING")
    print("=" * 60)
    print("📅 Objetivo: Predecir toneladas para febrero 2020 (t+2)")
    print("⚡ Validación: Backtesting automático con 3 ventanas")
    print("⏱️ Entrenamiento: Límite de 24 horas")
    print("=" * 60)
    
    start_total = time.time()
    
    # Paso 1: Construir dataset base
    print("\n🏗️ PASO 1: Construcción del dataset base")
    base_df = build_base_dataset()
    
    # Paso 2: Preparar datos para AutoGluon
    print("\n🔄 PASO 2: Preparación para AutoGluon")
    historical_data, _ = prepare_timeseries_data(base_df)
    
    # Paso 3: Entrenamiento unificado con backtesting
    print("\n📚 PASO 3: Entrenamiento y validación con Backtesting")
    predictor = train_autogluon_with_backtesting(historical_data)
    
    # Paso 4: Generar predicciones y artefactos
    print("\n📊 PASO 4: Generación de predicciones y artefactos")
    generate_final_predictions(predictor, historical_data, base_df)
    
    end_total = time.time()
    
    print("\n" + "="*60)
    print(f"🎉 PIPELINE COMPLETADO EN {(end_total - start_total) / 3600:.2f} HORAS 🎉")
    print("="*60)


if __name__ == "__main__":
    try:
        main()
    except ImportError as e:
        print(f"❌ Error de importación: {e}. Asegúrate de tener las librerías instaladas.")
        print("   Intenta ejecutar: pip install pandas 'autogluon.timeseries' torch matplotlib")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error inesperado en la ejecución: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)