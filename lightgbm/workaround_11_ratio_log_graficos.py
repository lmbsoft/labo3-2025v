# -*- coding: utf-8 -*-
"""
Generador de Visualizaciones de Series Temporales y Forecasts
=============================================================

Este script está diseñado para generar visualizaciones detalladas de las series
temporales de ventas y las predicciones generadas por los modelos.

Funcionalidades principales:
- Carga los datos históricos y los archivos de predicción (buscando el mejor
  ensamble disponible).
- Para cada producto a predecir, genera un gráfico que muestra:
  - La serie temporal histórica de ventas.
  - La predicción para el período objetivo (Febrero 2020).
  - Estadísticas clave y detalles del método de modelado.
- Compila todos los gráficos individuales en un único archivo PDF para una
  revisión consolidada.
- Genera un archivo de texto con un resumen del proceso de visualización.

Parámetros Principales de Configuración (dentro de la clase Config):
--------------------------------------------------------------------
- GRANULARITY: 'product' o 'customer'. Define el nivel de agregación.
- FEATURE_VERSION: Versión de las features, para alinear con los artefactos
  de predicción correctos.
- GRAFICOS_PATH: Directorio donde se guardarán los gráficos y el PDF.

"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import logging
import warnings
from datetime import datetime
import glob
from typing import List, Optional

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')

class Config:
    GRANULARITY     = 'product'       # 'product' | 'customer'
    FEATURE_VERSION = 'v6_log_ratio_advanced'  # Alineado con workaround_11
    CACHE_PATH      = 'cache_log_ratio_advanced'
    ARTIFACTS_PATH  = 'artifacts_log_ratio_advanced'
    GRAFICOS_PATH  = os.path.join('artifacts_log_ratio_advanced', 'graficos')
    
    # URLs para descargar datos
    SELLIN_URL = 'https://storage.googleapis.com/open-courses/austral2025-af91/labo3v/sell-in.txt.gz'
    PRODUCTOS_URL = 'https://storage.googleapis.com/open-courses/austral2025-af91/labo3v/tb_productos.txt'
    PRODUCTOS_A_PREDECIR_URL = 'https://storage.googleapis.com/open-courses/austral2025-af91/labo3v/product_id_apredecir201912.txt'
    
    # Configuración de gráficos
    FIGURE_SIZE = (15, 8)
    DPI = 150
    DATE_FORMAT = '%Y-%m'
    
def get_logger():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger()

logger = get_logger()

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

def load_historical_data(cfg, birth_dates):
    """Carga datos históricos con filtro de fechas de nacimiento."""
    logger.info("Cargando datos históricos...")
    
    # Cargar datos principales
    sellin = pd.read_csv(
        cfg.SELLIN_URL,
        sep="\t", 
        compression='gzip',
        dtype={'periodo': str, 'customer_id': str, 'product_id': str}
    )
    sellin['periodo'] = pd.to_datetime(sellin['periodo'], format='%Y%m')
    
    productos = pd.read_csv(
        cfg.PRODUCTOS_URL,
        sep="\t", 
        dtype={'product_id': str}
    )
    
    # Agregar por granularidad
    gcols = ['periodo', 'product_id']
    if cfg.GRANULARITY == 'customer':
        gcols.append('customer_id')
    
    df = sellin.groupby(gcols)['tn'].sum().reset_index()
    
    # Crear grid completo
    periods = pd.date_range(sellin['periodo'].min(), sellin['periodo'].max(), freq='MS')
    products = sellin['product_id'].unique()
    
    if cfg.GRANULARITY == 'customer':
        customers = sellin['customer_id'].unique()
        grid = pd.MultiIndex.from_product([periods, products, customers],
                                          names=['periodo','product_id','customer_id'])
    else:
        grid = pd.MultiIndex.from_product([periods, products],
                                          names=['periodo','product_id'])
    
    df = df.set_index(gcols).reindex(grid, fill_value=0).reset_index()
    
    # Merge con productos para tener descripción
    df = df.merge(productos, on='product_id', how='left')
    
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
    
    # Eliminar columnas auxiliares
    cols_to_drop = ['birth_date_product']
    if cfg.GRANULARITY == 'customer':
        cols_to_drop.append('birth_date_customer')
    df = df.drop(columns=cols_to_drop)
    
    return df

def load_productos_a_predecir(cfg):
    """Carga la lista de productos a predecir."""
    logger.info("Cargando lista de productos a predecir...")
    productos_a_predecir = pd.read_csv(
        cfg.PRODUCTOS_A_PREDECIR_URL,
        sep='\t', 
        dtype={'product_id': str}
    )
    return productos_a_predecir

def find_best_ensemble_prediction(cfg):
    """Encuentra el mejor archivo de predicción (ensemble prioritario, luego individual)."""
    # 1. Buscar ensemble prioritario (semillerío)
    ensemble_pattern = os.path.join(
        cfg.ARTIFACTS_PATH, 
        f'{cfg.GRANULARITY}_{cfg.FEATURE_VERSION}_semillerio_*.csv'
    )
    ensemble_files = glob.glob(ensemble_pattern)
    
    if ensemble_files:
        # Tomar el ensemble con más modelos (número más alto en el nombre)
        ensemble_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        best_file = ensemble_files[-1]
        logger.info(f"Usando predicción ensemble: {os.path.basename(best_file)}")
        return best_file
    
    # 2. Si no hay ensemble, buscar submission individual
    single_pattern = os.path.join(
        cfg.ARTIFACTS_PATH, 
        f'submission_202002_{cfg.GRANULARITY}_{cfg.FEATURE_VERSION}.csv'
    )
    if os.path.exists(single_pattern):
        logger.info(f"Usando predicción individual: {os.path.basename(single_pattern)}")
        return single_pattern
    
    # 3. Buscar cualquier submission con features avanzadas
    fallback_pattern = os.path.join(
        cfg.ARTIFACTS_PATH, 
        f'submission_202002_{cfg.GRANULARITY}_*advanced*.csv'
    )
    fallback_files = glob.glob(fallback_pattern)
    if fallback_files:
        best_file = fallback_files[0]  # Tomar el primero
        logger.info(f"📂 Usando predicción fallback: {os.path.basename(best_file)}")
        return best_file
    
    logger.error("No se encontró ningún archivo de predicción")
    logger.error("Ejecuta primero workaround_11_ratio_log.py o workaround_11_semillerio_ratio_log.py")
    return None

def prepare_time_series_data(df, cfg):
    """Prepara los datos de series temporales por producto."""
    logger.info("Preparando datos de series temporales...")
    
    if cfg.GRANULARITY == 'customer':
        # Agregar por producto
        ts_data = df.groupby(['periodo', 'product_id'])['tn'].sum().reset_index()
    else:
        ts_data = df[['periodo', 'product_id', 'tn']].copy()
    
    # También incluir descripción del producto
    productos_info = df[['product_id', 'descripcion']].drop_duplicates()
    ts_data = ts_data.merge(productos_info, on='product_id', how='left')
    
    return ts_data

def create_product_chart(product_data, prediction_value, product_info, cfg):
    """Crea un gráfico mejorado para un producto específico y devuelve la figura."""
    
    fig, ax = plt.subplots(figsize=cfg.FIGURE_SIZE, dpi=cfg.DPI)
    
    # Datos históricos
    x_dates = product_data['periodo']
    y_values = product_data['tn']
    
    # Línea principal de la serie temporal con estilo mejorado
    ax.plot(x_dates, y_values, linewidth=2.8, color='steelblue', alpha=0.9, 
            label='Ventas Históricas', marker='o', markersize=3, markerfacecolor='navy', markeredgewidth=0)
    
    # Área bajo la curva para mejor visualización
    ax.fill_between(x_dates, y_values, alpha=0.25, color='lightblue')

    # Etiqueta para el último valor de la serie
    if not y_values.empty:
        last_date = x_dates.iloc[-1]
        last_value = y_values.iloc[-1]
        ax.text(last_date, last_value, f'{last_value:.3g}', 
                ha='center', va='bottom', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.7, edgecolor='orange'))
    
    # Punto de predicción con estilo mejorado
    pred_date = pd.to_datetime('2020-02-01')
    ax.scatter([pred_date], [prediction_value], color='red', s=150, zorder=5, 
               label=f'Predicción Feb 2020: {prediction_value:.2f} tn',
               marker='D', edgecolors='darkred', linewidth=2)

    # Etiqueta para el valor de la predicción con mejor posicionamiento
    offset = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.03
    ax.text(pred_date, prediction_value + offset, f'{prediction_value:.3g}', 
            ha='center', va='bottom', fontsize=10, fontweight='bold', color='darkred',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='red', alpha=0.8))
    
    # Línea vertical indicando la predicción
    ax.axvline(x=pred_date, color='red', linestyle='--', alpha=0.6, linewidth=2)
    
    # Configuración de ejes mejorada
    ax.set_xlabel('Período', fontsize=13, fontweight='bold')
    ax.set_ylabel('Toneladas Vendidas', fontsize=13, fontweight='bold')
    
    # Título con información del producto
    product_id = product_info['product_id']
    descripcion = product_info['descripcion']
    
    # Truncar descripción si es muy larga
    if len(descripcion) > 55:
        descripcion = descripcion[:52] + "..."
    
    title = f'Producto: {product_id} | Features Avanzadas v11\n{descripcion}'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Formato de fechas en eje X mejorado
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_minor_locator(mdates.MonthLocator((1, 4, 7, 10)))  # Cada trimestre
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_minor_formatter(mdates.DateFormatter('%m'))
    
    # Rotar etiquetas para mejor legibilidad
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha='center', fontweight='bold')
    
    # Grid mejorado
    ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Leyenda mejorada
    ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True, 
              fontsize=11, framealpha=0.9)
    
    # Estadísticas en el gráfico con mejor formato
    max_val = y_values.max()
    min_val = y_values.min()
    mean_val = y_values.mean()
    std_val = y_values.std()
    
    stats_text = (f'Estadísticas Históricas:\n'
                  f'Máx: {max_val:.2f} tn\n'
                  f'Min: {min_val:.2f} tn\n'
                  f'Promedio: {mean_val:.2f} tn\n'
                  f'Std Dev: {std_val:.2f} tn')
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', 
            bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.85, edgecolor='brown'))
    
    # Información adicional sobre el método
    method_text = ('Método: Log-ratio + Features Avanzadas\n'
                   'Cache: Nivel 3 (_ii)\n'
                   'Categorías: 3 niveles + Brand\n'
                   'Estacionalidad: Fourier + Eventos')
    
    ax.text(0.98, 0.02, method_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightcyan', alpha=0.8, edgecolor='teal'))
    
    # Ajustar layout
    plt.tight_layout()
    
    return fig

def generate_visualizations(ts_data, predictions, productos_a_predecir, cfg):
    """Genera todas las visualizaciones y el PDF con mejoras."""
    
    logger.info("Generando visualizaciones avanzadas...")
    
    # Crear directorio de gráficos
    os.makedirs(cfg.GRAFICOS_PATH, exist_ok=True)
    
    # Lista para almacenar rutas de imágenes
    image_paths = []
    productos_procesados = 0
    productos_sin_datos = 0
    productos_sin_prediccion = 0
    
    # PDF para compilar todas las imágenes
    pdf_path = os.path.join(cfg.GRAFICOS_PATH, f'series_temporales_{cfg.GRANULARITY}_advanced.pdf')
    
    with PdfPages(pdf_path) as pdf:
        for i, row in enumerate(productos_a_predecir.iterrows()):
            _, row_data = row
            product_id = row_data['product_id']
            
            # Obtener datos históricos del producto
            product_data = ts_data[ts_data['product_id'] == product_id].copy()
            
            if product_data.empty:
                logger.debug(f"No hay datos históricos para producto {product_id}")
                productos_sin_datos += 1
                continue
            
            # Obtener predicción
            pred_row = predictions[predictions['product_id'] == product_id]
            if pred_row.empty:
                logger.debug(f"No hay predicción para producto {product_id}")
                productos_sin_prediccion += 1
                continue
            
            prediction_value = pred_row['tn'].iloc[0]
            
            # Información del producto
            product_info = {
                'product_id': product_id,
                'descripcion': product_data['descripcion'].iloc[0] if not product_data['descripcion'].isna().all() 
                              else f'Producto {product_id}'
            }
            
            # Ruta para guardar la imagen
            image_filename = f'producto_{product_id}_advanced.png'
            image_path = os.path.join(cfg.GRAFICOS_PATH, image_filename)
            
            # Crear gráfico mejorado (la función ahora devuelve la figura)
            fig = create_product_chart(product_data, prediction_value, product_info, cfg)
            
            # Guardar la figura en formato PNG
            fig.savefig(image_path, dpi=cfg.DPI, bbox_inches='tight', facecolor='white',
                       edgecolor='none', pad_inches=0.1)
            
            # Guardar la misma figura en el PDF
            pdf.savefig(fig, bbox_inches='tight', facecolor='white', edgecolor='none')
            
            # Cerrar la figura para liberar memoria
            plt.close(fig)
            
            image_paths.append(image_path)
            productos_procesados += 1
            
            if (i + 1) % 100 == 0:
                logger.info(f"Procesados {productos_procesados} de {len(productos_a_predecir)} productos...")
    
    logger.info(f"Visualizaciones completadas:")
    logger.info(f"  Productos procesados: {productos_procesados}")
    logger.info(f"  Productos sin datos: {productos_sin_datos}")
    logger.info(f"  Productos sin predicción: {productos_sin_prediccion}")
    logger.info(f"  Imágenes generadas: {len(image_paths)}")
    logger.info(f"  PDF compilado: {pdf_path}")
    
    # Generar archivo de resumen mejorado
    summary_path = os.path.join(cfg.GRAFICOS_PATH, 'resumen_visualizaciones_advanced.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Resumen Visualizaciones - Features Avanzadas v11\n")
        f.write("="*70 + "\n")
        f.write(f"Granularidad: {cfg.GRANULARITY}\n")
        f.write(f"Feature Version: {cfg.FEATURE_VERSION}\n")
        f.write(f"Cache Level: Nivel 3 (_ii) - Features Avanzadas\n")
        f.write(f"Estrategia: y = log1p(tn_future) - log1p(tn_current) + Advanced Features\n")
        f.write(f"Fecha generación: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total productos a predecir: {len(productos_a_predecir)}\n")
        f.write(f"Productos procesados: {productos_procesados}\n")
        f.write(f"Productos sin datos históricos: {productos_sin_datos}\n")
        f.write(f"Productos sin predicción: {productos_sin_prediccion}\n")
        f.write(f"Período histórico: 2017-01 a 2019-12\n")
        f.write(f"Predicción para: 2020-02\n")
        f.write(f"\nFeatures Avanzadas Implementadas:\n")
        f.write(f"- A. Productos Relacionados: contexto 3 niveles categorías\n")
        f.write(f"- B. Estacionalidad Avanzada: fourier + eventos argentinos\n")
        f.write(f"- C. Momentum/Aceleración: slopes + cambios volatilidad\n")
        f.write(f"- D. Patrones Consumo: skewness + regularidad temporal\n")
        f.write(f"\nArchivos generados:\n")
        f.write(f"  - PDF compilado: {os.path.basename(pdf_path)}\n")
        f.write(f"  - Imágenes individuales: {len(image_paths)} archivos PNG\n")
        f.write(f"\nUso recomendado:\n")
        f.write(f"  - Revisar PDF para análisis secuencial completo\n")
        f.write(f"  - Usar PNG individuales para consultas a LLM multimodal\n")
        f.write(f"  - Validar razonabilidad de predicciones con features avanzadas\n")
        f.write(f"  - Comparar vs. versiones anteriores para evaluar mejoras\n")
        f.write(f"\nObjetivo:\n")
        f.write(f"  - Superar meseta de optimización con features de alto impacto\n")
        f.write(f"  - Reducir Total Error Rate mediante contexto categorial\n")
        f.write(f"  - Mejorar captación de patrones estacionales complejos\n")
    
    return image_paths, pdf_path

def main():
    logger.info("INICIANDO GRÁFICOS - FEATURES AVANZADAS v11")
    
    cfg = Config()
    
    # Crear directorios
    for p in [cfg.CACHE_PATH, cfg.ARTIFACTS_PATH, cfg.GRAFICOS_PATH]:
        os.makedirs(p, exist_ok=True)
    
    # Cargar datos necesarios
    logger.info("--- FASE 1: Cargando Datos ---")
    birth_dates = calculate_birth_dates(cfg)
    historical_data = load_historical_data(cfg, birth_dates)
    productos_a_predecir = load_productos_a_predecir(cfg)
    
    # Preparar series temporales
    logger.info("--- FASE 2: Preparando Series Temporales ---")
    ts_data = prepare_time_series_data(historical_data, cfg)
    
    # Cargar predicciones
    logger.info("--- FASE 3: Cargando Predicciones Avanzadas ---")
    prediction_file = find_best_ensemble_prediction(cfg)
    if prediction_file is None:
        logger.error("No se pudo encontrar archivo de predicciones")
        logger.error("Ejecuta primero workaround_11_ratio_log.py o workaround_11_semillerio_ratio_log.py")
        return
    
    predictions = pd.read_csv(prediction_file, dtype={'product_id': str})
    logger.info(f"Predicciones cargadas: {len(predictions)} productos")
    logger.info(f"Total predicho: {predictions['tn'].sum():,.2f} toneladas")
    
    # Generar visualizaciones
    logger.info("--- FASE 4: Generando Visualizaciones Avanzadas ---")
    image_paths, pdf_path = generate_visualizations(ts_data, predictions, productos_a_predecir, cfg)
    
    logger.info("\nGENERACIÓN GRÁFICOS AVANZADOS COMPLETADA")
    logger.info(f"📂 Archivos generados en: {cfg.GRAFICOS_PATH}")
    logger.info(f"PDF compilado: {os.path.basename(pdf_path)}")
    logger.info(f"Imágenes individuales: {len(image_paths)} archivos PNG")
    logger.info("\nUso sugerido:")
    logger.info("  Revisar PDF para análisis secuencial de features avanzadas")
    logger.info("  Usar PNG para consultas específicas a LLM multimodal")
    logger.info("  Validar mejoras vs. versiones anteriores")
    logger.info("  Evaluar impacto de features A+B+C+D en predicciones")
    logger.info("Objetivo: Verificar que features avanzadas mejoran calidad predictiva")

if __name__ == '__main__':
    main()