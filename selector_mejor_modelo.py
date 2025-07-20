# -*- coding: utf-8 -*-
"""
Selector de Mejor Modelo por Producto y Generador de Ensemble Inteligente
=========================================================================

Este script implementa un sistema de selección inteligente de modelos que evalúa
múltiples algoritmos de forecasting por producto individual y genera un ensemble
final basado en el rendimiento histórico.

Funcionalidades principales:
1. Carga automáticamente todos los modelos disponibles en el directorio actual
   (archivos con formato *_diciembre.csv y *_febrero.csv).
2. Calcula modelos baseline basados en promedios históricos de 12 meses.
3. Evalúa el rendimiento de cada modelo usando datos reales de diciembre 2019.
4. Para cada producto, selecciona el modelo con menor error absoluto.
5. Genera un ensemble final combinando las mejores predicciones por producto.
6. Calcula métricas WAPE (Weighted Absolute Percentage Error) por modelo.
7. Produce tablas comparativas y archivos de entrega listos para submission.

Parámetros Principales de Configuración (dentro de la clase Config):
--------------------------------------------------------------------
- DICIEMBRE_2019: Período de referencia para evaluar modelos ('201912').
- FEBRERO_2020: Período objetivo de predicción ('202002').
- OUTPUT_DIR: Directorio donde se guardan los resultados y análisis.
- Archivos de salida: ensemble final, tabla detallada de comparación, tabla WAPE.

Detección Automática de Modelos:
--------------------------------
El script detecta automáticamente todos los archivos CSV en el directorio actual
que sigan el patrón: nombre_modelo_diciembre.csv y nombre_modelo_febrero.csv

"""
import os
import pandas as pd
import numpy as np
import glob
import logging
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

class Config:
    # URLs para descargar datos (mismas que en los scripts originales)
    SELLIN_URL = 'https://storage.googleapis.com/open-courses/austral2025-af91/labo3v/sell-in.txt.gz'
    PRODUCTOS_A_PREDECIR_URL = 'https://storage.googleapis.com/open-courses/austral2025-af91/labo3v/product_id_apredecir201912.txt'
    
    # Configuración de análisis
    DICIEMBRE_2019 = '201912'
    FEBRERO_2020 = '202002'
    FECHA_REFERENCIA_DIC = '201910'  # Octubre 2019 para baseline diciembre
    FECHA_REFERENCIA_FEB = '201912'  # Diciembre 2019 para baseline febrero
    
    # Salidas
    OUTPUT_DIR = 'analisis_modelos'
    ENSEMBLE_FILENAME = 'ensemble_final_febrero2020.csv'
    TABLA_DETALLADA_FILENAME = 'tabla_comparacion_detallada.csv'
    TABLA_WAPE_FILENAME = 'tabla_wape_por_modelo.csv'

def get_logger():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger()

logger = get_logger()

def load_base_data(cfg):
    """Carga los datos base: sell-in y productos a predecir."""
    logger.info("Descargando datos base...")
    
    # Cargar sell-in
    logger.info("Cargando sell-in...")
    sellin = pd.read_csv(
        cfg.SELLIN_URL,
        sep="\t", 
        compression='gzip',
        dtype={'periodo': str, 'customer_id': str, 'product_id': str}
    )
    sellin['periodo'] = pd.to_datetime(sellin['periodo'], format='%Y%m')
    
    # Cargar productos a predecir
    logger.info("Cargando productos a predecir...")
    productos_a_predecir = pd.read_csv(
        cfg.PRODUCTOS_A_PREDECIR_URL,
        sep='\t', 
        dtype={'product_id': str}
    )
    
    logger.info(f"Datos cargados: {len(sellin):,} registros sell-in, {len(productos_a_predecir)} productos a predecir")
    return sellin, productos_a_predecir

def create_base_table(sellin, productos_a_predecir, cfg):
    """Crea la tabla base con información histórica por producto."""
    logger.info("Creando tabla base con información histórica...")
    
    # Agregar por producto y período
    product_history = sellin.groupby(['product_id', 'periodo'])['tn'].sum().reset_index()
    
    # Estadísticas históricas por producto
    stats_by_product = []
    
    for product_id in productos_a_predecir['product_id']:
        product_data = product_history[product_history['product_id'] == product_id]
        
        if product_data.empty:
            # Producto sin datos históricos
            stats = {
                'product_id': product_id,
                'meses_presencia': 0,
                'total_tn_historico': 0.0,
                'primera_venta': None,
                'ultima_venta': None,
                'promedio_tn': 0.0,
                'max_tn': 0.0,
                'min_tn': 0.0,
                'std_tn': 0.0,
                'tn_dic2019_real': 0.0
            }
        else:
            # Calcular estadísticas
            active_months = product_data[product_data['tn'] > 0]
            
            # Valor real diciembre 2019
            dic_2019_data = product_data[product_data['periodo'] == pd.to_datetime(cfg.DICIEMBRE_2019, format='%Y%m')]
            tn_dic2019_real = dic_2019_data['tn'].iloc[0] if not dic_2019_data.empty else 0.0
            
            stats = {
                'product_id': product_id,
                'meses_presencia': len(active_months),
                'total_tn_historico': product_data['tn'].sum(),
                'primera_venta': active_months['periodo'].min() if not active_months.empty else None,
                'ultima_venta': active_months['periodo'].max() if not active_months.empty else None,
                'promedio_tn': product_data['tn'].mean(),
                'max_tn': product_data['tn'].max(),
                'min_tn': product_data['tn'].min(),
                'std_tn': product_data['tn'].std() if len(product_data) > 1 else 0.0,
                'tn_dic2019_real': tn_dic2019_real
            }
        
        stats_by_product.append(stats)
    
    base_table = pd.DataFrame(stats_by_product)
    logger.info(f"Tabla base creada: {len(base_table)} productos")
    return base_table, product_history

def calculate_baseline_models(product_history, base_table, cfg):
    """Calcula los modelos baseline basados en promedios de 12 meses."""
    logger.info("Calculando modelos baseline (avg_12m)...")
    
    baseline_predictions = []
    
    for _, row in base_table.iterrows():
        product_id = row['product_id']
        product_data = product_history[product_history['product_id'] == product_id]
        
        # Baseline para diciembre 2019: promedio oct2018-oct2019
        fecha_fin_dic = pd.to_datetime(cfg.FECHA_REFERENCIA_DIC, format='%Y%m')
        fecha_inicio_dic = fecha_fin_dic - pd.DateOffset(months=11)  # 12 meses atrás
        
        data_12m_dic = product_data[
            (product_data['periodo'] >= fecha_inicio_dic) & 
            (product_data['periodo'] <= fecha_fin_dic)
        ]
        avg_12m_dic = data_12m_dic['tn'].mean() if not data_12m_dic.empty else 0.0
        
        # Baseline para febrero 2020: promedio dic2018-dic2019
        fecha_fin_feb = pd.to_datetime(cfg.FECHA_REFERENCIA_FEB, format='%Y%m')
        fecha_inicio_feb = fecha_fin_feb - pd.DateOffset(months=11)  # 12 meses atrás
        
        data_12m_feb = product_data[
            (product_data['periodo'] >= fecha_inicio_feb) & 
            (product_data['periodo'] <= fecha_fin_feb)
        ]
        avg_12m_feb = data_12m_feb['tn'].mean() if not data_12m_feb.empty else 0.0
        
        baseline_predictions.append({
            'product_id': product_id,
            'avg_12m_dic': avg_12m_dic,
            'avg_12m_feb': avg_12m_feb,
            'avg_12m_099_dic': avg_12m_dic * 0.99,
            'avg_12m_099_feb': avg_12m_feb * 0.99,
            'avg_12m_101_dic': avg_12m_dic * 1.01,
            'avg_12m_101_feb': avg_12m_feb * 1.01
        })
    
    baseline_df = pd.DataFrame(baseline_predictions)
    logger.info("Modelos baseline calculados")
    return baseline_df

def detect_model_files(directory='.'):
    """Detecta automáticamente archivos de modelos en formato *_diciembre.csv y *_febrero.csv."""
    logger.info(f"Detectando modelos en directorio: {directory}")
    
    # Buscar archivos diciembre y febrero
    dic_files = glob.glob(os.path.join(directory, '*_diciembre.csv'))
    feb_files = glob.glob(os.path.join(directory, '*_febrero.csv'))
    
    # Extraer nombres de modelos
    dic_models = set()
    feb_models = set()
    
    for file in dic_files:
        model_name = os.path.basename(file).replace('_diciembre.csv', '')
        dic_models.add(model_name)
    
    for file in feb_files:
        model_name = os.path.basename(file).replace('_febrero.csv', '')
        feb_models.add(model_name)
    
    # Encontrar modelos que tienen ambos archivos
    complete_models = dic_models.intersection(feb_models)
    
    model_files = {}
    for model in complete_models:
        dic_file = os.path.join(directory, f'{model}_diciembre.csv')
        feb_file = os.path.join(directory, f'{model}_febrero.csv')
        
        if os.path.exists(dic_file) and os.path.exists(feb_file):
            model_files[model] = {
                'diciembre': dic_file,
                'febrero': feb_file
            }
    
    logger.info(f"Modelos detectados: {list(model_files.keys())}")
    return model_files

def load_model_predictions(model_files):
    """Carga las predicciones de todos los modelos detectados."""
    logger.info("Cargando predicciones de modelos...")
    
    all_predictions = {}
    
    for model_name, files in model_files.items():
        try:
            # Cargar predicciones diciembre
            dic_pred = pd.read_csv(files['diciembre'], dtype={'product_id': str})
            dic_pred = dic_pred.rename(columns={'tn': f'{model_name}_dic'})
            
            # Cargar predicciones febrero
            feb_pred = pd.read_csv(files['febrero'], dtype={'product_id': str})
            feb_pred = feb_pred.rename(columns={'tn': f'{model_name}_feb'})
            
            # Combinar
            model_pred = pd.merge(dic_pred, feb_pred, on='product_id', how='outer').fillna(0)
            all_predictions[model_name] = model_pred
            
            logger.info(f"  {model_name}: {len(model_pred)} productos")
            
        except Exception as e:
            logger.warning(f"  Error cargando {model_name}: {e}")
    
    return all_predictions

def calculate_errors_and_ensemble(base_table, baseline_df, all_predictions):
    """Calcula errores para diciembre y crea el ensemble final."""
    logger.info("Calculando errores y creando ensemble...")
    
    # Comenzar con tabla base
    result_table = base_table.copy()
    
    # Agregar baselines
    result_table = pd.merge(result_table, baseline_df, on='product_id', how='left')
    
    # Columnas de modelos para almacenar nombres
    all_model_names = []
    
    # Agregar baselines como modelos
    baseline_models = ['avg_12m', 'avg_12m_099', 'avg_12m_101']
    for model in baseline_models:
        all_model_names.append(model)
        # Calcular error baseline
        result_table[f'{model}_error'] = abs(result_table['tn_dic2019_real'] - result_table[f'{model}_dic'])
    
    # Agregar modelos detectados
    for model_name, model_pred in all_predictions.items():
        all_model_names.append(model_name)
        
        # Merge predicciones
        result_table = pd.merge(result_table, model_pred, on='product_id', how='left')
        
        # Calcular error
        result_table[f'{model_name}_error'] = abs(result_table['tn_dic2019_real'] - result_table[f'{model_name}_dic'].fillna(0))
    
    # Crear ensemble eligiendo el mejor modelo por producto
    logger.info("Creando ensemble por mejor modelo...")
    
    ensemble_predictions = []
    best_models = []
    
    for _, row in result_table.iterrows():
        # Encontrar modelo con menor error
        errors = {}
        for model in all_model_names:
            error_col = f'{model}_error'
            if error_col in result_table.columns:
                errors[model] = row[error_col]
        
        if errors:
            best_model = min(errors.keys(), key=lambda x: errors[x])
            feb_col = f'{best_model}_feb'
            
            if feb_col in result_table.columns:
                ensemble_pred = row[feb_col]
            else:
                ensemble_pred = 0.0
        else:
            best_model = 'ninguno'
            ensemble_pred = 0.0
        
        ensemble_predictions.append(ensemble_pred)
        best_models.append(best_model)
    
    result_table['ensemble_feb2020'] = ensemble_predictions
    result_table['modelo_seleccionado'] = best_models
    
    logger.info(f"Ensemble creado para {len(result_table)} productos")
    return result_table, all_model_names

def calculate_wape_table(result_table, all_model_names):
    """Calcula tabla de WAPE por modelo."""
    logger.info("Calculando tabla de WAPE por modelo...")
    
    wape_results = []
    
    # Filtrar productos con datos reales > 0 para WAPE
    valid_products = result_table[result_table['tn_dic2019_real'] > 0]
    total_real = valid_products['tn_dic2019_real'].sum()
    
    for model in all_model_names:
        dic_col = f'{model}_dic'
        feb_col = f'{model}_feb'
        error_col = f'{model}_error'
        
        if dic_col in result_table.columns:
            # WAPE para diciembre
            valid_model_data = valid_products.dropna(subset=[dic_col])
            
            if not valid_model_data.empty:
                total_error = valid_model_data[error_col].sum()
                wape = total_error / total_real if total_real > 0 else np.nan
                
                total_pred = valid_model_data[dic_col].sum()
                productos_evaluados = len(valid_model_data)
            else:
                wape = np.nan
                total_pred = 0
                productos_evaluados = 0
            
            wape_results.append({
                'modelo': model,
                'wape_diciembre': wape,
                'productos_evaluados': productos_evaluados,
                'total_tn_real': total_real,
                'total_tn_pred': total_pred
            })
    
    # Agregar WAPE del ensemble
    ensemble_error = abs(valid_products['tn_dic2019_real'] - valid_products['ensemble_feb2020'])
    ensemble_wape = ensemble_error.sum() / total_real if total_real > 0 else np.nan
    
    wape_results.append({
        'modelo': 'ENSEMBLE',
        'wape_diciembre': ensemble_wape,
        'productos_evaluados': len(valid_products),
        'total_tn_real': total_real,
        'total_tn_pred': valid_products['ensemble_feb2020'].sum()
    })
    
    wape_table = pd.DataFrame(wape_results).sort_values('wape_diciembre')
    logger.info("Tabla WAPE calculada")
    return wape_table

def generate_final_ensemble_file(result_table, cfg):
    """Genera el archivo final de ensemble para entrega."""
    logger.info("Generando archivo final de ensemble...")
    
    ensemble_file = result_table[['product_id', 'ensemble_feb2020']].copy()
    ensemble_file = ensemble_file.rename(columns={'ensemble_feb2020': 'tn'})
    
    output_path = os.path.join(cfg.OUTPUT_DIR, cfg.ENSEMBLE_FILENAME)
    ensemble_file.to_csv(output_path, index=False)
    
    total_pred = ensemble_file['tn'].sum()
    logger.info(f"Ensemble final guardado: {output_path}")
    logger.info(f"Total predicho para febrero 2020: {total_pred:,.2f} toneladas")
    
    return output_path

def main():
    logger.info("INICIANDO SELECTOR DE MEJOR MODELO")
    
    cfg = Config()
    
    # Crear directorio de salida
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    # 1. Cargar datos base
    logger.info("--- FASE 1: Cargando Datos Base ---")
    sellin, productos_a_predecir = load_base_data(cfg)
    
    # 2. Crear tabla base con estadísticas históricas
    logger.info("--- FASE 2: Creando Tabla Base ---")
    base_table, product_history = create_base_table(sellin, productos_a_predecir, cfg)
    
    # 3. Calcular modelos baseline
    logger.info("--- FASE 3: Calculando Modelos Baseline ---")
    baseline_df = calculate_baseline_models(product_history, base_table, cfg)
    
    # 4. Detectar modelos automáticamente
    logger.info("--- FASE 4: Detectando Modelos ---")
    model_files = detect_model_files()
    
    if not model_files:
        logger.warning("No se encontraron archivos de modelos en formato *_diciembre.csv / *_febrero.csv")
        logger.info("Continuando solo con modelos baseline...")
        all_predictions = {}
    else:
        # 5. Cargar predicciones de modelos
        logger.info("--- FASE 5: Cargando Predicciones de Modelos ---")
        all_predictions = load_model_predictions(model_files)
    
    # 6. Calcular errores y crear ensemble
    logger.info("--- FASE 6: Calculando Errores y Creando Ensemble ---")
    result_table, all_model_names = calculate_errors_and_ensemble(base_table, baseline_df, all_predictions)
    
    # 7. Calcular tabla de WAPE
    logger.info("--- FASE 7: Calculando Tabla de WAPE ---")
    wape_table = calculate_wape_table(result_table, all_model_names)
    
    # 8. Generar archivos de salida
    logger.info("--- FASE 8: Generando Archivos de Salida ---")
    
    # Archivo detallado
    detailed_path = os.path.join(cfg.OUTPUT_DIR, cfg.TABLA_DETALLADA_FILENAME)
    result_table.to_csv(detailed_path, index=False)
    logger.info(f"Tabla detallada guardada: {detailed_path}")
    
    # Tabla WAPE
    wape_path = os.path.join(cfg.OUTPUT_DIR, cfg.TABLA_WAPE_FILENAME)
    wape_table.to_csv(wape_path, index=False)
    logger.info(f"Tabla WAPE guardada: {wape_path}")
    
    # Ensemble final
    ensemble_path = generate_final_ensemble_file(result_table, cfg)
    
    # 9. Mostrar resumen
    logger.info("\n" + "="*60)
    logger.info("RESUMEN FINAL")
    logger.info("="*60)
    
    # Resumen de modelos detectados
    logger.info(f"Modelos detectados: {len(model_files)}")
    for model in model_files.keys():
        logger.info(f"  - {model}")
    
    logger.info(f"Modelos baseline: {len(['avg_12m', 'avg_12m_099', 'avg_12m_101'])}")
    
    # Top 5 mejores modelos por WAPE
    logger.info("\nTop 5 Modelos por WAPE (Diciembre 2019):")
    for i, (_, row) in enumerate(wape_table.head(5).iterrows()):
        logger.info(f"  {i+1}. {row['modelo']}: {row['wape_diciembre']:.4f}")
    
    # Distribución de modelos seleccionados
    logger.info("\nDistribución de Modelos Seleccionados:")
    model_counts = result_table['modelo_seleccionado'].value_counts()
    for model, count in model_counts.head(10).items():
        percentage = (count / len(result_table)) * 100
        logger.info(f"  - {model}: {count} productos ({percentage:.1f}%)")
    
    # Total de ensemble
    total_ensemble = result_table['ensemble_feb2020'].sum()
    logger.info(f"\nTotal Ensemble Febrero 2020: {total_ensemble:,.2f} toneladas")
    
    logger.info("\nPROCESO COMPLETADO")
    logger.info(f"Revisa los archivos en: {cfg.OUTPUT_DIR}")

if __name__ == '__main__':
    main()