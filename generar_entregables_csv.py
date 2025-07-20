# -*- coding: utf-8 -*-
"""
Generador de Entregables y Ensambles de Forecasts
=================================================

Este script procesa múltiples archivos de forecast en formato CSV desde un 
directorio específico. Su objetivo es consolidar las predicciones, generar 
diferentes tipos de ensambles y producir un reporte resumido.

Funcionalidades principales:
- Lee todos los archivos CSV de un directorio.
- Valida que los archivos contengan las columnas 'product_id' y 'tn'.
- Crea ensambles por mediana, media y media con filtro de outliers.
- Guarda cada ensamble como un nuevo archivo CSV.
- Genera un archivo de texto (.txt) con estadísticas y un resumen de los resultados.

Parámetros de configuración:
---------------------------
- El script se ejecuta sobre el directorio desde donde es llamado, o bien,
  se puede pasar la ruta a un directorio como primer argumento en la línea 
  de comandos.
  Ejemplo: python generar_entregables_csv.py /ruta/a/mis_forecasts/

"""

import pandas as pd
import numpy as np
import os
import glob

def process_forecasts_all_csv(directory):
    """
    Procesa archivos CSV con columnas product_id y tn para crear ensambles.
    Genera estadísticas y múltiples tipos de ensambles sobre los forecasts.
    """
    pattern = os.path.join(directory, "*.csv")
    files = glob.glob(pattern)

    if not files:
        print(f"No se encontraron archivos CSV en: {directory}")
        return

    # Leer los archivos y validar estructura
    dfs = []
    valid_files = []
    for file in files:
        try:
            df = pd.read_csv(file)
        except Exception as e:
            print(f"Error leyendo {file}: {e}")
            continue
        if set(['product_id', 'tn']).issubset(df.columns):
            dfs.append(df)
            valid_files.append(file)
            print(f"Archivo válido incluido: {os.path.basename(file)}")
        else:
            print(f"Archivo ignorado por formato inesperado: {os.path.basename(file)}")

    if len(dfs) < 2:
        print("Se necesitan al menos 2 archivos válidos para ensamblar.")
        return

    print(f"\nEnsamblando {len(dfs)} archivos...")

    # Combinar forecasts
    combined_df = pd.DataFrame({'product_id': dfs[0]['product_id']})
    for i, df in enumerate(dfs):
        combined_df[f'forecast_{i}'] = df['tn'].values

    # Calcular estadísticas
    forecast_cols = [col for col in combined_df.columns if col.startswith('forecast_')]
    combined_df['median'] = combined_df[forecast_cols].median(axis=1)
    combined_df['mean'] = combined_df[forecast_cols].mean(axis=1)
    combined_df['std'] = combined_df[forecast_cols].std(axis=1)

    # Ensamble por mediana
    median_ensemble = combined_df[['product_id', 'median']].rename(columns={'median': 'tn'})
    median_filename = os.path.join(directory, f"ensemble_median.csv")
    median_ensemble.to_csv(median_filename, index=False)

    # Ensamble promedio simple
    mean_ensemble = pd.DataFrame({
        'product_id': combined_df['product_id'],
        'tn': combined_df['mean']
    })
    mean_filename = os.path.join(directory, f"ensemble_mean.csv")
    mean_ensemble.to_csv(mean_filename, index=False)

    # Remover outliers (z-score > 2) y promediar
    z_scores = (combined_df[forecast_cols] - combined_df['mean'].values[:, None]) / (combined_df['std'].values[:, None])
    mask = np.abs(z_scores) < 2.0
    filtered_means = combined_df[forecast_cols].where(mask).mean(axis=1)
    outlier_filtered_mean = pd.DataFrame({'product_id': combined_df['product_id'], 'tn': filtered_means})
    outlier_filename = os.path.join(directory, f"ensemble_outlier_filtered_mean.csv")
    outlier_filtered_mean.to_csv(outlier_filename, index=False)

    # Estadísticas de los ensambles
    print("\nArchivos generados:")
    print(f" - {os.path.basename(median_filename)}")
    print(f" - {os.path.basename(mean_filename)}")
    print(f" - {os.path.basename(outlier_filename)}")
    
    print("\nEstadísticas de predicciones totales:")
    print(f" - Mediana ensemble: {median_ensemble['tn'].sum():,.2f} toneladas")
    print(f" - Media ensemble: {mean_ensemble['tn'].sum():,.2f} toneladas")
    print(f" - Media sin outliers: {outlier_filtered_mean['tn'].sum():,.2f} toneladas")
    
    # Generar resumen detallado
    summary_filename = os.path.join(directory, f"ensemble_summary.txt")
    with open(summary_filename, 'w', encoding='utf-8') as f:
        f.write(f"Resumen de Ensambles - CSV válidos procesados\n")
        f.write("="*50 + "\n")
        f.write(f"Archivos procesados: {len(dfs)}\n\n")
        
        f.write("Archivos incluidos:\n")
        for file in valid_files:
            f.write(f"  - {os.path.basename(file)}\n")
        
        f.write(f"\nTotales predichos:\n")
        f.write(f"  - Mediana: {median_ensemble['tn'].sum():,.2f} toneladas\n")
        f.write(f"  - Media: {mean_ensemble['tn'].sum():,.2f} toneladas\n")
        f.write(f"  - Media sin outliers: {outlier_filtered_mean['tn'].sum():,.2f} toneladas\n")
        
        # Análisis por producto (top 10)
        f.write("\nTop 10 productos por volumen (ensemble mediana):\n")
        top_products = median_ensemble.nlargest(10, 'tn')
        for _, row in top_products.iterrows():
            f.write(f"  - Producto {row['product_id']}: {row['tn']:,.2f} tn\n")
    
    print(f"\nResumen guardado en: {os.path.basename(summary_filename)}")

# Ejecutar en el directorio actual o pasado por argumento
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        directory = sys.argv[1]
    else:
        directory = "."
    if not os.path.exists(directory):
        print(f"El directorio '{directory}' no existe.")
        print("Uso: python generar_entregables_csv.py [directorio]")
        sys.exit(1)
    print(f"Procesando archivos en: {directory}\n")
    process_forecasts_all_csv(directory)
