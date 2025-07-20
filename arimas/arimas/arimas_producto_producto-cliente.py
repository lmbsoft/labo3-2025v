import os
import pandas as pd
import numpy as np
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
from datetime import datetime
import urllib.request
from typing import Dict, List, Tuple, Optional, Any
import logging
import json
import time
import psutil
import multiprocessing
from dataclasses import dataclass, asdict
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')  # Usar backend sin display
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
import warnings
warnings.filterwarnings('ignore')

# Opcional: usar Polars para operaciones más rápidas
try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False
    print("Polars no está instalado. Usando pandas para todas las operaciones.")

@dataclass
class SeriesStats:
    """Estadísticas de una serie temporal"""
    series_id: str
    total_periods: int
    non_zero_periods: int
    mean_value: float
    std_value: float
    min_value: float
    max_value: float
    has_all_zeros: bool
    processing_time: float = 0.0
    forecast_values: List[float] = None
    error_message: str = None

@dataclass
class ProcessingReport:
    """Reporte de procesamiento completo"""
    start_time: str
    end_time: str
    total_duration: float
    phase: str
    granularity: str
    total_series: int
    processed_series: int
    zero_series: int
    failed_series: int
    short_series: int
    cpu_usage_avg: float
    memory_usage_mb: float
    series_per_second: float
    forecast_results: Dict[str, Any]
    phase1_metrics: Dict[str, float] = None

class ReportingSystem:
    """Sistema de reporting y logging para pronósticos"""
    
    def __init__(self, log_dir: str = "forecast_logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Configurar logging
        self.setup_logging()
        
        # Inicializar contadores y estadísticas
        self.reset_stats()
        
        # Información del sistema
        self.system_info = self.get_system_info()
        
    def setup_logging(self):
        """Configura el sistema de logging"""
        log_file = os.path.join(self.log_dir, f"forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
        # Configurar formato de logging
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        # Configurar handlers
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()  # También mostrar en consola
            ]
        )
        
        self.logger = logging.getLogger('ForecastSystem')
        self.logger.info("="*80)
        self.logger.info("Sistema de Pronóstico Iniciado")
        self.logger.info("="*80)
        
    def reset_stats(self):
        """Reinicia estadísticas para nueva ejecución"""
        self.stats = {
            'series_processed': 0,
            'series_zero': 0,
            'series_failed': 0,
            'series_short': 0,
            'series_smoothed': 0,
            'processing_times': [],
            'memory_usage': [],
            'cpu_usage': [],
            'series_details': {},
            'phase1_errors': {},
            'phase1_metrics': None,
            'negative_forecasts_corrected': 0,
            'smoothing_time': 0
        }
        
    def get_system_info(self) -> Dict:
        """Obtiene información del sistema"""
        info = {
            'cpu_count': multiprocessing.cpu_count(),
            'cpu_count_physical': psutil.cpu_count(logical=False),
            'memory_gb': psutil.virtual_memory().total / (1024**3),
            'platform': os.name,
            'polars_available': POLARS_AVAILABLE
        }
        
        self.logger.info(f"Sistema: CPUs={info['cpu_count']}, RAM={info['memory_gb']:.2f}GB, Polars={'Sí' if POLARS_AVAILABLE else 'No'}")
        return info
    
    def log_series_analysis(self, series_id: str, stats: SeriesStats):
        """Registra análisis de una serie"""
        if stats.has_all_zeros:
            self.logger.warning(f"Serie {series_id}: Todos valores cero - Pronóstico directo = 0")
            self.stats['series_zero'] += 1
        elif stats.error_message:
            self.logger.error(f"Serie {series_id}: {stats.error_message}")
            self.stats['series_failed'] += 1
        else:
            self.logger.debug(f"Serie {series_id}: Procesada OK - Media={stats.mean_value:.2f}")
            self.stats['series_processed'] += 1
            
        self.stats['series_details'][series_id] = asdict(stats)
        
    def generate_summary_report(self, phase: str, granularity: str) -> ProcessingReport:
        """Genera reporte resumen de procesamiento"""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        # Calcular métricas de fase 1 si existen
        phase1_metrics = None
        if phase == 'phase1':
            # Usar métricas ya calculadas si están disponibles
            if self.stats['phase1_metrics']:
                phase1_metrics = self.stats['phase1_metrics']
            elif self.stats['phase1_errors']:
                # Fallback: calcular desde errores guardados
                errors = list(self.stats['phase1_errors'].values())
                phase1_metrics = {
                    'mae': np.mean(np.abs(errors)),
                    'rmse': np.sqrt(np.mean(np.square(errors))),
                    'total_error': np.sum(np.abs(errors)),
                    'n_comparisons': len(errors)
                }
        
        report = ProcessingReport(
            start_time=self.start_time.isoformat(),
            end_time=end_time.isoformat(),
            total_duration=duration,
            phase=phase,
            granularity=granularity,
            total_series=sum([
                self.stats['series_processed'],
                self.stats['series_zero'],
                self.stats['series_failed'],
                self.stats['series_short']
            ]),
            processed_series=self.stats['series_processed'],
            zero_series=self.stats['series_zero'],
            failed_series=self.stats['series_failed'],
            short_series=self.stats['series_short'],
            cpu_usage_avg=np.mean(self.stats['cpu_usage']) if self.stats['cpu_usage'] else 0,
            memory_usage_mb=np.mean(self.stats['memory_usage']) if self.stats['memory_usage'] else 0,
            series_per_second=self.stats['series_processed'] / duration if duration > 0 else 0,
            forecast_results={},
            phase1_metrics=phase1_metrics
        )
        
        return report
    
    def save_report(self, report: ProcessingReport, filename: str = None):
        """Guarda reporte en archivo JSON"""
        if filename is None:
            filename = f"report_{report.phase}_{report.granularity}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = os.path.join(self.log_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(asdict(report), f, indent=2, default=str)
            
        self.logger.info(f"Reporte guardado en: {filepath}")
        
    def print_summary(self, report: ProcessingReport):
        """Imprime resumen del reporte"""
        print("\n" + "="*80)
        print(f"RESUMEN DE PROCESAMIENTO - {report.phase} ({report.granularity})")
        print("="*80)
        print(f"Tiempo total: {report.total_duration:.2f} segundos")
        print(f"Series totales: {report.total_series}")
        print(f"  - Procesadas: {report.processed_series}")
        print(f"  - Con todos ceros: {report.zero_series}")
        print(f"  - Series cortas: {report.short_series}")
        print(f"  - Fallidas: {report.failed_series}")
        print(f"  - Series suavizadas: {self.stats.get('series_smoothed', 0)}")
        print(f"  - Tiempo de suavizado: {self.stats.get('smoothing_time', 0):.2f}s")
        print(f"  - Pronósticos negativos corregidos: {self.stats.get('negative_forecasts_corrected', 0)}")
        print(f"Velocidad: {report.series_per_second:.2f} series/segundo")
        print(f"CPU promedio: {report.cpu_usage_avg:.1f}%")
        print(f"Memoria promedio: {report.memory_usage_mb:.1f} MB")
        
        # Métricas de error para Fase 1
        if report.phase1_metrics:
            print("\nMÉTRICAS DE ERROR (Fase 1 - Comparación con 2019-12 real):")
            print(f"  - MAE: {report.phase1_metrics['mae']:.4f}")
            print(f"  - RMSE: {report.phase1_metrics['rmse']:.4f}")
            print(f"  - Error Total: {report.phase1_metrics['total_error']:.2f}")
            if 'wape_modificado' in report.phase1_metrics:
                print(f"  - WAPE Modificado: {report.phase1_metrics['wape_modificado']:.4f}")
            print(f"  - Series comparadas: {report.phase1_metrics['n_comparisons']}")
        
        print("="*80 + "\n")


class ForecastSystemWithReporting:
    """Sistema de pronóstico con reporting completo"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.reporter = ReportingSystem()
        self.data = {}
        self.zero_series_cache = {}  # Cache para series con todos ceros
        
    def wape_modificado(self, actual: np.ndarray, forecast: np.ndarray) -> float:
        """
        Calcula el WAPE (Weighted Absolute Percentage Error) modificado
        
        Formula: WAPE = Σ|actual - forecast| / Σ|actual|
        
        Args:
            actual: valores reales
            forecast: valores pronosticados
            
        Returns:
            float: WAPE modificado (0-∞, donde 0 es perfecto)
        """
        actual = np.array(actual)
        forecast = np.array(forecast)
        
        # Evitar división por cero
        if np.sum(np.abs(actual)) == 0:
            return np.nan
            
        return np.sum(np.abs(actual - forecast)) / np.sum(np.abs(actual))
    
    def apply_smoothing_vectorized(self, df: pd.DataFrame, method: str = 'exponential', alpha: float = 0.3) -> pd.DataFrame:
        """
        Aplica suavizado vectorizado a todas las series de manera eficiente
        
        Args:
            df: DataFrame con columnas 'unique_id', 'periodo', 'tn'
            method: 'exponential' o 'moving_average'
            alpha: parámetro para suavizado exponencial
            
        Returns:
            DataFrame con valores suavizados
        """
        start_time = time.time()
        self.reporter.logger.info(f"Aplicando suavizado vectorizado ({method})...")
        
        # Guardar valores originales
        df['tn_original'] = df['tn'].copy()
        
        # Contar series que serán suavizadas (con suficientes datos y variabilidad)
        series_stats = df.groupby('unique_id')['tn'].agg(['count', 'std'])
        valid_series = series_stats[(series_stats['count'] >= 6) & (series_stats['std'] > 0)].index
        self.reporter.stats['series_smoothed'] = len(valid_series)
        
        if len(valid_series) == 0:
            self.reporter.logger.warning("No hay series válidas para suavizar")
            return df
        
        # Filtrar solo series válidas para suavizado
        mask_valid = df['unique_id'].isin(valid_series)
        
        if method == 'exponential':
            # Suavizado exponencial vectorizado
            df.loc[mask_valid, 'tn'] = df[mask_valid].groupby('unique_id')['tn'].transform(
                lambda x: x.ewm(alpha=alpha, adjust=False).mean()
            )
        else:  # moving_average
            # Media móvil vectorizada
            window = 3
            df.loc[mask_valid, 'tn'] = df[mask_valid].groupby('unique_id')['tn'].transform(
                lambda x: x.rolling(window=window, center=True, min_periods=1).mean()
            )
        
        smoothing_time = time.time() - start_time
        self.reporter.stats['smoothing_time'] = smoothing_time
        self.reporter.logger.info(f"Suavizado completado en {smoothing_time:.2f}s para {len(valid_series)} series")
        
        return df
    
    def apply_smoothing_polars(self, df: pd.DataFrame, method: str = 'exponential', alpha: float = 0.3) -> pd.DataFrame:
        """
        Versión con Polars para mayor eficiencia (si está disponible)
        """
        if not POLARS_AVAILABLE:
            return self.apply_smoothing_vectorized(df, method, alpha)
            
        start_time = time.time()
        self.reporter.logger.info(f"Aplicando suavizado con Polars ({method})...")
        
        # Convertir a Polars
        df_pl = pl.from_pandas(df)
        
        # Identificar series válidas - usar group_by en lugar de groupby
        series_stats = df_pl.group_by('unique_id').agg([
            pl.col('tn').count().alias('count'),
            pl.col('tn').std().alias('std')
        ])
        
        valid_series = series_stats.filter(
            (pl.col('count') >= 6) & (pl.col('std') > 0)
        )['unique_id'].to_list()
        
        self.reporter.stats['series_smoothed'] = len(valid_series)
        
        # Aplicar suavizado
        if method == 'exponential':
            df_smoothed = df_pl.with_columns([
                pl.when(pl.col('unique_id').is_in(valid_series))
                .then(
                    pl.col('tn').ewm_mean(alpha=alpha).over('unique_id')
                )
                .otherwise(pl.col('tn'))
                .alias('tn_smoothed')
            ])
        else:  # moving_average
            df_smoothed = df_pl.with_columns([
                pl.when(pl.col('unique_id').is_in(valid_series))
                .then(
                    pl.col('tn').rolling_mean(window_size=3).over('unique_id')
                )
                .otherwise(pl.col('tn'))
                .alias('tn_smoothed')
            ])
        
        # Convertir de vuelta a pandas
        df_result = df_smoothed.to_pandas()
        df_result['tn_original'] = df_result['tn'].copy()
        df_result['tn'] = df_result['tn_smoothed']
        df_result = df_result.drop('tn_smoothed', axis=1)
        
        smoothing_time = time.time() - start_time
        self.reporter.stats['smoothing_time'] = smoothing_time
        self.reporter.logger.info(f"Suavizado con Polars completado en {smoothing_time:.2f}s")
        
        return df_result
        
    def analyze_series(self, series_data: pd.Series) -> SeriesStats:
        """Analiza una serie temporal y retorna estadísticas"""
        non_zero = series_data[series_data != 0]
        
        stats = SeriesStats(
            series_id=series_data.name if hasattr(series_data, 'name') else 'unknown',
            total_periods=len(series_data),
            non_zero_periods=len(non_zero),
            mean_value=series_data.mean(),
            std_value=series_data.std(),
            min_value=series_data.min(),
            max_value=series_data.max(),
            has_all_zeros=(len(non_zero) == 0)
        )
        
        return stats
    
    def detect_zero_series(self, df: pd.DataFrame, id_col: str = 'unique_id') -> Tuple[pd.DataFrame, List[str]]:
        """
        Detecta y separa series con todos valores cero
        
        Returns:
            - DataFrame sin series cero
            - Lista de IDs de series con todos ceros
        """
        self.reporter.logger.info("Detectando series con todos valores cero...")
        
        # Agrupar por serie y verificar si todos son ceros
        series_sums = df.groupby(id_col)['y'].sum()
        zero_series = series_sums[series_sums == 0].index.tolist()
        
        if zero_series:
            self.reporter.logger.warning(f"Encontradas {len(zero_series)} series con todos valores cero")
            for series_id in zero_series[:10]:  # Mostrar primeras 10
                self.reporter.logger.debug(f"  - Serie cero: {series_id}")
            if len(zero_series) > 10:
                self.reporter.logger.debug(f"  ... y {len(zero_series) - 10} más")
        
        # Filtrar DataFrame
        df_filtered = df[~df[id_col].isin(zero_series)]
        
        # Guardar en cache
        self.zero_series_cache.update({sid: 0.0 for sid in zero_series})
        
        return df_filtered, zero_series
    
    def load_all_data(self):
        """Carga todos los datasets con logging"""
        self.reporter.logger.info("=== CARGA DE DATOS ===")
        
        for dataset_name, dataset_config in self.config['files'].items():
            start_time = time.time()
            
            # Descargar si es necesario
            path = self._download_file(
                dataset_config['local'], 
                dataset_config['url']
            )
            
            # Cargar datos
            self.data[dataset_name] = pd.read_csv(
                path, 
                **dataset_config['read_args']
            )
            
            load_time = time.time() - start_time
            
            self.reporter.logger.info(
                f"{dataset_name}: {self.data[dataset_name].shape} "
                f"(cargado en {load_time:.2f}s)"
            )
    
    def _download_file(self, local_path: str, url: str) -> str:
        """Descarga archivo si no existe localmente"""
        if not os.path.exists(local_path):
            self.reporter.logger.info(f"Descargando {local_path}...")
            urllib.request.urlretrieve(url, local_path)
        return local_path
    
    def correct_negative_forecasts(self, forecasts: pd.DataFrame) -> pd.DataFrame:
        """Corrige pronósticos negativos llevándolos a 0"""
        negative_mask = forecasts['forecast'] < 0
        n_negative = negative_mask.sum()
        
        if n_negative > 0:
            self.reporter.logger.warning(f"Corrigiendo {n_negative} pronósticos negativos a 0")
            self.reporter.stats['negative_forecasts_corrected'] += n_negative
            forecasts.loc[negative_mask, 'forecast'] = 0.0
            
        return forecasts
    
    def calculate_phase1_errors(
        self, 
        forecasts: pd.DataFrame, 
        actuals: pd.DataFrame,
        granularity: str
    ) -> Dict[str, float]:
        """Calcula errores para Fase 1 comparando con valores reales"""
        self.reporter.logger.info("Calculando métricas de error (Fase 1)...")
        
        # Preparar datos reales para 2019-12
        target_date = pd.to_datetime('2019-12-01')
        
        if granularity == 'product':
            actuals_dec = actuals[actuals['periodo'] == target_date].copy()
            actuals_dec = actuals_dec.groupby('product_id')['tn'].sum().reset_index()
            actuals_dec['unique_id'] = actuals_dec['product_id']
        else:
            actuals_dec = actuals[actuals['periodo'] == target_date].copy()
            actuals_dec['unique_id'] = (
                actuals_dec['product_id'].astype(str) + "_" + 
                actuals_dec['customer_id'].astype(str)
            )
            actuals_dec = actuals_dec.groupby('unique_id')['tn'].sum().reset_index()
        
        # Filtrar pronósticos para 2019-12
        forecasts_dec = forecasts[forecasts['ds'] == target_date].copy()
        
        # Merge
        comparison = pd.merge(
            forecasts_dec[['unique_id', 'forecast']],
            actuals_dec[['unique_id', 'tn']],
            on='unique_id',
            how='inner'
        )
        
        # Calcular errores
        comparison['error'] = comparison['tn'] - comparison['forecast']
        comparison['abs_error'] = np.abs(comparison['error'])
        
        # Guardar errores por serie
        for _, row in comparison.iterrows():
            self.reporter.stats['phase1_errors'][row['unique_id']] = row['error']
        
        # Métricas agregadas
        mae = comparison['abs_error'].mean()
        rmse = np.sqrt((comparison['error'] ** 2).mean())
        total_error = comparison['abs_error'].sum()
        
        # Calcular WAPE modificado
        wape_mod = self.wape_modificado(comparison['tn'].values, comparison['forecast'].values)
        
        self.reporter.logger.info(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, Error Total: {total_error:.2f}, WAPE Modificado: {wape_mod:.4f}")
        
        # Crear y guardar métricas completas
        metrics = {
            'mae': mae,
            'rmse': rmse,
            'total_error': total_error,
            'wape_modificado': wape_mod,
            'n_comparisons': len(comparison)
        }
        
        # Guardar en el reporter para usar en el reporte final
        self.reporter.stats['phase1_metrics'] = metrics
        
        return metrics
    
    def train_and_forecast_with_monitoring(
        self,
        df: pd.DataFrame,
        train_end: str,
        horizon: int = 2,
        granularity: str = 'product',
        n_jobs: int = -1,
        phase: str = None,
        base_data: pd.DataFrame = None,
        apply_smoothing: bool = True
    ) -> pd.DataFrame:
        """
        Entrena y pronostica con monitoreo completo
        """
        self.reporter.start_time = datetime.now()
        self.reporter.reset_stats()
        
        self.reporter.logger.info(f"\n=== ENTRENAMIENTO Y PRONÓSTICO ({granularity}) ===")
        self.reporter.logger.info(f"Periodo de entrenamiento hasta: {train_end}")
        self.reporter.logger.info(f"Suavizado de series: {'Activado' if apply_smoothing else 'Desactivado'}")
        
        # Preparar datos
        train_end_date = pd.to_datetime(train_end)
        df_train = df[df['periodo'] <= train_end_date].copy()
        
        # Crear unique_id
        if granularity == 'product':
            df_train['unique_id'] = df_train['product_id']
        else:
            df_train['unique_id'] = (
                df_train['product_id'].astype(str) + "_" + 
                df_train['customer_id'].astype(str)
            )
        
        # Aplicar suavizado si está habilitado
        if apply_smoothing:
            smoothing_method = self.config.get('smoothing', {}).get('method', 'exponential')
            smoothing_alpha = self.config.get('smoothing', {}).get('alpha', 0.3)
            
            # Usar versión vectorizada/Polars según disponibilidad
            if POLARS_AVAILABLE and len(df_train['unique_id'].unique()) > 10000:
                df_train = self.apply_smoothing_polars(df_train, smoothing_method, smoothing_alpha)
            else:
                df_train = self.apply_smoothing_vectorized(df_train, smoothing_method, smoothing_alpha)
        
        # Formatear para StatsForecast
        sf_input = df_train.rename(columns={
            'periodo': 'ds',
            'tn': 'y'
        })[['unique_id', 'ds', 'y']]
        
        # Verificar longitud de series
        series_length = sf_input.groupby('unique_id').size()
        short_series = series_length[series_length < self.config['model']['min_series_length']].index
        self.reporter.stats['series_short'] = len(short_series)
        
        if len(short_series) > 0:
            self.reporter.logger.warning(f"Encontradas {len(short_series)} series muy cortas (<24 periodos)")
        
        # Filtrar series válidas
        sf_input = sf_input[~sf_input['unique_id'].isin(short_series)]
        
        # Detectar y separar series con todos ceros
        sf_input_filtered, zero_series = self.detect_zero_series(sf_input)
        
        n_series = sf_input_filtered['unique_id'].nunique()
        self.reporter.logger.info(f"Series a procesar: {n_series} (excluidas {len(zero_series)} con todos ceros)")
        
        if n_series == 0:
            self.reporter.logger.warning("No hay series válidas para procesar")
            return pd.DataFrame()
        
        # Monitorear recursos durante entrenamiento
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024
        
        # Entrenar modelo
        self.reporter.logger.info("Iniciando entrenamiento de modelos...")
        
        model = AutoARIMA(
            season_length=self.config['model']['season_length'],
            **self.config['model'].get('autoarima_params', {})
        )
        
        sf = StatsForecast(
            models=[model], 
            freq='MS', 
            n_jobs=n_jobs,
            verbose=False
        )
        
        # Entrenar con monitoreo de progreso
        train_start = time.time()
        sf.fit(sf_input_filtered)
        train_time = time.time() - train_start
        
        self.reporter.logger.info(f"Entrenamiento completado en {train_time:.2f}s")
        
        # Generar pronósticos
        self.reporter.logger.info("Generando pronósticos...")
        forecast_start = time.time()
        forecasts = sf.predict(h=horizon)
        forecast_time = time.time() - forecast_start
        
        self.reporter.logger.info(f"Pronósticos generados en {forecast_time:.2f}s")
        
        # Renombrar columna
        forecasts = forecasts.rename(columns={'AutoARIMA': 'forecast'})
        
        # Corregir pronósticos negativos
        forecasts = self.correct_negative_forecasts(forecasts)
        
        # Agregar pronósticos cero para series con todos ceros
        if zero_series:
            self.reporter.logger.info(f"Agregando pronósticos cero para {len(zero_series)} series")
            zero_forecasts = []
            
            forecast_dates = forecasts['ds'].unique()
            for series_id in zero_series:
                for date in forecast_dates:
                    zero_forecasts.append({
                        'unique_id': series_id,
                        'ds': date,
                        'forecast': 0.0
                    })
            
            zero_df = pd.DataFrame(zero_forecasts)
            forecasts = pd.concat([forecasts, zero_df], ignore_index=True)
        
        # Calcular métricas para Fase 1
        if phase == 'phase1' and base_data is not None:
            self.calculate_phase1_errors(forecasts, base_data, granularity)
        
        # Registrar uso de recursos
        final_memory = process.memory_info().rss / 1024 / 1024
        self.reporter.stats['memory_usage'].append(final_memory - initial_memory)
        self.reporter.stats['cpu_usage'].append(process.cpu_percent(interval=0.1))
        self.reporter.stats['processing_times'].append(train_time + forecast_time)
        
        # Guardar datos para generación de PDFs
        if hasattr(self, 'series_data_for_plots'):
            # Para Fase 1, necesitamos incluir TODOS los datos históricos (hasta 2019-12)
            if phase == 'phase1':
                # Obtener datos completos hasta 2019-12
                df_full = df[df['periodo'] <= pd.to_datetime('2019-12-31')].copy()
                if granularity == 'product':
                    df_full['unique_id'] = df_full['product_id']
                else:
                    df_full['unique_id'] = (
                        df_full['product_id'].astype(str) + "_" + 
                        df_full['customer_id'].astype(str)
                    )
                
                sf_input_for_plots = df_full.rename(columns={
                    'periodo': 'ds',
                    'tn': 'y'
                })[['unique_id', 'ds', 'y']]
            else:
                # Para Fase 2, usar solo datos hasta el periodo de entrenamiento
                if apply_smoothing and 'tn_original' in df_train.columns:
                    sf_input_for_plots = df_train.rename(columns={
                        'periodo': 'ds',
                        'tn_original': 'y'  # Usar valores originales para los gráficos
                    })[['unique_id', 'ds', 'y']]
                else:
                    sf_input_for_plots = sf_input_filtered
                
            self.series_data_for_plots[granularity] = {
                'historical': sf_input_for_plots,
                'forecasts': forecasts,
                'zero_series': zero_series,
                'train_end': train_end_date
            }
        
        return forecasts
    
    def save_forecasts_with_metadata(
        self, 
        forecasts: pd.DataFrame, 
        target_date: str,
        filename: str,
        phase: str,
        granularity: str
    ):
        """Guarda pronósticos con metadatos adicionales"""
        target = pd.to_datetime(target_date)
        df_save = forecasts[forecasts['ds'] == target].copy()
        
        # Agregar metadatos
        df_save['forecast_generated'] = datetime.now().isoformat()
        df_save['phase'] = phase
        df_save['model'] = 'AutoARIMA'
        
        # Marcar series que tenían todos ceros
        df_save['all_zeros_series'] = df_save['unique_id'].isin(self.zero_series_cache.keys())
        
        # Reformatear según granularidad
        if granularity == 'product_customer':
            df_save[['product_id', 'customer_id']] = df_save['unique_id'].str.split('_', expand=True)
            columns_order = ['product_id', 'customer_id', 'forecast', 'all_zeros_series', 
                           'forecast_generated', 'phase', 'model']
        else:
            df_save = df_save.rename(columns={'unique_id': 'product_id'})
            columns_order = ['product_id', 'forecast', 'all_zeros_series', 
                           'forecast_generated', 'phase', 'model']
        
        # Renombrar forecast a tn y actualizar columns_order
        df_save = df_save.rename(columns={'forecast': 'tn'})
        columns_order = [col if col != 'forecast' else 'tn' for col in columns_order]
        
        # Guardar CSV
        df_save[columns_order].to_csv(filename, index=False)
        self.reporter.logger.info(f"Pronósticos guardados en: {filename}")
        
        # Para estrategia 2 (producto-cliente), generar CSV agregado por producto
        if granularity == 'product_customer' and phase == 'phase2':
            self.save_aggregated_by_product(df_save, filename.replace('.csv', '_aggregated_by_product.csv'))
        
        # Estadísticas
        n_zero = df_save['all_zeros_series'].sum()
        n_total = len(df_save)
        self.reporter.logger.info(
            f"Total pronósticos: {n_total} "
            f"(incluyendo {n_zero} series con todos ceros)"
        )
    
    def save_aggregated_by_product(self, df: pd.DataFrame, filename: str):
        """Guarda pronósticos agregados por producto (suma de todos los clientes)"""
        self.reporter.logger.info("Generando agregación por producto...")
        
        # Agrupar por producto y sumar
        df_agg = df.groupby('product_id')['tn'].sum().reset_index()
        df_agg['forecast_generated'] = datetime.now().isoformat()
        df_agg['aggregation'] = 'sum_all_customers'
        
        # Guardar
        df_agg.to_csv(filename, index=False)
        self.reporter.logger.info(f"Pronósticos agregados por producto guardados en: {filename}")
        
        # Mostrar resumen
        self.reporter.logger.info(f"Total productos: {len(df_agg)}")
        self.reporter.logger.info(f"Pronóstico total (todas las toneladas): {df_agg['tn'].sum():.2f}")
    
    def generate_pdf_plots(
        self, 
        phase: str, 
        granularity: str,
        max_plots_per_pdf: Optional[int] = None
    ):
        """Genera PDFs con gráficos de series temporales y pronósticos"""
        
        # Obtener porcentaje de gráficos a generar desde configuración
        plot_percentages = self.config.get('execution', {}).get('plot_percentages', {})
        plot_percentage = plot_percentages.get(granularity, 1.0)  # Default 100%
        
        self.reporter.logger.info(f"\nGenerando PDFs para {phase} - {granularity}...")
        self.reporter.logger.info(f"Porcentaje de gráficos a generar: {plot_percentage*100:.1f}%")
        
        # Obtener datos guardados
        if not hasattr(self, 'series_data_for_plots') or granularity not in self.series_data_for_plots:
            self.reporter.logger.warning("No hay datos disponibles para generar gráficos")
            return
        
        data = self.series_data_for_plots[granularity]
        historical = data['historical']
        forecasts = data['forecasts']
        zero_series = data['zero_series']
        train_end = data.get('train_end', pd.to_datetime('2019-10-31'))
        
        # Directorio para PDFs
        pdf_dir = os.path.join(self.reporter.log_dir, 'plots')
        os.makedirs(pdf_dir, exist_ok=True)
        
        # Nombre del archivo PDF
        pdf_filename = os.path.join(pdf_dir, f"{phase}_{granularity}_plots.pdf")
        
        # Series únicas (excluyendo las de todos ceros)
        unique_series = historical['unique_id'].unique()
        unique_series = [s for s in unique_series if s not in zero_series]
        
        if len(unique_series) == 0:
            self.reporter.logger.warning("No hay series válidas para graficar")
            return
        
        # Aplicar porcentaje de gráficos a generar
        total_series = len(unique_series)
        n_plots_to_generate = max(1, int(total_series * plot_percentage))  # Al menos 1 gráfico
        
        if plot_percentage < 1.0:
            # Tomar una muestra aleatoria para representatividad
            import random
            random.seed(42)  # Seed fijo para reproducibilidad
            unique_series = random.sample(list(unique_series), n_plots_to_generate)
            self.reporter.logger.info(f"Generando muestra de {n_plots_to_generate} gráficos de {total_series} series ({plot_percentage*100:.1f}%)")
        else:
            self.reporter.logger.info(f"Generando gráficos para todas las {total_series} series válidas")
        
        # Aplicar límite adicional si se especifica max_plots_per_pdf
        if max_plots_per_pdf is not None and len(unique_series) > max_plots_per_pdf:
            self.reporter.logger.info(f"Aplicando límite adicional: {max_plots_per_pdf} gráficos máximo")
            unique_series = unique_series[:max_plots_per_pdf]
        
        with PdfPages(pdf_filename) as pdf:
            # Configurar estilo
            try:
                plt.style.use('seaborn-v0_8-darkgrid')
            except:
                plt.style.use('default')  # Fallback al estilo por defecto
            
            for i, series_id in enumerate(unique_series):
                if i % 10 == 0:
                    self.reporter.logger.info(f"Generando gráfico {i+1}/{len(unique_series)}...")
                
                # Datos históricos de la serie
                series_hist = historical[historical['unique_id'] == series_id].sort_values('ds')
                
                # Pronósticos de la serie
                series_forecast = forecasts[forecasts['unique_id'] == series_id].sort_values('ds')
                
                # Crear figura
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Si es Fase 1, separar datos de entrenamiento vs reales
                if phase == 'phase1':
                    # Datos de entrenamiento (hasta train_end)
                    train_data = series_hist[series_hist['ds'] <= train_end]
                    # Datos posteriores al entrenamiento (nov y dic 2019)
                    test_data = series_hist[series_hist['ds'] > train_end]
                    
                    # Graficar serie de entrenamiento
                    ax.plot(train_data['ds'], train_data['y'], 
                           'b-', linewidth=1.5, label='Histórico (entrenamiento)')
                    
                    # Graficar valores reales post-entrenamiento
                    if not test_data.empty:
                        ax.plot(test_data['ds'], test_data['y'], 
                               'g-', linewidth=1.5, label='Valores reales')
                        ax.plot(test_data['ds'], test_data['y'], 
                               'go', markersize=8)
                        
                        # Línea conectando último punto de entrenamiento con primer punto real
                        if not train_data.empty:
                            last_train = train_data.iloc[-1]
                            first_test = test_data.iloc[0]
                            ax.plot([last_train['ds'], first_test['ds']], 
                                   [last_train['y'], first_test['y']], 
                                   'g--', linewidth=1, alpha=0.5)
                    
                    # Graficar pronósticos
                    ax.plot(series_forecast['ds'], series_forecast['forecast'], 
                           'ro--', markersize=8, linewidth=1, label='Pronóstico')
                    
                    # Conectar pronóstico con valor real para visualizar error
                    forecast_dec = series_forecast[series_forecast['ds'] == pd.to_datetime('2019-12-01')]
                    real_dec = test_data[test_data['ds'] == pd.to_datetime('2019-12-01')] if not test_data.empty else pd.DataFrame()
                    
                    if not forecast_dec.empty and not real_dec.empty:
                        # Línea de error
                        ax.plot([forecast_dec['ds'].iloc[0], real_dec['ds'].iloc[0]], 
                               [forecast_dec['forecast'].iloc[0], real_dec['y'].iloc[0]], 
                               'r:', linewidth=2, alpha=0.7)
                        
                        # Anotar el error
                        error = abs(real_dec['y'].iloc[0] - forecast_dec['forecast'].iloc[0])
                        mid_y = (real_dec['y'].iloc[0] + forecast_dec['forecast'].iloc[0]) / 2
                        ax.annotate(f'Error: {error:.1f}', 
                                   xy=(forecast_dec['ds'].iloc[0], mid_y),
                                   xytext=(10, 0), textcoords='offset points',
                                   fontsize=9, color='red', alpha=0.7)
                else:
                    # Para Fase 2, graficar todo como histórico
                    ax.plot(series_hist['ds'], series_hist['y'], 
                           'b-', linewidth=1.5, label='Histórico')
                    
                    # Graficar pronósticos
                    ax.plot(series_forecast['ds'], series_forecast['forecast'], 
                           'ro--', markersize=8, linewidth=1, label='Pronóstico')
                
                # Título y etiquetas
                if granularity == 'product':
                    title = f'Producto: {series_id}'
                else:
                    parts = series_id.split('_')
                    title = f'Producto: {parts[0]} - Cliente: {parts[1]}'
                
                ax.set_title(title, fontsize=14, fontweight='bold')
                ax.set_xlabel('Periodo', fontsize=12)
                ax.set_ylabel('Toneladas (tn)', fontsize=12)
                
                # Formato de fechas
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
                plt.xticks(rotation=45)
                
                # Leyenda
                ax.legend(loc='upper left')
                
                # Grid
                ax.grid(True, alpha=0.3)
                
                # Ajustar layout
                plt.tight_layout()
                
                # Guardar en PDF
                pdf.savefig(fig)
                plt.close(fig)
            
            # Página de resumen con métricas si es Fase 1
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.axis('off')
            
            summary_text = f"""
            RESUMEN DE PRONÓSTICOS
            
            Fase: {phase}
            Granularidad: {granularity}
            Fecha de generación: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            
            Series graficadas: {len(unique_series)}
            Series con todos ceros (excluidas): {len(zero_series)}
            
            Configuración del modelo:
            - Modelo: AutoARIMA
            - Longitud de temporada: {self.config['model']['season_length']}
            - Series mínimas requeridas: {self.config['model']['min_series_length']} periodos
            - Suavizado: {self.config.get('smoothing', {}).get('enabled', True)}
            - Método de suavizado: {self.config.get('smoothing', {}).get('method', 'exponential')}
            - Tiempo de suavizado: {self.reporter.stats.get('smoothing_time', 0):.2f}s
            """
            
            # Agregar métricas si es Fase 1
            if phase == 'phase1' and self.reporter.stats.get('phase1_metrics'):
                metrics = self.reporter.stats['phase1_metrics']
                summary_text += f"""
            
            MÉTRICAS DE ERROR (Comparación con 2019-12):
            - Error Total: {metrics['total_error']:.2f} toneladas
            - MAE: {metrics['mae']:.4f}
            - RMSE: {metrics['rmse']:.4f}
            - WAPE Modificado: {metrics['wape_modificado']:.4f}
            - Series comparadas: {metrics['n_comparisons']}
            """
            
            ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, 
                   fontsize=12, verticalalignment='top', fontfamily='monospace')
            
            pdf.savefig(fig)
            plt.close(fig)
        
        self.reporter.logger.info(f"PDF generado: {pdf_filename}")
    
    def run_complete_pipeline(self):
        """Ejecuta pipeline completo con reporting"""
        start_time = datetime.now()
        
        print("\n" + "="*80)
        print("SISTEMA DE PRONÓSTICO DE VENTAS - INICIANDO")
        print("="*80)
        print(f"Hora de inicio: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Inicializar almacenamiento para plots
        self.series_data_for_plots = {}
        
        # Cargar datos
        self.load_all_data()
        base_data = self._prepare_base_data()
        
        # Diccionario para almacenar todos los reportes
        all_reports = {}
        
        # Obtener configuración de suavizado
        apply_smoothing = self.config.get('smoothing', {}).get('enabled', True)
        
        # Obtener configuración de ejecución
        execution_config = self.config.get('execution', {})
        enabled_granularities = execution_config.get('granularities', ['product', 'product_customer'])
        enabled_phases = execution_config.get('phases', ['phase1', 'phase2'])
        
        print(f"\nConfiguración de ejecución:")
        print(f"  - Estrategias habilitadas: {enabled_granularities}")
        print(f"  - Fases habilitadas: {enabled_phases}")
        print(f"  - Suavizado: {'Activado' if apply_smoothing else 'Desactivado'}")
        
        # Ejecutar solo las combinaciones configuradas
        for phase_name, phase_config in self.config['phases'].items():
            # Filtrar solo fases habilitadas
            if phase_name not in enabled_phases:
                print(f"\n⏭️  Saltando {phase_name} (no habilitado en configuración)")
                continue
                
            for granularity in enabled_granularities:
                
                print(f"\n{'='*50}")
                print(f"Procesando: {phase_name} - {granularity}")
                print('='*50)
                
                # Preparar series temporales
                df_series = self._prepare_time_series(
                    base_data,
                    self.config['dates']['start'],
                    phase_config['data_end'],
                    granularity
                )
                
                # Entrenar y pronosticar con monitoreo
                forecasts = self.train_and_forecast_with_monitoring(
                    df_series,
                    phase_config['train_end'],
                    phase_config['horizon'],
                    granularity,
                    phase=phase_name,
                    base_data=base_data,
                    apply_smoothing=apply_smoothing
                )
                
                if not forecasts.empty:
                    # Guardar pronósticos
                    output_file = f"forecast_{granularity}_{phase_name}.csv"
                    self.save_forecasts_with_metadata(
                        forecasts,
                        phase_config['target_date'],
                        output_file,
                        phase_name,
                        granularity
                    )
                    
                    # Generar PDFs
                    try:
                        self.generate_pdf_plots(phase_name, granularity)
                    except Exception as e:
                        self.reporter.logger.error(f"Error generando PDFs: {str(e)}")
                
                # Generar y guardar reporte
                report = self.reporter.generate_summary_report(phase_name, granularity)
                self.reporter.save_report(report)
                self.reporter.print_summary(report)
                
                all_reports[f"{phase_name}_{granularity}"] = report
        
        # Resumen final
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()
        
        print("\n" + "="*80)
        print("PROCESAMIENTO COMPLETO")
        print("="*80)
        print(f"Hora de finalización: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Duración total: {total_duration:.2f} segundos ({total_duration/60:.1f} minutos)")
        
        # Guardar reporte consolidado
        consolidated_report = {
            'execution_summary': {
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'total_duration_seconds': total_duration,
                'system_info': self.reporter.system_info,
                'smoothing_config': self.config.get('smoothing', {})
            },
            'phase_reports': {k: asdict(v) for k, v in all_reports.items()}
        }
        
        report_path = os.path.join(
            self.reporter.log_dir, 
            f"consolidated_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        with open(report_path, 'w') as f:
            json.dump(consolidated_report, f, indent=2, default=str)
        
        print(f"\nReporte consolidado guardado en: {report_path}")
        print("\n=== ARCHIVOS GENERADOS ===")
        print("CSVs de pronósticos:")
        print("  - forecast_product_phase1.csv")
        print("  - forecast_product_phase2.csv") 
        print("  - forecast_product_customer_phase1.csv")
        print("  - forecast_product_customer_phase2.csv")
        print("  - forecast_product_customer_phase2_aggregated_by_product.csv (suma por producto)")
        print("\nPDFs con gráficos en: forecast_logs/plots/")
        print("\nReportes JSON en: forecast_logs/")
        print("="*80 + "\n")
        
        return all_reports
    
    def _prepare_base_data(self) -> pd.DataFrame:
        """Prepara y une los datos base"""
        self.reporter.logger.info("\n=== PREPARACIÓN DE DATOS BASE ===")
        
        # Realizar joins
        df = pd.merge(
            self.data['sellin'], 
            self.data['productos'], 
            on='product_id', 
            how='left'
        )
        df = pd.merge(
            df, 
            self.data['stocks'], 
            on=['periodo', 'product_id'], 
            how='left'
        )
        
        # Convertir periodo a datetime
        df['periodo'] = pd.to_datetime(df['periodo'], format='%Y%m')
        
        # Filtrar productos a predecir
        productos_predecir = self.data['productos_a_predecir']['product_id'].unique()
        df = df[df['product_id'].isin(productos_predecir)]
        
        self.reporter.logger.info(f"Datos base preparados: {df.shape}")
        return df
    
    def _prepare_time_series(
        self, 
        df: pd.DataFrame,
        start_date: str,
        end_date: str,
        granularity: str = 'product'
    ) -> pd.DataFrame:
        """Prepara series temporales completas"""
        self.reporter.logger.info(f"\nPreparando series temporales ({granularity})")
        
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        # Filtrar rango
        df_filtered = df[(df['periodo'] >= start) & (df['periodo'] <= end)]
        
        # Agrupar según granularidad
        if granularity == 'product':
            group_cols = ['product_id', 'periodo']
            id_cols = ['product_id']
        else:
            group_cols = ['product_id', 'customer_id', 'periodo']
            id_cols = ['product_id', 'customer_id']
        
        df_agg = df_filtered.groupby(group_cols)['tn'].sum().reset_index()
        
        # Completar serie con todos los meses
        all_months = pd.date_range(start=start, end=end, freq='MS')
        
        if granularity == 'product':
            unique_ids = df_agg['product_id'].unique()
            all_combinations = pd.MultiIndex.from_product(
                [unique_ids, all_months],
                names=['product_id', 'periodo']
            ).to_frame(index=False)
        else:
            unique_pairs = df_agg[['product_id', 'customer_id']].drop_duplicates()
            all_combinations = unique_pairs.assign(key=1).merge(
                pd.DataFrame({'periodo': all_months, 'key': 1}), 
                on='key'
            ).drop('key', axis=1)
        
        # Merge y rellenar
        df_complete = pd.merge(
            all_combinations, 
            df_agg, 
            on=group_cols, 
            how='left'
        )
        df_complete['tn'] = df_complete['tn'].fillna(0)
        
        self.reporter.logger.info(f"Series preparadas: {df_complete[id_cols].drop_duplicates().shape[0]} series únicas")
        
        return df_complete


# Configuración actualizada con parámetros de optimización y suavizado
CONFIG = {
    'execution': {
        # Controla qué estrategias ejecutar
        #'granularities': ['product', 'product_customer'],  # ['product', 'product_customer'] para ambas
        'granularities': ['product'],  # ['product', 'product_customer'] para ambas
        # Controla qué fases ejecutar
        'phases': ['phase1', 'phase2'],  # ['phase1'] solo validación, ['phase2'] solo producción
        # Porcentaje de gráficos a generar por estrategia (0.0 a 1.0)
        'plot_percentages': {
            'product': 1.0,        # 100% - Estrategia 1: Mostrar todos los gráficos (~300-400 series)
            'product_customer': 0.001  # 0.1% - Estrategia 2: Solo una muestra (~50-100 gráficos de ~50,000 series)
        }
    },
    'files': {
        'sellin': {
            'local': 'sell-in.txt.gz',
            'url': 'https://storage.googleapis.com/open-courses/austral2025-af91/labo3v/sell-in.txt.gz',
            'read_args': {
                'sep': '\t', 
                'compression': 'gzip', 
                'dtype': {'periodo': str, 'customer_id': str, 'product_id': str}
            }
        },
        'productos': {
            'local': 'tb_productos.txt',
            'url': 'https://storage.googleapis.com/open-courses/austral2025-af91/labo3v/tb_productos.txt',
            'read_args': {'sep': '\t', 'dtype': {'product_id': str}}
        },
        'stocks': {
            'local': 'tb_stocks.txt',
            'url': 'https://storage.googleapis.com/open-courses/austral2025-af91/labo3v/tb_stocks.txt',
            'read_args': {'sep': '\t', 'dtype': {'periodo': str, 'product_id': str}}
        },
        'productos_a_predecir': {
            'local': 'product_id_apredecir201912.txt',
            'url': 'https://storage.googleapis.com/open-courses/austral2025-af91/labo3v/product_id_apredecir201912.txt',
            'read_args': {'sep': '\t', 'dtype': {'product_id': str}}
        }
    },
    'dates': {
        'start': '2017-01-01',
    },
    'phases': {
        'phase1': {
            'train_end': '2019-10-31',
            'data_end': '2019-12-31',
            'target_date': '2019-12-01',
            'horizon': 2
        },
        'phase2': {
            'train_end': '2019-12-31', 
            'data_end': '2019-12-31',
            'target_date': '2020-02-01',
            'horizon': 2
        }
    },
    'model': {
        'season_length': 12,
        'min_series_length': 24,
        'autoarima_params': {
            'max_p': 3,
            'max_q': 3,
            'max_P': 2,
            'max_Q': 2,
            'stepwise': True,
            'approximation': True
        }
    },
    'smoothing': {
        'enabled': False,  # Activar/desactivar suavizado
        'method': 'exponential',  # 'exponential' o 'moving_average'
        'alpha': 0.3  # Factor de suavizado para método exponencial (0-1)
    }
}

# Script principal
if __name__ == "__main__":
    # Crear sistema con reporting
    system = ForecastSystemWithReporting(CONFIG)
    
    # Ejecutar pipeline completo
    reports = system.run_complete_pipeline()
    
    print("\nProceso completado exitosamente!")
    print(f"Revisa los logs y reportes en: {system.reporter.log_dir}/")

