import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from prophet import Prophet
from pmdarima import auto_arima
import warnings
import os
import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from numpy.polynomial.polynomial import Polynomial
import io
from sklearn.model_selection import TimeSeriesSplit

# Setup logging
warnings.filterwarnings("ignore")
logging.basicConfig(filename='app.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Set page configuration for wider layout
st.set_page_config(layout="wide", page_title="Property Price Forecast", page_icon="📈")

# Configuration
CONFIG = {
    'prophet_params': {
        'yearly_seasonality': True,
        'weekly_seasonality': False,
        'daily_seasonality': False,
        'changepoint_prior_scale': 0.05,
        'seasonality_mode': 'additive',
        'interval_width': 0.95
    },
    'arima_params': {
        'seasonal': True,
        'm': 12,
        'stepwise': True,
        'suppress_warnings': True,
        'start_p': 1,
        'start_q': 1,
        'max_p': 20,
        'max_q': 20,
        'max_d': 2,
        'start_P': 0,
        'start_Q': 0,
        'max_P': 2,
        'max_Q': 2,
        'max_order': 40,
        'information_criterion': 'aic',
        'seasonal_test': 'ch'
    },
    'default_periods': 12
}

# 1. Data Preparation
def load_and_preprocess(data_path=None):
    """Load and preprocess Excel data with additional lags."""
    if data_path is None:
        data_path = os.getenv('DATA_PATH', 'Open Transaction Data 2021-2024.xlsx')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    df = pd.read_excel(data_path)
    logging.debug(f"Raw data loaded: {df.shape} rows, columns: {df.columns.tolist()}")
    
    required_columns = ['Price Per Sq Ft', 'MHPI %', 'Transaction Year', 'Month', 'District', 'State', 'Property Type']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")
    
    df['Property Type'] = df['Property Type'].str.replace(r'[^\w\s]', '_', regex=True)
    df['Price Per Sq Ft'] = (df['Price Per Sq Ft'].astype(str)
                             .str.replace(r'[^\d.]', '', regex=True)
                             .replace('', np.nan)
                             .astype(float))
    df = df[df['Price Per Sq Ft'] > 0]
    
    q1 = df['Price Per Sq Ft'].quantile(0.25)
    q3 = df['Price Per Sq Ft'].quantile(0.75)
    iqr = q3 - q1
    df = df[(df['Price Per Sq Ft'] >= q1 - 4 * iqr) & (df['Price Per Sq Ft'] <= q3 + 4 * iqr)]
    logging.debug(f"After outlier removal: {df.shape} rows")
    
    df['Transaction Date'] = pd.to_datetime(df['Transaction Year'].astype(str) + '-' + 
                                          df['Month'].astype(str).str.zfill(2) + '-01', errors='coerce')
    df = df.dropna(subset=['Transaction Date', 'Price Per Sq Ft', 'MHPI %'])
    
    duplicates = df[df.duplicated(subset=['Transaction Date', 'District', 'State', 'Property Type'], keep=False)]
    if not duplicates.empty:
        logging.warning(f"Duplicate entries found: {len(duplicates)} rows")
        df = df.drop_duplicates(subset=['Transaction Date', 'District', 'State', 'Property Type'], keep='last')
    
    df = df.sort_values('Transaction Date')
    df['Price_Lag1'] = df['Price Per Sq Ft'].shift(1)
    df['Price_Lag2'] = df['Price Per Sq Ft'].shift(2)
    df['Price_Lag3'] = df['Price Per Sq Ft'].shift(3)
    df = df.dropna()
    
    logging.debug(f"Preprocessed data with lags: {df.shape} rows, unique dates: {df['Transaction Date'].nunique()}")
    logging.debug(f"Price Per Sq Ft stats: mean={df['Price Per Sq Ft'].mean():.2f}, std={df['Price Per Sq Ft'].std():.2f}")
    
    return df.sort_values('Transaction Date').reset_index(drop=True)

# 2. Forecasting Model
class PropertyForecaster:
    def __init__(self, data):
        self.data = data
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    def __del__(self):
        self.executor.shutdown()
    
    def get_subset(self, geo_level, geo_value, prop_type):
        """Get data subset for a geographic level (district or state) and property type with rolling statistics."""
        subset = self.data[
            (self.data[geo_level] == geo_value) & 
            (self.data['Property Type'] == prop_type)
        ]
        if len(subset) < 3:
            logging.error(f"Insufficient data for {geo_level}={geo_value}, {prop_type}: {len(subset)} rows")
            raise ValueError(f"Insufficient data: {len(subset)} rows, need at least 3")
        
        subset = subset.sort_values('Transaction Date')
        logging.debug(f"Subset for {geo_level}={geo_value}, {prop_type}: {len(subset)} rows, dates: {subset['Transaction Date'].tolist()}")
        
        subset['Price_Rolling_Mean'] = subset['Price Per Sq Ft'].rolling(window=3, min_periods=1).mean()
        subset['Price_Rolling_Std'] = subset['Price Per Sq Ft'].rolling(window=3, min_periods=1).std().fillna(0)
        subset['Price_Upper'] = subset['Price_Rolling_Mean'] + 1.5 * subset['Price_Rolling_Std']
        subset['Price_Lower'] = subset['Price_Rolling_Mean'] - 1.5 * subset['Price_Rolling_Std']
        
        if subset['Price_Rolling_Std'].isna().all() or (subset['Price_Rolling_Std'] == 0).all():
            logging.warning(f"Rolling std failed for {geo_level}={geo_value}, {prop_type}, using min/max range")
            subset['Price_Upper'] = subset['Price Per Sq Ft'].rolling(window=3, min_periods=1).max()
            subset['Price_Lower'] = subset['Price Per Sq Ft'].rolling(window=3, min_periods=1).min()
        
        # Fill NaN values in rolling statistics to avoid plot gaps
        subset['Price_Rolling_Mean'] = subset['Price_Rolling_Mean'].fillna(method='ffill').fillna(0)
        subset['Price_Upper'] = subset['Price_Upper'].fillna(method='ffill').fillna(0)
        subset['Price_Lower'] = subset['Price_Lower'].fillna(method='ffill').fillna(0)
        
        logging.debug(f"Subset stats: mean_price={subset['Price Per Sq Ft'].mean():.2f}, std_price={subset['Price Per Sq Ft'].std():.2f}")
        return subset
    
    def train_prophet(self, subset, geo_level, geo_value, prop_type):
        """Train Prophet model with optimized parameters and cross-validation."""
        train_df = subset[['Transaction Date', 'Price Per Sq Ft', 'MHPI %', 'Price_Lag1', 'Price_Lag2', 'Price_Lag3']].copy()
        train_df.columns = ['ds', 'y', 'MHPI', 'lag1', 'lag2', 'lag3']
        if train_df.empty or train_df['y'].isna().all() or train_df['ds'].isna().all():
            logging.error(f"No valid data for Prophet training for {geo_level}={geo_value}, {prop_type}")
            raise ValueError("No valid data for Prophet training")
        
        tscv = TimeSeriesSplit(n_splits=3)
        val_rmse_scores = []
        
        for train_idx, test_idx in tscv.split(train_df):
            train_data = train_df.iloc[train_idx]
            test_data = train_df.iloc[test_idx]
            
            model_val = Prophet(**CONFIG['prophet_params'])
            for col in ['MHPI', 'lag1', 'lag2', 'lag3']:
                model_val.add_regressor(col)
            try:
                model_val.fit(train_data)
                pred = model_val.predict(test_data[['ds', 'MHPI', 'lag1', 'lag2', 'lag3']])
                rmse = mean_squared_error(test_data['y'], pred['yhat'], squared=False)
                val_rmse_scores.append(rmse)
            except Exception as e:
                logging.error(f"Prophet CV error for {geo_level}={geo_value}, {prop_type}: {str(e)}")
                val_rmse_scores.append(np.nan)
        
        rmse = np.nanmean(val_rmse_scores)
        logging.debug(f"Prophet cross-validated RMSE for {geo_level}={geo_value}, {prop_type}: {rmse:.2f}")
        
        model_final = Prophet(**CONFIG['prophet_params'])
        for col in ['MHPI', 'lag1', 'lag2', 'lag3']:
            model_final.add_regressor(col)
        try:
            model_final.fit(train_df)
        except Exception as e:
            logging.error(f"Prophet final fit error for {geo_level}={geo_value}, {prop_type}: {str(e)}")
            raise
        return model_final, rmse
    
    def train_arima(self, subset, geo_level, geo_value, prop_type):
        """Train ARIMA model with enhanced parameters and dynamic tuning."""
        train_df = subset[['Transaction Date', 'Price Per Sq Ft', 'MHPI %', 'Price_Lag1', 'Price_Lag2', 'Price_Lag3']].copy()
        train_df.columns = ['Transaction Date', 'y', 'MHPI', 'lag1', 'lag2', 'lag3']
        train_df['Transaction Date'] = pd.to_datetime(train_df['Transaction Date'])
        y = train_df['y'].dropna()
        
        from statsmodels.tsa.stattools import adfuller
        adf_result = adfuller(y)
        d = 1 if adf_result[1] > 0.05 else 0
        if d == 0 and len(y) > 12:
            y_diff = y.diff().dropna()
            adf_diff = adfuller(y_diff)
            if adf_diff[1] > 0.05:
                d = 2
        if len(y) > 24 and adf_result[1] > 0.01:
            d = max(1, d)
        
        exog = train_df[['MHPI', 'lag1', 'lag2', 'lag3']].dropna()
        if len(exog) != len(y):
            logging.warning(f"Exog data mismatch for {geo_level}={geo_value}, {prop_type}: y={len(y)}, exog={len(exog)}")
            exog = exog.reindex(y.index, method='ffill')
        
        try:
            model = auto_arima(
                y,
                exogenous=exog,
                seasonal=True,
                m=12,
                stepwise=True,
                suppress_warnings=True,
                start_p=1,
                start_q=1,
                max_p=20,
                max_q=20,
                d=d,
                max_d=2,
                start_P=0,
                start_Q=0,
                max_P=2,
                max_Q=2,
                max_order=40,
                maxiter=50,
                information_criterion='aic',
                trace=True,
                error_action='ignore',
                seasonal_test='ch'
            )
        except Exception as e:
            logging.error(f"ARIMA fit error for {geo_level}={geo_value}, {prop_type}: {str(e)}")
            raise
        
        train_size = max(3, int(0.7 * len(y)))
        train_y, test_y = y.iloc[:train_size], y.iloc[train_size:]
        train_exog, test_exog = exog.iloc[:train_size], exog.iloc[train_size:]
        
        if len(test_y) > 0:
            try:
                model.fit(train_y, exogenous=train_exog)
                pred = model.predict(n_periods=len(test_y), exogenous=test_exog)
                rmse = mean_squared_error(test_y, pred, squared=False)
            except Exception as e:
                logging.error(f"ARIMA validation error for {geo_level}={geo_value}, {prop_type}: {str(e)}")
                rmse = np.nan
        else:
            rmse = np.nan
        logging.debug(f"ARIMA validation RMSE for {geo_level}={geo_value}, {prop_type}: {rmse:.2f}, Order: {model.order}, Seasonal Order: {model.seasonal_order}")
        
        model.fit(y, exogenous=exog)
        return model, rmse
    
    def predict(self, geo_level, geo_value, prop_type, periods=CONFIG['default_periods']):
        try:
            subset = self.get_subset(geo_level, geo_value, prop_type)
            logging.debug(f"Subset columns: {subset.columns.tolist()}")
            last_date = subset['Transaction Date'].max()
            future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=periods, freq='MS')
            
            trend_mhpi = Polynomial.fit(range(len(subset)), subset['MHPI %'], deg=2)
            future_mhpi = [max(0, trend_mhpi(i + len(subset))) for i in range(periods)]
            last_lag1, last_lag2, last_lag3 = subset['Price_Lag1'].iloc[-1], subset['Price_Lag2'].iloc[-1], subset['Price_Lag3'].iloc[-1]
            future_lag1 = [last_lag1] * periods
            future_lag2 = [last_lag2] * periods
            future_lag3 = [last_lag3] * periods
            
            future_df = pd.DataFrame({'ds': future_dates, 'MHPI': future_mhpi, 'lag1': future_lag1, 'lag2': future_lag2, 'lag3': future_lag3})
            future = self.executor.submit(self._run_prediction, subset, future_df, periods, geo_level, geo_value, prop_type)
            forecast_result = future.result(timeout=60)  # Increased timeout to 60 seconds
        except TimeoutError:
            logging.error(f"Prediction timed out for {geo_level}={geo_value}, {prop_type}")
            raise
        except Exception as e:
            logging.error(f"Prediction error for {geo_level}={geo_value}, {prop_type}: {str(e)}")
            raise
        
        metrics = forecast_result['metrics']
        prophet_rmse = metrics['prophet_val_rmse'] if not np.isnan(metrics['prophet_val_rmse']) else float('inf')
        arima_rmse = metrics['arima_val_rmse'] if not np.isnan(metrics['arima_val_rmse']) else float('inf')
        total_rmse = prophet_rmse + arima_rmse
        prophet_weight = (1 - prophet_rmse / total_rmse) if total_rmse > 0 else 0.5
        arima_weight = (1 - arima_rmse / total_rmse) if total_rmse > 0 else 0.5
        forecast_result['future']['Ensemble'] = prophet_weight * forecast_result['future']['Prophet'] + arima_weight * forecast_result['future']['ARIMA']
        return forecast_result
    
    def _run_prediction(self, subset, future_df, periods, geo_level, geo_value, prop_type):
        prophet_model, prophet_val_rmse = self.train_prophet(subset, geo_level, geo_value, prop_type)
        arima_model, arima_val_rmse = self.train_arima(subset, geo_level, geo_value, prop_type)
        
        train_df = subset[['Transaction Date', 'Price Per Sq Ft', 'MHPI %', 'Price_Lag1', 'Price_Lag2', 'Price_Lag3']].copy()
        train_df.columns = ['ds', 'y', 'MHPI', 'lag1', 'lag2', 'lag3']
        
        prophet_pred = prophet_model.predict(train_df)
        rmse_prophet = mean_squared_error(train_df['y'], prophet_pred['yhat'], squared=False)
        mae_prophet = mean_absolute_error(train_df['y'], prophet_pred['yhat'])
        
        exog = train_df[['MHPI', 'lag1', 'lag2', 'lag3']]
        arima_pred = arima_model.predict(n_periods=len(subset), exogenous=exog)
        rmse_arima = mean_squared_error(subset['Price Per Sq Ft'], arima_pred, squared=False)
        mae_arima = mean_absolute_error(subset['Price Per Sq Ft'], arima_pred)
        
        prophet_forecast = prophet_model.predict(future_df)
        arima_forecast = arima_model.predict(n_periods=periods, exogenous=future_df[['MHPI', 'lag1', 'lag2', 'lag3']])
        
        future_length = min(len(future_df['ds']), len(prophet_forecast['yhat']), len(arima_forecast))
        forecast_future_df = pd.DataFrame({
            'Date': future_df['ds'].values[:future_length],
            'Prophet': prophet_forecast['yhat'].values[:future_length],
            'ARIMA': arima_forecast[:future_length],
            'Ensemble': np.nan
        })
        
        # Ensure historical predictions match subset length
        historical_length = len(subset['Transaction Date'])
        if len(prophet_pred['yhat']) > historical_length:
            prophet_pred = prophet_pred.iloc[:historical_length]
        if len(arima_pred) > historical_length:
            arima_pred = arima_pred[:historical_length]
        
        forecast_historical_df = pd.DataFrame({
            'ProphetHistorical': prophet_pred['yhat'].values,
            'ARIMAHistorical': arima_pred
        })
        
        metrics = {
            'prophet_rmse': float(rmse_prophet),
            'prophet_mae': float(mae_prophet),
            'arima_rmse': float(rmse_arima),
            'arima_mae': float(mae_arima),
            'prophet_val_rmse': float(prophet_val_rmse) if not np.isnan(prophet_val_rmse) else np.nan,
            'arima_val_rmse': float(arima_val_rmse) if not np.isnan(arima_val_rmse) else np.nan
        }
        
        return {'future': forecast_future_df, 'historical': forecast_historical_df, 'metrics': metrics}

# 3. Streamlit App
def main():
    st.title("Property Price Forecast Dashboard")
    
    data_path = r'C:\Users\steffiephang\OneDrive - LBS Bina Holdings Sdn Bhd\Desktop\Steffie\ADHD_Project\AVM\Open Transaction Data 2021-2024.xlsx'
    try:
        data = load_and_preprocess(data_path)
    except Exception as e:
        logging.error(f"Data loading error: {str(e)}")
        st.error(f"⚠️ Data Error: {str(e)}. Please check the file and try again.")
        return
    
    forecaster = PropertyForecaster(data)
    
    st.sidebar.header("Filters")
    geo_level = st.sidebar.radio("Forecast by", ['District', 'State'])
    if geo_level == 'District':
        geo_value = st.sidebar.selectbox("Select District", sorted(data['District'].unique()), index=0)
    else:  # State
        geo_value = st.sidebar.selectbox("Select State", sorted(data['State'].unique()), index=0)
    prop_type = st.sidebar.selectbox("Select Property Type", sorted(data['Property Type'].unique()), index=0)
    periods = st.sidebar.slider("Forecast Periods", min_value=6, max_value=24, value=12)
    model_choice = st.sidebar.radio("Select Model", ['Ensemble', 'Prophet', 'ARIMA'])
    
    try:
        logging.info(f"Updating dashboard for {geo_level}={geo_value}, {prop_type} at {datetime.now()}")
        
        historical = forecaster.get_subset(geo_level, geo_value, prop_type)
        forecast_result = forecaster.predict(geo_level, geo_value, prop_type, periods)
        
        forecast_future_df = forecast_result['future']
        forecast_historical_df = forecast_result['historical']
        metrics = forecast_result['metrics']
        
        # Create plot
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=historical['Transaction Date'],
            y=historical['Price Per Sq Ft'],
            mode='lines',
            name='Actual Price',
            line=dict(color='#9b59b6')
        ))
        
        fig.add_trace(go.Scatter(
            x=historical['Transaction Date'],
            y=historical['Price_Upper'],
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        fig.add_trace(go.Scatter(
            x=historical['Transaction Date'],
            y=historical['Price_Lower'],
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(52, 152, 219, 0.2)',
            name='Historical Range',
            hoverinfo='skip'
        ))
        
        fig.add_trace(go.Scatter(
            x=historical['Transaction Date'],
            y=historical['Price_Rolling_Mean'],
            mode='lines',
            name='Historical Mean',
            line=dict(color='#3498db')
        ))
        
        if len(historical['Transaction Date']) == len(forecast_historical_df['ProphetHistorical']):
            fig.add_trace(go.Scatter(
                x=historical['Transaction Date'],
                y=forecast_historical_df['ProphetHistorical'],
                mode='lines',
                name='Prophet Fit',
                line=dict(color='#e74c3c', dash='dot')
            ))
        if len(historical['Transaction Date']) == len(forecast_historical_df['ARIMAHistorical']):
            fig.add_trace(go.Scatter(
                x=historical['Transaction Date'],
                y=forecast_historical_df['ARIMAHistorical'],
                mode='lines',
                name='ARIMA Fit',
                line=dict(color='#f39c12', dash='dot')
            ))
        
        y_data = forecast_future_df[model_choice] if model_choice in forecast_future_df.columns else forecast_future_df['Ensemble']
        fig.add_trace(go.Scatter(
            x=forecast_future_df['Date'],
            y=y_data,
            mode='lines',
            name=f'{model_choice} Forecast',
            line=dict(color='#2ecc71', dash='dash')
        ))
        
        fig.update_layout(
            title=f"{geo_value} - {prop_type} Property Price Forecast",
            xaxis_title='Date',
            yaxis_title='Price Per Sq Ft (RM)',
            template='plotly_white',
            xaxis=dict(
                gridcolor='lightgrey',
                tickformat='%b %Y',
                rangeslider_visible=True,  # Add slider for better navigation
                type='date'
            ),
            yaxis=dict(
                gridcolor='lightgrey',
                tickprefix='RM',
                automargin=True
            ),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
            autosize=True,
            width=None,
            height=600,
            margin=dict(l=50, r=50, t=100, b=50)  # Adjust margins for better fit
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        if len(historical) > 3:
            prophet_residuals = historical['Price Per Sq Ft'] - forecast_historical_df['ProphetHistorical']
            st.line_chart(pd.DataFrame({'Date': historical['Transaction Date'], 'Prophet Residuals': prophet_residuals}))
        
        csv = forecast_future_df.to_csv(index=False)
        st.download_button(label="Download Forecast Data", data=csv, file_name=f"{geo_value}_{prop_type}_forecast.csv", mime="text/csv")
        
        st.subheader("Forecast Summary")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Next Month", f"RM{y_data.iloc[0]:.2f}", delta_color="normal")
        with col2:
            st.metric("6-Month Avg", f"RM{y_data.head(6).mean():.2f}", delta_color="normal")
        with col3:
            st.metric("12-Month", f"RM{y_data.iloc[-1]:.2f}", delta_color="normal")
        with col4:
            st.text("Model Metrics:")
            st.text(f"Prophet RMSE: {metrics['prophet_val_rmse']:.2f} (Val)" if not np.isnan(metrics['prophet_val_rmse']) else "Prophet RMSE: N/A")
            st.text(f"ARIMA RMSE: {metrics['arima_val_rmse']:.2f} (Val)" if not np.isnan(metrics['arima_val_rmse']) else "ARIMA RMSE: N/A")
    
    except ValueError as ve:
        logging.error(f"ValueError for {geo_level}={geo_value}, {prop_type}: {str(ve)}")
        st.error(f"⚠️ Data Error: {str(ve)}\nPlease select a different {geo_level.lower()} or property type.")
    except Exception as e:
        logging.error(f"Unexpected error for {geo_level}={geo_value}, {prop_type}: {str(e)}")
        st.error(f"⚠️ System Error: {str(e)}. Please try again or contact support.")

if __name__ == '__main__':
    main()