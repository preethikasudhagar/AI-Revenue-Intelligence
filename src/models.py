import os
import pickle
import pandas as pd
import numpy as np
from datetime import timedelta

# Try importing models, define fallbacks if packages are not yet installed
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

from sklearn.ensemble import GradientBoostingRegressor

# Simple fallback time series model in case Prophet is missing (e.g. during fresh setup)
class FallbackProphet:
    def __init__(self):
        self.daily_seasonality = {}
        self.weekly_seasonality = {}
        self.overall_mean = 0
        self.trend_slope = 0
        self.min_date = None

    def fit(self, df):
        # Expects 'ds' and 'y'
        df = df.copy()
        df['ds'] = pd.to_datetime(df['ds'])
        df = df.sort_values('ds').reset_index(drop=True)
        
        self.min_date = df['ds'].min()
        days_since_start = (df['ds'] - self.min_date).dt.days
        
        # Fit a simple linear trend
        if len(df) > 1:
            coef = np.polyfit(days_since_start, df['y'], 1)
            self.trend_slope = coef[0]
            self.overall_mean = coef[1]
        else:
            self.trend_slope = 0
            self.overall_mean = df['y'].mean() if len(df) > 0 else 0
            
        df['dayofweek'] = df['ds'].dt.dayofweek
        self.weekly_seasonality = df.groupby('dayofweek')['y'].mean().to_dict()
        df['month'] = df['ds'].dt.month
        self.monthly_seasonality = df.groupby('month')['y'].mean().to_dict()
        
        # Normalize seasonalities
        mean_val = df['y'].mean() if len(df) > 0 else 1.0
        if mean_val == 0: mean_val = 1.0
        self.weekly_seasonality = {k: v / mean_val for k, v in self.weekly_seasonality.items()}
        self.monthly_seasonality = {k: v / mean_val for k, v in self.monthly_seasonality.items()}

    def predict(self, df):
        df = df.copy()
        df['ds'] = pd.to_datetime(df['ds'])
        days_since_start = (df['ds'] - self.min_date).dt.days
        
        preds = []
        for i, row in df.iterrows():
            day_idx = days_since_start.iloc[i]
            dow = row['ds'].dayofweek
            month = row['ds'].month
            
            trend = self.overall_mean + self.trend_slope * day_idx
            w_factor = self.weekly_seasonality.get(dow, 1.0)
            m_factor = self.monthly_seasonality.get(month, 1.0)
            
            preds.append(max(0.0, trend * w_factor * m_factor))
            
        return pd.DataFrame({'ds': df['ds'], 'yhat': preds})

class RevenueForecastingEnsemble:
    def __init__(self, weights={'prophet': 0.3, 'xgb': 0.4, 'lgbm': 0.3}):
        self.weights = weights
        self.models = {}
        self.feature_cols = []
        self.target_name = None

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    def train(self, df_train, target_name, feature_cols):
        self.target_name = target_name
        self.feature_cols = feature_cols
        
        X = df_train[feature_cols]
        y = df_train[target_name]
        
        # 1. Train Prophet
        df_prophet = df_train[['Date', target_name]].rename(columns={'Date': 'ds', target_name: 'y'})
        if PROPHET_AVAILABLE:
            prophet_model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                interval_width=0.8
            )
            prophet_model.fit(df_prophet)
            self.models['prophet'] = prophet_model
        else:
            prophet_model = FallbackProphet()
            prophet_model.fit(df_prophet)
            self.models['prophet'] = prophet_model
            
        # 2. Train XGBoost
        if XGB_AVAILABLE:
            xgb_model = xgb.XGBRegressor(
                n_estimators=150,
                max_depth=5,
                learning_rate=0.05,
                random_state=42
            )
            xgb_model.fit(X, y)
            self.models['xgb'] = xgb_model
        
        # 3. Train LightGBM
        if LGBM_AVAILABLE:
            lgb_model = lgb.LGBMRegressor(
                n_estimators=150,
                max_depth=5,
                learning_rate=0.05,
                random_state=42,
                verbosity=-1
            )
            lgb_model.fit(X, y)
            self.models['lgbm'] = lgb_model
            
        # 4. Train Quantile Regressors (P10 and P90)
        # We fit standard scikit-learn Gradient Boosting Regressors with quantile loss
        qr_10 = GradientBoostingRegressor(loss='quantile', alpha=0.10, n_estimators=100, learning_rate=0.05, random_state=42)
        qr_90 = GradientBoostingRegressor(loss='quantile', alpha=0.90, n_estimators=100, learning_rate=0.05, random_state=42)
        
        qr_10.fit(X, y)
        qr_90.fit(X, y)
        
        self.models['qr_10'] = qr_10
        self.models['qr_90'] = qr_90
        
        print(f"Ensemble models successfully trained for target: {target_name}")

    def predict_point(self, X_row, date_t):
        """
        Generates point predictions from Prophet, XGBoost, and LightGBM, and blends them.
        """
        # Prophet prediction
        df_p_test = pd.DataFrame({'ds': [pd.to_datetime(date_t)]})
        if PROPHET_AVAILABLE:
            p_pred = self.models['prophet'].predict(df_p_test)['yhat'].values[0]
        else:
            p_pred = self.models['prophet'].predict(df_p_test)['yhat'].values[0]
            
        # ML Predictions
        xgb_pred = 0.0
        lgbm_pred = 0.0
        
        # If ML models are missing or unavailable, fallback to Prophet
        xgb_avail = XGB_AVAILABLE and 'xgb' in self.models
        lgbm_avail = LGBM_AVAILABLE and 'lgbm' in self.models
        
        if xgb_avail:
            xgb_pred = self.models['xgb'].predict(X_row)[0]
        if lgbm_avail:
            lgbm_pred = self.models['lgbm'].predict(X_row)[0]
            
        # Combine using weights
        if xgb_avail and lgbm_avail:
            blend = (self.weights['prophet'] * p_pred +
                     self.weights['xgb'] * xgb_pred +
                     self.weights['lgbm'] * lgbm_pred)
        elif xgb_avail:
            blend = 0.4 * p_pred + 0.6 * xgb_pred
        elif lgbm_avail:
            blend = 0.4 * p_pred + 0.6 * lgbm_pred
        else:
            blend = p_pred
            
        return max(0.0, blend)

    def predict_quantiles(self, X_row, point_prediction):
        """
        Predicts P10 and P90 intervals using Quantile Regression, bounded to ensure P10 <= P50 <= P90.
        """
        p10 = self.models['qr_10'].predict(X_row)[0]
        p90 = self.models['qr_90'].predict(X_row)[0]
        
        # Ensure logical bounds
        p10 = min(point_prediction, p10)
        p90 = max(point_prediction, p90)
        
        return max(0.0, p10), max(0.0, p90)
