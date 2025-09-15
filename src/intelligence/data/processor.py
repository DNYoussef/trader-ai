"""
Production-ready data processing pipeline
Comprehensive feature engineering for financial data
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer
import ta  # Technical Analysis library

logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Comprehensive data processing pipeline for financial data
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._get_default_config()
        self.scalers = {}
        self.imputers = {}
        self.feature_names = []

    def _get_default_config(self) -> Dict:
        """Default data processing configuration"""
        return {
            'scaling': {
                'method': 'robust',  # 'standard', 'robust', 'minmax'
                'feature_range': (0, 1)
            },
            'imputation': {
                'strategy': 'median',  # 'mean', 'median', 'most_frequent', 'knn'
                'knn_neighbors': 5
            },
            'features': {
                'technical_indicators': True,
                'statistical_features': True,
                'time_features': True,
                'rolling_windows': [5, 10, 20, 50]
            },
            'outlier_detection': {
                'method': 'iqr',  # 'iqr', 'zscore'
                'threshold': 3.0
            }
        }

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Complete preprocessing pipeline

        Args:
            df: Raw dataframe with financial data

        Returns:
            Processed dataframe ready for ML
        """
        logger.info("Starting data preprocessing pipeline")

        # Make a copy to avoid modifying original
        processed_df = df.copy()

        # Basic data validation
        processed_df = self._validate_data(processed_df)

        # Handle missing values
        processed_df = self._handle_missing_values(processed_df)

        # Generate features
        processed_df = self._generate_features(processed_df)

        # Handle outliers
        processed_df = self._handle_outliers(processed_df)

        # Scale features
        processed_df = self._scale_features(processed_df)

        # Final validation
        processed_df = self._final_validation(processed_df)

        logger.info(f"Preprocessing complete. Shape: {processed_df.shape}")
        return processed_df

    def _validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic data validation and cleaning"""
        logger.info("Validating input data")

        # Ensure we have required columns
        required_cols = ['price'] if 'price' in df.columns else []
        if 'timestamp' in df.columns or 'date' in df.columns:
            # Sort by timestamp if available
            time_col = 'timestamp' if 'timestamp' in df.columns else 'date'
            df = df.sort_values(time_col).reset_index(drop=True)

        # Remove duplicate rows
        initial_rows = len(df)
        df = df.drop_duplicates()
        if len(df) < initial_rows:
            logger.warning(f"Removed {initial_rows - len(df)} duplicate rows")

        # Ensure numeric columns are properly typed
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values using configured strategy"""
        logger.info("Handling missing values")

        numeric_columns = df.select_dtypes(include=[np.number]).columns
        missing_info = df[numeric_columns].isnull().sum()

        if missing_info.sum() > 0:
            logger.info(f"Found missing values: {missing_info[missing_info > 0].to_dict()}")

            if self.config['imputation']['strategy'] == 'knn':
                imputer = KNNImputer(n_neighbors=self.config['imputation']['knn_neighbors'])
            else:
                imputer = SimpleImputer(strategy=self.config['imputation']['strategy'])

            df[numeric_columns] = imputer.fit_transform(df[numeric_columns])
            self.imputers['numeric'] = imputer

        return df

    def _generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate comprehensive feature set"""
        logger.info("Generating features")

        feature_df = df.copy()

        # Technical indicators
        if self.config['features']['technical_indicators']:
            feature_df = self._add_technical_indicators(feature_df)

        # Statistical features
        if self.config['features']['statistical_features']:
            feature_df = self._add_statistical_features(feature_df)

        # Time-based features
        if self.config['features']['time_features']:
            feature_df = self._add_time_features(feature_df)

        # Rolling window features
        feature_df = self._add_rolling_features(feature_df)

        return feature_df

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical analysis indicators"""
        if 'price' not in df.columns:
            # If no price column, create one from available data
            if 'close' in df.columns:
                df['price'] = df['close']
            elif len(df.select_dtypes(include=[np.number]).columns) > 0:
                # Use the first numeric column as price
                price_col = df.select_dtypes(include=[np.number]).columns[0]
                df['price'] = df[price_col]
            else:
                return df

        try:
            # Price-based indicators
            if 'volume' not in df.columns:
                df['volume'] = 1000000  # Default volume for synthetic data

            # Simple Moving Averages
            df['sma_5'] = ta.trend.sma_indicator(df['price'], window=5)
            df['sma_10'] = ta.trend.sma_indicator(df['price'], window=10)
            df['sma_20'] = ta.trend.sma_indicator(df['price'], window=20)

            # Exponential Moving Averages
            df['ema_12'] = ta.trend.ema_indicator(df['price'], window=12)
            df['ema_26'] = ta.trend.ema_indicator(df['price'], window=26)

            # MACD
            df['macd'] = ta.trend.macd_diff(df['price'])
            df['macd_signal'] = ta.trend.macd_signal(df['price'])

            # RSI
            df['rsi'] = ta.momentum.rsi(df['price'], window=14)

            # Bollinger Bands
            bollinger = ta.volatility.BollingerBands(df['price'])
            df['bb_high'] = bollinger.bollinger_hband()
            df['bb_low'] = bollinger.bollinger_lband()
            df['bb_width'] = df['bb_high'] - df['bb_low']

            # Price position in Bollinger Bands
            df['bb_position'] = (df['price'] - df['bb_low']) / df['bb_width']

            # Stochastic Oscillator
            df['stoch_k'] = ta.momentum.stoch(df['price'], df['price'], df['price'])
            df['stoch_d'] = ta.momentum.stoch_signal(df['price'], df['price'], df['price'])

            # Average True Range
            df['atr'] = ta.volatility.average_true_range(df['price'], df['price'], df['price'])

            # Volume indicators (if volume is available)
            if 'volume' in df.columns:
                df['volume_sma'] = ta.volume.volume_sma(df['price'], df['volume'])
                df['volume_ratio'] = df['volume'] / df['volume_sma']

        except Exception as e:
            logger.warning(f"Error adding technical indicators: {e}")

        return df

    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add statistical features"""
        numeric_columns = df.select_dtypes(include=[np.number]).columns

        for col in numeric_columns:
            if col in ['price', 'returns']:
                # Returns
                df[f'{col}_returns'] = df[col].pct_change()

                # Log returns
                df[f'{col}_log_returns'] = np.log(df[col] / df[col].shift(1))

                # Volatility (rolling standard deviation)
                df[f'{col}_volatility_10'] = df[f'{col}_returns'].rolling(10).std()
                df[f'{col}_volatility_20'] = df[f'{col}_returns'].rolling(20).std()

                # Skewness and Kurtosis
                df[f'{col}_skew_20'] = df[f'{col}_returns'].rolling(20).skew()
                df[f'{col}_kurt_20'] = df[f'{col}_returns'].rolling(20).kurt()

        return df

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        if 'timestamp' in df.columns:
            time_col = 'timestamp'
        elif 'date' in df.columns:
            time_col = 'date'
        else:
            # Create synthetic timestamp
            df['timestamp'] = pd.date_range('2020-01-01', periods=len(df), freq='H')
            time_col = 'timestamp'

        # Ensure datetime format
        df[time_col] = pd.to_datetime(df[time_col])

        # Extract time features
        df['hour'] = df[time_col].dt.hour
        df['day_of_week'] = df[time_col].dt.dayofweek
        df['day_of_month'] = df[time_col].dt.day
        df['month'] = df[time_col].dt.month
        df['quarter'] = df[time_col].dt.quarter

        # Cyclical encoding for time features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        return df

    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling window statistical features"""
        numeric_columns = [col for col in df.select_dtypes(include=[np.number]).columns
                          if col not in ['hour', 'day_of_week', 'day_of_month', 'month', 'quarter']]

        for window in self.config['features']['rolling_windows']:
            for col in numeric_columns[:3]:  # Limit to first 3 columns to avoid explosion
                try:
                    df[f'{col}_roll_mean_{window}'] = df[col].rolling(window).mean()
                    df[f'{col}_roll_std_{window}'] = df[col].rolling(window).std()
                    df[f'{col}_roll_min_{window}'] = df[col].rolling(window).min()
                    df[f'{col}_roll_max_{window}'] = df[col].rolling(window).max()

                    # Relative position in rolling window
                    df[f'{col}_roll_pos_{window}'] = (
                        (df[col] - df[f'{col}_roll_min_{window}']) /
                        (df[f'{col}_roll_max_{window}'] - df[f'{col}_roll_min_{window}'] + 1e-8)
                    )
                except Exception as e:
                    logger.warning(f"Error adding rolling features for {col}, window {window}: {e}")

        return df

    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers using configured method"""
        logger.info("Handling outliers")

        numeric_columns = df.select_dtypes(include=[np.number]).columns

        for col in numeric_columns:
            if self.config['outlier_detection']['method'] == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                # Cap outliers instead of removing them
                df[col] = np.clip(df[col], lower_bound, upper_bound)

            elif self.config['outlier_detection']['method'] == 'zscore':
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                threshold = self.config['outlier_detection']['threshold']

                # Cap outliers
                outlier_mask = z_scores > threshold
                if outlier_mask.sum() > 0:
                    median_val = df[col].median()
                    df.loc[outlier_mask, col] = median_val

        return df

    def _scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale features using configured method"""
        logger.info("Scaling features")

        # Identify numeric columns to scale (exclude certain columns)
        exclude_cols = ['timestamp', 'date', 'target']
        numeric_columns = [col for col in df.select_dtypes(include=[np.number]).columns
                          if col not in exclude_cols]

        if self.config['scaling']['method'] == 'standard':
            scaler = StandardScaler()
        elif self.config['scaling']['method'] == 'robust':
            scaler = RobustScaler()
        elif self.config['scaling']['method'] == 'minmax':
            scaler = MinMaxScaler(feature_range=self.config['scaling']['feature_range'])
        else:
            logger.warning("Invalid scaling method, skipping scaling")
            return df

        # Fit and transform
        df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
        self.scalers['features'] = scaler
        self.feature_names = numeric_columns

        return df

    def _final_validation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Final data validation and cleanup"""
        logger.info("Final validation")

        # Remove any remaining NaN/inf values
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)

        # Ensure all numeric columns are finite
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if not np.isfinite(df[col]).all():
                logger.warning(f"Non-finite values found in {col}, filling with 0")
                df[col] = df[col].fillna(0)

        return df

    def transform_new_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted scalers and imputers"""
        logger.info("Transforming new data")

        # Apply same preprocessing steps
        processed_df = df.copy()
        processed_df = self._validate_data(processed_df)

        # Use fitted imputers
        if 'numeric' in self.imputers:
            numeric_columns = processed_df.select_dtypes(include=[np.number]).columns
            processed_df[numeric_columns] = self.imputers['numeric'].transform(processed_df[numeric_columns])

        # Generate features (same as training)
        processed_df = self._generate_features(processed_df)
        processed_df = self._handle_outliers(processed_df)

        # Use fitted scalers
        if 'features' in self.scalers and self.feature_names:
            # Ensure we have the same features
            missing_features = set(self.feature_names) - set(processed_df.columns)
            if missing_features:
                logger.warning(f"Missing features in new data: {missing_features}")
                # Add missing features with zeros
                for feature in missing_features:
                    processed_df[feature] = 0

            # Apply scaling
            processed_df[self.feature_names] = self.scalers['features'].transform(
                processed_df[self.feature_names]
            )

        processed_df = self._final_validation(processed_df)
        return processed_df