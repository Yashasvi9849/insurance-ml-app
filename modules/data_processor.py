import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Tuple, List, Dict
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import FeatureConfig, ModelConfig, PROCESSED_DIR, MODELS_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProcessor:
    
    def __init__(self):
        self.feature_config = FeatureConfig()
        self.model_config = ModelConfig()
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def load_data(self, csv_path: str) -> pd.DataFrame:
        logger.info(f"Loading data from: {csv_path}")
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Cleaning data...")
        
        df_clean = df.copy()
        
        df_clean = self._handle_missing_values(df_clean)
        df_clean = self._handle_outliers(df_clean)
        df_clean = self._convert_data_types(df_clean)
        
        logger.info(f"Data cleaned: {len(df_clean)} records remaining")
        return df_clean
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Handling missing values...")
        
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        if self.feature_config.TARGET in numeric_features:
            numeric_features.remove(self.feature_config.TARGET)
        
        for col in numeric_features:
            if df[col].isnull().sum() > 0:
                median_value = df[col].median()
                df[col].fillna(median_value, inplace=True)
                logger.debug(f"Filled {col} with median: {median_value}")
        
        categorical_features = df.select_dtypes(include=['object', 'bool']).columns.tolist()
        for col in categorical_features:
            if df[col].isnull().sum() > 0:
                df[col].fillna('Unknown', inplace=True)
                logger.debug(f"Filled {col} with 'Unknown'")
        
        return df
    
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Handling outliers...")
        
        numeric_cols = ['monthly_rent', 'amount_of_claim', 'approved_benefit_amount']
        
        for col in numeric_cols:
            if col in df.columns and df[col].notna().sum() > 0:
                Q1 = df[col].quantile(0.01)
                Q3 = df[col].quantile(0.99)
                
                original_count = len(df)
                df = df[(df[col] >= Q1) & (df[col] <= Q3) | df[col].isna()]
                removed = original_count - len(df)
                
                if removed > 0:
                    logger.debug(f"Removed {removed} outliers from {col}")
        
        return df
    
    def _convert_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Converting data types...")
        
        bool_cols = ['is_2nd_tenant', 'is_3rd_tenant']
        for col in bool_cols:
            if col in df.columns:
                df[col] = df[col].astype(bool)
        
        date_cols = ['claim_date', 'approval_date', 'lease_start_date', 'lease_end_date', 'move_out_date']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Engineering additional features...")
        
        df_eng = df.copy()
        
        if 'claim_date' in df_eng.columns:
            df_eng['claim_month'] = pd.to_datetime(df_eng['claim_date'], errors='coerce').dt.month
            df_eng['claim_year'] = pd.to_datetime(df_eng['claim_date'], errors='coerce').dt.year
            df_eng['claim_day_of_week'] = pd.to_datetime(df_eng['claim_date'], errors='coerce').dt.dayofweek
        
        if 'monthly_rent' in df_eng.columns and 'amount_of_claim' in df_eng.columns:
            df_eng['claim_to_rent_ratio'] = df_eng['amount_of_claim'] / (df_eng['monthly_rent'] + 1)
        
        if 'late_fees_total' in df_eng.columns and 'amount_of_claim' in df_eng.columns:
            df_eng['late_fee_percentage'] = (df_eng['late_fees_total'] / (df_eng['amount_of_claim'] + 1)) * 100
        
        if 'repair_costs_total' in df_eng.columns and 'amount_of_claim' in df_eng.columns:
            df_eng['repair_percentage'] = (df_eng['repair_costs_total'] / (df_eng['amount_of_claim'] + 1)) * 100
        
        df_eng['total_fees'] = df_eng[['late_fees_total', 'repair_costs_total', 'legal_fees_total', 'admin_fees_total']].fillna(0).sum(axis=1)
        
        logger.info(f"Added engineered features. Total columns: {len(df_eng.columns)}")
        return df_eng
    
    def encode_categorical(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        logger.info("Encoding categorical features...")
        
        df_encoded = df.copy()
        
        categorical_cols = df_encoded.select_dtypes(include=['object']).columns.tolist()
        
        exclude_cols = ['filename', 'pm_explanation', 'hold_reason', 'missing_fields']
        categorical_cols = [col for col in categorical_cols if col not in exclude_cols]
        
        for col in categorical_cols:
            if fit:
                self.label_encoders[col] = LabelEncoder()
                df_encoded[col] = self.label_encoders[col].fit_transform(df_encoded[col].astype(str))
            else:
                if col in self.label_encoders:
                    df_encoded[col] = self.label_encoders[col].transform(df_encoded[col].astype(str))
                else:
                    logger.warning(f"No encoder found for {col}, creating new one")
                    self.label_encoders[col] = LabelEncoder()
                    df_encoded[col] = self.label_encoders[col].fit_transform(df_encoded[col].astype(str))
        
        logger.info(f"Encoded {len(categorical_cols)} categorical features")
        return df_encoded
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        logger.info("Preparing features for modeling...")
        
        target_col = self.feature_config.TARGET
        
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataframe")
        
        df_valid = df[df[target_col].notna()].copy()
        
        if len(df_valid) == 0:
            raise ValueError(f"No valid target values found in '{target_col}'")
        
        logger.info(f"Valid samples with target: {len(df_valid)}/{len(df)}")
        
        exclude_cols = [
            target_col,
            'filename',
            'tracking_number',
            'pm_explanation',
            'hold_reason',
            'missing_fields',
            'extraction_confidence',
            'claim_date',
            'approval_date',
            'lease_start_date',
            'lease_end_date',
            'move_out_date'
        ]
        
        feature_cols = [col for col in df_valid.columns if col not in exclude_cols]
        
        X = df_valid[feature_cols].copy()
        y = df_valid[target_col].copy()
        
        X = X.select_dtypes(include=[np.number])
        
        logger.info(f"Feature matrix shape: {X.shape}")
        logger.info(f"Target variable shape: {y.shape}")
        logger.info(f"Features used: {list(X.columns)}")
        
        self.feature_names = list(X.columns)
        
        return X, y
    
    def scale_features(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        logger.info("Scaling features...")
        
        if fit:
            X_scaled = pd.DataFrame(
                self.scaler.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
        else:
            X_scaled = pd.DataFrame(
                self.scaler.transform(X),
                columns=X.columns,
                index=X.index
            )
        
        logger.info("Feature scaling complete")
        return X_scaled
    
    def split_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        logger.info("Splitting data into train and test sets...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.model_config.TEST_SIZE,
            random_state=self.model_config.RANDOM_STATE
        )
        
        logger.info(f"Train set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        
        return X_train, X_test, y_train, y_test
    
    def save_preprocessor(self, filename: str = "preprocessor.pkl"):
        save_path = MODELS_DIR / filename
        
        preprocessor_data = {
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names
        }
        
        joblib.dump(preprocessor_data, save_path)
        logger.info(f"Preprocessor saved to: {save_path}")
    
    def load_preprocessor(self, filename: str = "preprocessor.pkl"):
        load_path = MODELS_DIR / filename
        
        if not load_path.exists():
            raise FileNotFoundError(f"Preprocessor not found: {load_path}")
        
        preprocessor_data = joblib.load(load_path)
        
        self.scaler = preprocessor_data['scaler']
        self.label_encoders = preprocessor_data['label_encoders']
        self.feature_names = preprocessor_data['feature_names']
        
        logger.info(f"Preprocessor loaded from: {load_path}")
    
    def process_pipeline(self, csv_path: str, save_preprocessor: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        
        logger.info("="*60)
        logger.info("DATA PREPROCESSING PIPELINE")
        logger.info("="*60)
        
        df = self.load_data(csv_path)
        
        df = self.clean_data(df)
        
        df = self.engineer_features(df)
        
        df = self.encode_categorical(df, fit=True)
        
        X, y = self.prepare_features(df)
        
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        
        X_train = self.scale_features(X_train, fit=True)
        X_test = self.scale_features(X_test, fit=False)
        
        if save_preprocessor:
            self.save_preprocessor()
        
        logger.info("="*60)
        logger.info("PREPROCESSING COMPLETE")
        logger.info("="*60)
        
        return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    processor = DataProcessor()
    
    X_train, X_test, y_train, y_test = processor.process_pipeline(
        csv_path="data/processed/extracted_features.csv"
    )
    
    print(f"\nTraining data ready:")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")