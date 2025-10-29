import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Tuple, Any
import joblib
import json

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score
import xgboost as xgb

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import ModelConfig, MODELS_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    
    def __init__(self):
        self.model_config = ModelConfig()
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.training_history = {}
        
    def initialize_models(self):
        logger.info("Initializing models...")
        
        self.models = {
            'random_forest': RandomForestRegressor(
                **self.model_config.MODELS['random_forest']
            ),
            'gradient_boosting': GradientBoostingRegressor(
                **self.model_config.MODELS['gradient_boosting']
            ),
            'xgboost': xgb.XGBRegressor(
                **self.model_config.MODELS['xgboost']
            )
        }
        
        logger.info(f"Initialized {len(self.models)} models")
    
    def train_model(self, model_name: str, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
        logger.info(f"Training {model_name}...")
        
        model = self.models[model_name]
        model.fit(X_train, y_train)
        
        logger.info(f"{model_name} training complete")
        return model
    
    def evaluate_model(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series, model_name: str) -> Dict:
        logger.info(f"Evaluating {model_name}...")
        
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1))) * 100
        
        metrics = {
            'model_name': model_name,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2_score': r2,
            'mape': mape
        }
        
        logger.info(f"{model_name} Metrics:")
        logger.info(f"  RMSE: ${rmse:.2f}")
        logger.info(f"  MAE: ${mae:.2f}")
        logger.info(f"  R2 Score: {r2:.4f}")
        logger.info(f"  MAPE: {mape:.2f}%")
        
        return metrics
    
    def cross_validate(self, model: Any, X: pd.DataFrame, y: pd.Series, model_name: str) -> Dict:
        logger.info(f"Cross-validating {model_name}...")
        
        try:
            cv_folds = min(self.model_config.CV_FOLDS, len(X))
            if cv_folds < 2:
                logger.warning(f"Not enough samples for CV, skipping")
                return {
                    'cv_rmse_mean': 0,
                    'cv_rmse_std': 0,
                    'cv_scores': []
                }
            
            cv_scores = cross_val_score(
                model, X, y,
                cv=cv_folds,
                scoring='neg_mean_squared_error'
            )
            
            cv_rmse = np.sqrt(-cv_scores)
            
            cv_metrics = {
                'cv_rmse_mean': cv_rmse.mean(),
                'cv_rmse_std': cv_rmse.std(),
                'cv_scores': cv_rmse.tolist()
            }
            
            logger.info(f"{model_name} CV RMSE: ${cv_rmse.mean():.2f} (+/- ${cv_rmse.std():.2f})")
            
            return cv_metrics
        except Exception as e:
            logger.warning(f"Cross-validation failed: {e}")
            return {
                'cv_rmse_mean': 0,
                'cv_rmse_std': 0,
                'cv_scores': []
            }
    
    def get_feature_importance(self, model: Any, feature_names: list, model_name: str, top_n: int = 15) -> pd.DataFrame:
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            
            feature_imp = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            logger.info(f"\n{model_name} - Top {top_n} Features:")
            for idx, row in feature_imp.head(top_n).iterrows():
                logger.info(f"  {row['feature']}: {row['importance']:.4f}")
            
            return feature_imp
        else:
            logger.warning(f"{model_name} does not support feature importance")
            return None
    
    def train_all_models(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                         y_train: pd.Series, y_test: pd.Series) -> Dict:
        
        logger.info("="*60)
        logger.info("TRAINING ALL MODELS")
        logger.info("="*60)
        
        self.initialize_models()
        
        all_results = {}
        
        for model_name in self.models.keys():
            logger.info(f"\n{'='*60}")
            logger.info(f"MODEL: {model_name.upper()}")
            logger.info(f"{'='*60}")
            
            model = self.train_model(model_name, X_train, y_train)
            
            test_metrics = self.evaluate_model(model, X_test, y_test, model_name)
            
            cv_metrics = self.cross_validate(model, X_train, y_train, model_name)
            
            feature_importance = self.get_feature_importance(
                model, 
                X_train.columns.tolist(), 
                model_name
            )
            
            results = {
                'model': model,
                'test_metrics': test_metrics,
                'cv_metrics': cv_metrics,
                'feature_importance': feature_importance
            }
            
            all_results[model_name] = results
            self.training_history[model_name] = {
                'test_metrics': test_metrics,
                'cv_metrics': cv_metrics
            }
        
        self._select_best_model(all_results)
        
        return all_results
    
    def _select_best_model(self, all_results: Dict):
        logger.info("\n" + "="*60)
        logger.info("MODEL COMPARISON")
        logger.info("="*60)
        
        comparison = []
        for model_name, results in all_results.items():
            comparison.append({
                'model': model_name,
                'test_rmse': results['test_metrics']['rmse'],
                'test_r2': results['test_metrics']['r2_score'],
                'cv_rmse': results['cv_metrics']['cv_rmse_mean']
            })
        
        comparison_df = pd.DataFrame(comparison).sort_values('test_rmse')
        
        logger.info("\nModel Rankings (by Test RMSE):")
        for idx, row in comparison_df.iterrows():
            logger.info(f"  {row['model']}: RMSE=${row['test_rmse']:.2f}, R2={row['test_r2']:.4f}")
        
        best_row = comparison_df.iloc[0]
        self.best_model_name = best_row['model']
        self.best_model = all_results[self.best_model_name]['model']
        
        logger.info(f"\nBest Model: {self.best_model_name.upper()}")
        logger.info(f"  Test RMSE: ${best_row['test_rmse']:.2f}")
        logger.info(f"  Test R2: {best_row['test_r2']:.4f}")
        logger.info("="*60)
    
    def save_model(self, model_name: str = None, filename: str = None):
        
        if model_name is None:
            model_name = self.best_model_name
            model = self.best_model
        else:
            model = self.models.get(model_name)
        
        if model is None:
            raise ValueError(f"Model '{model_name}' not found")
        
        if filename is None:
            filename = f"{model_name}_model.pkl"
        
        model_path = MODELS_DIR / filename
        joblib.dump(model, model_path)
        
        logger.info(f"Model saved to: {model_path}")
        
        metadata_path = MODELS_DIR / f"{model_name}_metadata.json"
        metadata = {
            'model_name': model_name,
            'metrics': self.training_history.get(model_name, {}),
            'saved_at': pd.Timestamp.now().isoformat()
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Metadata saved to: {metadata_path}")
    
    def load_model(self, filename: str):
        model_path = MODELS_DIR / filename
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        model = joblib.load(model_path)
        logger.info(f"Model loaded from: {model_path}")
        
        return model
    
    def predict(self, model: Any, X: pd.DataFrame) -> np.ndarray:
        predictions = model.predict(X)
        return predictions
    
    def save_predictions(self, predictions: np.ndarray, filenames: list, output_path: str):
        results_df = pd.DataFrame({
            'filename': filenames,
            'predicted_approved_amount': predictions
        })
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        results_df.to_csv(output_path, index=False)
        logger.info(f"Predictions saved to: {output_path}")


if __name__ == "__main__":
    from data_processor import DataProcessor
    
    processor = DataProcessor()
    X_train, X_test, y_train, y_test = processor.process_pipeline(
        csv_path="data/processed/extracted_features.csv"
    )
    
    trainer = ModelTrainer()
    results = trainer.train_all_models(X_train, X_test, y_train, y_test)
    
    trainer.save_model()
    
    print("\nModel training complete!")
    print(f"Best model: {trainer.best_model_name}")