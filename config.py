"""
Configuration module for Insurance Claim Processing Pipeline
Centralized settings for paths, model parameters, and feature definitions
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict

# =============================================================================
# DIRECTORY STRUCTURE
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_PDF_DIR = DATA_DIR / "raw_pdfs"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_PDF_DIR, PROCESSED_DIR, MODELS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# =============================================================================
# FEATURE DEFINITIONS
# =============================================================================

@dataclass
class FeatureConfig:
    """Define all possible features from claim documents"""
    
    # Categorical Features
    CATEGORICAL_FEATURES = [
        'termination_type',
        'lease_state',
        'tenant_contacted',
        'tenant_collection_status',
        'collection_status',
        'is_2nd_tenant',
        'is_3rd_tenant',
        'property_management_company'
    ]
    
    # Numeric Features
    NUMERIC_FEATURES = [
        'monthly_rent',
        'amount_of_claim',
        'max_benefit',
        'lease_duration_days',
        'days_to_moveout',
        'days_claim_to_approval',
        'late_fees_total',
        'repair_costs_total',
        'legal_fees_total'
    ]
    
    # Date Features (will be engineered into numeric)
    DATE_FEATURES = [
        'claim_date',
        'approval_date',
        'lease_start_date',
        'lease_end_date',
        'move_out_date'
    ]
    
    # Text Features (for NLP extraction)
    TEXT_FEATURES = [
        'pm_explanation',
        'hold_reason',
        'document_full_text'
    ]
    
    # Target Variable
    TARGET = 'approved_benefit_amount'

# =============================================================================
# EXTRACTION PATTERNS
# =============================================================================

class ExtractionPatterns:
    """Regular expressions and patterns for information extraction"""
    
    # Currency patterns
    CURRENCY_PATTERNS = [
        r'\$\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',  # $1,200.00 or $1200.00
        r'(?:Total|Amount|Balance|Due)[:|\s]+\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
        r'(\d{1,3}(?:,\d{3})*\.\d{2})\s*(?:USD|dollars?)?'
    ]
    
    # Date patterns
    DATE_PATTERNS = [
        r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',  # 07/08/2022 or 7-8-22
        r'(\d{4}[/-]\d{1,2}[/-]\d{1,2})',    # 2022-07-08
        r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4})'
    ]
    
    # Address patterns
    ADDRESS_PATTERN = r'(\d+\s+[A-Za-z0-9\s]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Way|Court|Ct|Boulevard|Blvd))'
    
    # Termination types
    TERMINATION_KEYWORDS = {
        'eviction': ['eviction', 'evicted', 'forcible', 'unlawful detainer'],
        'skip': ['skip', 'skipped', 'abandoned', 'absconded'],
        'mutual': ['mutual', 'agreement', 'consensual'],
        'lease_break': ['lease break', 'early termination', 'broken lease']
    }
    
    # Fee types for itemization
    FEE_KEYWORDS = {
        'late_fee': ['late fee', 'late charge', 'late payment'],
        'repair': ['repair', 'damage', 'maintenance', 'fix'],
        'legal': ['legal', 'attorney', 'court', 'filing fee'],
        'admin': ['administrative', 'admin fee', 'processing']
    }

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

@dataclass
class ModelConfig:
    """Machine learning model hyperparameters"""
    
    # Train/Test Split
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    
    # Cross-validation
    CV_FOLDS = 5
    
    # Model candidates
    MODELS = {
        'random_forest': {
            'n_estimators': 100,
            'max_depth': 15,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': RANDOM_STATE
        },
        'gradient_boosting': {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 5,
            'random_state': RANDOM_STATE
        },
        'xgboost': {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 6,
            'random_state': RANDOM_STATE
        }
    }
    
    # Feature engineering
    MAX_CATEGORIES = 50  # Max unique values for categorical encoding
    MISSING_VALUE_THRESHOLD = 0.5  # Drop features with >50% missing
    
    # Scaling
    SCALE_FEATURES = True

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'file': {
            'level': 'INFO',
            'class': 'logging.FileHandler',
            'filename': str(LOGS_DIR / 'pipeline.log'),
            'formatter': 'standard',
        },
        'console': {
            'level': 'INFO',
            'class': 'logging.StreamHandler',
            'formatter': 'standard',
        },
    },
    'loggers': {
        '': {
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': True
        }
    }
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_model_path(model_name: str) -> Path:
    """Get path for saving/loading model"""
    return MODELS_DIR / f"{model_name}.pkl"

def get_processed_data_path(filename: str) -> Path:
    """Get path for processed data files"""
    return PROCESSED_DIR / filename

def validate_pdf_directory():
    """Validate that PDF directory exists and contains files"""
    if not RAW_PDF_DIR.exists():
        raise FileNotFoundError(f"PDF directory not found: {RAW_PDF_DIR}")
    
    pdf_files = list(RAW_PDF_DIR.glob("*.pdf"))
    if len(pdf_files) == 0:
        raise ValueError(f"No PDF files found in {RAW_PDF_DIR}")
    
    return pdf_files

# =============================================================================
# EXPORT
# =============================================================================

__all__ = [
    'BASE_DIR',
    'DATA_DIR',
    'RAW_PDF_DIR',
    'PROCESSED_DIR',
    'MODELS_DIR',
    'LOGS_DIR',
    'FeatureConfig',
    'ExtractionPatterns',
    'ModelConfig',
    'LOGGING_CONFIG',
    'get_model_path',
    'get_processed_data_path',
    'validate_pdf_directory'
]