import json
import os
from pathlib import Path
from types import SimpleNamespace

DATASET_ALIASES = {
    'ch-sims': 'sims',
    'ch_sims': 'sims',
    'chsims': 'sims',
    'ch-sims-v1': 'sims',
}


def normalize_dataset_name(dataset_name):
    dataset_name = dataset_name.lower()
    return DATASET_ALIASES.get(dataset_name, dataset_name)


def dict_to_namespace(d):
    """
    Recursively convert a dictionary to SimpleNamespace for dot notation access.
    """
    if isinstance(d, dict):
        for key, value in d.items():
            d[key] = dict_to_namespace(value)
        return SimpleNamespace(**d)
    elif isinstance(d, list):
        return [dict_to_namespace(item) for item in d]
    else:
        return d


def namespace_to_dict(ns):
    """
    Recursively convert a SimpleNamespace back to dictionary.
    """
    if isinstance(ns, SimpleNamespace):
        return {key: namespace_to_dict(value) for key, value in vars(ns).items()}
    elif isinstance(ns, list):
        return [namespace_to_dict(item) for item in ns]
    else:
        return ns


def get_config_regression(model_name, dataset_name, config_file="", data_dir=None):
    """
    Get configuration for regression tasks
    """
    dataset_name = normalize_dataset_name(dataset_name)
    if config_file == "":
        config_file = Path(__file__).parent / "config" / "dec_config.json"
    
    # Default configuration
    default_config = {
        'model_name': model_name.lower(),
        'dataset_name': dataset_name.lower(),
        'train_mode': 'regression',
        
        # Model architecture
        'use_bert': True,
        'use_finetune': True,
        'transformers': 'bert',
        'pretrained': 'bert-base-uncased',
        'need_data_aligned': False,
        
        # Feature dimensions (will be updated by dataloader)
        'feature_dims': [768, 74, 35],  # BERT, audio, video
        'dst_feature_dim_nheads': [40, 10],  # feature_dim, num_heads
        'nlevels': 5,
        
        # Conv1D parameters
        'conv1d_kernel_size_l': 1,
        'conv1d_kernel_size_a': 1, 
        'conv1d_kernel_size_v': 1,
        
        # Loss weights
        # 'alpha1': 0.1,  # decoupling loss weight
        # 'alpha2': 0.1,  # alignment loss weight
        'task_loss': 'mae',
        'dec_loss_weight': 1.0,
        'num_classes': 1,
        'class_weight': 'none',
        'iemocap_protocol': 'session_valid',
        
        # Training parameters
        'batch_size': 24,
        'learning_rate': 1e-3,
        'weight_decay': 0.0,
        'num_epochs': 100,
        'patience': 20,
        'scheduler_patience': None,
        'early_stop_patience': None,
        'selection_metric': None,
        'scheduler_metric': None,
        'grad_clip': 0.8,
        'factor': 0.1,
        
        # Data paths (to be set by user)
        'featurePath': '',
        'feature_T': '',
        'feature_A': '', 
        'feature_V': '',
    }
    
    # Set data directory
    if data_dir is None:
        data_dir = './data'

    # Dataset specific configurations
    if dataset_name.lower() == 'mosi':
        mosi_candidates = [
            os.path.join(data_dir, 'MOSI/aligned_50.pkl'),
            os.path.join(data_dir, 'MOSI/Processed/aligned_50.pkl'),
            os.path.join(data_dir, 'MOSI/mosi_data.pkl'),
        ]
        mosi_feature_path = next((p for p in mosi_candidates if os.path.exists(p)), mosi_candidates[0])
        dataset_config = {
            'featurePath': mosi_feature_path,
            'seq_lens': [50, 50, 50],
            'feature_dims': [768, 5, 20],
            'train_mode': 'regression',
        }
    elif dataset_name.lower() == 'mosei':
        mosei_candidates = [
            os.path.join(data_dir, 'MOSEI/aligned_50.pkl'),
            os.path.join(data_dir, 'MOSEI/Processed/aligned_50.pkl'),
            os.path.join(data_dir, 'MOSEI/mosei_data.pkl'),
        ]
        mosei_feature_path = next((p for p in mosei_candidates if os.path.exists(p)), mosei_candidates[0])
        dataset_config = {
            'featurePath': mosei_feature_path,
            'seq_lens': [50, 50, 50],
            'feature_dims': [768, 74, 35],
            'train_mode': 'regression',
        }
    elif dataset_name.lower() == 'iemocap':
        dataset_config = {
            'featurePath': os.path.join(data_dir, 'IEMOCAP/iemocap_data.pkl'),
            'seq_lens': [1, 1, 1],
            'feature_dims': [1024, 512, 1024],
            'use_bert': False,
            'need_data_aligned': True,
            'train_mode': 'classification',  # IEMOCAP is classification task
            'num_classes': 6,
            'selection_metric': 'WAF1',
            'scheduler_metric': 'WAF1',
            'selection_mode': 'max',
            'scheduler_mode': 'max',
        }
    elif dataset_name.lower() == 'sims':
        sims_candidates = [
            os.path.join(data_dir, 'CH-SIMS/chsims.pkl'),
            os.path.join(data_dir, 'CH-SIMS', 'chsims.pkl'),
            os.path.join(data_dir, 'SIMS/unaligned_39.pkl'),
            os.path.join(data_dir, 'SIMS/Processed/unaligned_39.pkl'),
            os.path.join(data_dir, 'chsims.pkl'),
            os.path.join(data_dir, 'unaligned_39.pkl'),
        ]
        sims_feature_path = next((p for p in sims_candidates if os.path.exists(p)), sims_candidates[0])
        dataset_config = {
            # DecAlign README uses CH-SIMS/chsims.pkl, while MMSA stores the same
            # preprocessed feature file as SIMS/Processed/unaligned_39.pkl.
            'featurePath': sims_feature_path,
            'seq_lens': [39, 400, 55],
            'feature_dims': [768, 33, 709],
            'train_samples': 1368,
            'num_classes': 3,
            'language': 'cn',
            'use_bert': True,
            'use_finetune': True,
            'transformers': 'bert',
            'pretrained': 'bert-base-chinese',
            'need_data_aligned': False,
            'need_feature_standardized': True,
            'feature_standardize_modalities': ['audio', 'vision'],
            'feature_clip_percentiles': [0.1, 99.9],
            'train_mode': 'regression',
            # Optimized CH-SIMS seed=1 hyperparameters. Architecture settings
            # below remain the paper CH-SIMS setup.
            'attn_dropout': 0.4,
            'attn_dropout_a': 0.3,
            'attn_dropout_v': 0.1,
            'relu_dropout': 0.1,
            'embed_dropout': 0.3,
            'res_dropout': 0.1,
            'output_dropout': 0.35,
            'text_dropout': 0.25,
            'dst_feature_dim_nheads': [32, 8],
            'conv1d_kernel_size_l': 5,
            'conv1d_kernel_size_a': 5,
            'conv1d_kernel_size_v': 5,
            'nlevels': 4,
            'batch_size': 24,
            'learning_rate': 3.5e-5,
            'bert_learning_rate': 7.5e-6,
            'weight_decay': 0.005,
            'task_loss': 'mse',
            'selection_metric': 'F1_score',
            'scheduler_metric': 'F1_score',
            'dec_loss_weight': 1.0,
            'alpha': 0.0175,
            'beta': 0.0525,
            'ct_weight': 0.01,
            'ct_temperature': 0.07,
            'binary_task_weight': 0.01,
            'num_epochs': 55,
            'patience': 5,
            'early_stop_patience': 11,
            'scheduler_patience': 5,
            'clip': 0.5,
            'factor': 0.1,
        }
    else:
        dataset_config = {}

    default_config.update(dataset_config)
    
    # Load from file if exists
    if config_file and os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                file_config = json.load(f)
            default_config.update(file_config)
        except Exception as e:
            print(f"Warning: Could not load config file {config_file}: {e}")

    # Dataset paths and task contracts should remain stable even when the
    # shared config file supplies model hyperparameters.
    default_config['model_name'] = model_name.lower()
    default_config['dataset_name'] = dataset_name.lower()

    # Convert to SimpleNamespace for dot notation access
    return dict_to_namespace(default_config)

def save_config(config, save_path):
    """Save configuration to JSON file"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(config, f, indent=4)

def load_config(config_path):
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        return json.load(f)
