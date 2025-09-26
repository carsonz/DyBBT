"""
Custom utilities for MSDialog dataset
Adapted from original custom_util.py for MSDialog format compatibility
"""

import json
import logging
import os
import time
from typing import Dict, List, Any, Union
import numpy as np
from datetime import datetime


class timeout:
    """Timeout context manager"""
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        import signal
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        import signal
        signal.alarm(0)


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def flatten_acts(acts: List[Dict]) -> List[str]:
    """Flatten dialogue acts for MSDialog format"""
    flattened = []
    for act in acts:
        if isinstance(act, dict):
            # MSDialog format: {'act': 'inform', 'slot': 'movie_name', 'value': 'Inception'}
            act_str = f"{act.get('act', '')}"
            if 'slot' in act and act['slot']:
                act_str += f"({act['slot']}"
                if 'value' in act and act['value']:
                    act_str += f"={act['value']})"
                else:
                    act_str += ")"
            flattened.append(act_str)
        elif isinstance(act, str):
            flattened.append(act)
    return flattened


def load_config_file(config_path: str) -> Dict:
    """Load and validate configuration file for MSDialog"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Validate required modules for MSDialog
    required_modules = ['model', 'vectorizer_sys', 'nlu_sys']
    for module in required_modules:
        if module not in config:
            raise ValueError(f"Config missing required module: {module}")
    
    return config


def save_config(config: Dict, config_path: str):
    """Save configuration to file"""
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2, cls=NumpyEncoder)


def set_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    import random
    import torch
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def init_logging(log_dir: str, log_level: int = logging.INFO) -> str:
    """Initialize logging for MSDialog training"""
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'msdialog_train_{timestamp}.log')
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return log_file


def log_start_args(args: Any):
    """Log training arguments"""
    logger = logging.getLogger(__name__)
    logger.info("Starting MSDialog training with arguments:")
    for arg in vars(args):
        logger.info(f"  {arg}: {getattr(args, arg)}")


def save_best(model: Any, optimizer: Any, scheduler: Any, metrics: Dict, 
             save_path: str, is_best: bool = False):
    """Save model checkpoint with MSDialog-specific metadata"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
        'dataset_type': 'msdialog',  # MSDialog specific
        'timestamp': datetime.now().isoformat()
    }
    
    torch.save(checkpoint, save_path)
    
    if is_best:
        best_path = os.path.join(os.path.dirname(save_path), 'best_model.pt')
        torch.save(checkpoint, best_path)


def eval_policy(policy: Any, env: Any, episodes: int = 100, 
               goal_generator: Any = None, distributed: bool = False) -> Dict:
    """Evaluate policy for MSDialog environment"""
    logger = logging.getLogger(__name__)
    logger.info(f"Evaluating policy on MSDialog dataset for {episodes} episodes")
    
    # MSDialog-specific evaluation setup
    if goal_generator is None:
        # Create simple goal generator for MSDialog
        from convlab.policy.rule.multiwoz import GoalGenerator
        goal_generator = GoalGenerator()
    
    try:
        if distributed:
            from convlab.util.evaluation import evaluate_distributed
            results = evaluate_distributed(policy, env, goal_generator, episodes)
        else:
            from convlab.util.evaluation import evaluate
            results = evaluate(policy, env, goal_generator, episodes)
        
        # Log MSDialog-specific metrics
        logger.info(f"MSDialog evaluation results: {results}")
        return results
        
    except Exception as e:
        logger.error(f"Error during MSDialog policy evaluation: {e}")
        return {
            'success_rate': 0.0,
            'reward': 0.0,
            'turn': 0,
            'error': str(e)
        }


def load_msdialog_data(data_path: str, domain: str = None) -> List[Dict]:
    """Load MSDialog dataset with optional domain filtering"""
    logger = logging.getLogger(__name__)
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"MSDialog data not found: {data_path}")
    
    logger.info(f"Loading MSDialog data from {data_path}")
    if domain:
        logger.info(f"Filtering for domain: {domain}")
    
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    # Filter by domain if specified
    if domain:
        filtered_data = []
        for dialog in data:
            if dialog.get('domain') == domain:
                filtered_data.append(dialog)
        logger.info(f"Loaded {len(filtered_data)} dialogues for domain {domain}")
        return filtered_data
    
    logger.info(f"Loaded {len(data)} dialogues")
    return data


def create_msdialog_env(config: Dict, data: List[Dict] = None) -> Any:
    """Create MSDialog environment"""
    logger = logging.getLogger(__name__)
    logger.info("Creating MSDialog environment")
    
    try:
        from convlab.task.msdialog import MSDialog
        from convlab.env.msdialog_env import MSDialogEnv
        
        # Load data if not provided
        if data is None:
            data_path = config.get('data_path', 'data/msdialog.json')
            data = load_msdialog_data(data_path, config.get('domain'))
        
        # Create environment
        task = MSDialog(data)
        env = MSDialogEnv(config, task)
        
        logger.info("MSDialog environment created successfully")
        return env
        
    except ImportError:
        logger.warning("MSDialog modules not found, using fallback environment")
        from convlab.env import Environment
        return Environment(config)


def setup_msdialog_training(config: Dict, args: Any) -> Dict:
    """Setup training configuration for MSDialog"""
    logger = logging.getLogger(__name__)
    
    # Update config with MSDialog-specific settings
    config.update({
        'dataset_type': 'msdialog',
        'domain': getattr(args, 'domain', None),
        'max_turns': getattr(args, 'max_turns', 20),  # MSDialog typically has shorter dialogues
        'reward_scale': getattr(args, 'reward_scale', 1.0)
    })
    
    logger.info(f"MSDialog training configuration: {config}")
    return config