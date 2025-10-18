# utils/config_loader.py

import os
import yaml
import logging
from typing import Dict, Any, Optional

from .db_connector import db
from .db_schema import JudgeModel

_local_judge_configs: Optional[Dict[str, Any]] = None

def _load_local_judge_configs(config_file: str = 'config/judge_model_creds.yaml') -> Dict[str, Any]:
    """Loads and caches judge configurations from the local YAML file."""
    global _local_judge_configs
    if _local_judge_configs is None:
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                _local_judge_configs = yaml.safe_load(f) or {}
                logging.info(f"Loaded {len(_local_judge_configs)} local judge configurations from {config_file}.")
        except FileNotFoundError:
            logging.info(f"Local judge config file not found: {config_file}. Will rely on database.")
            _local_judge_configs = {}
    return _local_judge_configs

def get_judge_config(judge_name: str) -> Dict[str, Any]:
    """
    Retrieves configuration for a judge, prioritizing local YAML file over the database.
    Resolves 'env:VAR_NAME' syntax for API keys.
    """
    local_configs = _load_local_judge_configs()

    config = None
    if judge_name in local_configs:
        logging.debug(f"Using local YAML configuration for judge '{judge_name}'.")
        config = local_configs[judge_name]
    else:
        logging.debug(f"Checking database for judge '{judge_name}' configuration.")
        with db.get_session() as session:
            judge_model_db = session.query(JudgeModel).filter_by(name=judge_name).first()
            if judge_model_db:
                config = {
                    "model_id": judge_model_db.model_id,
                    "provider": judge_model_db.provider,
                    "api_key": judge_model_db.api_key,
                    "base_url": judge_model_db.base_url,
                    "system_prompt": judge_model_db.system_prompt,
                }

    if config is None:
        raise ValueError(f"Configuration for judge '{judge_name}' not found in local file or database.")

    # Resolve API key from environment if specified
    api_key = config.get('api_key')
    if isinstance(api_key, str) and api_key.startswith('env:'):
        env_var = api_key.split(':', 1)[1]
        resolved_key = os.getenv(env_var)
        if not resolved_key:
            raise ValueError(f"Environment variable '{env_var}' for judge '{judge_name}' API key is not set.")
        config['api_key'] = resolved_key

    return config