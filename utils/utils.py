
import logging
import os
from typing import Any, Dict

import tensorflow as tf
import torch


def restore_checkpoint(ckpt_dir: str, state: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
  """
  Restores a checkpoint from the given directory and returns the state dictionary.
  
  Parameters
  ----------
  ckpt_dir : str
      The path to the checkpoint directory.
  state : Dict[str, Any]
      The state dictionary to restore the checkpoint to.
  device : torch.device
      The device to load the checkpoint on.
  
  Returns
  -------
  state : Dict[str, Any]
      The state dictionary with the restored checkpoint.
  """
  if not tf.io.gfile.exists(ckpt_dir):
    tf.io.gfile.makedirs(os.path.dirname(ckpt_dir))
    logging.warning(f"No checkpoint found at {ckpt_dir}. "
                    f"Returned the same state as input")
    return state
  else:
    loaded_state = torch.load(ckpt_dir, map_location=device)
    state['optimizer'].load_state_dict(loaded_state['optimizer'])
    state['model'].load_state_dict(loaded_state['model'], strict=False)
    state['ema'].load_state_dict(loaded_state['ema'])
    state['step'] = loaded_state['step']
    return state


def save_checkpoint(ckpt_dir: str, state: Dict[str, Any]):
  """
  Saves the given state dictionary to the given directory.
  
  Parameters
  ----------
  ckpt_dir : str
      The path to the checkpoint directory.
  state : Dict[str, Any]
      The state dictionary to save.
  """
  saved_state = {
    'optimizer': state['optimizer'].state_dict(),
    'model': state['model'].state_dict(),
    'ema': state['ema'].state_dict(),
    'step': state['step']
  }
  torch.save(saved_state, ckpt_dir)