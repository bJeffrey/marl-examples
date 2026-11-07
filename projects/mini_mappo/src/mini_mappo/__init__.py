"""Educational, modular MAPPO implementation."""

from .agent import MAPPOAgent
from .config import Config
from .environment import GridWorldEnv
from .trainer import train

__all__ = ["Config", "GridWorldEnv", "MAPPOAgent", "train"]
