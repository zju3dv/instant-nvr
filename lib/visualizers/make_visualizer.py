import os
import importlib

def make_visualizer(cfg, name=None, split='test'):
    if hasattr(getattr(cfg, split), "visualizer_module"):
        module = getattr(getattr(cfg, split), "visualizer_module")
    else:
        module = cfg.visualizer_module
    visualizer = importlib.import_module(module).Visualizer(name)
    return visualizer
