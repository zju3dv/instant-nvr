import os
import importlib
from termcolor import colored

def make_renderer(cfg, network, vis=False, split='train'):
    if hasattr(getattr(cfg, split), 'renderer_module'):
        module = getattr(cfg, split).renderer_module
        renderer = importlib.import_module(module).Renderer(network)
        return renderer

    module = cfg.renderer_module
    if not vis:
        renderer = importlib.import_module(module).Renderer(network)
    else:
        renderer = importlib.import_module(cfg.renderer_vis_module).Renderer(network)
    return renderer
