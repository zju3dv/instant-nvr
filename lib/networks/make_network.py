import os
import importlib


def make_network(cfg):
    module = cfg.network_module
    network = importlib.import_module(module).Network()
    return network


def make_part_network(gcfg, partname, pid):
    from lib.networks.bw_deform.part_base_network import Network
    cfg = getattr(gcfg.partnet, partname)
    module = cfg.module
    network: Network = importlib.import_module(module, partname).Network(partname, pid)
    return network


def make_embedder(cfg):
    module = cfg.embedder.module
    kwargs = cfg.embedder.kwargs
    embedder = importlib.import_module(module).Embedder(**kwargs)
    return embedder


def make_part_embedder(gcfg, partname, pid):
    this_cfg = getattr(gcfg.partnet, partname)
    bbox = this_cfg.bbox
    module = this_cfg.embedder.module
    kwargs = this_cfg.embedder.kwargs
    embedder = importlib.import_module(module).Embedder(bbox=bbox, pid=pid, partname=partname, **kwargs)
    return embedder


def make_viewdir_embedder(cfg):
    module = cfg.viewdir_embedder.module
    kwargs = cfg.viewdir_embedder.kwargs
    embedder = importlib.import_module(module).Embedder(**kwargs)
    return embedder


def make_deformer(cfg):
    module = cfg.tpose_deformer.module
    deformer = importlib.import_module(module).Deformer(deformer_cfg=cfg.tpose_deformer)
    return deformer


def make_residual(cfg):
    if 'color_residual' in cfg:
        module = cfg.color_residual.module
        kwargs = cfg.color_residual.kwargs
        residual = importlib.import_module(module).Residual(**kwargs)
        return residual
    else:
        from lib.networks.residuals.zero_residual import Residual
        return Residual()


def make_color_network(cfg, **kargs):
    if "color_network" in cfg:
        module = cfg.color_network.module
        kwargs = cfg.color_network.kwargs
    elif "network" in cfg and "color" in cfg.network:
        if "module" in cfg.network.color:
            module = cfg.network.color.module
        else:
            module = "lib.networks.bw_deform.inb_network"
        kwargs = cfg.network.color

    full_args = dict(kwargs, **kargs)
    color_network = importlib.import_module(module).ColorNetwork(**full_args)
    return color_network


def make_part_color_network(gcfg, partname, **kargs):
    this_cfg = None
    try:
        this_cfg = getattr(gcfg.partnet, partname)
        assert "color_network" in this_cfg
    except:
        pass

    module = "lib.networks.bw_deform.part_base_network"
    kwargs = {}
    if this_cfg is not None:
        if "color_network" in this_cfg:
            if hasattr(this_cfg.color_network, 'module'):
                module = this_cfg.color_network.module
            if hasattr(this_cfg.color_network, 'kwargs'):
                kwargs = this_cfg.color_network.kwargs
        elif "network" in this_cfg and "color" in this_cfg.network:
            if "module" in this_cfg.network.color:
                module = this_cfg.network.color.module
            if 'kwargs' in this_cfg.network.color:
                kwargs = this_cfg.network.color

    full_args = dict(kwargs, **kargs)
    color_network = importlib.import_module(module).ColorNetwork(**full_args)
    return color_network

# def _make_module_factory(cfgname, classname):
#     def make_sth(cfg, **kargs):
#         module = getattr(cfg, cfgname).module
#         kwargs = getattr(cfg, cfgname).kwargs
#         full_args = dict(kwargs, **kargs)
#         sth = importlib.import_module(module).__dict__[classname](**full_args)
#         return sth
#     return make_sth

# possible_modules = [
#     {
#         "cfgname": "color_network",
#         "classname": "ColorNetwork",
#     }
# ]
