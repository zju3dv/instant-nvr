from lib.config import cfg, args
import warnings
warnings.filterwarnings("ignore", category=UserWarning) 


def run_dataset():
    from lib.datasets import make_data_loader
    import tqdm

    cfg.train.num_workers = 0
    data_loader = make_data_loader(cfg, split='test')
    for batch in tqdm.tqdm(data_loader):
        pass


def run_network():
    from lib.networks import make_network
    from lib.datasets import make_data_loader
    from lib.utils.net_utils import load_network
    import tqdm
    import torch
    import time

    network = make_network(cfg).cuda()
    load_network(network, cfg.trained_model_dir, epoch=cfg.test.epoch)
    network.eval()

    data_loader = make_data_loader(cfg, split='test')
    total_time = 0
    for batch in tqdm.tqdm(data_loader):
        for k in batch:
            if k != 'meta':
                batch[k] = batch[k].cuda()
        with torch.no_grad():
            torch.cuda.synchronize()
            start = time.time()
            network(batch)
            torch.cuda.synchronize()
            total_time += time.time() - start
    print(total_time / len(data_loader))

def run_exportdecoder():
    from lib.networks import make_network
    from lib.utils.net_utils import load_network
    import torch

    network = make_network(cfg).cuda()
    load_network(network, cfg.trained_model_dir, epoch=cfg.test.epoch)
    network.tpose_human.save_part_decoders()

def run_exportpart():
    from lib.networks import make_network
    from lib.utils.net_utils import load_network
    import torch

    network = make_network(cfg).cuda()
    load_network(network, cfg.trained_model_dir, epoch=cfg.test.epoch)
    network.tpose_human.save_parts()


def run_evaluate():
    from lib.datasets import make_data_loader
    from lib.evaluators import make_evaluator
    import tqdm
    import torch
    from lib.networks import make_network
    from lib.utils import net_utils
    from lib.networks.renderer import make_renderer

    cfg.perturb = 0
    cfg.eval = True

    network = make_network(cfg).cuda()
    net_utils.load_network(network,
                           cfg.trained_model_dir,
                           resume=cfg.resume,
                           epoch=cfg.test.epoch)
    network.eval()

    data_loader = make_data_loader(cfg, split='test')
    renderer = make_renderer(cfg, network)
    evaluator = make_evaluator(cfg)
    for batch in tqdm.tqdm(data_loader):
        for k in batch:
            if k != 'meta':
                batch[k] = batch[k].cuda()
        with torch.no_grad():
            output = renderer.render(batch)
        evaluator.evaluate(output, batch)
    evaluator.summarize()

def to_cuda(batch):
    if isinstance(batch, dict):
        for k in batch:
            if k == 'meta' or k == 'obj':
                continue
            elif isinstance(batch[k], tuple) or isinstance(batch[k], list): 
                batch[k] = [to_cuda(b) for b in batch[k]]
            elif isinstance(batch[k], dict):
                batch[k] = to_cuda(batch[k])
            else:
                batch[k] = batch[k].cuda()
        return batch
    else:
        return batch.cuda()


def run_vis():
    from lib.networks import make_network
    from lib.datasets import make_data_loader
    from lib.utils.net_utils import load_network
    from lib.utils import net_utils
    import tqdm
    import torch
    from lib.visualizers import make_visualizer
    from lib.networks.renderer import make_renderer

    cfg.perturb = 0

    network = make_network(cfg).cuda()
    load_network(network,
                 cfg.trained_model_dir,
                 resume=cfg.resume,
                 epoch=cfg.test.epoch,
                 strict=False)
    network.train()

    data_loader = make_data_loader(cfg, split='test')
    renderer = make_renderer(cfg, network)
    visualizer = make_visualizer(cfg)
    for batch in tqdm.tqdm(data_loader):
        batch = to_cuda(batch)
        with torch.no_grad():
            output = renderer.render(batch)
            visualizer.visualize(output, batch)

def run_prune():
    from lib.networks import make_network
    from lib.datasets import make_data_loader
    from lib.utils.net_utils import load_network
    from lib.utils import net_utils
    import tqdm
    import torch
    from lib.visualizers import make_visualizer
    from lib.networks.renderer import make_renderer

    cfg.perturb = 0

    network = make_network(cfg).cuda()
    load_network(network,
                 cfg.trained_model_dir,
                 resume=cfg.resume,
                 epoch=cfg.prune.epoch,
                 strict=False)
    network.eval()

    data_loader = make_data_loader(cfg, split='prune')
    renderer = make_renderer(cfg, network)
    visualizer = make_visualizer(cfg)
    for batch in tqdm.tqdm(data_loader):
        batch = to_cuda(batch)
        with torch.no_grad():
            output = renderer.render(batch)
            visualizer.visualize(output, batch, split = 'prune')

def run_tmesh():
    from lib.networks import make_network
    from lib.datasets import make_data_loader
    from lib.utils.net_utils import load_network
    from lib.utils import net_utils
    import tqdm
    import torch
    from lib.visualizers import make_visualizer
    from lib.networks.renderer import make_renderer

    breakpoint()

    cfg.perturb = 0

    network = make_network(cfg).cuda()
    load_network(network,
                 cfg.trained_model_dir,
                 resume=cfg.resume,
                 epoch=cfg.tmesh.epoch,
                 strict=False)
    network.eval()

    data_loader = make_data_loader(cfg, split='tmesh')
    renderer = make_renderer(cfg, network, split='tmesh')
    visualizer = make_visualizer(cfg)
    for batch in tqdm.tqdm(data_loader):
        batch = to_cuda(batch)
        with torch.no_grad():
            output = renderer.render(batch)
            visualizer.visualize(output, batch, split='tmesh')

    
def run_tdmesh():
    from lib.networks import make_network
    from lib.datasets import make_data_loader
    from lib.utils.net_utils import load_network
    from lib.utils import net_utils
    import tqdm
    import torch
    from lib.visualizers import make_visualizer
    from lib.networks.renderer import make_renderer

    breakpoint()

    cfg.perturb = 0

    network = make_network(cfg).cuda()
    load_network(network,
                 cfg.trained_model_dir,
                 resume=cfg.resume,
                 epoch=cfg.tdmesh.epoch,
                 strict=False)
    network.eval()

    data_loader = make_data_loader(cfg, split='tdmesh')
    renderer = make_renderer(cfg, network, split='tdmesh')
    visualizer = make_visualizer(cfg)
    for batch in tqdm.tqdm(data_loader):
        batch = to_cuda(batch)
        with torch.no_grad():
            output = renderer.render(batch)
            visualizer.visualize(output, batch, split='tdmesh')


def run_light_stage():
    from lib.utils.light_stage import ply_to_occupancy
    ply_to_occupancy.ply_to_occupancy()
    # ply_to_occupancy.create_voxel_off()


def run_evaluate_nv():
    from lib.datasets import make_data_loader
    from lib.evaluators import make_evaluator
    import tqdm
    from lib.utils import net_utils

    data_loader = make_data_loader(cfg, split='test')
    evaluator = make_evaluator(cfg)
    for batch in tqdm.tqdm(data_loader):
        for k in batch:
            if k != 'meta':
                batch[k] = batch[k].cuda()
        evaluator.evaluate(batch)
    evaluator.summarize()


def run_animation():
    from tools import animate_mesh
    animate_mesh.animate()


def run_raster():
    from tools import rasterizer_mesh
    renderer = rasterizer_mesh.Renderer()
    renderer.render()


def run_lpips():
    from tools import calculate_lpips
    calculate_lpips.run()

def run_other(type):
    from lib.networks import make_network
    from lib.datasets import make_data_loader
    from lib.utils.net_utils import load_network
    from lib.utils import net_utils
    import tqdm
    import torch
    from lib.visualizers import make_visualizer
    from lib.networks.renderer import make_renderer

    cfg.perturb = 0

    network = make_network(cfg).cuda()
    epoch = load_network(network,
                 cfg.trained_model_dir,
                 resume=cfg.resume,
                 epoch=cfg.test.epoch,
                 strict=False)
    network.train()

    data_loader = make_data_loader(cfg, split=type)
    renderer = make_renderer(cfg, network, split=type)
    visualizer = make_visualizer(cfg, split=type)
    for batch in tqdm.tqdm(data_loader):
        batch = to_cuda(batch)
        with torch.no_grad():
            output = renderer.render(batch)
            visualizer.visualize(output, batch, split=type)
    if type == 'bullet':
        visualizer.merge_into_video(epoch)

if __name__ == '__main__':
    cfg.split = args.type
    if 'run_' + args.type in globals():
        globals()['run_' + args.type]()
    else:
        run_other(args.type)
