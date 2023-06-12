import time
import datetime
import torch
import tqdm
from torch.nn import DataParallel
from lib.config import cfg
from lib.networks.renderer.make_renderer import make_renderer
from lib.visualizers.make_visualizer import make_visualizer
from termcolor import colored

from lib.utils.base_utils import DotDict

# global cuda context dict
cuda_context = DotDict()  # will replace with previous stream once training started


class Trainer(object):
    def __init__(self, network):
        device = torch.device('cuda:{}'.format(cfg.local_rank))
        network = network.to(device)
        if cfg.distributed:
            network = torch.nn.parallel.DistributedDataParallel(
                network,
                device_ids=[cfg.local_rank],
                output_device=cfg.local_rank
            )
        self.network = network
        self.local_rank = cfg.local_rank
        self.device = device

        self.scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp)
        self.timer = {
            'begin': 0,
            'end': 0
        }

    def reduce_loss_stats(self, loss_stats):
        reduced_losses = {k: torch.mean(v) for k, v in loss_stats.items()}
        return reduced_losses

    def to_cuda(self, batch):
        if isinstance(batch, tuple) or isinstance(batch, list):
            batch = [self.to_cuda(b) for b in batch]
            return batch

        for k in batch:
            if k == 'meta':
                continue
            if isinstance(batch[k], tuple) or isinstance(batch[k], list):
                batch[k] = [b.to(self.device, non_blocking=True) for b in batch[k]]
            else:
                batch[k] = batch[k].to(self.device, non_blocking=True)

        return batch

    def add_iter_step(self, batch, iter_step):
        if isinstance(batch, tuple) or isinstance(batch, list):
            for batch_ in batch:
                self.add_iter_step(batch_, iter_step)

        if isinstance(batch, dict):
            batch['iter_step'] = iter_step

    def train(self, epoch, data_loader, optimizer, recorder):
        global prev_stream
        ep_iter = cfg.ep_iter
        self.network.train()
        end = time.time()

        device_prefetch = cfg.device_prefetch
        assert len(data_loader) >= device_prefetch
        # assert device_prefetch >= 2

        iterator = iter(data_loader)
        not_finished = True

        stream_index = 0  # current stream + 1

        if cfg.multi_stream:
            cuda_streams = [torch.cuda.Stream() for _ in range(device_prefetch)]
            cuda_context.cuda_streams = cuda_streams

            next_datas = []
            for i in range(device_prefetch):
                with torch.cuda.stream(cuda_streams[i]):
                    next_datas.append(self.to_cuda(next(iterator)))
        else:
            next_datas = [self.to_cuda(next(iterator)) for _ in range(device_prefetch)]

        for index in range(ep_iter):  # last round of data and we end up in the first data (will this skip a step?)
            if not next_datas:  # no more data, do not train
                return

            if cfg.profiling:
                from train_net import prof
                # print(f'profiler_id: {id(prof)}')
                prof.step()

            stream_index = (stream_index + 1) % device_prefetch

            if cfg.multi_stream:
                cuda_context.prev_stream = cuda_streams[stream_index-1]  # utilizing python -1 machanism
                cuda_context.curr_stream = cuda_streams[stream_index]
                cuda_context.stream_index = stream_index
                torch.cuda.set_stream(cuda_streams[stream_index])
                # torch.cuda.current_stream().wait_stream(cuda_streams[prev_index])  # wait for previous stream update to finish

            batch = next_datas[stream_index]

            # forward
            data_time = time.time() - end  # including moving to cuda time

            iteration = (index + 1) % ep_iter
            self.add_iter_step(batch, index + 1)

            with torch.cuda.amp.autocast(enabled=cfg.use_amp):
                output, loss, loss_stats, image_stats = self.network(batch, epoch, split='train')

            if iteration == 1 and self.timer['begin'] == 0:
                print(colored(f"[*] First forward pass is done, this usually means cuda is initialized", 'green'))
                print(colored(f"Start timer", 'green'))
                self.timer['begin'] = time.time()

            if not_finished:  # asynchronous data prefetching to GPU
                try:
                    if cfg.multi_stream:
                        # utilizing python -1 machanism
                        with torch.cuda.stream(cuda_streams[stream_index - 1]):
                            next_datas[stream_index-1] = self.to_cuda(next(iterator))  # non blocking loading within backward pass
                    else:
                        # utilizing python -1 machanism
                        next_datas[stream_index-1] = self.to_cuda(next(iterator))  # non blocking loading within backward pass

                except StopIteration:
                    not_finished = False

            # backward
            loss = loss.mean()

            optimizer.zero_grad(set_to_none=True)
            # loss.backward()
            self.scaler.scale(loss).backward()

            # if cfg.use_amp:
            #     self.scaler.unscale_(optimizer)
            #     torch.nn.utils.clip_grad_value_(self.network.parameters(), 40.0)  # this is slow...
            # optimizer.step()
            self.scaler.step(optimizer)
            self.scaler.update()

            if cfg.local_rank > 0:
                continue

            # data recording stage: loss_stats, time, image_stats
            recorder.step += 1

            batch_time = time.time() - end
            end = time.time()

            if (iteration % cfg.log_interval == 0 or iteration == (ep_iter - 1)) and not cfg.silent:
                loss_stats = self.reduce_loss_stats(loss_stats)
                recorder.update_loss_stats(loss_stats)
                # print training state
                eta_seconds = batch_time * (ep_iter - iteration)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                lr = optimizer.param_groups[0]['lr']
                memory = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0

                recorder.update_loss_stats({"lr": torch.Tensor([lr]), 'data': torch.Tensor([data_time]), 'batch': torch.Tensor([batch_time])})

                training_state = '  '.join(['expname: {}, eta: {}', '{}', 'max_mem: {:.0f}', 'gpu: {}'])
                training_state = training_state.format(colored(cfg.exp_name, "cyan"), eta_string, str(recorder), memory, cfg.gpus)
                if "thresh" in batch:
                    print(batch['thresh'])
                print(training_state)

            if iteration % cfg.record_interval == 0 or iteration == (ep_iter - 1):
                # record loss_stats and image_dict
                recorder.update_image_stats(image_stats)
                recorder.record('train')

            if cfg.debug and iteration > 1000:
                exit(-1)

            data_loader.dataset.update_global(output, batch)

    def val(self, epoch, data_loader, evaluator=None, recorder=None):
        cfg.global_test_switch = True
        self.network.eval()
        val_loss_stats = {}
        data_size = len(data_loader)
        val_image_stats = {}
        outputs = []
        batches = []
        losses = []
        images = []

        for batch in tqdm.tqdm(data_loader):
            batch = self.to_cuda(batch)
            with torch.no_grad():
                output, loss, loss_stats, image_stats = self.network(batch, split='val')

            outputs.append(output)
            batches.append(batch)
            losses.append(loss_stats)
            images.append(image_stats)

        for index in tqdm.tqdm(range(data_size)):
            with torch.no_grad():
                if evaluator is not None:
                    evaluator.evaluate(outputs[index], batches[index], epoch)
            loss_stats = self.reduce_loss_stats(losses[index])
            image_stats = images[index]

            for k, v in loss_stats.items():
                val_loss_stats.setdefault(k, 0)
                val_loss_stats[k] += v

            for k, v in image_stats.items():
                val_image_stats.setdefault(k, [])
                val_image_stats[k].append(v)

        loss_state = []
        for k in val_loss_stats.keys():
            val_loss_stats[k] /= data_size
            loss_state.append('{}: {:.4f}'.format(k, val_loss_stats[k]))
        print(loss_state)

        if evaluator is not None:
            result = evaluator.summarize(epoch=epoch)
            val_loss_stats.update(result)

        if recorder:
            recorder.record('val', (epoch + 1) * cfg.ep_iter, val_loss_stats, val_image_stats)

        cfg.global_test_switch = False

    def vis(self, epoch, data_loader):
        self.network.eval()
        net = self.network.net
        renderer = make_renderer(cfg, net, vis=True)
        visualizer = make_visualizer(cfg, str(epoch))

        print(colored("Start visualization...", "blue"))

        cfg.global_test_switch = True

        with torch.no_grad():
            for batch in tqdm.tqdm(data_loader):
                batch = self.to_cuda(batch)
                with torch.no_grad():
                    output = renderer.render(batch, test=True)
                    visualizer.visualize(output, batch)
                # break

        cfg.global_test_switch = False

    def tmesh(self, epoch, data_loader):
        self.network.eval()
        net = self.network.net
        renderer = make_renderer(cfg, net, split="tmesh")
        visualizer = make_visualizer(cfg, str(epoch))

        print(colored("Start extracting tmesh...", "blue"))

        cfg.global_test_switch = True

        for batch in tqdm.tqdm(data_loader):
            batch = self.to_cuda(batch)
            with torch.no_grad():
                output = renderer.render(batch, test=True)
                visualizer.visualize(output, batch, split='tmesh')
            # break

        cfg.global_test_switch = False

    def print_time_elapsed(self):
        print(colored('Training finished. Time elapsed: {} seconds'.format((self.timer['end'] - self.timer['begin'])), 'green'))