import os
import sys
import hydra
import wandb
import math
import torch
import random
import time
import numpy as np
from easydict import EasyDict
from triton.testing import do_bench
import torch.distributed as dist
from omegaconf import OmegaConf
from omegaconf.listconfig import ListConfig
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(project_root)
sys.path.insert(0, project_root)
from src.utils.registry import get_all_registries
registries = get_all_registries()
import src.model
import src.task
from src.task.task import LMTask
import src.optim
import src.data  # to load all the things into registries
for registry in registries:
    registry._is_register = False
from src.utils import config as util_config
from src.model.backbone.cache import AttentionCache, LocalAttentionCache, RNNCache, RATCache
from src.utils.registry import (
    data_registry,
    task_registry,
    lr_scheduler_registry,
    optimizer_registry,
    metric_registry
)
import src.utils.gen as gen_util


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


@torch.no_grad()
@torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16)
def bench_gen(seq_len, batch_size, config, task: LMTask, seq_start: int):
    task.eval()
    config.data.batch_size = batch_size
    cache = gen_util.get_cache(config)
    def prepare_cache():
        if isinstance(cache, tuple):
            cache[0].reset_cache()
            cache[1].reset_cache()
        else:
            cache.reset_cache()
        if isinstance(cache, tuple) and isinstance(cache[0], LocalAttentionCache):
            cache[0].seq_start = seq_start - 1
        elif isinstance(cache, tuple) and isinstance(cache[1], LocalAttentionCache):
            cache[1].seq_start = seq_start - 1
        elif isinstance(cache, LocalAttentionCache):
            cache.seq_start = seq_start - 1
    prepare_cache()
    # warmup
    cur_inp = torch.ones(batch_size, 1, dtype=torch.long).cuda()
    for i in range(seq_start, seq_len):
        if isinstance(cache, tuple):
            cache[0].set_seq(i)
            cache[1].set_seq(i)
        else:
            cache.set_seq(i)
        task.step(cur_inp, seq_start=i, cache=cache)
    print("finish warmup!")
    prepare_cache()

    torch.cuda.synchronize()
    start_time = time.time()
    for i in range(seq_start, seq_len):
        if isinstance(cache, tuple):
            cache[0].set_seq(i)
            cache[1].set_seq(i)
        else:
            cache.set_seq(i)
        task.step(cur_inp, seq_start=i, cache=cache)
    torch.cuda.synchronize()
    end_time = time.time()
    total_time = end_time - start_time
    return total_time


@torch.no_grad()
@torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16)
def check_mx_bs(task: LMTask, config, seq_start):

    for batch_size in range(400000, 500000, 16):
        torch._dynamo.reset()
        try:
            config.data.batch_size = batch_size
            cache = gen_util.get_cache(config)
            if isinstance(cache, tuple):
                cache[0].reset_cache()
                cache[1].reset_cache()
            else:
                cache.reset_cache()
            cur_inp = torch.ones(batch_size, 1, dtype=torch.long).cuda()

            if isinstance(cache, tuple) and isinstance(cache[0], LocalAttentionCache):
                cache[0].seq_start = seq_start - 1
            elif isinstance(cache, tuple) and isinstance(cache[1], LocalAttentionCache):
                cache[1].seq_start = seq_start - 1
            elif isinstance(cache, LocalAttentionCache):
                cache.seq_start = seq_start - 1
            if isinstance(cache, tuple):
                cache[0].set_seq(seq_start)
                cache[1].set_seq(seq_start)
            else:
                cache.set_seq(seq_start)
            task.step(input_ids=cur_inp, seq_start=seq_start, cache=cache)
            del cache
            torch.cuda.empty_cache()
        except RuntimeError as e:
            torch.cuda.empty_cache()
            print("the batch size is {}".format(batch_size - 16))
            raise e


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="config",
)
def main(config):
    print(config)
    set_seed(1005)
    config = EasyDict(OmegaConf.to_container(config, resolve=True, throw_on_missing=True))
    task = (util_config.instantiate(task_registry,
                                    config=config.task,
                                    model_config=config.model,
                                    device="cuda",
                                    dtype=torch.float32))
    print(task)
    task = task.to("cuda").to(torch.bfloat16)
    torch._dynamo.reset()
    task = torch.compile(task)
    seq_start = 3072
    # check_mx_bs(task, config, seq_start)
    torch.cuda.empty_cache()
    gen_ms = bench_gen(config.data.seq_len, batch_size=736, config=config, task=task, seq_start=seq_start)
    # print("the context is {:.2f}ms, and the gen is {:.2f}s".format(context_ms, gen_ms))
    print("the gen is {:.2f}s".format(gen_ms))


if __name__ == "__main__":
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.register_new_resolver("int", int)
    gpu_id = int(os.getenv("RANK", -1))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", -1))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", world_size=world_size, rank=gpu_id, init_method="env://")
    main()
    dist.destroy_process_group()
