# put in LongBench/LongBench pred. we use v1
import os
import torch
import json
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)
import numpy as np
import random
import argparse
from tqdm import tqdm
from easydict import EasyDict
from datasets import load_dataset
from omegaconf import OmegaConf
from transformers import LlamaTokenizerFast
import hydra
# from config import dataset2maxlen_dict, dataset2prompt_dict, datasettype_dict
model_root = "/data8/zhangxin/rat/fineweb_llama4096-lm-lmposinterrope10000-sequenced2048l24-ratl16-ffn-lm/"
sys.path.insert(0, model_root)
import src.utils.config as util_config
from src.utils.registry import task_registry
from src.utils import convert_load_ckpt
from src.utils import gen as gen_util
from src.model.backbone.cache import LocalAttentionCache
from config import dataset2maxlen_dict, dataset2prompt_dict, datasettype_dict

def load_model_from_hydra_config(hydra_overrides):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.register_new_resolver("int", int)
    #with hydra.initialize(config_path="../../../sequence_models/configs/", version_base=None):
    with hydra.initialize(config_path="../../configs/"):
        config = hydra.compose(
            config_name="experiment/fineweb_edu/rat-xl",
            overrides=[x for x in hydra_overrides.split(',')]
        )
    config = EasyDict(OmegaConf.to_container(config, resolve=True, throw_on_missing=True))
    return config


class ModelWrapper:

    def __init__(self, hydra_overrides):
        self.config = load_model_from_hydra_config(hydra_overrides)
        torch.serialization.add_safe_globals([EasyDict])
        self.model = (util_config.instantiate(task_registry, 
                                              config=self.config.task,
                                              model_config=self.config.model,
                                              device="cuda", dtype=torch.float32))
        if self.config.trainer.pretrained_path is not None:
            convert_load_ckpt.convert(self.model, self.config.trainer.pretrained_path)
        self.model = self.model.to("cuda").to(torch.float32)
        self.model = torch.compile(self.model)
        self.model.eval()
        print(self.model)
        self.repetition_penalty = 1.0

    @torch.no_grad()
    @torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16)
    def generate_single(self, input_ids, max_new_tokens, enc):
        cache = gen_util.get_cache(self.config)
        if isinstance(cache, tuple):
            cache[0].reset_cache()
            cache[1].reset_cache()
        else:
            cache.reset_cache()
        pos = input_ids[0].tolist().index(self.config.data.ignore_input_index) - 1
        preds = self.model(input_ids=input_ids, seq_start=0, cache=cache, seq_end=pos + 1).to(torch.float32)
        if isinstance(cache, tuple) and isinstance(cache[0], LocalAttentionCache):
            cache[0].seq_start = min(cache[0].window_size - 1, pos)
        elif isinstance(cache, tuple) and isinstance(cache[1], LocalAttentionCache):
            cache[1].seq_start = min(cache[1].window_size - 1, pos)
        elif isinstance(cache, LocalAttentionCache):
            cache.seq_start = min(cache.window_size - 1, pos)
        start_token = torch.argmax(preds[:, pos: pos + 1], dim=-1)
        generated = gen_util.generate_greedy_search(self.model, cache, self.config, start_token, pos + 1, max_new_tokens, enc, self.repetition_penalty)
        return generated


def get_pred(data, model: ModelWrapper, enc, prompt_template: str, max_prefill, max_gen, max_len, out_path):
    print(f"max_prefill {max_prefill}, max_gen {max_gen}, max gen {max_gen}")
    fout = open(out_path, "w", encoding="utf-8")
    for json_obj in tqdm(data):
        prompt = prompt_template.format_map(json_obj)
        ids = enc(prompt, truncation=False, padding=False, add_special_tokens=False)["input_ids"]
        # truncation the middle part
        if len(ids) >= max_prefill:
            half = (max_prefill - 1) // 2
            second_half = max_prefill - 1 - half
            ids = ids[:half] + ids[-second_half:]
            assert len(ids) < max_prefill, len(ids)
        # we run prefill by using max len to avoid triton compile error
        input_ids = ids + [2] * (max_len - len(ids))
        input_ids = torch.tensor(input_ids).unsqueeze(0).cuda()
        pred = model.generate_single(
            input_ids=input_ids,
            max_new_tokens=max_gen,
            enc = enc) # we do post process in evaluation stage
        json.dump({"pred": pred, "answers":  json_obj["answers"]}, fout, ensure_ascii=False)
        fout.write('\n')
    fout.close()


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hydra_overrides', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--max_len', type=int, default=4096)
    parser.add_argument('--data_type', type=int, default=1) 
    args = parser.parse_args()
    seed_everything(42)
    # build model
    model = ModelWrapper(args.hydra_overrides)
    enc = LlamaTokenizerFast.from_pretrained("/data8/zhangxin/ljc/RAT/llama-7b-tokenizer")
    # save_dir = "pred_wxy"
    save_dir = "pred_wxy/sft"
    datasets = []
    datasets = datasettype_dict.get(args.data_type)
    # for i in range(1, 6):
    #     datasets = datasets + datasettype_dict.get(i)
    for dataset in datasets:
        datafile = f"/data8/zhangxin/ljc/RAT/datasets/LongBench/data/{dataset}.jsonl"
        data = load_dataset('json', data_files=datafile, split='train')
        #data = load_dataset('THUDM/LongBench', dataset, split='test',trust_remote_code=True)
        if not os.path.exists(f"{save_dir}/{args.model_name}/"):
            os.makedirs(f"{save_dir}/{args.model_name}")
        out_path = f"{save_dir}/{args.model_name}/{dataset}.jsonl"
        prompt_template = dataset2prompt_dict[dataset]
        print(prompt_template.format_map(data[0]))
        get_pred(data, model, enc,
                 prompt_template,
                 max_gen=dataset2maxlen_dict[dataset], 
                 max_len=args.max_len,
                 max_prefill=args.max_len - dataset2maxlen_dict[dataset],
                 out_path=out_path)