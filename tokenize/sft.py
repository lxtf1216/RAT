from argparse import ArgumentParser
import os
import copy
import numpy as np
from tqdm import tqdm
from transformers import LlamaTokenizer,LlamaTokenizerFast
from sft_util import PROMPT_DICT, load_sft_dataset

prefill_max_len_dict = {
    "narrativeqa_summary": {
        4096: 3840,
        16384: 16256,
    },
    "narrativeqa_text": {
        4096: 3840,
        16384: 16256,
    },
    "narrativeqa_full": {
        4096: 3840,
        16384: 16256,
    },
    "qmsum": {
        4096: 3712, # to generate 384 tokens at most 300 generated tokens
        16384: 16000,
    },
    "wikisum": {
        4096: 3840,
        16384: 16128,
    }
}


cache_dir = "/data8/zhangxin/ljc/RAT/datasets_tokenized/sft"


def tokenize_answer_only(dataset, task, num_proc, max_length, split="train"):
    org_max_length = max_length
    # enc = LlamaTokenizer.from_pretrained("/data8/zhangxin/ljc/RAT/llama-7b-tokenizer")
    enc = LlamaTokenizerFast.from_pretrained("/data8/zhangxin/ljc/RAT/llama-7b-tokenizer")
    if split == "test":
        max_length = prefill_max_len_dict.get(task).get(max_length)
    def tokenize_process(example):
        prompt_template = PROMPT_DICT[task]
        prompt = prompt_template.format_map(example)
        ids = enc(prompt, truncation=False, padding=False, add_special_tokens=False)["input_ids"]
        labels = [-100] * (len(ids) - 1)
        if split != "test":
            answer_ids = enc(example["answer"], truncation=False, padding=False, add_special_tokens=False)["input_ids"]
            ids = ids + answer_ids
            labels = labels + answer_ids
        # truncation the middle part
        if len(ids) >= max_length:
            half = (max_length - 1) // 2
            second_half = max_length - 1 - half
            ids = ids[:half] + ids[-second_half:]
            labels = labels[:half] + labels[-second_half + 1:]
            assert len(ids) < max_length, len(ids)
        input_ids = ids + [2] * (org_max_length - len(ids))
        labels = labels + [2] + [-100] * (org_max_length - len(ids))
        assert len(input_ids) == len(labels)
        out = {'input_ids': input_ids, 'labels': labels}
        return out

    tokenized = dataset.map(
        tokenize_process,
        desc="tokenizing the splits",
        num_proc=num_proc,
    )
    print(tokenized)
    print(tokenized["input_ids"][0], tokenized["labels"][0])
    return tokenized


def save_to_npmemmap(split, dset, path, max_length):
    filename = os.path.join(cache_dir, f"{path}.bin")
    arr_len = dset.num_rows
    dtype = np.int16  # (can do since enc.max_token_value == 32000 is < 2**16)
    os.makedirs(os.path.dirname(filename), exist_ok=True) 
    arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len, max_length))
    total_batches = 10

    idx = 0
    for batch_idx in tqdm(range(total_batches), desc=f"writing {filename}"):
        # Batch together samples for faster write
        batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True)
        # Write into mmap
        arr_batch = np.stack(batch[split])
        arr[idx : idx + arr_batch.shape[0], :] = arr_batch
        idx += arr_batch.shape[0]
    arr.flush()


def main(args):
    global cache_dir
    cache_dir = os.path.join(cache_dir, args.task)
    dset = load_sft_dataset(args.task, args.split)
    print(dset)
    new_dset = tokenize_answer_only(dset, args.task, args.num_proc, args.max_length, args.split) # get the train split here
    save_to_npmemmap("input_ids", new_dset, f"llama-{args.split}-{args.max_length}-inputs", args.max_length)
    save_to_npmemmap("labels", new_dset, f"llama-{args.split}-{args.max_length}-labels", args.max_length)


if __name__ == "__main__":
    parser = ArgumentParser(description="Convert dataset into MDS format, optionally concatenating and tokenizing")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--num_proc", type=int, required=True, default=None)
    parser.add_argument("--max_length", type=int, required=True, default=32768)
    parser.add_argument("--split", type=str, required=True)
    main(parser.parse_args())
