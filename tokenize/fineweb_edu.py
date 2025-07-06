from datasets import load_dataset
from transformers import AutoTokenizer
from argparse import ArgumentParser
import os
import numpy as np
from tqdm import tqdm
from transformers import LlamaTokenizer


long_path = "/capstor/store/cscs/swissai/a06/datasets_raw/fineweb-edu-sample-100BT/sample/100BT/"
cache_dir = "/capstor/store/cscs/swissai/a10/datasets_tokenized/fineweb_edu_100B/llama_files/"


def tokenize(tokenizer, num_proc, dataset):
    if tokenizer == "llama":
        enc = LlamaTokenizer.from_pretrained("huggyllama/llama-7b")
        eos_tokens = enc("</s>", truncation=False, padding=False, add_special_tokens=False)["input_ids"]
        def tokenize_process(example):
            ids = enc(example["text"], truncation=False, padding=False, add_special_tokens=False)["input_ids"]
            ids = ids + eos_tokens
            out = {'ids': ids, 'len': len(ids)}
            return out
    else:
        raise NotImplementedError

    tokenized = dataset.map(
        tokenize_process,
        remove_columns=["text", "url", "dump", "token_count"],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )
    print(tokenized)
    return tokenized


def save_to_npmemmap(split, dset, tokenizer, path):
    print(split)
    filename = os.path.join(cache_dir, f"{path}.bin")
    dtype = np.uint16  # (can do since enc.max_token_value == 32000 is < 2**16)
    arr_len = np.sum(dset['len'], dtype=np.uint64)
    arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))
    total_batches = 1024

    idx = 0
    for batch_idx in tqdm(range(total_batches), desc=f"writing {filename}"):
        # Batch together samples for faster write
        batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format("numpy")
        # Write into mmap
        arr_batch = np.concatenate(batch["ids"])
        arr[idx : idx + len(arr_batch)] = arr_batch
        idx += arr_batch.shape[0]
    arr.flush()


def parse_args():
    parser = ArgumentParser(description="Convert dataset into MDS format, optionally concatenating and tokenizing")
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--num_proc", type=int, required=True, default=None)
    return parser.parse_args()


def main(args):
    print(args.num_proc)
    new_dataset = []
    for i in range(14):
        for j in range(10):
            str_name = f"0{i}_0000{j}" if i >= 10 else f"00{i}_0000{j}"
            fineweb_chunk = load_dataset(path=long_path, split="train", data_files=f"{str_name}.parquet")
            print("begin to tokenize!")
            new_dataset = tokenize(args.tokenizer, args.num_proc, fineweb_chunk)
            new_dataset = new_dataset.train_test_split(test_size=0.005, seed=1005, shuffle=True)

            save_to_npmemmap("train", new_dataset["train"], args.tokenizer, str_name + "_train")
            save_to_npmemmap("val", new_dataset["test"], args.tokenizer, str_name + "_val")
            print(f"Finish {i} and {j}")


def aggregate_files(split):
    num_tokens = 0
    dtype = np.uint16
    for i in range(14):
        for j in range(10):
            str_name = f"0{i}_0000{j}" if i >= 10 else f"00{i}_0000{j}"
            path = str_name + f"_{split}"
            filename = os.path.join(cache_dir, f"{path}.bin")
            arr = np.memmap(filename, dtype=dtype, mode="r")
            num_tokens += len(arr)
    print(num_tokens)
    new_path = "/capstor/store/cscs/swissai/a10/datasets_tokenized/fineweb_edu_100B"
    new_file = os.path.join(new_path, f"llama-{split}.bin")
    new_arr = np.memmap(new_file, dtype=dtype, mode="w+", shape=(num_tokens, ))

    idx = 0
    for i in range(14):
        for j in range(10):
            str_name = f"0{i}_0000{j}" if i >= 10 else f"00{i}_0000{j}"
            path = str_name + f"_{split}"
            filename = os.path.join(cache_dir, f"{path}.bin")
            arr = np.memmap(filename, dtype=dtype, mode="r")

            new_arr[idx : idx + len(arr)] = arr[:]
            idx += len(arr)
        new_arr.flush()


if __name__ == "__main__":
    # main(parse_args())
    aggregate_files("train")
    aggregate_files("val")
