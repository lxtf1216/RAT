from datasets import load_dataset, load_from_disk, concatenate_datasets
import os
import random
import json
from collections import defaultdict


cache_dir = "/claire-rcp-scratch/shared/xwei/dataset/sft/downloads"


PROMPT_DICT = {
    "narrativeqa_summary": "Story: {context}\n\nQuestion: {input}\n\nAnswer:",
    "narrativeqa_text": "Story: {context}\n\nQuestion: {input}\n\nAnswer:",
    "narrativeqa_full": "Story: {context}\n\nQuestion: {input}\n\nAnswer:",
    "wikisum": "Article: {context}\n\nQuestion: {input}\n\nAnswer:",
    "qmsum": "Transcript: {context}\n\nQuestion: {input}\n\nAnswer:",
}


def save_test_answer(ds, task):
    path = os.path.join(os.path.dirname(cache_dir), f"{task}/test.jsonl")
    with open(path, 'w', encoding='utf-8') as f:
        for answer in ds['answer']:
            json.dump({"answer": answer}, f, ensure_ascii=False)
            f.write('\n')


def load_sft_dataset(task, split="train"):
    if task in ["narrativeqa_summary", "narrativeqa_text", "narrativeqa_full"]:
        ds = load_dataset("deepmind/narrativeqa", split=split)
        path = os.path.join(os.path.dirname(cache_dir), f"narrativeqa/{split}")
        if os.path.isdir(path):
            ds = load_from_disk(path)
        else:
            answer_choice = "answer1"
            def preprocess(example):
                summary = example["document"]["summary"]["text"]
                text = example["document"]["text"]
                full_text = summary + "\n\n" + text
                question = example["question"]["text"]
                answer1 = example["answers"][0]["text"]
                answer2 = example["answers"][1]["text"]
                if answer_choice == "answer1":
                    return {"question": question, "summary": summary, "text": text, "full_text": full_text, "answer": answer1}
                elif answer_choice == "answer2":
                    return {"question": question, "summary": summary, "text": text, "full_text": full_text, "answer": answer2}
                else:
                    return {"question": question, "summary": summary, "text": text, "full_text": full_text, "answer": [answer1, answer2]}
            if split == "test":
                answer_choice = "both"
                ds = ds.map(preprocess, remove_columns=ds.column_names, batched=False).shuffle(seed=1234)
                random.seed(1234)
                indices = random.sample(range(len(ds)), 200) # same as LongBench
                ds = ds.select(indices)
            else:
                answer_choice = "answer1"
                ds1 = ds.map(preprocess, remove_columns=ds.column_names, batched=False)
                answer_choice = "answer2"
                ds2 = ds.map(preprocess, remove_columns=ds.column_names, batched=False)
                ds = concatenate_datasets([ds1, ds2]).shuffle(seed=1234)
            ds.save_to_disk(path)
        if task == "narrativeqa_summary":
            ds = ds.rename_columns({"question": "input", "summary": "context", "answer": "answer"})
        elif task == "narrativeqa_text":
            ds = ds.rename_columns({"question": "input", "text": "context", "answer": "answer"})
        else:
            ds = ds.rename_columns({"question": "input", "full_text": "context", "answer": "answer"})
        print(ds["input"][0], ds["answer"][0])
        if split == "test":
            save_test_answer(ds, task)
        return ds
    elif task == "qmsum":
        if split == "test":
            ds = load_dataset('THUDM/LongBench', task, split=split, cache_dir=cache_dir)
            ds = ds.rename_columns({"input": "input", "context": "context", "answers": "answer"})
            save_test_answer(ds, task)
            return ds
        else:
            ds = load_dataset("pszemraj/qmsum-cleaned", "no-prefix", cache_dir=cache_dir, split=split).shuffle(seed=1234)
            ds = ds.rename_columns({"prompt": "input", "input": "context", "output": "answer"})
            return ds
    elif task == "wikisum":
        ds = load_dataset("d0rj/wikisum", split=split)
        ds = ds.rename_columns({"article": "context", "title": "input", "summary": "answer"})
        if split == "test":
            random.seed(1234)
            indices = random.sample(range(len(ds)), 200) # same as LongBench
            ds = ds.select(indices)
            save_test_answer(ds, task)
            return ds
        else:
            return ds.shuffle(seed=1234)
    else:
        raise NotImplementedError