# put in RULER/scripts/eval
import os
import json
import argparse
import numpy as np


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--task', type=str, default=None)
    return parser.parse_args(args)


import re


def extract_first_isolated_number(predict_str: str) -> str | None:
    predict_str = predict_str.strip()
    match = re.search(r'\b(\d+)\b', predict_str)
    return match.group(1) if match else None


def clean_and_extract_uid(predict_str: str) -> str | None:
    first_part = re.split(r'[\n\.]', predict_str, maxsplit=1)[0]
    first_part = first_part.strip().lstrip(':：\"\' ')

    return first_part

def scorer(dataset, predictions, answers):
    total_score = 0.
    for (prediction, ground_truths) in zip(predictions, answers):
        # if dataset in ["niah_single_1", "niah_single_2"]:
        #     prediction = extract_first_isolated_number(prediction)
        # elif dataset in ["niah_single_3"]:
        #     prediction = clean_and_extract_uid(prediction)
        prediction =  sorted(x.strip() for x in prediction.split(","))
        ground_truths = sorted(x.strip() for x in ground_truths)
        print(prediction, ground_truths)
        # if prediction == ground_truths:
        #     total_score += 1
        overlap = set(prediction) & set(ground_truths)
        total_score += len(overlap) / len(ground_truths)
    print(total_score)
    return round(100 * total_score / len(predictions), 2)



if __name__ == '__main__':
    args = parse_args()
    scores = dict()
    cache_dir = f"/home/xwei/fake_path/sequence_model/exp/ckpt/needle_eval/"
    path_dict = {
        "rnn": os.path.join(cache_dir, f"{args.task}4096-lm-lmposempty-sequenced2048l24-rnn-ffn-lm/"),
        "ratl16": os.path.join(cache_dir, f"{args.task}4096-lm-lmposinterrope10000-sequenced2048l24-ratl16-ffn-lm/"),
        "ratl64": os.path.join(cache_dir, f"{args.task}4096-lm-lmposinterrope10000-sequenced2048l24-ratl64-ffn-lm/"),
        "attention": os.path.join(cache_dir, f"{args.task}4096-lm-lmposrope10000-sequenced2048l24-attention-ffn-lm/"),
    }
    path = path_dict.get(args.model)
    all_lr = os.listdir(path)
    print("Evaluating on:", all_lr)
    for lr_dir in all_lr:
        predictions, answers, lengths = [], [], []
        dataset = args.task
        if not os.path.exists(f"{path}{lr_dir}/{args.task}.jsonl"):
            continue
        index = 0
        with open(f"{path}{lr_dir}/{args.task}.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                predictions.append(data["pred"])
                answers.append(data["outputs"])
        score = scorer(dataset, predictions, answers)
        scores[lr_dir] = score
    out_path = os.path.join(path, "result.json") # f"pred/{args.model}/result.json"
    with open(out_path, "w") as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)
    print(scores)