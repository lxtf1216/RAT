# After generating answers for SFT datasets, we use LongBench's metric functions to evaluate answers' quality.
# put it in LongBench/LongBench
import os
import json
import argparse
import numpy as np
import evaluate

from metrics import (
    qa_f1_score,
    rouge_zh_score,
    qa_f1_zh_score,
    rouge_score,
    classification_score,
    retrieval_score,
    retrieval_zh_score,
    count_score,
    code_sim_score,
)

dataset2metric = {
    "narrativeqa_summary": qa_f1_score,
    "narrativeqa_text": qa_f1_score,
    "narrativeqa_full": qa_f1_score,
    "wikisum": rouge_score,
    "qmsum": rouge_score,
}

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--task', type=str, default=None)
    return parser.parse_args(args)

def scorer(dataset, predictions, answers):
    total_score = 0.
    for (prediction, ground_truths) in zip(predictions, answers):
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth))
        total_score += score
    return round(100 * total_score / len(predictions), 2)


if __name__ == '__main__':
    args = parse_args()
    scores = dict()
    cache_dir = f"/home/xwei/fake_path/sequence_model/exp/ckpt/sft_{args.task}/"
    path_dict = {
        "rnn": os.path.join(cache_dir, f"{args.task}4096-lm-lmposempty-sequenced2048l24-rnn-ffn-lm/"),
        "ratl16": os.path.join(cache_dir, f"{args.task}4096-lm-lmposinterrope10000-sequenced2048l24-ratl16-ffn-lm/"),
        "attention": os.path.join(cache_dir, f"{args.task}4096-lm-lmposrope10000-sequenced2048l24-attention-ffn-lm/"),
        "rat_localattention": os.path.join(cache_dir, f"{args.task}4096-lm-lmposrope_interrope10000-sequence_interleaved2048l24-local_attention-ratl16-ffn-lm/"),
        "attention_localattention": os.path.join(cache_dir, f"{args.task}4096-lm-lmposrope10000-sequence_interleaved2048l24-local_attention-attention-ffn-lm/"),
    }
    path = path_dict.get(args.model)
    available_files = dataset2metric.keys()
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
                if not isinstance(data["answers"], list):
                    answers.append([data["answers"]])
                else:
                    answers.append(data["answers"])
        score = scorer(dataset, predictions, answers)
        scores[lr_dir] = score
    out_path = os.path.join(path, "result.json") # f"pred/{args.model}/result.json"
    with open(out_path, "w") as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)
    print(scores)