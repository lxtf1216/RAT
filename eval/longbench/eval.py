# put in LongBench/LongBench/ we use v1
import os
import json
import argparse
import numpy as np
from datasets import load_dataset
# from config.config import postprocess_dict
from config import postprocess_dict
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
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--dataset', type=str, default=None)
    return parser.parse_args(args)


    
def scorer(dataset, predictions, answers, all_classes):
    total_score = 0.
    for (prediction, ground_truths) in zip(predictions, answers):
        score = 0.
        prediction = postprocess_dict.get(dataset)(prediction)
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        total_score += score
    return round(100 * total_score / len(predictions), 2)


if __name__ == '__main__':
    args = parse_args()
    scores = dict()
    save_dir = "pred_wxy"
    path = f"pred_wxy/{args.model}/"
    all_files = os.listdir(path)
    print("Evaluating on:", all_files)
    for filename in all_files:
        if not filename.endswith("jsonl"):
            continue
        predictions, answers, lengths = [], [], []
        dataset = filename.split('.')[0]
        if dataset != args.dataset and args.dataset is not None:
            continue
        if dataset == "trec":
            # orig_dataset = load_dataset('THUDM/LongBench', dataset, split='test')
            # datafile = f"/data8/zhangxin/ljc/RAT/datasets/LongBench/data/{dataset}.jsonl"
            orig_dataset = load_dataset('json', data_files=f"/data8/zhangxin/ljc/RAT/datasets/LongBench/data/{dataset}.jsonl", split='train')
            classes = list(orig_dataset["all_classes"])
        else:
            classes = None
        index = 0
        with open(f"{path}{filename}", "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                all_classes = None if classes is None else classes[index]
                index += 1
                predictions.append(data["pred"])
                answers.append(data["answers"])
        if len(predictions) != 0:
            score = scorer(dataset, predictions, answers, all_classes=all_classes)
        print(dataset, score)
        scores[dataset] = score
    out_path = f"{save_dir}/{args.model}/result.json"
    with open(out_path, "w") as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)
    print(scores)