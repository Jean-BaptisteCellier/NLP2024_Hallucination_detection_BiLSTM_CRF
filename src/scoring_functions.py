from typing import Dict, List
import random
from dataclasses import dataclass
from scipy.stats import spearmanr
import argparse as ap
import json
import pandas as pd
import numpy as np


# Code taken from https://helsinki-nlp.github.io/shroom/scorer.py

@dataclass
class SoftLabel:
    start: int
    end: int
    prob: float

    def to_dict(self) -> Dict[str, any]:
        return {
            "start": self.start,
            "end": self.end,
            "prob": self.prob
        }

@dataclass
class HardLabel:
    start: int
    end: int

    def to_list(self) -> List[int]:
        return [self.start, self.end]


@dataclass
class PredictionResult:
    id: str
    hard_labels: List[HardLabel]
    soft_labels: List[SoftLabel]

    def to_dict(self) -> Dict[str, any]:
        return {
            "id": self.id,
            "soft_labels": [soft_label.to_dict() for soft_label in self.soft_labels],
            "hard_labels": [hard_label.to_list() for hard_label in self.hard_labels]
        }

# Code taken from https://helsinki-nlp.github.io/shroom/scorer.py

def recompute_hard_labels(soft_labels):
    """optionally, infer hard labels from the soft labels provided"""
    hard_labels = []
    prev_end = -1
    for start, end in (
        (lbl['start'], lbl['end'])
        for lbl in sorted(soft_labels, key=lambda span: (span['start'], span['end']))
        if lbl['prob'] > 0.5
    ):
        if start == prev_end:
            hard_labels[-1][-1] = end
        else:
            hard_labels.append([start, end])
        prev_end = end
    return hard_labels

def prepare_ref_df(df: pd.DataFrame):
    if 'hard_labels' not in df.columns:
        df['hard_labels'] = df.soft_labels.apply(recompute_hard_labels)
    # adding an extra column for convenience
    df['text_len'] = df.model_output_text.apply(len)
    df = df[['id', 'soft_labels', 'hard_labels', 'text_len']]
    return df.sort_values('id').to_dict(orient='records'), df['text_len'].values

def prepare_result_df(df: pd.DataFrame, text_lens):
    if 'hard_labels' not in df.columns:
        df['hard_labels'] = df.soft_labels.apply(recompute_hard_labels)
    df = df[['id', 'soft_labels', 'hard_labels']]
    df['text_lens'] = text_lens
    return df.sort_values('id').to_dict(orient='records')

def score_iou(ref_dict, pred_dict):
    """computes intersection-over-union between reference and predicted hard labels, for a single datapoint.
    inputs:
    - ref_dict: a gold reference datapoint,
    - pred_dict: a model's prediction
    returns:
    the IoU, or 1.0 if neither the reference nor the prediction contain hallucinations
    """
    # ensure the prediction is correctly matched to its reference
    assert ref_dict['id'] == pred_dict['id']
    # convert annotations to sets of indices
    ref_indices = {idx for span in ref_dict['hard_labels'] for idx in range(*span)}
    pred_indices = {idx for span in pred_dict['hard_labels'] for idx in range(*span)}
    # avoid division by zero
    if not pred_indices and not ref_indices: return 1.
    # otherwise compute & return IoU
    return len(ref_indices & pred_indices) / len(ref_indices | pred_indices)

def score_cor(ref_dict, pred_dict):
    """computes Spearman correlation between predicted and reference soft labels, for a single datapoint.
    inputs:
    - ref_dict: a gold reference datapoint,
    - pred_dict: a model's prediction
    returns:
    the Spearman correlation, or a binarized exact match (0.0 or 1.0) if the reference or prediction contains no variation
    """
    # ensure the prediction is correctly matched to its reference
    assert ref_dict['id'] == pred_dict['id']
    # convert annotations to vectors of observations
    ref_vec = [0.] * ref_dict['text_len']
    pred_vec = [0.] * ref_dict['text_len']
    for span in ref_dict['soft_labels']:
        for idx in range(span['start'], span['end']):
            ref_vec[idx] = span['prob']
    for span in pred_dict['soft_labels']:
        for idx in range(span['start'], span['end']):
            pred_vec[idx] = span['prob']
    # constant series (i.e., no hallucination) => cor is undef
    if len({round(flt, 8) for flt in pred_vec}) == 1 or len({round(flt, 8) for flt in ref_vec}) == 1 :
        return float(len({round(flt, 8) for flt in ref_vec}) == len({round(flt, 8) for flt in pred_vec}))
    # otherwise compute Spearman's rho
    return spearmanr(ref_vec, pred_vec).correlation

# def score_result(ref_file, predictions: List[PredictionResult]) -> Dict[str, float]:
#     ref_dataframe, text_lens = prepare_ref_df(pd.read_json(ref_file, lines=True))
#     prediction_df = prepare_result_df(pd.DataFrame([result.to_dict() for result in predictions]), text_lens)
#     ious = np.array([score_iou(r, d) for r, d in zip(ref_dataframe, prediction_df)])
#     cors = np.array([score_cor(r, d) for r, d in zip(ref_dataframe, prediction_df)])
#     return {
#         'IoU': ious.mean(),
#         'Cor': cors.mean()
#     }

def score_result(ref: List[Dict], predictions: List[PredictionResult]) -> Dict[str, float]:
    ref_dataframe, text_lens = prepare_ref_df(pd.DataFrame(ref))
    prediction_df = prepare_result_df(pd.DataFrame([result.to_dict() for result in predictions]), text_lens)
    ious = np.array([score_iou(r, d) for r, d in zip(ref_dataframe, prediction_df)])
    cors = np.array([score_cor(r, d) for r, d in zip(ref_dataframe, prediction_df)])
    return {
        'IoU': ious.mean(),
        'Cor': cors.mean()
    }


def generate_random_ranges(text: str, coverage: float) -> List[HardLabel]:
    """
    Generate random ranges covering the specified percentage of the text length.
    """
    text_length = len(text)
    num_indices = int(text_length * coverage)
    indices = sorted(random.sample(range(text_length), num_indices))

    # Create contiguous ranges
    ranges = []
    start = indices[0]
    for i in range(1, len(indices)):
        if indices[i] != indices[i - 1] + 1:
            ranges.append(HardLabel(start=start, end=indices[i - 1] + 1))
            start = indices[i]
    ranges.append(HardLabel(start=start, end=indices[-1] + 1))

    return ranges


def create_random_baseline_instances(data: str) -> List[PredictionResult]:

    # Create PredictionResult instances
    prediction_results = []
    for entry in data:
        coverage = random.uniform(0.5, 0.90)  # Random coverage between 20-70%
        hard_labels = generate_random_ranges(entry['model_output_text'], coverage)
        prediction_results.append(PredictionResult(id=entry['id'], hard_labels=hard_labels, soft_labels=[]))

    return prediction_results


def merge_ranges(ranges: List[List[float]]) -> List[List[float]]:
    """Merge overlapping or adjacent ranges."""
    if not ranges:
        return []

    # Sort ranges by start value
    ranges.sort(key=lambda x: x[0])
    merged = [ranges[0]]
    
    for current in ranges[1:]:
        last = merged[-1]
        # Merge if the gap is 1 or less
        if current[0] - last[1] <= 1:
            last[1] = max(last[1], current[1])
        else:
            merged.append(current)
    return merged


def process_json_lines(file_path: str) -> List[PredictionResult]:
    data_by_id = {}

    # Read lines from file
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            entry = json.loads(line)
            if entry['predicted_class'] == 1.0:
                record_id = entry['id']
                if record_id not in data_by_id:
                    data_by_id[record_id] = []
                data_by_id[record_id].append([entry['start'], entry['end']])

    # Merge ranges and create HardLabels
    results = []
    for record_id, ranges in data_by_id.items():
        merged_ranges = merge_ranges(ranges)
        hard_labels = [HardLabel(start=int(start), end=int(end)) for start, end in merged_ranges]
        results.append(PredictionResult(id=record_id, hard_labels=hard_labels, soft_labels=[]))

    return results


def filter_truth_by_prediction_ids(truth_path: str, prediction_results: List[PredictionResult]) -> List[Dict]:
    # Extract the set of IDs from prediction results
    prediction_ids = {result.id for result in prediction_results}

    # Read the truth file and filter by ID
    filtered_truth = []
    with open(truth_path, 'r') as file:
        for line in file:
            entry = json.loads(line)
            if entry['id'] in prediction_ids:
                filtered_truth.append(entry)
    
    return filtered_truth