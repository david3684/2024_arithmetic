import os
import pickle

import numpy as np
import torch

def get_logits(inputs, classifier):
    assert callable(classifier)
    if hasattr(classifier, "to"):
        classifier = classifier.to(inputs.device)
    return classifier(inputs)

def normalize_weight(weight, rotation_matrix, fnorm):
    """Normalize the finetuned weight."""
    weight = weight.view(-1)
    assert rotation_matrix.shape[1] == weight.shape[0], "Rotation matrix and weight shape mismatch"
    weight = torch.matmul(weight, rotation_matrix)
    weight = weight / fnorm
    return weight

def find_optimal_coef(
    results,
    metric="avg_normalized_top1",
    minimize=False,
    control_metric=None,
    control_metric_threshold=0.0,
):
    best_coef = None
    if minimize:
        best_metric = 1
    else:
        best_metric = 0
    for scaling_coef in results.keys():
        if control_metric is not None:
            if results[scaling_coef][control_metric] < control_metric_threshold:
                print(f"Control metric fell below {control_metric_threshold} threshold")
                continue
        if minimize:
            if results[scaling_coef][metric] < best_metric:
                best_metric = results[scaling_coef][metric]
                best_coef = scaling_coef
        else:
            if results[scaling_coef][metric] > best_metric:
                best_metric = results[scaling_coef][metric]
                best_coef = scaling_coef
    return best_coef