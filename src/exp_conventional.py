import torch
import sys
import numpy as np
from src.eval import eval_single_dataset_with_prediction, eval_single_dataset
from src.main import save_scale_factors
from src.args import parse_arguments
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.datasets.registry import get_dataset
from src.task_vectors import TaskVector
from tqdm import tqdm
from copy import deepcopy
import matplotlib.pyplot as plt
from datetime import datetime
import pickle
import os
import open_clip


def calculate_weight_difference(original_state_dict, formatted_state_dict):

    differences = {}

    for key in original_state_dict.keys():
        if key in formatted_state_dict:
            # import ipdb; ipdb.set_trace()
            original_weight = original_state_dict[key]
            formatted_weight = formatted_state_dict[key]
            # difference = torch.abs(original_weight - formatted_weight).sum().item()
            difference = torch.mean(
                (original_weight - formatted_weight)**2).item()
            differences[key] = difference
        else:
            print(f"Key {key} not found in formatted model state dict.")

    return differences


args = parse_arguments()
args.model = 'ViT-L-14'
args.tasks = ['DTD', 'SUN397']
args.device = 'cuda'
args.task_scale_factors = None
args.save = 'checkpoints/ViT-L-14'
args.data_location = '/data2/david3684/data'
args.no_shared_weight = True
args.pretrained_model = 'openai'
args.num_test_samples = 2048
zero_shot_encoder = torch.load(
    "/data2/david3684/2024_arithmetic/checkpoints/ViT-L-14/zeroshot.pt").to(args.device)

finetuned_model_0 = torch.load(
    f"/data2/david3684/2024_arithmetic/checkpoints/ViT-L-14/{args.tasks[0]}/finetuned.pt").to(args.device)
finetuned_model_1 = torch.load(
    f"/data2/david3684/2024_arithmetic/checkpoints/ViT-L-14/{args.tasks[1]}/finetuned.pt").to(args.device)
_, _, val_preprocess = open_clip.create_model_and_transforms(
    args.model, pretrained='openai', cache_dir=args.openclip_cachedir)
dataset_2 = get_dataset(
    args.tasks[1],
    val_preprocess,
    location=args.data_location,
    batch_size=args.batch_size,
    num_workers=16,
    num_test_samples=args.num_test_samples,
)
dataloader_2 = get_dataloader(
    dataset_2, is_train=False, args=args, image_encoder=None)

task_vector_0 = TaskVector(
    args, zero_shot_encoder.state_dict(), finetuned_model_0.state_dict())
task_vector_1 = TaskVector(
    args, zero_shot_encoder.state_dict(), finetuned_model_1.state_dict())
task_vectors = {args.tasks[0]: task_vector_0, args.tasks[1]: task_vector_1}

single_task_encoder_0 = task_vector_0.apply_to(
    deepcopy(zero_shot_encoder), scaling_coef=1.0).to(args.device)

single_task_encoder_1 = task_vector_1.apply_to(
    deepcopy(zero_shot_encoder), scaling_coef=1.0).to(args.device)
eval_single_dataset(
    single_task_encoder_1, args.tasks[0], args)
_, _, _ = eval_single_dataset_with_prediction(
    single_task_encoder_1, args.tasks[1], dataloader_2, args)
task_vector_sum = sum(task_vectors.values())
for task in args.tasks:
    multitask_image_encoder = task_vector_sum.apply_to(
        deepcopy(zero_shot_encoder), scaling_coef=1.0).to(args.device)
    eval_single_dataset(multitask_image_encoder, task, args)
