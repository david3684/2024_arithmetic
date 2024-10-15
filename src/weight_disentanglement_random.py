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

def error_rate(y_true, y_pred):
    error = 0
    count = 0
    for i in range(len(y_true)):
        if y_true[i] != y_pred[i]:
            error += 1
        count += 1
    return error / count


current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = f'/data2/david3684/2024_arithmetic/logs/disentanglement_{current_time}.log'
log_dir = os.path.dirname(log_file)


if not os.path.exists(log_dir):
    os.makedirs(log_dir)


log_f = open(log_file, 'w')

def log(message):
    print(message)
    log_f.write(message + '\n')

args = parse_arguments()
args.model = 'ViT-L-14'
args.tasks = ['DTD', 'SUN397']
args.device = 'cuda'
args.low_rank_mode = 'SoRA'
args.initial_rank_ratio = 0.5
args.task_scale_factors = None
args.save = 'checkpoints/ViT-L-14'
args.data_location = '/data2/david3684/data'
args.n_eval_points = 11
args.num_test_samples = 2048
args.no_shared_weight = True

_, _, val_preprocess = open_clip.create_model_and_transforms(
            args.model, pretrained='openai', cache_dir=args.openclip_cachedir)

dataset_1 = get_dataset(
        args.tasks[0],
        val_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
        num_workers=16,
        num_test_samples=None,
    )
dataloader_1 = get_dataloader(
    dataset_1, is_train=False, args=args, image_encoder=None)

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


zero_shot_encoder= torch.load("/data2/david3684/2024_arithmetic/checkpoints/ViT-L-14/zeroshot.pt")


n=args.n_eval_points
alpha_range = np.linspace(-2, 2, n)

disentanglement_errors = np.zeros((n, n))

predicts_0_file = "predicts_0_conv.pkl"
predicts_1_file = "predicts_1_conv.pkl"

    
finetuned_model_0 = torch.load(f"/data2/david3684/2024_arithmetic/checkpoints/ViT-L-14/{args.tasks[0]}/finetuned.pt")
finetuned_model_1 = torch.load(f"/data2/david3684/2024_arithmetic/checkpoints/ViT-L-14/{args.tasks[1]}/finetuned.pt")

finetuned_state_dict_0 = finetuned_model_0.state_dict()
finetuned_state_dict_1 = finetuned_model_1.state_dict()

task_vector_0 = TaskVector(args, finetuned_state_dict_0, zero_shot_encoder.state_dict())
task_vector_1 = TaskVector(args, finetuned_state_dict_1, zero_shot_encoder.state_dict())
task_vectors = {args.tasks[0]: task_vector_0, args.tasks[1]: task_vector_1}

head_1 = torch.load(f"/data2/david3684/2024_arithmetic/checkpoints/ViT-L-14/head_{args.tasks[0]}.pt")
head_2 = torch.load(f"/data2/david3684/2024_arithmetic/checkpoints/ViT-L-14/head_{args.tasks[1]}.pt")

if os.path.exists(predicts_0_file) and os.path.exists(predicts_1_file):

    with open(predicts_0_file, 'rb') as f:
        predicts_0 = pickle.load(f)
    with open(predicts_1_file, 'rb') as f:
        predicts_1 = pickle.load(f)
else:       
    predicts_0 = {}
    for i, alpha_0 in enumerate(tqdm(alpha_range)):
        log(f"Saving predictions for alpha: {alpha_0}")
        single_task_image_encoder_1 = task_vectors[args.tasks[0]].apply_to(deepcopy(zero_shot_encoder), scaling_coef=alpha_0).to(args.device)
        _, predicts_0[i], labels = eval_single_dataset_with_prediction(single_task_image_encoder_1, head_1, args.tasks[0], dataloader_1, args)

        

    predicts_1 = {}
    for i, alpha_1 in enumerate(tqdm(alpha_range)):
        log(f"Saving predictions for alpha: {alpha_1}")
        single_task_image_encoder_2 = task_vectors[args.tasks[1]].apply_to(deepcopy(zero_shot_encoder), scaling_coef=alpha_1).to(args.device)
        _, predicts_1[i], labels = eval_single_dataset_with_prediction(single_task_image_encoder_2, head_2, args.tasks[1], dataloader_2, args)


    with open(predicts_0_file, 'wb') as f:
        pickle.dump(predicts_0, f)
    with open(predicts_1_file, 'wb') as f:
        pickle.dump(predicts_1, f)

log("Calculating disentanglement errors")
for i, alpha_0 in enumerate(tqdm(alpha_range)):
    for j, alpha_1 in enumerate(alpha_range):
        log(f"Calculating disentanglement error for alpha_0: {alpha_0}, alpha_1: {alpha_1}")
        task_vector_sum = task_vectors[args.tasks[0]].multiply(alpha_0) + task_vectors[args.tasks[1]].multiply(alpha_1)
        zero_shot_encoder_copy = deepcopy(zero_shot_encoder)
        multitask_image_encoder = task_vector_sum.apply_to(zero_shot_encoder_copy, scaling_coef=1).to(args.device)
        
        error = 0
        total_count=0
        for task in args.tasks:
            dataloader = dataloader_1 if task == args.tasks[0] else dataloader_2
            head = head_1 if task == args.tasks[0] else head_2
            _, multitask_pred, multitask_labels = eval_single_dataset_with_prediction(multitask_image_encoder, head, task, dataloader, args)
            if task == args.tasks[0]:
                print(multitask_labels)
                total_count += len(multitask_pred) 
                for k in range(len(multitask_pred)):
                    if multitask_pred[k] != predicts_0[i][k]:
                        error += 1
                log(f"Error for task {task} at alpha_0: {alpha_0}: {error}")
            elif task == args.tasks[1]:
                print(multitask_labels)
                total_count += len(multitask_pred)
                for k in range(len(multitask_pred)):
                    if multitask_pred[k] != predicts_1[j][k]:
                        error += 1
                log(f"Error for task {task} at alpha_1: {alpha_1}: {error}")
        
        disentanglement_errors[i, j] = error/total_count
        log(f"Disentanglement error for alpha_0: {alpha_0}, alpha_1: {alpha_1}: {disentanglement_errors[i, j]}")
rank = 0.5
np.save(f"disentanglement_errors_{rank}_noshared.npy", disentanglement_errors)
log_f.close()