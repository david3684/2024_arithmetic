import torch
import sys
import numpy as np
from src.eval import eval_single_dataset_with_prediction, eval_single_dataset
from src.main import save_scale_factors
from src.args import parse_arguments
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.datasets.registry import get_dataset
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

# 현재 시간으로 로그 파일 이름 생성
current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = f'/data2/david3684/2024_arithmetic/logs/disentanglement_{current_time}.log'
log_dir = os.path.dirname(log_file)

# 디렉토리가 존재하지 않으면 생성
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# 로그 파일 열기
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
args.n_eval_points = 10
args.num_test_samples = 2048

shared_weight_model = torch.load('/data2/david3684/2024_arithmetic/shared_weight/20241010_vanilla/rankmin_config_20241010_uni_vanilla_2.bin') 
zero_shot_encoder_org = torch.load("/data2/david3684/2024_arithmetic/checkpoints/ViT-L-14/zeroshot.pt")
zero_shot_encoder = torch.load(f"/data2/david3684/2024_arithmetic/checkpoints/ViT-L-14/{args.tasks[0]}_{args.tasks[1]}_shared_weight_openclip.pt")
zero_shot_encoder = zero_shot_encoder.to(args.device)

scale_factor_1, scale_factor_2 = save_scale_factors(shared_weight_model['scale_dict'])
args.task_scale_factors = {args.tasks[0]: scale_factor_1, args.tasks[1]: scale_factor_2}

task_vectors = {}
for task in args.tasks:
    task_vector_path = f"/data2/david3684/2024_arithmetic/checkpoints/ViT-L-14/{task}/{task}_vector_from_shared_{args.low_rank_mode}_rank{args.initial_rank_ratio}.pt"
    task_vectors[task] = torch.load(task_vector_path).to(args.device)

n=11
alpha_range = np.linspace(-2, 2, n)

disentanglement_errors = np.zeros((n, n))

predicts_0_file = "predicts_0.pkl"
predicts_1_file = "predicts_1.pkl"

_, _, val_preprocess = open_clip.create_model_and_transforms(
            args.model, pretrained='openai', cache_dir=args.openclip_cachedir)

dataset_1 = get_dataset(
        args.tasks[0],
        val_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
        num_test_samples=args.num_test_samples,
    )
dataloader_1 = get_dataloader(
    dataset_1, is_train=False, args=args, image_encoder=None)

dataset_2 = get_dataset(
        args.tasks[1],
        val_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
        num_test_samples=args.num_test_samples,
    )
dataloader_2 = get_dataloader(
    dataset_2, is_train=False, args=args, image_encoder=None)
    

if os.path.exists(predicts_0_file) and os.path.exists(predicts_1_file):

    with open(predicts_0_file, 'rb') as f:
        predicts_0 = pickle.load(f)
    with open(predicts_1_file, 'rb') as f:
        predicts_1 = pickle.load(f)
else:       
    predicts_0 = {}
    for alpha_0 in tqdm(alpha_range):
        log(f"Saving predictions for alpha: {alpha_0}")
        single_task_image_encoder_1 = task_vectors[args.tasks[0]].apply_to(deepcopy(zero_shot_encoder), scaling_coef=alpha_0).to(args.device)
        predicts_0[alpha_0] = eval_single_dataset_with_prediction(single_task_image_encoder_1, args.tasks[0], dataloader_1, args)[1]

    predicts_1 = {}
    for alpha_1 in tqdm(alpha_range):
        log(f"Saving predictions for alpha: {alpha_1}")
        single_task_image_encoder_2 = task_vectors[args.tasks[1]].apply_to(deepcopy(zero_shot_encoder), scaling_coef=alpha_1).to(args.device)
        predicts_1[alpha_1] = eval_single_dataset_with_prediction(single_task_image_encoder_2, args.tasks[1], dataloader_2, args)[1]

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
            _, multitask_pred, multitask_label = eval_single_dataset_with_prediction(multitask_image_encoder, task, dataloader, args)
            if task == args.tasks[0]:
                total_count += len(multitask_pred) 
                print(len(multitask_pred))
                for k in range(len(multitask_pred)):
                    if multitask_pred[k] != predicts_0[alpha_0][k]:
                        error += 1
                log(f"Error for task {task} at alpha_0: {alpha_0}: {error}")
            elif task == args.tasks[1]:
                total_count += len(multitask_pred)
                for k in range(len(multitask_pred)):
                    if multitask_pred[k] != predicts_1[alpha_1][k]:
                        error += 1
                log(f"Error for task {task} at alpha_1: {alpha_1}: {error}")
        
        disentanglement_errors[i, j] = error/total_count
        log(f"Disentanglement error for alpha_0: {alpha_0}, alpha_1: {alpha_1}: {disentanglement_errors[i, j]}")
rank = 0.5
np.save(f"disentanglement_errors_{rank}.npy", disentanglement_errors)
log_f.close()