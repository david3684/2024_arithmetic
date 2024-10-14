import torch
import sys
import numpy as np
from src.eval import eval_single_dataset_with_prediction, eval_single_dataset
from src.main import save_scale_factors
from src.args import parse_arguments
from tqdm import tqdm
from copy import deepcopy
import matplotlib.pyplot as plt
from datetime import datetime
import pickle
import os

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

if os.path.exists(predicts_0_file) and os.path.exists(predicts_1_file):

    with open(predicts_0_file, 'rb') as f:
        predicts_0 = pickle.load(f)
    with open(predicts_1_file, 'rb') as f:
        predicts_1 = pickle.load(f)
else:       
    predicts_0 = {}
    for alpha_1 in tqdm(alpha_range):
        log(f"Saving predictions for alpha: {alpha_1}")
        single_task_image_encoder_1 = task_vectors[args.tasks[0]].apply_to(deepcopy(zero_shot_encoder), scaling_coef=alpha_1).to(args.device)
        predicts_0[alpha_1] = eval_single_dataset_with_prediction(single_task_image_encoder_1, args.tasks[0], args)[1]

    predicts_1 = {}
    for alpha_2 in tqdm(alpha_range):
        log(f"Saving predictions for alpha: {alpha_2}")
        single_task_image_encoder_2 = task_vectors[args.tasks[1]].apply_to(deepcopy(zero_shot_encoder), scaling_coef=alpha_2).to(args.device)
        predicts_1[alpha_2] = eval_single_dataset_with_prediction(single_task_image_encoder_2, args.tasks[1], args)[1]

    with open(predicts_0_file, 'wb') as f:
        pickle.dump(predicts_0, f)
    with open(predicts_1_file, 'wb') as f:
        pickle.dump(predicts_1, f)

        
log("Calculating disentanglement errors")
for i, alpha_1 in enumerate(tqdm(alpha_range)):
    for j, alpha_2 in enumerate(alpha_range):
        log(f"Calculating disentanglement error for alpha_1: {alpha_1}, alpha_2: {alpha_2}")
        task_vector_sum = task_vectors[args.tasks[0]].multiply(alpha_1) + task_vectors[args.tasks[1]].multiply(alpha_2)
        zero_shot_encoder_copy = deepcopy(zero_shot_encoder)
        multitask_image_encoder = task_vector_sum.apply_to(zero_shot_encoder_copy, scaling_coef=1).to(args.device)
        
        error = 0
        total_count=0
        for task in args.tasks:
            _, multitask_pred, multitask_label = eval_single_dataset_with_prediction(multitask_image_encoder, task, args)
            if task == args.tasks[0]:
                total_count += len(multitask_pred) 
                for k in range(len(multitask_pred)):
                    if multitask_pred[i] != predicts_0[alpha_1][i]:
                        error += 1
                log(f"Error for task {task} at alpha_1: {alpha_1}: {error}")
            elif task == args.tasks[1]:
                total_count += len(multitask_pred)
                for k in range(len(multitask_pred)):
                    if multitask_pred[i] != predicts_1[alpha_2][i]:
                        error += 1
                log(f"Error for task {task} at alpha_2: {alpha_2}: {error}")
        
        disentanglement_errors[i, j] = error/total_count
        log(f"Disentanglement error for alpha_1: {alpha_1}, alpha_2: {alpha_2}: {disentanglement_errors[i, j]}")
rank = 0.5
np.save(f"disentanglement_errors_{rank}.npy", disentanglement_errors)
log_f.close()