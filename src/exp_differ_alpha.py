import torch
import logging
import os
import sys
import numpy as np
from datetime import datetime
from src.eval import evaluate, eval_single_dataset
from src.main import save_scale_factors
from args import parse_arguments

current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = f'experiment_{current_time}.log'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[
    logging.FileHandler(log_file),
    logging.StreamHandler()
])
logger = logging.getLogger(__name__)

class LoggerWriter:
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level
        self.buffer = ''

    def write(self, message):
        if message != '\n':
            self.buffer += message
        if '\n' in message:
            self.logger.log(self.level, self.buffer.strip())
            self.buffer = ''

    def flush(self):
        pass

sys.stdout = LoggerWriter(logger, logging.INFO)
sys.stderr = LoggerWriter(logger, logging.ERROR)

args = parse_arguments()
args.model = 'ViT-L-14'
args.tasks = ['DTD', 'SUN397']
args.device = 'cuda'
args.low_rank_mode = 'SoRA'
args.initial_rank_ratio = 0.16
args.task_scale_factors = None
args.save = 'checkpoints/ViT-L-14'
args.data_location = '/data2/david3684/data'
args.n_eval_points = 10

logger.info("Loading shared weight model")
shared_weight_model = torch.load('/data2/david3684/2024_arithmetic/shared_weight/20241010_vanilla/rankmin_config_20241010_uni_vanilla_2.bin') 


logger.info("Loading zero-shot encoder")
zero_shot_encoder_org = torch.load("/data2/david3684/2024_arithmetic/checkpoints/ViT-L-14/zeroshot.pt")
zero_shot_encoder = torch.load(f"/data2/david3684/2024_arithmetic/checkpoints/ViT-L-14/{args.tasks[0]}_{args.tasks[1]}_shared_weight_openclip.pt")
zero_shot_encoder = zero_shot_encoder.to(args.device)

# logger.info("Evaluating original pretrained weight")
# eval_single_dataset(zero_shot_encoder_org, args.tasks[0], args)
# eval_single_dataset(zero_shot_encoder_org, args.tasks[1], args) 
# logger.info("Evaluating finetuned models")
# eval_single_dataset(finetuned_model1, args.tasks[0], args)
# eval_single_dataset(finetuned_model2, args.tasks[1], args)


logger.info("Saving scale factors")
scale_factor_1, scale_factor_2 = save_scale_factors(shared_weight_model['scale_dict'])
args.task_scale_factors = {args.tasks[0]: scale_factor_1, args.tasks[1]: scale_factor_2}
rank_minimization_mode = 'SoRA'
initial_rank_ratios = [0.5, 0.32, 0.16, 0.08, 0.05, 0.02, 0.01, 0.005, 0.001, 0]

alphas = np.linspace(0, 1, args.n_eval_points)

import copy

for alpha in alphas:
    logger.info(f"Starting experiment with alpha: {alpha}")
    task_vectors = {}
    initial_rank_ratio = 0.5
    for i, task in enumerate(args.tasks):
        task_vector_path = f"/data2/david3684/2024_arithmetic/checkpoints/ViT-L-14/{task}/{task}_vector_from_shared_{rank_minimization_mode}_rank{initial_rank_ratio}.pt"
        logger.info(f"Loading task vector for task: {task} from {task_vector_path}")
        task_vectors[task] = torch.load(task_vector_path).to(args.device)
    task_vector_sum = sum(task_vectors.values())
    
    zero_shot_encoder_copy = copy.deepcopy(zero_shot_encoder) # zero_shot_encoder의 state_dict가 바뀌는 것을 막기 위해 copy
    multitask_image_encoder = task_vector_sum.apply_to(zero_shot_encoder_copy, scaling_coef=alpha).to(args.device)

    logger.info(f"Evaluating multitask image encoder with alpha {alpha}")
    for task in args.tasks:
        logger.info(f"Evaluating task {task} with alpha {alpha}")
        eval_single_dataset(multitask_image_encoder, task, args)
        logger.info(f"Completed evaluation for task {task} with alpha {alpha}")

logger.info("All experiments completed")