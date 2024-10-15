import torch
import logging
import os
import sys
from datetime import datetime
from src.eval import evaluate, eval_single_dataset
from src.main import save_scale_factors
from args import parse_arguments
from src.modeling import ImageEncoder
from src.task_vectors import TaskVector

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
def average_weights(state_dict1, state_dict2):
    """Average the weights of two state dicts."""
    averaged_state_dict = {}
    for key in state_dict1:
        averaged_state_dict[key] = (state_dict1[key] + state_dict2[key]) / 2
    return averaged_state_dict

def create_model_with_averaged_weights(args, state_dict1, state_dict2):
    """Create a model with averaged weights."""
    averaged_state_dict = average_weights(state_dict1, state_dict2)
    model = ImageEncoder(args, keep_lang=False)
    model.load_state_dict(averaged_state_dict)
    return model

sys.stdout = LoggerWriter(logger, logging.INFO)
sys.stderr = LoggerWriter(logger, logging.ERROR)

args = parse_arguments()
args.model = 'ViT-L-14'
args.tasks = ['DTD', 'SUN397']
args.device = 'cuda'
args.task_scale_factors = None
args.save = 'checkpoints/ViT-L-14'
args.data_location = '/data2/david3684/data'
args.no_shared_weight = True

logger.info("Loading shared weight model")


finetuned_model_0 = torch.load(f"/data2/david3684/2024_arithmetic/checkpoints/ViT-L-14/{args.tasks[0]}/finetuned.pt")
finetuned_model_1 = torch.load(f"/data2/david3684/2024_arithmetic/checkpoints/ViT-L-14/{args.tasks[1]}/finetuned.pt")

finetuned_state_dict_0 = finetuned_model_0.state_dict()
finetuned_state_dict_1 = finetuned_model_1.state_dict()

averaged_model = create_model_with_averaged_weights(args, finetuned_state_dict_0, finetuned_state_dict_1)
for task in args.tasks:
    eval_single_dataset(averaged_model, task, args)

task_vector_0 = TaskVector(args, finetuned_state_dict_0, averaged_model.state_dict())
task_vector_1 = TaskVector(args, finetuned_state_dict_1, averaged_model.state_dict())

task_vector_sum = (task_vector_0+task_vector_1)
multitask_image_encoder = task_vector_sum.apply_to(averaged_model, scaling_coef=1.0).to(args.device)
for task in args.tasks:
    eval_single_dataset(multitask_image_encoder, task, args)






