import ipdb
import open_clip
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from src.task_vectors import TaskVector
from src.modeling import ImageEncoder, ImageClassifier
from src.datasets.registry import get_dataset
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.args import parse_arguments
from src.main import save_scale_factors
from src.eval import eval_single_dataset_with_prediction, eval_single_dataset
import numpy as np
import torch
import torch
import sys
import os
print(os.getcwd())
module_path = os.path.abspath(os.path.join('/data2/david3684/2024_arithmetic'))
if module_path not in sys.path:
    sys.path.append(module_path)


class Args:
    def __init__(self):
        self.model = 'ViT-L-14'
        self.tasks = ['DTD', 'SUN397']
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.task_scale_factors = None
        self.save = '/data2/david3684/2024_arithmetic/checkpoints/ViT-L-14'
        self.data_location = '/data2/david3684/data'
        self.no_shared_weights = True
        self.eval_datasets = None
        self.train_dataset = None
        self.exp_name = None
        self.results_db = None
        self.batch_size = 128
        self.lr = 0.001
        self.wd = 0.1
        self.ls = 0.0
        self.warmup_length = 500
        self.epochs = 10
        self.load = None
        self.cache_dir = None
        self.openclip_cachedir = '/data2/david3684/.cache/open_clip'
        self.initial_rank_ratio = 1.0
        self.low_rank_mode = 'SoRA'
        self.pretrained_model = 'openai'
        self.scale_shared_weight = True
        self.no_shared_weight = True
        self.num_test_samples = 2048


args = Args()

model_1 = torch.load(
    '/data2/david3684/2024_arithmetic/checkpoints/ViT-L-14/DTDVal/finetuned_laion2b_s32b_b82k.pt')
model_2 = torch.load(
    '/data2/david3684/2024_arithmetic/checkpoints/ViT-L-14/SUN397/finetuned.pt')


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


averaged_model = create_model_with_averaged_weights(
    args, model_1.state_dict(), model_2.state_dict())


def transform_key(old_key):
    if old_key.startswith('shared.attn.layer') or old_key.startswith('clip_vit'):
        parts = old_key.split('.')
        layer_idx = parts[3]
        # print(layer_idx)
        sub_key = parts[4]
        if sub_key in ['q', 'k', 'v']:
            return f'model.visual.transformer.resblocks.{layer_idx}.attn.{sub_key}_weight'
        elif sub_key == 'out_proj':
            return f'model.visual.transformer.resblocks.{layer_idx}.attn.out_proj.weight'
        elif sub_key == 'c_fc' or sub_key == 'c_proj':
            return f'model.visual.transformer.resblocks.{layer_idx}.mlp.{sub_key}.weight'
    return old_key


def save_scale_factors(scale_dict):
    qkv_scale_store_task1 = {}
    qkv_scale_store_task2 = {}
    scale_factors_1 = {}
    scale_factors_2 = {}
    for scale_dict_key, value in scale_dict.items():
        transformed_scale_dict_key = transform_key(scale_dict_key)
        if 'clip_vit_1' in scale_dict_key:
            subkey = scale_dict_key.split('.')[-1]
            index = scale_dict_key.split('.')[-2]
            if index not in qkv_scale_store_task1:
                qkv_scale_store_task1[index] = {
                    'q': None, 'k': None, 'v': None}
            if subkey == 'q':
                q_scale = value.unsqueeze(0)
                qkv_scale_store_task1[index]['q'] = q_scale
            elif subkey == 'k':
                k_scale = value.unsqueeze(0)
                qkv_scale_store_task1[index]['k'] = k_scale
            elif subkey == 'v':
                v_scale = value.unsqueeze(0)
                qkv_scale_store_task1[index]['v'] = v_scale
            else:
                scale_factors_1[transformed_scale_dict_key +
                                '.scale'] = value  # scale factor 저장
        elif 'clip_vit_2' in scale_dict_key:
            subkey = scale_dict_key.split('.')[-1]
            index = scale_dict_key.split('.')[-2]
            if index not in qkv_scale_store_task2:
                qkv_scale_store_task2[index] = {
                    'q': None, 'k': None, 'v': None}
            if subkey == 'q':
                q_scale = value.unsqueeze(0)
                qkv_scale_store_task2[index]['q'] = q_scale
            elif subkey == 'k':
                k_scale = value.unsqueeze(0)
                qkv_scale_store_task2[index]['k'] = k_scale
            elif subkey == 'v':
                v_scale = value.unsqueeze(0)
                qkv_scale_store_task2[index]['v'] = v_scale
            else:
                scale_factors_2[transformed_scale_dict_key +
                                '.scale'] = value  # scale factor 저장

    for layer_idx, qkv in qkv_scale_store_task1.items():
        # print(layer_idx, qkv)
        if qkv['q'] is not None and qkv['k'] is not None and qkv['v'] is not None:
            concat_scale = torch.cat([qkv['q'], qkv['k'], qkv['v']], dim=0)
            # print('hi')
            scale_factors_1[f'model.visual.transformer.resblocks.{layer_idx}.attn.in_proj_weight.scale'] = concat_scale
    for layer_idx, qkv in qkv_scale_store_task1.items():
        if qkv['q'] is not None and qkv['k'] is not None and qkv['v'] is not None:
            concat_scale = torch.cat([qkv['q'], qkv['k'], qkv['v']], dim=0)
            scale_factors_2[f'model.visual.transformer.resblocks.{layer_idx}.attn.in_proj_weight' +
                            '.scale'] = concat_scale

    return scale_factors_1, scale_factors_2


def format_shared_weight(shared_weight_state_dict, open_clip_state_dict_template):
    qkv_store = {}
    for old_key, value in shared_weight_state_dict.items():
        if 'diff' in old_key or 'scale_dict' in old_key:
            continue

        new_key = transform_key(old_key)
        layer_idx = new_key.split('.')[4]

        if layer_idx not in qkv_store:
            qkv_store[layer_idx] = {'q': None, 'k': None, 'v': None}

        weight_type = new_key.split('.')[-1]
        # in_proj.weight (q, k, v)
        if weight_type in ['q_weight', 'k_weight', 'v_weight']:
            if args.scale_shared_weight:
                print('Scaling Shared Weight Down')
                scaled_value = value / torch.norm(value, p='fro')
                qkv_store[layer_idx][weight_type[0]] = scaled_value
            else:
                qkv_store[layer_idx][weight_type[0]] = value
        else:  # out_proj.weight, c_fc.weight, c_proj.weight
            assert new_key in open_clip_state_dict_template
            open_clip_state_dict_template[new_key] = value / \
                torch.norm(value, p='fro')

    for layer_idx, qkv in qkv_store.items():
        if all(v.bool().all().item() for v in qkv.values()):
            in_proj_weight = torch.cat([qkv['q'], qkv['k'], qkv['v']], dim=0)
            # concat qkv into 3072*1024 tensor
            new_key = f'model.visual.transformer.resblocks.{layer_idx}.attn.in_proj_weight'
            assert new_key in open_clip_state_dict_template
            open_clip_state_dict_template[new_key] = in_proj_weight
        else:
            print(
                f"Missing q, k, or v for layer {layer_idx}. q: {qkv['q']}, k: {qkv['k']}, v: {qkv['v']}")

    return open_clip_state_dict_template

# 나머지 처리 필요


shared_weight_state_dict = torch.load(
    '/data2/david3684/2024_arithmetic/checkpoints/rankmin_config_20241017_uni_vanilla_1.bin')

scale_factors_1 = {}
for keys in model_1.state_dict():
    if 'weight' in keys:
        if 'attn.in_proj' in keys:
            q, k, v = model_1.state_dict()[keys].chunk(3, dim=0)
            q_scale = torch.norm(q, p='fro').unsqueeze(0)
            k_scale = torch.norm(k, p='fro').unsqueeze(0)
            v_scale = torch.norm(v, p='fro').unsqueeze(0)
            scale_key = keys + '.scale'
            scale_factors_1[scale_key] = torch.cat(
                [q_scale, k_scale, v_scale], dim=0)
            print(scale_factors_1[scale_key])
        elif 'out_proj' in keys or 'c_fc' in keys or 'c_proj' in keys:
            scale_factors_1[keys +
                            '.scale'] = torch.norm(model_1.state_dict()[keys], p='fro')
print(scale_factors_1.keys())
scale_factors_2 = {}

for keys in model_2.state_dict():
    if 'weight' in keys:
        if 'attn.in_proj' in keys:
            q, k, v = model_2.state_dict()[keys].chunk(3, dim=0)
            q_scale = torch.norm(q, p='fro').unsqueeze(0)
            k_scale = torch.norm(k, p='fro').unsqueeze(0)
            v_scale = torch.norm(v, p='fro').unsqueeze(0)
            scale_key = keys + '.scale'
            scale_factors_2[scale_key] = torch.cat(
                [q_scale, k_scale, v_scale], dim=0)
            print(scale_factors_2[scale_key])
        elif 'out_proj' in keys or 'c_fc' in keys or 'c_proj' in keys:
            scale_factors_2[keys +
                            '.scale'] = torch.norm(model_2.state_dict()[keys], p='fro')
print(scale_factors_2.keys())

args.task_scale_factors = {
    'DTD': scale_factors_1, 'SUN397': scale_factors_2}

zero_shot_encoder = ImageEncoder(args, keep_lang=False)

# 이러면 pretrained checkpoint에 있는 ln, bias 등으로 초기화 될것이다.

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

args.scale_shared_weight = False
formatted_shared_weight = format_shared_weight(
    shared_weight_state_dict, zero_shot_encoder.state_dict())

zero_shot_encoder.load_state_dict(formatted_shared_weight)

# args.task_scale_factors = {
#     'DTD': scale_factors_1, 'SUN397': scale_factors_2}
# args.task_scale_factors = None
# args.pretrained_model = 'openai'
# args.no_shared_weight = True
# args.save = '/data2/david3684/2024_arithmetic/checkpoints/ViT-L-14'
# task_vector_temp = TaskVector(
#     args, zero_shot_encoder.state_dict(), model_2.state_dict(), 'SUN397')
# eval_single_dataset_with_prediction(model_2, 'SUN397', dataloader_2, args)

# args.task_scale_factors = {
#     'DTD': scale_factors_1, 'SUN397': scale_factors_2}
# args.scale_shared_weight = True
# args.pretrained_model = 'openai'
# args.no_shared_weight = False
# task_vector_temp = TaskVector(
#     args, zero_shot_encoder.state_dict(), model_2.state_dict(), 'SUN397')

# eval_single_dataset_with_prediction(model_2, 'SUN397', dataloader_2, args)

# ipdb.set_trace()
# eval_single_dataset(model_1, 'DTD', args)
# single_task_encoder = task_vector_temp.apply_to(deepcopy(averaged_model), scaling_coef=1.0)
# single_task_encoder = task_vector_temp.apply_to(
#     deepcopy(zero_shot_encoder), scaling_coef=1.0)
# eval_single_dataset_with_prediction(
#     single_task_encoder, 'SUN397', dataloader_2, args)

low_rank_task_vectors = {}
for task in args.tasks:
    finetuned_state_dict = model_1.state_dict(
    ) if task == 'DTD' else model_2.state_dict()
    args.no_shared_weights = False
    low_rank_task_vectors[task] = TaskVector(
        args, zero_shot_encoder.state_dict(), finetuned_state_dict, task)
low_rank_task_vector_sum = sum(low_rank_task_vectors.values())
low_rank_multi_task_encoder = low_rank_task_vector_sum.apply_to(
    deepcopy(zero_shot_encoder), scaling_coef=1.0)

for task in args.tasks:
    if task == 'DTD':
        args.pretrained_model = 'laion2b_s32b_b82k'
    else:
        args.pretrained_model = 'openai'
    args.task_scale_factors = {
        'DTD': scale_factors_1, 'SUN397': scale_factors_2}
    # print(args.task_scale_factors)
    # print(type(low_rank_multi_task_encoder))
    loader = dataloader_1 if task == 'DTD' else dataloader_2
    eval_single_dataset_with_prediction(
        low_rank_multi_task_encoder, task, loader, args)
