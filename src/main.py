import torch
from src.eval import evaluate, eval_single_dataset
from src.modeling import ImageEncoder, ImageClassifier
from src.task_vectors import TaskVector
from args import parse_arguments
from src.utils import cosine_lr, LabelSmoothing
import os
from src.eval import evaluate, eval_single_dataset

def transform_key(old_key):
    if old_key.startswith('shared.attn.layer'):
        parts = old_key.split('.')
        layer_idx = parts[3]
        # print(layer_idx)
        sub_key = parts[4]
        
        if sub_key in ['q', 'k', 'v']:
            return f'model.visual.transformer.resblocks.{layer_idx}.attn.in_proj_weight'
        elif sub_key == 'out_proj':
            return f'model.visual.transformer.resblocks.{layer_idx}.attn.out_proj.weight'
        elif sub_key == 'c_fc':
            return f'model.visual.transformer.resblocks.{layer_idx}.mlp.c_fc.weight'
        elif sub_key == 'c_proj':
            return f'model.visual.transformer.resblocks.{layer_idx}.mlp.c_proj.weight'
    return old_key


def main(args):
    
    # Config
    shared_weight = torch.load(args.shared_weight)
    pretrained_checkpoint = f'checkpoints/{args.model}/zeroshot.pt'
    
    formatted_shared_weight_path = f"/data2/david3684/2024_arithmetic/checkpoints/ViT-L-14/DTD_SUN_shared_weight_openclip.pt"
    if os.path.exists(formatted_shared_weight_path):
        shared_weight_formatted = torch.load(formatted_shared_weight_path)
        shared_state_dict_formatted = shared_weight_formatted.state_dict()
    else:
        shared_weight_formatted = torch.load(pretrained_checkpoint)
        shared_state_dict_formatted = {}
        finetuned_state_dict = {}
        for task in args.tasks:
            finetuned_state_dict[f'{task}'] = torch.load(f'/data2/david3684/2024_arithmetic/checkpoints/ViT-L-14/{task}/finetuned.pt').state_dict()
        qkv_store = {}
        
        for old_key, value in shared_weight.items():
            if old_key == 'scale_dict':
                continue
            if 'diff' in old_key:
                continue
            new_key = transform_key(old_key)
            if 'in_proj_weight' in new_key:
                layer_idx = new_key.split('.')[4]
                if layer_idx not in qkv_store:
                    qkv_store[layer_idx] = {'q': None, 'k': None, 'v': None}
                if 'q' in old_key:
                    qkv_store[layer_idx]['q'] = value
                elif 'k' in old_key:
                    qkv_store[layer_idx]['k'] = value
                elif 'v' in old_key:
                    qkv_store[layer_idx]['v'] = value
            else:
                shared_state_dict_formatted[new_key] = value
        
        for layer_idx, qkv in qkv_store.items():
            print(layer_idx)
            if qkv['q'] is not None and qkv['k'] is not None and qkv['v'] is not None:
                in_proj_weight = torch.cat([qkv['q'], qkv['k'], qkv['v']], dim=0)
                new_key = f'model.visual.transformer.resblocks.{layer_idx}.attn.in_proj_weight' #concat qkv into 3072*1024 tensor
                print(new_key)
                shared_state_dict_formatted[new_key] = in_proj_weight
        for scale_dict_key, value in shared_weight['scale_dict'].items():
            transformed_scale_dict_key = transform_key(scale_dict_key)
            if 'clip_vit_1' in scale_dict_key:
                finetuned_state_dict['DTD'][transformed_scale_dict_key + '.scale'] = value
            elif 'clip_vit_2' in scale_dict_key:
                finetuned_state_dict['SUN397'][transformed_scale_dict_key + '.scale'] = value
            elif 'shared' in scale_dict_key:
                transformed_scale_dict_key = transform_key(scale_dict_key)
                shared_state_dict_formatted[transformed_scale_dict_key + '.scale'] = value    
        
        temp_state_dict = shared_weight_formatted.state_dict()
        temp_state_dict.update(shared_state_dict_formatted)
        shared_weight_formatted.load_state_dict(temp_state_dict)
        torch.save(shared_weight_formatted, f"/data2/david3684/2024_arithmetic/checkpoints/ViT-L-14/DTD_SUN_shared_weight_openclip.pt")
    
    # Obtain task vectors from shared weight and finetuned weights
    # for key in finetuned_state_dict['SUN397']:
    #     print(key)
    
    low_rank_vectors = {}
    for task in args.tasks:
        task_vector_path = f"/data2/david3684/2024_arithmetic/checkpoints/ViT-L-14/{task}/vector_from_shared.pt"
        if os.path.exists(task_vector_path):
            low_rank_vectors[f'{task}'] = torch.load(task_vector_path)
        else:
            print(f"Building task vectors for task {task}")
            finetuned_state_dict[f'{task}'] = torch.load(f'/data2/david3684/2024_arithmetic/checkpoints/ViT-L-14/{task}/finetuned.pt').state_dict()
            low_rank_vectors[f'{task}'] = TaskVector(args, shared_state_dict_formatted, finetuned_state_dict[f'{task}'], vector=None)
            torch.save(low_rank_vectors[f'{task}'], task_vector_path)

    task_vector_sum = sum(low_rank_vectors.values())

    # 잘 구했다 치고 evaluate로    
    multitask_image_encoder = task_vector_sum.apply_to(shared_weight_formatted, scaling_coef=0.8)
    
    for dataset in args.tasks:
        eval_single_dataset(multitask_image_encoder, dataset, args)
    
    
if __name__ == "__main__":
    datasets = ['DTD', 'SUN397']
    model = 'ViT-L-14'
    args = parse_arguments()
    args.data_location = '/data2/david3684/data'
    args.model = model
    args.save = f'checkpoints/{model}'
    args.save_file_name = '_'.join(datasets) + '_shared_openclip.pt'
    args.low_rank_mode = 'SoRA'
    args.tasks = datasets
    args.initial_rank_ratio = 0.5
    pretrained_checkpoint = f'checkpoints/{model}/zeroshot.pt'
    args.shared_weight = '/data2/david3684/2024_arithmetic/shared_weight/20241007_vanilla/rankmin_config_20241009_uni_vanilla_2.bin'
    main(args)
    

