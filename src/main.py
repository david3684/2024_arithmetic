import torch
from src.eval import evaluate, eval_single_dataset
from src.modeling import ImageEncoder, ImageClassifier
from src.task_vectors import TaskVector
from args import parse_arguments
from src.utils import cosine_lr, LabelSmoothing
import os
from src.eval import evaluate, eval_single_dataset


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


def calculate_weight_difference(original_state_dict, formatted_state_dict):

    differences = {}

    for key in original_state_dict.keys():
        if key in formatted_state_dict:
            import ipdb
            ipdb.set_trace()
            original_weight = original_state_dict[key]
            formatted_weight = formatted_state_dict[key]
            # difference = torch.abs(original_weight - formatted_weight).sum().item()
            difference = torch.mean(
                (original_weight - formatted_weight)**2).item()
            differences[key] = difference
        else:
            print(f"Key {key} not found in formatted model state dict.")

    return differences


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
                scale_key = f'shared.attn.layer.{layer_idx}.{weight_type[0]}'
                if scale_key in shared_weight_state_dict['scale_dict']:
                    weight_scale_factor = shared_weight_state_dict['scale_dict'][scale_key]
                    scaled_value = value / weight_scale_factor
                    qkv_store[layer_idx][weight_type[0]] = scaled_value
                else:
                    print(f"Scale key {scale_key} not found in scale_dict.")
            else:
                qkv_store[layer_idx][weight_type[0]] = value
        else:  # out_proj.weight, c_fc.weight, c_proj.weight
            assert new_key in open_clip_state_dict_template
            weight_scale_factor = shared_weight_state_dict['scale_dict'][old_key]
            open_clip_state_dict_template[new_key] = value / \
                weight_scale_factor

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


def main(args):

    # Config
    shared_weight_state_dict = torch.load(args.shared_weight)
    zero_shot_checkpoint = f'checkpoints/{args.model}/zeroshot.pt'

    formatted_shared_weight_path = f"/data2/david3684/2024_arithmetic/checkpoints/ViT-L-14/DTD_SUN397_shared_weight_openclip.pt"
    finetuned_model_each_task = {}
    for task in args.tasks:
        finetuned_model_each_task[f'{task}'] = torch.load(
            f'/data2/david3684/2024_arithmetic/checkpoints/ViT-L-14/{task}/finetuned.pt')

    if os.path.exists(formatted_shared_weight_path):
        print("Loading formatted shared weight from path")
        zero_shot_encoder = torch.load(formatted_shared_weight_path)  # model
    else:
        print("No formatted shared weight found. Formatting shared weight into Openclip format")
        zero_shot_encoder = torch.load(zero_shot_checkpoint)
        formatted_shared_weight_state_dict = format_shared_weight(
            shared_weight_state_dict, zero_shot_encoder.state_dict())
        zero_shot_encoder.load_state_dict(formatted_shared_weight_state_dict)
        torch.save(zero_shot_encoder,
                   f"/data2/david3684/2024_arithmetic/checkpoints/ViT-L-14/DTD_SUN397_shared_weight_openclip.pt")

    zero_shot_encoder = zero_shot_encoder.to(args.device)

    scale_factors_1, scale_factors_2 = save_scale_factors(
        shared_weight_state_dict['scale_dict'])
    args.task_scale_factors = {
        'DTD': scale_factors_1, 'SUN397': scale_factors_2}

    # args.task_scale_factors = None
    # eval_single_dataset(finetuned_model_each_task['DTD'], 'DTD', args)

    low_rank_vectors = {}
    for initial_rank_ratio in [1]:
        for task in args.tasks:
            finetuned_model_each_task[f'{task}'].to(args.device)
            args.initial_rank_ratio = initial_rank_ratio
            task_vector_path = f"/data2/david3684/2024_arithmetic/checkpoints/ViT-L-14/{task}/{task}_vector_from_shared_{args.low_rank_mode}_rank{args.initial_rank_ratio}.pt"
            if os.path.exists(task_vector_path):
                low_rank_vectors[f'{task}'] = torch.load(task_vector_path)
            else:
                print(f"Building task vectors for task {task}")
                low_rank_vectors[f'{task}'] = TaskVector(args, zero_shot_encoder.state_dict(
                ), finetuned_model_each_task[f'{task}'].state_dict(), task=task, vector=None)
                torch.save(low_rank_vectors[f'{task}'], task_vector_path)

    # low_rank_vectors['DTD'].to(args.device)
    # low_rank_vectors['SUN397'].to(args.device)
    # zero_shot_encoder.to(args.device)

    # task_vector_sum = sum(low_rank_vectors.values()).to(args.device)

    # multitask_image_encoder = task_vector_sum.apply_to(zero_shot_encoder, scaling_coef=1)
    # # eval_single_dataset(multitask_image_encoder, 'DTD', args)

    # for task in args.tasks:
    #     eval_single_dataset(multitask_image_encoder, task, args)


if __name__ == "__main__":
    datasets = ['DTD', 'SUN397']
    model = 'ViT-L-14'
    args = parse_arguments()
    args.device = 'cuda'
    args.data_location = '/data2/david3684/data'
    args.model = model
    args.save = f'checkpoints/{model}'
    args.save_file_name = '_'.join(datasets) + '_shared_openclip.pt'
    args.low_rank_mode = 'SoRA'
    args.tasks = datasets
    args.initial_rank_ratio = 0.5
    args.scale_shared_weight = True
    args.task_scale_factors = None
    zero_shot_checkpoint = f'checkpoints/{model}/zeroshot.pt'
    args.shared_weight = '/data2/david3684/2024_arithmetic/shared_weight/20241010_vanilla/rankmin_config_20241010_uni_vanilla_2.bin'
    main(args)
