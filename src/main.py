import torch
from src.eval import evaluate, eval_single_dataset
from src.modeling import ImageEncoder, ImageClassifier
from src.task_vectors import TaskVector
from args import parse_arguments
from src.utils import cosine_lr, LabelSmoothing

def transform_key(old_key):
    if old_key.startswith('shared.attn.layer'):
        parts = old_key.split('.')
        layer_idx = parts[3]
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
    org_openclip_weight = torch.load(f'checkpoints/{args.model}/zeroshot.pt').state_dict()
    task1_weight = torch.load('/data2/david3684/2024_arithmetic/checkpoints/ViT-L-14/DTD/finetuned.pt').state_dict()
    task2_weight = torch.load('/data2/david3684/2024_arithmetic/checkpoints/ViT-L-14/SUN397/finetuned.pt').state_dict()
    
    # shared_weight_state_dict = shared_weight['model']
    
    qkv_store = {}
    new_checkpoint = {}
    
    for old_key, value in shared_weight.items():
        if old_key == 'scale_dict':
            continue
        if 'diff' in old_key:
            continue
        print(f"transforming {old_key}")
        new_key = transform_key(old_key)
        if 'in_proj_weight' in new_key:
            layer_idx = new_key.split('.')[3]
            if layer_idx not in qkv_store:
                qkv_store[layer_idx] = {'q': None, 'k': None, 'v': None}
            if 'q' in old_key:
                qkv_store[layer_idx]['q'] = value
            elif 'k' in old_key:
                qkv_store[layer_idx]['k'] = value
            elif 'v' in old_key:
                qkv_store[layer_idx]['v'] = value
        else:
            org_openclip_weight[new_key] = value
    
    for layer_idx, qkv in qkv_store.items():
        if qkv['q'] is not None and qkv['k'] is not None and qkv['v'] is not None:
            in_proj_weight = torch.cat([qkv['q'], qkv['k'], qkv['v']], dim=0)
            new_key = f'model.visual.transformer.resblocks.{layer_idx}.attn.in_proj_weight' #concat qkv into 3072*1024 tensor
            org_openclip_weight[new_key] = in_proj_weight

    for scale_dict_key, value in shared_weight['scale_dict'].items():
        transformed_scale_dict_key = transform_key(scale_dict_key)
        print(f'transformed_scale_dict_key: {transformed_scale_dict_key}')
        if 'clip_vit_1' in scale_dict_key:
            task1_weight[transformed_scale_dict_key + '.scale'] = value
        elif 'clip_vit_2' in scale_dict_key:
            task2_weight[transformed_scale_dict_key + '.scale'] = value
        elif 'shared' in scale_dict_key:
            transformed_scale_dict_key = transform_key(scale_dict_key)
            org_openclip_weight[transformed_scale_dict_key + '.scale'] = value    
        

    for key in task1_weight:
        print(key)

    low_rank_vector_task1= TaskVector(args, org_openclip_weight, task1_weight, vector = None)
    low_rank_vector_task2= TaskVector(args, org_openclip_weight, task1_weight, vector = None)

    # torch.save(org_openclip_weight, f"{args.save}/{args.save_file_name}")
    
if __name__ == "__main__":
    datasets = ['DTD', 'SUN397']
    model = 'ViT-L-14'
    args = parse_arguments()
    args.data_location = '/path/to/data'
    args.model = model
    args.save = f'checkpoints/{model}'
    args.save_file_name = '_'.join(datasets) + '_shared_openclip.pt'
    args.low_rank_mode = 'SoRA'
    args.initial_rank_ratio = 0.5
    pretrained_checkpoint = f'checkpoints/{model}/zeroshot.pt'
    args.shared_weight = '/data2/david3684/2024_arithmetic/shared_weight/20241007_vanilla/rankmin_config_20241009_uni_vanilla_2.bin'
    main(args)
    

