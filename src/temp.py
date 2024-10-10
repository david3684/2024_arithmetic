import torch
from src.modeling import ImageEncoder, ImageClassifier
from src.utils import cosine_lr, LabelSmoothing
import open_clip.model

# 체크포인트 파일 경로
checkpoint_path = f'/data2/david3684/2024_arithmetic/shared_weight/20241007_vanilla/rankmin_config_20241009_uni_vanilla_2.bin'

# 체크포인트 파일 로드
with torch.no_grad():
    encoder_state_dict = torch.load(checkpoint_path)

# checkpoint = torch.load(checkpoint_path, map_location='cpu')

# 체크포인트 파일의 키와 값을 출력하여 구조 확인
def print_checkpoint_structure(checkpoint, indent=0):
    for key, value in checkpoint.items():
        print(' ' * indent + f'Key: {key}')
        if isinstance(value, dict):
            print(' ' * indent + f'Value: Dictionary with {len(value)} keys')
            print_checkpoint_structure(value, indent + 2)
        elif isinstance(value, torch.Tensor):
            print(' ' * indent + f'Value: Tensor with shape {value.shape}')
        else:
            print(' ' * indent + f'Value: {type(value)}')

print_checkpoint_structure(encoder_state_dict)