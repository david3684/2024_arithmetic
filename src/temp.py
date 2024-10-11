import torch
from src.modeling import ImageEncoder, ImageClassifier
from src.utils import cosine_lr, LabelSmoothing, get_logits
import open_clip.model
from src.heads import get_classification_head
from src.datasets.common import get_dataloader, maybe_dictionarize
from tqdm import tqdm
from src.datasets.registry import get_dataset
from src.args import parse_arguments

# 체크포인트 파일 경로
checkpoint_path = '/data2/david3684/2024_arithmetic/checkpoints/ViT-L-14/DTD/finetuned.pt'
image_encoder = torch.load(checkpoint_path)
args = parse_arguments()
args.device = 'cuda'
args.data_location = '/data2/david3684/data'
args.batch_size = 32
args.save = '/data2/david3684/2024_arithmetic/checkpoints/ViT-L-14'

classification_head = get_classification_head(args, dataset = 'DTD')
model = ImageClassifier(image_encoder, classification_head)
# 이 근방에서 scaling 처리 해줘야함.
model.eval()

dataset = get_dataset(
    "DTD",
    model.val_preprocess,
    location='/data2/david3684/data',
    batch_size=32
)
dataloader = get_dataloader(
    dataset, is_train=False, args=args, image_encoder=None)
device = args.device

with torch.no_grad():
    top1, correct, n = 0., 0., 0.
    for i, data in enumerate(tqdm(dataloader)):
        data = maybe_dictionarize(data)
        x = data['images'].to(device)
        y = data['labels'].to(device)

        logits = get_logits(x, model)

        pred = logits.argmax(dim=1, keepdim=True).to(device)

        correct += pred.eq(y.view_as(pred)).sum().item()
        
        n += y.size(0)

    top1 = correct / n

metrics = {'top1': top1}
print(f'Done evaluating on DTD. Accuracy: {100*top1:.2f}%')

# # 체크포인트 파일 로드
# with torch.no_grad():
#     encoder_state_dict = torch.load(checkpoint_path).state_dict()

# # checkpoint = torch.load(checkpoint_path, map_location='cpu')

# # 체크포인트 파일의 키와 값을 출력하여 구조 확인
# def print_checkpoint_structure(checkpoint, indent=0):
#     for key, value in checkpoint.items():
#         print(' ' * indent + f'Key: {key}')
#         if isinstance(value, dict):
#             print(' ' * indent + f'Value: Dictionary with {len(value)} keys')
#             print_checkpoint_structure(value, indent + 2)
#         elif isinstance(value, torch.Tensor):
#             print(' ' * indent + f'Value: Tensor with shape {value.shape}')
#         else:
#             print(' ' * indent + f'Value: {type(value)}')

# print_checkpoint_structure(encoder_state_dict)