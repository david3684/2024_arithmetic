import json
import os, tqdm
import numpy as np
import open_clip
import torch
import torchvision.utils as utils
from core.datasets.common import get_dataloader, maybe_dictionarize
from core.heads import get_classification_head
from core.datasets.registry import get_dataset

from utils import find_optimal_coef
import argparse
from core.task_vectors import NonLinearTaskVector

class ImageClassifier(torch.nn.Module):
    def __init__(self, image_encoder, classification_head):
        super().__init__()
        self.image_encoder = image_encoder
        self.classification_head = classification_head
        if self.image_encoder is not None:
            self.train_preprocess = self.image_encoder.train_preprocess
            self.val_preprocess = self.image_encoder.val_preprocess

    def freeze_head(self):
        self.classification_head.weight.requires_grad_(False)
        self.classification_head.bias.requires_grad_(False)

    def forward(self, inputs):
        features = self.image_encoder(inputs)
        outputs = self.classification_head(features)
        return outputs

    def __call__(self, inputs):
        return self.forward(inputs)

    def save(self, filename):
        print(f"Saving image classifier to {filename}")
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f"Loading image classifier from {filename}")
        return utils.torch_load(filename)
    
class ImageEncoder(torch.nn.Module):
    def __init__(self, args, keep_lang=False, rotation_matrix=None, fnorm=None):
        super().__init__()

        print(f"Loading {args.model} pre-trained weights.")
        if "__pretrained__" in args.model:
            name, pretrained = args.model.split("__pretrained__")
        elif "__init__" in args.model:
            print("Using random initialization.")
            name, pretrained = args.model.split("__init__")[0], None
        else:
            name = args.model
            pretrained = "openai"
        (
            self.model,
            self.train_preprocess,
            self.val_preprocess,
        ) = open_clip.create_model_and_transforms(
            name, pretrained=pretrained, cache_dir=args.openclip_cachedir
        )

        self.cache_dir = args.cache_dir

        if not keep_lang and hasattr(self.model, "transformer"):
            delattr(self.model, "transformer")

    def forward(self, images):
        assert self.model is not None
        if self.rotation_matrix and self.fnorm is not None:
            # rescale and rotate the weights
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if 'bias' in name or 'layernorm' in name.lower():
                        continue
                    original_shape = param.data.shape
                    param.data = param.data.view(-1)
                    param.data = torch.matmul(self.rotation_matrix.transpose, param.data)
                    param.data = param.data * self.scaling_factor
                    param.data = param.data.view(original_shape)
        return self.model.encode_image(images)

    def __call__(self, inputs):
        return self.forward(inputs)

    def save(self, filename):
        print(f"Saving image encoder to {filename}")
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, model_name, filename):
        print(f"Loading image encoder from {filename}")
        state_dict = torch.load(filename, map_location="cpu")
        return cls.load(model_name, state_dict)

        
def add_normalized_accuracy(results, args):
    for dataset_name in args.eval_datasets:
        results[dataset_name + ":normalized_top1"] = (
            results[dataset_name + ":top1"] / args.finetuning_accuracies[dataset_name]
        )

    return results

def eval_single_dataset(image_encoder, dataset_name, args):
    classification_head = get_classification_head(args, dataset_name)
    model = ImageClassifier(image_encoder, classification_head)

    model.eval()

    dataset = get_dataset(
        dataset_name,
        model.val_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
    )
    dataloader = get_dataloader(dataset, is_train=False, args=args, image_encoder=None)
    device = args.device

    with torch.no_grad():
        top1, correct, n = 0.0, 0.0, 0.0
        for _, data in enumerate(tqdm.tqdm(dataloader)):
            data = maybe_dictionarize(data)
            x = data["images"].to(device)
            y = data["labels"].to(device)

            logits = utils.get_logits(x, model)

            pred = logits.argmax(dim=1, keepdim=True).to(device)

            correct += pred.eq(y.view_as(pred)).sum().item()

            n += y.size(0)

        top1 = correct / n

    metrics = {"top1": top1}
    print(f"Done evaluating on {dataset_name}. Accuracy: {100*top1:.2f}%")

    return metrics
def evaluate(image_encoder, args):
    if args.eval_datasets is None:
        return
    per_dataset_results = {}
    eval_datasets = (
        args.eval_datasets
        if args.control_dataset is None
        else args.eval_datasets + [args.control_dataset]
    )
    for dataset_name in eval_datasets:
        print("Evaluating on", dataset_name)

        results = eval_single_dataset(image_encoder, dataset_name, args)

        print(f"{dataset_name} Top-1 accuracy: {results['top1']:.4f}")
        per_dataset_results[dataset_name + ":top1"] = results["top1"]

    return per_dataset_results
def evaluate_task_vector(
    task_vector, shared_weight, args
):
    info = {}
    for scaling_coef in np.linspace(0.0, 1.0, args.n_eval_points):
        print(f"Evaluating for scaling coefficient {scaling_coef:.2f}")
        info[scaling_coef] = evaluate_task_vector_at_coef(
            task_vector,
            shared_weight,
            args,
            scaling_coef,
        )

    return info

def evaluate_task_vector_at_coef(
    task_vector, shared_weight, args, scaling_coef
):
    # build encoder with task arithmetic and scaling coefficient
    image_encoder = task_vector.apply_to(
        shared_weight, scaling_coef=scaling_coef
    )
    
    coef_info = evaluate(image_encoder, args)

    coef_info = add_normalized_accuracy(coef_info, args)
    coef_info["avg_normalized_top1"] = np.mean(
        [coef_info[dataset + ":normalized_top1"] for dataset in args.eval_datasets]
    )
    coef_info["avg_top1"] = np.mean(
        [coef_info[dataset + ":top1"] for dataset in args.eval_datasets]
    )

    return coef_info

def main(args):

    eval_datasets = [
        "DTD",
        "SUN397",
    ]

    task_vectors = []
    rotation_matrices = []
    scaling_factors = [] # frobenius norm 

    shared_weight = f"{args.save}/shared.pt"
    rotation_matrices = torch.load(f"{args.save}/rotation_matrices.pth")
    scaling_factors = torch.load(f"{args.save}/scaling_factors.pth")
    
    for i, dataset in enumerate(eval_datasets):
        finetuned = f"{args.save}/{dataset}/finetuned_sora.pt"
        rotation_matrix = rotation_matrices[i]
        scaling_factor = scaling_factors[i]
        task_vectors.append(
            NonLinearTaskVector(shared_weight, finetuned, rotation_matrix, scaling_factor)
        )

    # single coeffcieint for all task vectors
    # sum up normalized and rotated task vectors
    task_vector = sum(task_vectors)

    # We use the validation set to choose the optimal coefficient.
    val_metrics = evaluate_task_vector(
        task_vector,
        shared_weight,
        args,
    )

    optimal_coef = find_optimal_coef(
        val_metrics,
        metric="avg_normalized_top1",
        minimize=False,
    )

    
    # Evaluate on the test set with the optimal coefficient.
    args.eval_datasets = [dataset for dataset in eval_datasets]
    test_metrics = evaluate_task_vector_at_coef(
        task_vector,
        shared_weight,
        args,
        float(optimal_coef),
    )

    print("=" * 100)
    print(f"Test normalized accuracy: {test_metrics['avg_normalized_top1']}")
    print(f"Test absolute accuracy: {test_metrics['avg_top1']}")
    additive_accuracies = {"test": test_metrics, "val": val_metrics}

    save_file = f"{args.save}/additions.json"
    with open(save_file, "w") as f:
        json.dump(additive_accuracies, f, indent=4)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate task vectors.")
    parser.add_argument("--save", type=str, help="Directory to save results.")
    parser.add_argument("--n-eval-points", type=int, default=21, help="Number of evaluation points used to find optimal coefficient in task arithmetic.")
    args = parser.parse_args()
    main(args)