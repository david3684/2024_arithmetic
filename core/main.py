import argparse
import json
import os
import torch

from core.eval import evaluate_task_vector, evaluate_task_vector_at_coef
from core.task_vectors import NonLinearTaskVector
from utils import find_optimal_coef

def main(args):

    if torch.cuda.is_available():
        args.device = "cuda"
        
    eval_datasets = [
        "DTD",
        "SUN397",
    ]
    task_vectors = []
    scaling_factors = {} # frobenius norm 
    rotation_matrices = {}

    shared_weight = f"{args.save}/shared.pt"
    loaded_scaling_factors = torch.load(f"{args.save}/scaling_factors.pth")
    
    if args.apply_rotation:
        loaded_rotation_matrices = torch.load(f"{args.save}/rotation_matrices.pth")
        
    for i, dataset in enumerate(eval_datasets):
        finetuned = f"{args.save}/{dataset}/finetuned_sora.pt"
        scaling_factors[dataset] = loaded_scaling_factors[i]
        
        if args.apply_rotation:
            rotation_matrices[dataset] = loaded_rotation_matrices[i]
        
        rotation_matrix = rotation_matrices[dataset] if args.apply_rotation else None
        scaling_factor = scaling_factors[dataset]
        
        task_vectors.append(
            NonLinearTaskVector(shared_weight, finetuned, rotation_matrix, scaling_factor)
        )

    # single coeffcieint for all task vectors
    # sum up normalized and rotated task vectors
    task_vector = sum(task_vectors)

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
    parser.add_argument(
        "--data-location",
        type=str,
        default=os.path.expanduser("~/data"),
        help="The root directory for the datasets.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ViT-L-14",
        help="The type of model (e.g. RN50, ViT-B-32).",
    )
    parser.add_argument("--save", type=str, help="Directory to save results.")
    parser.add_argument("--n-eval-points", type=int, default=21, help="Number of evaluation points used to find optimal coefficient in task arithmetic.")
    #parser.add_argument("--apply_rotation", type=bool, default=True, help="Whether to apply rotation matrices to task vectors.")
    args = parser.parse_args()
    main(args)