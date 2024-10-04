import argparse
import json

import torch

from core.eval import evaluate_task_vector, evaluate_task_vector_at_coef
from core.task_vectors import NonLinearTaskVector
from utils import find_optimal_coef

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