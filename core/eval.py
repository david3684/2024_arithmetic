import tqdm
import numpy as np
import torch
from core import utils
from core.datasets.common import get_dataloader, maybe_dictionarize
from core.heads import get_classification_head
from core.datasets.registry import get_dataset
from core.model import ImageClassifier, get_classification_head


        
def add_normalized_accuracy(results, args):
    for dataset_name in args.eval_datasets:
        results[dataset_name + ":normalized_top1"] = (
            results[dataset_name + ":top1"] / args.finetuning_accuracies[dataset_name]
        )

    return results

def evaluate(image_encoder, args, rotation_matrices, scaling_factors):
    if args.eval_datasets is None:
        return
    per_dataset_results = {}
    eval_datasets = args.eval_datasets

    for dataset_name in eval_datasets:
        print("Evaluating on", dataset_name)

        classification_head = model.get_classification_head(args, dataset_name)
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

                if dataset_name in rotation_matrices and dataset_name in scaling_factors:
                    x = torch.matmul(x, rotation_matrices[dataset_name].transpose(0, 1))
                    x = x * scaling_factors[dataset_name]
                logits = utils.get_scaled_logits(x, model)

                pred = logits.argmax(dim=1, keepdim=True).to(device)
                correct += pred.eq(y.view_as(pred)).sum().item()
                n += y.size(0)

            top1 = correct / n

        results = {"top1": top1}
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

