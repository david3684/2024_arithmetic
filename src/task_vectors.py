import torch

def make_task_vector_for_weight(args, finetuned_single_weight, pretrained_single_weight):
    """Create a task vector for a single weight tensor."""
    if args.low_rank_mode == 'SoRA':
        print(finetuned_single_weight, pretrained_single_weight)
        diff = finetuned_single_weight - pretrained_single_weight
        
        print(f"Shape of diff: {diff.shape}")
        U, s, V = torch.linalg.svd(diff.to(args.device), full_matrices=False)    
        U, s, V = U.to("cpu"), s.to("cpu"), V.to("cpu")
        dim = s.shape[0]
        parsed_dim = int(args.initial_rank_ratio * dim)
        sqrted_s = torch.sqrt(s[:parsed_dim])
        parsed_U = U[:, :parsed_dim] @ torch.diag(sqrted_s)
        parsed_V = torch.diag(sqrted_s) @ V[:parsed_dim, :]
        return parsed_U @ parsed_V

    return finetuned_single_weight - pretrained_single_weight

def make_task_vector(args, finetuned_state_dict, pretrained_state_dict):
    """Create a task vector from a finetuned and a pretrained tensor."""
    task_vector = {}
    for key in finetuned_state_dict:
        print(f"Making task vector for {key}")
        if 'attn' or 'mlp' in key:
            task_vector[key] = make_task_vector_for_weight(args, finetuned_state_dict[key], pretrained_state_dict[key])
        elif pretrained_state_dict[key].dtype in [torch.int64, torch.uint8]: 
            continue # from original setting
        elif 'ln' in key:
            task_vector[key] = finetuned_state_dict[key] # preserve the layer norm of finetuned weight
        else:
            task_vector[key] = finetuned_state_dict[key] - pretrained_state_dict[key]  # 이거 pretrained weight 모를 경우에 수정 해줘야 함.
            
    return task_vector

class TaskVector():
    def __init__(self, args, pretrained_checkpoint=None, finetuned_checkpoint=None, vector=None):
        """Initializes the task vector from a pretrained and a finetuned checkpoints.
        
        This can either be done by passing two state dicts (one corresponding to the
        pretrained model, and another to the finetuned model), or by directly passying in
        the task vector state dict.
        """
        if vector is not None:
            self.vector = vector
        else:
            assert pretrained_checkpoint is not None and finetuned_checkpoint is not None
            with torch.no_grad():
                # pretrained_state_dict = torch.load(pretrained_checkpoint).state_dict()
                pretrained_state_dict = pretrained_checkpoint
                finetuned_state_dict = finetuned_checkpoint
                self.vector = make_task_vector(args, finetuned_state_dict, pretrained_state_dict)

    
    def __add__(self, other):
        """Add two task vectors together."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                if key not in other.vector:
                    print(f'Warning, key {key} is not present in both task vectors.')
                    continue
                new_vector[key] = self.vector[key] + other.vector[key]
        return TaskVector(vector=new_vector)

    def __radd__(self, other):
        if other is None or isinstance(other, int):
            return self
        return self.__add__(other)

    def __neg__(self):
        """Negate a task vector."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = - self.vector[key]
        return TaskVector(vector=new_vector)

    def apply_to(self, pretrained_checkpoint, scaling_coef=1.0):
        """Apply a task vector to a pretrained model."""
        with torch.no_grad():
            pretrained_model = torch.load(pretrained_checkpoint)
            new_state_dict = {}
            pretrained_state_dict = pretrained_model.state_dict()
            for key in pretrained_state_dict:
                if key not in self.vector:
                    print(f'Warning: key {key} is present in the pretrained state dict but not in the task vector')
                    continue
                # add up할때는 bias ln 어떻게 해야 하나...
                new_state_dict[key] = pretrained_state_dict[key] + scaling_coef * self.vector[key]
        pretrained_model.load_state_dict(new_state_dict, strict=False)
        return pretrained_model

