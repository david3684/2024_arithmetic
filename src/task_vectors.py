import torch

def make_task_vector_for_weight(args, finetuned_single_weight, pretrained_single_weight):
    """Create a task vector for a single weight tensor."""
    if args.low_rank_mode == 'SoRA':
        #print(finetuned_single_weight, pretrained_single_weight)
        diff = finetuned_single_weight/torch.norm(finetuned_single_weight, p='fro')  - pretrained_single_weight
        
        #print(f"Shape of diff: {diff.shape}")
        U, s, V_T = torch.linalg.svd(diff.to(args.device), full_matrices=False)    
        U, s, V_T = U.to("cpu"), s.to("cpu"), V_T.to("cpu")
        dim = s.shape[0]
        parsed_dim = int(args.initial_rank_ratio * dim)
        sqrted_s = torch.sqrt(s[:parsed_dim])
        parsed_V_T = torch.diag(sqrted_s) @ V_T[:parsed_dim, :]
        parsed_U = U[:, :parsed_dim] @ torch.diag(sqrted_s)
        return parsed_U @ parsed_V_T

    return finetuned_single_weight/torch.norm(finetuned_single_weight, p='fro')  - pretrained_single_weight

        
def make_task_vector(args, finetuned_state_dict, pretrained_state_dict, task):
    """Create a task vector from a finetuned and a pretrained tensor."""
    task_vector = {}
    
    for key, value in finetuned_state_dict.items():
        print(f"Making task vector for {key}")
        if key == 'model.logit_scale':
            task_vector[key] = finetuned_state_dict[key]
        elif 'ln' in key:
            task_vector[key] = finetuned_state_dict[key] # preserve the layer norm of finetuned weight
        elif 'bias' in key:
            task_vector[key] = finetuned_state_dict[key] - pretrained_state_dict[key]  # zero? 이거 pretrained weight 모를 경우에 수정 해줘야 함.
        elif 'attn' in key or 'mlp' in key:
            task_vector[key] = make_task_vector_for_weight(args, finetuned_state_dict[key], pretrained_state_dict[key])
        elif isinstance(value, dict):
            continue # from original setting
        else:
            task_vector[key] = finetuned_state_dict[key] - pretrained_state_dict[key]  # positional embedding, class token, etc.
        
    return task_vector

class TaskVector():
    def __init__(self, args=None, pretrained_checkpoint=None, finetuned_checkpoint=None, task=None, vector=None):
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
                self.vector = make_task_vector(args, finetuned_state_dict, pretrained_state_dict, task)

    def to(self, device):
        for key in self.vector:
            self.vector[key] = self.vector[key].to(device)
        return self
    
    def __add__(self, other):
        """Add two task vectors together."""
        with torch.no_grad():
            new_vector = {}
            all_keys = set(self.vector.keys()).union(set(other.vector.keys()))    
            for key in all_keys:
                if key in self.vector and key in other.vector:
                    # 같은 키는 합침
                    new_vector[key] = self.vector[key] + other.vector[key]
                elif key in self.vector:
                    # self.vector에만 있는 키
                    new_vector[key] = self.vector[key]
                else:
                    # other.vector에만 있는 키
                    new_vector[key] = other.vector[key]
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
            if isinstance(pretrained_checkpoint, str):
                pretrained_model = torch.load(pretrained_checkpoint)
            elif isinstance(pretrained_checkpoint, torch.nn.Module):
                pretrained_model = pretrained_checkpoint
            else:
                raise ValueError("pretrained_checkpoint must be a file path or a model")
            state_dict = pretrained_model.state_dict()
            for key in state_dict:
                if key in self.vector:
                    state_dict[key] = state_dict[key] + scaling_coef * self.vector[key]
                else:
                    print(f"Key {key} not found in task vector. Copying from pretrained model.")
                    state_dict[key] = state_dict[key]
            for key in self.vector:
                if key not in state_dict:
                    print(f"Key {key} found only in task vector. Adding to new state dict.")
                    state_dict[key] = self.vector[key]
            pretrained_model.load_state_dict(state_dict, strict=True)
            return pretrained_model

