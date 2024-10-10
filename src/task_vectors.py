import torch

def make_task_vector_for_weight(args, finetuned_weight, pretrained_weight, low_rank=None):
    """Create a task vector for a single weight tensor."""
    if low_rank == 'SoRA':
        diff = finetuned_weight - pretrained_weight
    
        U, s, V = torch.linalg.svd(diff.to(args.device), full_matrices=False)    
        U, s, V = U.to("cpu"), s.to("cpu"), V.to("cpu")
        dim = s.shape[0]
        parsed_dim = int(args.initial_rank_ratio * dim)
        sqrted_s = torch.sqrt(s[:parsed_dim])
        parsed_U = U[:, :parsed_dim] @ torch.diag(sqrted_s)
        parsed_V = torch.diag(sqrted_s) @ V[:parsed_dim, :]
        return parsed_U@parsed_V
        # return parsed_U.to("cpu"), parsed_V.to("cpu")
    return finetuned_weight - pretrained_weight

def make_task_vector(args, finetuned_tensor, pretrained_tensor, low_rank=None):
    """Create a task vector from a finetuned and a pretrained tensor."""
    task_vector = {}
    for key in finetuned_tensor:
        if 'bias' in key:
            task_vector[key] = finetuned_tensor[key] - pretrained_tensor[key] 
        elif 'ln' in key:
            task_vector[key] = finetuned_tensor[key] # preserve the layer norm of finetuned weight
        else:
            task_vector[key] = make_task_vector_for_weight(args, finetuned_tensor[key], pretrained_tensor[key], low_rank)
            
    return task_vector

class TaskVector():
    def __init__(self, args, pretrained_checkpoint=None, finetuned_checkpoint=None, vector=None,  low_rank=None):
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
                pretrained_state_dict = torch.load(pretrained_checkpoint).state_dict()
                finetuned_state_dict = torch.load(finetuned_checkpoint).state_dict()
                self.vector = {}
                for key in pretrained_state_dict:
                    if pretrained_state_dict[key].dtype in [torch.int64, torch.uint8]:
                        continue
                    self.vector[key] = make_task_vector(args, finetuned_state_dict[key], pretrained_state_dict[key], low_rank)

    
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

