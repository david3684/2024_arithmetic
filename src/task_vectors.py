import torch


def make_task_vector_for_weight(args, finetuned_single_weight, pretrained_single_weight, key):
    """Create a task vector for a single weight tensor."""
    if args.low_rank_mode == 'SoRA':
        if 'in_proj' in key:
            # print(f"Shape of finetuned_single_weight: {finetuned_single_weight.shape}")
            q_1, k_1, v_1 = finetuned_single_weight.chunk(3, dim=0)
            q_2, k_2, v_2 = pretrained_single_weight.chunk(3, dim=0)
            q_diff = q_1/torch.norm(q_1, p='fro') - q_2
            k_diff = k_1/torch.norm(k_1, p='fro') - k_2
            v_diff = v_1/torch.norm(v_1, p='fro') - v_2
            diff = torch.cat([q_diff, k_diff, v_diff], dim=0)
        else:
            diff = finetuned_single_weight / \
                torch.norm(finetuned_single_weight, p='fro') - \
                pretrained_single_weight

        # print(f"Shape of diff: {diff.shape}")
        U, s, V_T = torch.linalg.svd(diff.to(args.device), full_matrices=False)
        U, s, V_T = U.to("cpu"), s.to("cpu"), V_T.to("cpu")
        dim = s.shape[0]
        parsed_dim = int(args.initial_rank_ratio * dim)
        sqrted_s = torch.sqrt(s[:parsed_dim])
        parsed_V_T = torch.diag(sqrted_s) @ V_T[:parsed_dim, :]
        parsed_U = U[:, :parsed_dim] @ torch.diag(sqrted_s)
        return parsed_U @ parsed_V_T
    else:
        if 'in_proj' in key:
            # print(f"Shape of finetuned_single_weight: {finetuned_single_weight.shape}")
            q_1, k_1, v_1 = finetuned_single_weight.chunk(3, dim=0)
            q_2, k_2, v_2 = pretrained_single_weight.chunk(3, dim=0)
            q_diff = q_1/torch.norm(q_1, p='fro') - q_2
            k_diff = k_1/torch.norm(k_1, p='fro') - k_2
            v_diff = v_1/torch.norm(v_1, p='fro') - v_2
            diff = torch.cat([q_diff, k_diff, v_diff], dim=0)
        else:
            diff = finetuned_single_weight / \
                torch.norm(finetuned_single_weight, p='fro') - \
                pretrained_single_weight
    return diff


def make_task_vector(args, finetuned_state_dict, pretrained_state_dict, task):
    """Create a task vector from a finetuned and a pretrained tensor."""
    task_vector = {}

    for key, value in finetuned_state_dict.items():
        # print(f"Making task vector for {key}")
        value.to(args.device)
        diff = finetuned_state_dict[key] - pretrained_state_dict[key]
        if key == 'model.logit_scale':
            task_vector[key] = diff  # zero out the logit scale
        elif 'ln' in key or 'bias' in key:
            # preserve the layer norm of finetuned weight or handle bias
            task_vector[key] = diff
        elif 'attn' in key or 'mlp' in key:
            task_vector[key] = make_task_vector_for_weight(
                args, finetuned_state_dict[key], pretrained_state_dict[key], key)
        elif isinstance(value, dict):
            continue  # from original setting
        else:
            # positional embedding, class token, etc.
            task_vector[key] = diff
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
                pretrained_state_dict = pretrained_checkpoint
                finetuned_state_dict = finetuned_checkpoint
                if args.no_shared_weight:
                    print('Building task vector with no shared weight')
                    self.vector = {}
                    for key in pretrained_state_dict:
                        if pretrained_state_dict[key].dtype in [torch.int64, torch.uint8]:
                            continue
                        self.vector[key] = finetuned_state_dict[key] - \
                            pretrained_state_dict[key]
                else:
                    print('Building task vector with shared weight')
                    self.vector = make_task_vector(
                        args, finetuned_state_dict, pretrained_state_dict, task)

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

    def multiply(self, scalar):
        """Multiply a task vector by a scalar."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = scalar * self.vector[key]
        return TaskVector(vector=new_vector)

    def apply_to(self, pretrained_checkpoint, scaling_coef=1.0):
        """Apply a task vector to a pretrained model."""
        with torch.no_grad():
            if isinstance(pretrained_checkpoint, str):
                pretrained_model = torch.load(pretrained_checkpoint)
            elif isinstance(pretrained_checkpoint, torch.nn.Module):
                pretrained_model = pretrained_checkpoint
            else:
                raise ValueError(
                    "pretrained_checkpoint must be a file path or a model")
            state_dict = pretrained_model.state_dict()
            for key in state_dict:
                if key in self.vector:
                    state_dict[key] = state_dict[key] + \
                        scaling_coef * self.vector[key]
                else:
                    print(
                        f"Key {key} not found in task vector. Copying from pretrained model.")
                    state_dict[key] = state_dict[key]
            for key in self.vector:
                if key not in state_dict:
                    print(
                        f"Key {key} found only in task vector. Adding to new state dict.")
                    state_dict[key] = self.vector[key]
            pretrained_model.load_state_dict(state_dict, strict=True)
            return pretrained_model
