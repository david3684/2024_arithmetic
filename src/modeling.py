import torch

import open_clip

from src import utils


class ImageEncoder(torch.nn.Module):
    def __init__(self, args, keep_lang=False):
        super().__init__()

        print(f'Loading {args.model} pre-trained weights.')
        if '__pretrained__' in args.model:
            name, pretrained = args.model.split('__pretrained__')
        else:
            name = args.model
            pretrained = 'openai'
        self.model, self.train_preprocess, self.val_preprocess = open_clip.create_model_and_transforms(
            name, pretrained=pretrained, cache_dir=args.openclip_cachedir)
        
        self.cache_dir = args.cache_dir

        if not keep_lang and hasattr(self.model, 'transformer'):
            delattr(self.model, 'transformer')

        if args.scale_model:
            self.apply_scaling_and_permutation(args.scaling_factor, args.permute_matrix)
        
    def apply_scaling(self, scaling_coef, permute_matrix):
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                param.data = torch.matmul(param.data, permute_matrix.Transpose(0, 1))
                param.data *= scaling_coef
                
    def forward(self, images, task=None, args=None):
        assert self.model is not None
        
        # Apply scaling factors to weights
        #### Fix ####
        if args.task_scale_factors is not None:
            # print('Forward pass with scaling factors')
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if 'weight' in name:
                        scale_name = 'model.'+name + '.scale' # load task specific scaling factor
                        if scale_name in args.task_scale_factors[task].keys():
                            # import ipdb; ipdb.set_trace()
                            scaling_factor = args.task_scale_factors[task][scale_name]
                            if 'attn.in_proj' in name:
                                q, k, v = param.data.chunk(3, dim=0)
                                q *= scaling_factor[0]
                                k *= scaling_factor[1]
                                v *= scaling_factor[2]
                                param.data = torch.cat([q, k, v], dim=0)
                            else:
                                param.data *= scaling_factor                        
            encoded_images = self.model.encode_image(images)
        
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if 'weight' in name:
                        scale_name = 'model.'+name + '.scale' # load task specific scaling factor
                        if scale_name in args.task_scale_factors[task].keys():
                            scaling_factor = args.task_scale_factors[task][scale_name]
                            if 'attn.in_proj' in name:
                                q, k, v = param.data.chunk(3, dim=0)
                                q /= scaling_factor[0]
                                k /= scaling_factor[1]
                                v /= scaling_factor[2]
                                param.data = torch.cat([q, k, v], dim=0)
                            else:
                                param.data /= scaling_factor
        else:
            encoded_images = self.model.encode_image(images)
        
        return encoded_images
    
    def __call__(self, inputs, dataset, args):
        return self.forward(inputs, dataset, args)

    def save(self, filename):
        print(f'Saving image encoder to {filename}')
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, model_name, filename):
        print(f'Loading image encoder from {filename}')
        state_dict = torch.load(filename)
        return cls.load(model_name, state_dict)

    @classmethod
    def load_from_state_dict(cls, model_name, state_dict):
        self.model, self.train_preprocess, self.val_preprocess = open_clip.create_model_and_transforms(
            name, pretrained=pretrained, cache_dir=args.openclip_cachedir)
        self.model.load_from_state_dict(state_dict)
        



class ClassificationHead(torch.nn.Linear):
    def __init__(self, normalize, weights, biases=None):
        output_size, input_size = weights.shape
        super().__init__(input_size, output_size)
        self.normalize = normalize
        if weights is not None:
            self.weight = torch.nn.Parameter(weights.clone())
        if biases is not None:
            self.bias = torch.nn.Parameter(biases.clone())
        else:
            self.bias = torch.nn.Parameter(torch.zeros_like(self.bias))

    def forward(self, inputs):
        if self.normalize:
            inputs = inputs / inputs.norm(dim=-1, keepdim=True)
        return super().forward(inputs)

    def __call__(self, inputs):
        return self.forward(inputs)

    def save(self, filename):
        print(f'Saving classification head to {filename}')
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f'Loading classification head from {filename}')
        return utils.torch_load(filename)


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

    def forward(self, inputs, dataset, args):
        features = self.image_encoder(inputs, dataset, args)
        outputs = self.classification_head(features)
        return outputs

    def __call__(self, inputs, dataset, args):
        return self.forward(inputs, dataset, args)

    def save(self, filename):
        print(f'Saving image classifier to {filename}')
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f'Loading image classifier from {filename}')
        return utils.torch_load(filename)


class MultiHeadImageClassifier(torch.nn.Module):
    def __init__(self, image_encoder, classification_heads):
        super().__init__()
        self.image_encoder = image_encoder
        self.classification_heads = torch.nn.ModuleList(classification_heads)
        if self.image_encoder is not None:
            self.train_preprocess = self.image_encoder.train_preprocess
            self.val_preprocess = self.image_encoder.val_preprocess

    def freeze_head(self):
        for idx in range(len(self.classification_heads)):
            self.classification_heads[idx].weight.requires_grad_(False)
            self.classification_heads[idx].bias.requires_grad_(False)

    def forward(self, inputs, head_idx):
        features = self.image_encoder(inputs)
        outputs = self.classification_heads[head_idx](features)
        return outputs

    def __call__(self, inputs, head_idx):
        return self.forward(inputs, head_idx)

    def save(self, filename):
        print(f'Saving image classifier to {filename}')
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f'Loading image classifier from {filename}')
        return utils.torch_load(filename)
