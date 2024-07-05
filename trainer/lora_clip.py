import math

# import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
# from einops import rearrange
# from safetensors import safe_open
# from safetensors.torch import save_file
# from timm.models.vision_transformer import VisionTransformer as timm_ViT
from torch import Tensor
from torch.nn.parameter import Parameter

# from base_vit import ViT


class _LoRALayer(nn.Module):
    def __init__(self, w: nn.Module, w_a: nn.Module, w_b: nn.Module):
        super().__init__()
        self.w = w
        self.w_a = w_a
        self.w_b = w_b

    def forward(self, x):
        x = self.w(x) + self.w_b(self.w_a(x))
        return x


class LoRA_clip(nn.Module):
    """Applies low-rank adaptation to a vision transformer.

    Args:
        vit_model: a vision transformer model, see base_vit.py
        r: rank of LoRA
        num_classes: how many classes the model output, default to the vit model
        lora_layer: which layer we apply LoRA.

    Examples::
        >>> model = ViT('B_16_imagenet1k')
        >>> lora_model = LoRA_ViT(model, r=4)
        >>> preds = lora_model(img)
        >>> print(preds.shape)
        torch.Size([1, 1000])
    """

    def __init__(self, vit_model, r: int, num_classes: int = 0, lora_layer=None):
        super(LoRA_clip, self).__init__()

        assert r > 0
        # vit_model.visual.transformer.resblocks #visual
        # vit_model.transformer.resblocks #text
        # base_vit_dim = vit_model.transformer.blocks[0].attn.proj_q.in_features
        # dim = base_vit_dim
        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(range(len(vit_model.transformer.resblocks)))
        # create for storage, then we can init them or load weights
        self.w_As = []  # These are linear layers
        self.w_Bs = []

        self.w_As1 = []  # These are linear layers
        self.w_Bs1 = []

        # lets freeze first
        for param in vit_model.parameters():
            param.requires_grad = False

        # Here, we do the surgery
        for t_layer_i, blk in enumerate(vit_model.transformer.resblocks):
            # If we only want few lora layer instead of all
            if t_layer_i not in self.lora_layer:
                continue
            # w_q_linear = blk.attn.q_proj_weight
            # w_v_linear = blk.attn.v_proj_weight
            w_in_linear = blk.attn.in_proj_weight
            # w_in_linear.requires_grad = False

            # self.dimq = w_q_linear.shape[0]
            self.dim1 = w_in_linear.shape[0] #dim1 = 3*dim2
            self.dim2 = w_in_linear.shape[1]

            # w_a_linear_q = nn.Linear(self.dimq , r, bias=False)
            # w_b_linear_q = nn.Linear(r, self.dimq , bias=False)
            # w_a_linear_v = nn.Linear(self.dimv, r, bias=False)
            # w_b_linear_v = nn.Linear(r, self.dimv, bias=False)

            self.w_a_linear_q = Parameter(torch.empty((self.dim2, r)),requires_grad=True)
            self.w_b_linear_q = Parameter(torch.empty((r, self.dim2)),requires_grad=True)
            self.w_a_linear_v = Parameter(torch.empty((self.dim2, r)),requires_grad=True)
            self.w_b_linear_v = Parameter(torch.empty((r, self.dim2 )),requires_grad=True)

            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)
            # blk.attn.proj_q = _LoRALayer(w_q_linear, w_a_linear_q, w_b_linear_q)
            # blk.attn.proj_v = _LoRALayer(w_v_linear, w_a_linear_v, w_b_linear_v)
            
            blk.attn.in_proj_weight.data[0:self.dim2,:] += (w_a_linear_q @ w_b_linear_q)
            blk.attn.in_proj_weight.data[-self.dim2:,:] += (w_a_linear_v @ w_b_linear_v)

            # blk.attn.q_proj_weight = w_q_linear + (w_a_linear_q @ w_b_linear_q)
            # blk.attn.v_proj_weight = w_v_linear + (w_a_linear_v @ w_b_linear_v)


        # Here, we do the surgery
        for t_layer_i, blk in enumerate(vit_model.transformer.resblocks):
            # If we only want few lora layer instead of all
            if t_layer_i not in self.lora_layer:
                continue
            # w_q_linear = blk.attn.q_proj_weight
            # w_v_linear = blk.attn.v_proj_weight
            w_in_linear = blk.attn.in_proj_weight
            # w_in_linear.requires_grad = False

            # self.dimq = w_q_linear.shape[0]
            self.dim1 = w_in_linear.shape[0] #dim1 = 3*dim2
            self.dim2 = w_in_linear.shape[1]

            # w_a_linear_q = nn.Linear(self.dimq , r, bias=False)
            # w_b_linear_q = nn.Linear(r, self.dimq , bias=False)
            # w_a_linear_v = nn.Linear(self.dimv, r, bias=False)
            # w_b_linear_v = nn.Linear(r, self.dimv, bias=False)

            self.w_a_linear_q = Parameter(torch.empty((self.dim2, r)),requires_grad=True)
            self.w_b_linear_q = Parameter(torch.empty((r, self.dim2)),requires_grad=True)
            self.w_a_linear_v = Parameter(torch.empty((self.dim2, r)),requires_grad=True)
            self.w_b_linear_v = Parameter(torch.empty((r, self.dim2 )),requires_grad=True)

            self.w_As1.append(w_a_linear_q)
            self.w_Bs1.append(w_b_linear_q)
            self.w_As1.append(w_a_linear_v)
            self.w_Bs1.append(w_b_linear_v)
            # blk.attn.proj_q = _LoRALayer(w_q_linear, w_a_linear_q, w_b_linear_q)
            # blk.attn.proj_v = _LoRALayer(w_v_linear, w_a_linear_v, w_b_linear_v)
            
            blk.attn.in_proj_weight.data[0:self.dim2,:] += (w_a_linear_q @ w_b_linear_q)
            blk.attn.in_proj_weight.data[-self.dim2:,:] += (w_a_linear_v @ w_b_linear_v)

        self.reset_parameters()
        self.lora_clip = vit_model
        if num_classes > 0:
            self.lora_vit.fc = nn.Linear(vit_model.fc.in_features, num_classes)

    def save_fc_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.
        """
        assert filename.endswith(".safetensors")
        _in = self.lora_vit.fc.in_features
        _out = self.lora_vit.fc.out_features
        fc_tensors = {f"fc_{_in}in_{_out}out": self.lora_vit.fc.weight}
        save_file(fc_tensors, filename)

    def load_fc_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.
        """

        assert filename.endswith(".safetensors")
        _in = self.lora_vit.fc.in_features
        _out = self.lora_vit.fc.out_features
        with safe_open(filename, framework="pt") as f:
            saved_key = f"fc_{_in}in_{_out}out"
            try:
                saved_tensor = f.get_tensor(saved_key)
                self.lora_vit.fc.weight = Parameter(saved_tensor)
            except ValueError:
                print("this fc weight is not for this model")

    def save_lora_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.
        """

        assert filename.endswith(".safetensors")

        num_layer = len(self.w_As)  # actually, it is half
        a_tensors = {f"w_a_{i:03d}": self.w_As[i].weight for i in range(num_layer)}
        b_tensors = {f"w_b_{i:03d}": self.w_Bs[i].weight for i in range(num_layer)}
        
        _in = self.lora_vit.fc.in_features
        _out = self.lora_vit.fc.out_features
        fc_tensors = {f"fc_{_in}in_{_out}out": self.lora_vit.fc.weight}
        
        merged_dict = {**a_tensors, **b_tensors, **fc_tensors}
        save_file(merged_dict, filename)

    def load_lora_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.
        """

        assert filename.endswith(".safetensors")

        with safe_open(filename, framework="pt") as f:
            for i, w_A_linear in enumerate(self.w_As):
                saved_key = f"w_a_{i:03d}"
                saved_tensor = f.get_tensor(saved_key)
                w_A_linear.weight = Parameter(saved_tensor)

            for i, w_B_linear in enumerate(self.w_Bs):
                saved_key = f"w_b_{i:03d}"
                saved_tensor = f.get_tensor(saved_key)
                w_B_linear.weight = Parameter(saved_tensor)
                
            _in = self.lora_vit.fc.in_features
            _out = self.lora_vit.fc.out_features
            saved_key = f"fc_{_in}in_{_out}out"
            try:
                saved_tensor = f.get_tensor(saved_key)
                self.lora_vit.fc.weight = Parameter(saved_tensor)
            except ValueError:
                print("this fc weight is not for this model")

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B)

    def forward(self, x: Tensor) -> Tensor:
        return self.lora_clip(x)
