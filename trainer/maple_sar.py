import os.path as osp
from collections import OrderedDict
import math
import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from . import models_mae, models_mae_vitae
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
import numpy as np
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from   core_qnn.quaternion_layers       import *
# from .lora_clip import LoRA_clip
from torch.nn import Parameter
_tokenizer = _Tokenizer()
from torch.nn.init import xavier_uniform_
from segment_anything import sam_model_registry
from timm.models.vision_transformer import vit_base_patch16_384

def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)
    print(model_path)
    try:
        print('yes')
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        print('yes')
        state_dict = torch.load(model_path, map_location="cpu")
    design_details = {"trainer": 'MaPLe',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0,
                      "maple_length": cfg.TRAINER.MAPLE.N_CTX}
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, compound_prompts_deeper_text):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        # Pass as the list, as nn.sequential cannot process multiple arguments in the forward pass
        combined = [x, compound_prompts_deeper_text, 0]  # third argument is the counter which denotes depth of prompt
        outputs = self.transformer(combined)
        x = outputs[0]  # extract the x back from here
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class MultiModalPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.MAPLE.N_CTX
        ctx_init = cfg.TRAINER.MAPLE.CTX_INIT
        
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        
        # dtype = clip_model.lora_clip.dtype
        # ctx_dim = clip_model.lora_clip.ln_final.weight.shape[0]
        # clip_imsize = clip_model.lora_clip.visual.input_resolution

        cfg_imsize = cfg.INPUT.SIZE[0]
        # Default is 1, which is compound shallow prompting
        assert cfg.TRAINER.MAPLE.PROMPT_DEPTH >= 1, "For MaPLe, PROMPT_DEPTH should be >= 1"
        self.compound_prompts_depth = cfg.TRAINER.MAPLE.PROMPT_DEPTH  # max=12, but will create 11 such shared prompts
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"
        self.dtype = clip_model.dtype
        if ctx_init and (n_ctx) <= 4:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = n_ctx
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        print('MaPLe design: Multi-modal Prompt Learning')
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of MaPLe context words (tokens): {n_ctx}")
        # These below, related to the shallow prompts
        # Linear layer so that the tokens will project to 512 and will be initialized from 768
        self.proj = nn.Linear(ctx_dim, 768)
        self.proj.half()
        self.ctx = nn.Parameter(ctx_vectors)
        # These below parameters related to the shared prompts
        # Define the compound prompts for the deeper layers

        # self.meta_net = nn.Sequential(OrderedDict([
        #     ("linear1", nn.Linear(512, 512 // 16)),
        #     ("relu", nn.ReLU(inplace=True)),
        #     ("linear2", nn.Linear(512 // 16, ctx_dim))
        # ]))
        self.meta_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(1000, 512 // 16)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(512 // 16, ctx_dim))
        ]))

        # self.visual_net_mrq = QuaternionLinearAutograd(512, ctx_dim,rotation=True)
        self.visual_net_mq = QuaternionLinearAutograd(ctx_dim*2, ctx_dim)
        # self.visual_net_mq = nn.Linear(ctx_dim, ctx_dim)
        self.visual_net_mq.half()
        if cfg.TRAINER.COCOOP.PREC == "fp16":
            self.meta_net.half()

        # Minimum can be 1, which defaults to shallow MaPLe
        # compound prompts
        self.compound_prompts_text = nn.ParameterList([nn.Parameter(torch.empty(n_ctx, 512))
                                                      for _ in range(self.compound_prompts_depth - 1)])
        for single_para in self.compound_prompts_text:
            nn.init.normal_(single_para, std=0.02)
        # Also make corresponding projection layers, for each prompt
        # single_layer = nn.Linear(ctx_dim, 768)
        # single_layer = nn.Linear(ctx_dim*2, 768)
        single_layer = QuaternionLinearAutograd(ctx_dim*2, 768)
        self.compound_prompt_projections = _get_clones(single_layer, self.compound_prompts_depth - 1)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self,maefeatures):
        ctx = self.ctx # (n_ctx, ctx_dim)

        bias = self.meta_net(maefeatures.type(self.dtype))  # (batch, ctx_dim)
        # bias = self.visual_net_mrq(maefeatures.float())
        mean = torch.mean(maefeatures)
        noise = torch.randn(bias.shape).cuda().type(self.dtype)*mean

        bias = bias.unsqueeze(1)  # (batch, 1, ctx_dim)
        ctx = ctx.unsqueeze(0)  # (1, n_ctx, ctx_dim)
        # ctx_shifted = ctx + bias   # (batch, n_ctx, ctx_dim)
        ctx_shifted = ctx + bias + noise.unsqueeze(1)  # (batch, n_ctx, ctx_dim)
        a = torch.zeros(ctx_shifted.shape)
        ctx_dim = ctx_shifted.shape[-1]
        a = a.type(ctx_shifted.dtype)
        ctx_shifted = torch.cat((a.cuda(),ctx_shifted),dim=-1)
        ctx_shifted = self.visual_net_mq(ctx_shifted)
        ctx_shifted = ctx_shifted.type(self.dtype)

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = []
        for ctx_shifted_i in ctx_shifted:
            ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_cls, -1, -1)
            pts_i = self.construct_prompts(ctx_i, prefix, suffix)  # (n_cls, n_tkn, ctx_dim)
            prompts.append(pts_i)
        prompts = torch.stack(prompts)

        # if ctx.dim() == 2:
        #     ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        # prefix = self.token_prefix
        # suffix = self.token_suffix
        # prompts = self.construct_prompts(ctx, prefix, suffix)

        # Before returning, need to transform
        # prompts to 768 for the visual side
        visual_deep_prompts = []
        for index, layer in enumerate(self.compound_prompt_projections):
            # visual_deep_prompts.append(layer(self.compound_prompts_text[index]))
            # print(self.compound_prompts_text[index].shape) #2,512

            cprompt_text = self.compound_prompts_text[index].unsqueeze(0) 
            cprompt_text = cprompt_text + bias  # (batch, n_ctx, ctx_dim)
            a = torch.zeros(cprompt_text.shape)
            a = a.type(cprompt_text.dtype)
            cprompt_text = torch.cat((a.cuda(),cprompt_text),dim=-1)
            cprompt_visual = layer(cprompt_text)
            visual_deep_prompts.append(cprompt_visual)

        # Now the other way around
        # We will project the textual prompts from 512 to 768
        return prompts, self.proj(self.ctx), self.compound_prompts_text, visual_deep_prompts   # pass here original, as for visual 768 is required

class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = MultiModalPromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        # print('loading remote MAE model')
        # self.remote_model = models_mae.__dict__['mae_vit_base_patch16'](norm_pix_loss=False)
        # model_path = '/code/multimodal-prompt-learning/premodels/vit-b-checkpoint-1599.pth'
        # self.remote_model.load_state_dict(torch.load(model_path)['model'])
        
        # self.remote_model = models_mae_vitae.__dict__['mae_vitae_base_patch16'](norm_pix_loss=False)
        # model_path = '/code/multimodal-prompt-learning/premodels/vitae-b-checkpoint-1599-transform-no-average.pth'
        # self.remote_model.load_state_dict(torch.load(model_path)['model'])

        print('loading sar model')
        # self.remote_model = models_mae.__dict__['mae_vit_base_patch16'](norm_pix_loss=False)
        self.remote_model = vit_base_patch16_384()
        # model_path = '/code/multimodal-prompt-learning/premodels/B2_vitb16_mae_ep99.pth'
        model_path = '/code/MAE-ViT-pytorch-main/MAE-ViT-pytorch-main/logs/best_model.pth'
        pretrained_weights = torch.load(model_path)
        encoder_weights = {}
        for key in pretrained_weights.keys():
            if key.startswith('encoder.'):
                new_key = key[len('encoder.'):]
                encoder_weights[new_key] = pretrained_weights[key]
        self.remote_model.load_state_dict(encoder_weights)

        # print('loading medical model')
        # MedSAM_CKPT_PATH = "/code/multimodal-prompt-learning/premodels/medsam_vit_b.pth"
        # self.medical_model = sam_model_registry['vit_b'](checkpoint=MedSAM_CKPT_PATH)


        self.visual_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(512, 512 // 16)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(512 // 16, 512))
        ]))
        # self.visual_net_rq = QuaternionLinearAutograd(512, 512,rotation=True)
        self.visual_net_q = QuaternionLinearAutograd(1024, 512)
        # self.p1 = nn.Parameter(torch.tensor([1.0]))
        # self.p2 = nn.Parameter(torch.tensor([0.0]))
        if cfg.TRAINER.COCOOP.PREC == "fp16":
            self.visual_net.half()
            # self.p1.half()
            # self.p2.half()
        
        # self.lora_constrain_matrix = nn.Parameter(torch.zeros(9, 4, 2)) 
        self.lora_relation = nn.Parameter(torch.zeros(2, 2)) 
        # self.lora_relation_layer = nn.Parameter(torch.zeros(2, 9, 9)) 
        # xavier_uniform_(self.lora_constrain_matrix)
        # self.lora_cm_layer = QuaternionLinearAutograd(4, 4)
        # self.lora_cm_layer = nn.Linear(2, 2)
    def forward(self, image, label=None):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()
        

        # 提取每个 resblock 中的 out_proj.lora_S 层的权重
        image_lora_s_weights = torch.stack([resblock.attn.in_proj_weight_lora_S for resblock in self.image_encoder.transformer.resblocks[:9]], dim=0)
        text_lora_s_weights = torch.stack([resblock.attn.in_proj_weight_lora_S for resblock in self.text_encoder.transformer.resblocks[:9]], dim=0)
        # 拼接 image_lora_s_weights 和 text_lora_s_weights
        lora_s_weights_concat = torch.cat([image_lora_s_weights.unsqueeze(-1), text_lora_s_weights.unsqueeze(-1)], dim=-1)
        lora_shape = lora_s_weights_concat.shape #9 4 2
        # 进行矩阵乘法
        # lora_s_weights_concat = torch.diagonal(lora_s_weights_concat, dim1=1, dim2=2).permute(0,2,1) #(9, 4, 4, 2) - (9, 4, 2)
        # lora_s_weights_concat = torch.matmul(lora_s_weights_concat.permute(2,1,0), self.lora_relation_layer.half()) #(2, 4, 9)
        new_lora_s_weights = torch.matmul(lora_s_weights_concat,  self.lora_relation.half()) #(9, 4, 2)
        #  new_lora_s_weights = torch.matmul(lora_s_weights_concat.permute(2,1,0),  self.lora_relation.half()) #(9, 4, 2)

        # new_lora_s_weights += lora_s_weights_concat.permute(2,1,0)        
        new_lora_s_weights += lora_s_weights_concat
        # new_lora_s_weights = new_lora_s_weights.sum(dim=-1) #(9, 4, 4, 2) - (9, 4, 4)
        # diagonal_values = torch.diagonal(new_lora_s_weights, dim1=-2, dim2=-1) 
        # diagonal_values = self.lora_cm_layer(diagonal_values)
        # new_lora_s_weights = torch.diag_embed(diagonal_values)#(9, 4, 4) 

        # 重塑为 8 * s * s * 2 的形状
        # new_lora_s_weights = new_lora_s_weights.view(9, 4, 4, 2)
        # new_lora_s_weights = new_lora_s_weights.view(-1, 2)
        # new_lora_s_weights = self.lora_cm_layer(new_lora_s_weights)
        # new_lora_s_weights = new_lora_s_weights.view(lora_shape)

        # a = torch.zeros(new_lora_s_weights.shape)
        # new_lora_s_weights = self.lora_cm_layer(torch.cat([new_lora_s_weights,a.cuda()],dim=-1))
        # new_lora_s_weights = new_lora_s_weights[:,:2].view(lora_shape).half()

        # 将结果重新赋值给 out_proj.lora_S
        for i, resblock in enumerate(self.image_encoder.transformer.resblocks[:9]):
            # resblock.attn.in_proj_weight_lora_S += nn.Parameter(new_lora_s_weights[i,:,:,0])
            resblock.attn.in_proj_weight_lora_S = nn.Parameter(new_lora_s_weights[i,:,0])
            # resblock.attn.in_proj_weight_lora_S.data += new_lora_s_weights[i,:,:]
            # resblock.attn.in_proj_weight_lora_S = torch.add(resblock.attn.in_proj_weight_lora_S, nn.Parameter(new_lora_s_weights[i,:,:,0], requires_grad=True))
        
        for i, resblock in enumerate(self.text_encoder.transformer.resblocks[:9]):
            resblock.attn.in_proj_weight_lora_S = nn.Parameter(new_lora_s_weights[i,:,1])
        #     # resblock.attn.in_proj_weight_lora_S.data += new_lora_s_weights[i,:,:,0]


        with torch.no_grad():
            remote_mae_feature = self.remote_model(nn.functional.interpolate(image, (384, 384)))
            # print(remote_mae_feature.shape)
            # remote_mae_feature = remote_mae_feature.mean(dim=1).type(self.dtype)
            # print(image.shape)  4*3*224*224
            # print(remote_mae_feature.shape) 4*196*512
            # print(image_features.shape) 4*196*512  
            
            # medical_feature = self.medical_model.image_encoder(nn.functional.interpolate(image, (1024, 1024)))
            # mb,mc,mh,mw = medical_feature.shape
            # medical_feature = medical_feature.view(mb,mc,mh*mw)
            # remote_mae_feature = medical_feature.mean(dim=-1).type(self.dtype)


        prompts, shared_ctx, deep_compound_prompts_text, deep_compound_prompts_vision = self.prompt_learner(remote_mae_feature.type(self.dtype))
        # text_features = self.text_encoder(prompts, tokenized_prompts, deep_compound_prompts_text)
        image_features = self.image_encoder(image.type(self.dtype), shared_ctx, deep_compound_prompts_vision)
        # index = np.random.choice([1,2,3],1)
        # imager = np.array(image.cpu())
        # imager  = np.rot90(imager,index, axes=(2,3))
        # imager = torch.tensor(np.ascontiguousarray(imager)).cuda()
        # image_features_rotation = self.image_encoder(imager.type(self.dtype), shared_ctx, deep_compound_prompts_vision)

        mean = torch.mean(image_features)
        # var = torch.var(image_features)
        # std = torch.sqrt(var)
        # noise = torch.normal(mean.cpu(), std.cpu(), size=image_features.shape)
        # noise = torch.tensor(np.random.normal(mean.cpu().detach().numpy(), std.cpu().detach().numpy(), size=image_features.shape)).cuda().type(self.dtype)
        noise = torch.randn(image_features.shape).cuda().type(self.dtype)*mean

        # image_features_bias = self.visual_net(remote_mae_feature.type(self.dtype))

        # # # image_features_bias_r = self.visual_net_rq(remote_mae_feature.float())
        # image_features1 = image_features  + image_features_rotation
        # a = torch.zeros(image_features.shape)
        # a = a.type(image_features.dtype)
        # image_features1 = torch.cat((a.cuda(),image_features1),dim=-1)
        # image_features1 = self.visual_net_q(image_features1)
        # image_features = image_features +image_features1

        # a = torch.zeros(image_features.shape)
        # a = a.type(image_features.dtype)
        # image_features1 = torch.cat((a.cuda(),image_features1),dim=-1)
        # image_features1 = self.visual_net_q(image_features1)

        # image_features1 = image_features1[:,512:]
        # image_features = image_features + image_features1

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        # text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logits = []
        for pts_i, imf_i in zip(prompts, image_features.type(self.dtype)):
            text_features = self.text_encoder(pts_i, tokenized_prompts,deep_compound_prompts_text)
            # text_mean = torch.mean(text_features)
            # noise = torch.randn(text_features.shape).cuda().type(self.dtype)*text_mean
            # text_features = text_features + noise


            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            l_i = logit_scale * imf_i @ text_features.t()
            logits.append(l_i)
        logits = torch.stack(logits)
        # logits = logit_scale * image_features @ text_features.t()

        if self.prompt_learner.training:
            return F.cross_entropy(logits, label)

        return logits


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


@TRAINER_REGISTRY.register()
class MaPLe(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.MAPLE.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.MAPLE.PREC == "fp32" or cfg.TRAINER.MAPLE.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        # Lora_clip = LoRA_clip(clip_model,r = 4)
        # clip_model = Lora_clip.lora_clip()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)
        # self.model = CustomCLIP(cfg, classnames, Lora_clip.lora_clip)
        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"

        for name, param in self.model.named_parameters():

            if name_to_update not in name:
                # Make sure that VPT prompts are updated
                if "VPT" in name  or 'lora_' in name:
                # if "VPT" in name or 'visual_net ' in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)
            if "visual_net" in name:
                param.requires_grad_(True)
            # if 'w_a' in name  or 'w_b' in name:
            #     param.requires_grad_(True)

        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("MultiModalPromptLearner", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.MAPLE.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        model = self.model
        optim = self.optim
        scaler = self.scaler

        prec = self.cfg.TRAINER.MAPLE.PREC
        if prec == "amp":
            with autocast():
                loss = model(image, label)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss = model(image, label)
            optim.zero_grad()
            loss.backward()
            optim.step()

        loss_summary = {"loss": loss.item()}

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]

            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
