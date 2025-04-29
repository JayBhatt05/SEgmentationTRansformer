import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


########## ViT Based Encoder ##########
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerLayer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, img_size, patch_size, embed_dim, num_layers, num_heads, mlp_ratio=4.,
                 qkv_bias=True, drop_rate=0., attn_drop_rate=0.):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)
        self.embed_dim = embed_dim
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Position embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerLayer(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        # Input shape: [B, C, H, W]
        B = x.shape[0]
        
        # Convert to patches
        x = self.patch_embed(x)  # [B, embed_dim, H/patch_size, W/patch_size]
        x = rearrange(x, 'b c h w -> b (h w) c')  # [B, num_patches, embed_dim]
        
        # Add position embeddings
        x = x + self.pos_embed
        
        # Apply transformer blocks
        features = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            features.append(x)
        
        # Apply final norm
        x = self.norm(x)
        
        return x, features
    
    
########## All Proposed Variants of the Decoder ##########

'''
1. SETR-Naive
'''
class SETRNaiveDecoder(nn.Module):
    def __init__(self, input_dim, num_classes, image_size, patch_size):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.output_size = (image_size[0] // patch_size, image_size[1] // patch_size)
        
        self.decoder = nn.Sequential(
            nn.Conv2d(input_dim, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )
        
    def forward(self, x):
        # x shape: [B, num_patches, embed_dim]
        B, N, C = x.shape
        H, W = self.output_size
        
        # Reshape to 2D feature map
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)  # [B, C, H, W]
        
        # Apply decoder
        x = self.decoder(x)
        
        # Upscale to original image size
        x = F.interpolate(x, size=self.image_size, mode='bilinear', align_corners=False)
        
        return x
    
'''
2. SETR-PUP
'''
class SETPUPDecoder(nn.Module):
    def __init__(self, input_dim, num_classes, image_size, patch_size):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.output_size = (image_size[0] // patch_size, image_size[1] // patch_size)
        
        # Progressive upsampling decoder
        self.decoder = nn.ModuleList([
            # 1/16 -> 1/8
            nn.Sequential(
                nn.Conv2d(input_dim, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            ),
            # 1/8 -> 1/4
            nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            ),
            # 1/4 -> 1/2
            nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            ),
            # 1/2 -> 1
            nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            ),
        ])
        
        # Final prediction head
        self.pred_head = nn.Conv2d(256, num_classes, kernel_size=1)
        
    def forward(self, x):
        # x shape: [B, num_patches, embed_dim]
        B, N, C = x.shape
        H, W = self.output_size
        
        # Reshape to 2D feature map
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)  # [B, C, H, W]
        
        # Apply progressive upsampling
        for decoder_stage in self.decoder:
            x = decoder_stage(x)
        
        # Apply final prediction head
        x = self.pred_head(x)
        
        return x
    
'''
3. SETR-MLA
'''
class SETRMLADecoder(nn.Module):
    def __init__(self, input_dim, num_classes, image_size, patch_size, num_layers=24, num_stages=4):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.output_size = (image_size[0] // patch_size, image_size[1] // patch_size)
        
        # Select layers for feature aggregation
        self.selected_layers = [num_layers // num_stages * i for i in range(1, num_stages + 1)]
        
        # First conv for each selected layer
        self.first_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(input_dim, input_dim // 2, kernel_size=1),
                nn.BatchNorm2d(input_dim // 2),
                nn.ReLU(inplace=True)
            ) for _ in range(num_stages)
        ])
        
        # Second conv with aggregation
        self.second_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(input_dim // 2, input_dim // 2, kernel_size=3, padding=1),
                nn.BatchNorm2d(input_dim // 2),
                nn.ReLU(inplace=True)
            ) for _ in range(num_stages)
        ])
        
        # Third conv
        self.third_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(input_dim // 2, input_dim // 4, kernel_size=3, padding=1),
                nn.BatchNorm2d(input_dim // 4),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
            ) for _ in range(num_stages)
        ])
        
        # Final prediction head
        self.pred_head = nn.Sequential(
            nn.Conv2d(input_dim, num_classes, kernel_size=1),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        )
        
    def forward(self, x, features):
        # x shape: [B, num_patches, embed_dim]
        B, N, C = x.shape
        H, W = self.output_size
        
        # Process selected features
        processed_features = []
        
        for i, layer_idx in enumerate(self.selected_layers):
            feature = features[layer_idx - 1]  # -1 because indices start at 0
            feature = feature.reshape(B, H, W, C).permute(0, 3, 1, 2)  # [B, C, H, W]
            
            # Apply first conv
            feature = self.first_convs[i](feature)
            
            # Top-down aggregation
            if i > 0:
                feature = feature + processed_features[-1]
            
            # Apply second conv
            feature = self.second_convs[i](feature)
            processed_features.append(feature)
            
        # Apply third conv to each processed feature
        upscaled_features = []
        for i, feature in enumerate(processed_features):
            upscaled_features.append(self.third_convs[i](feature))
        
        # Concatenate features
        concat_features = torch.cat(upscaled_features, dim=1)
        
        # Final prediction
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)  # [B, C, H, W]
        x = self.pred_head(x)
        
        return x


########## SETR Model ##########
# SETR Model
class SETR(nn.Module):
    def __init__(self, config):
        super().__init__()
        model_config = config.get_model_config(config.TRANSFORMER_VARIANT)
        
        # Initialize transformer encoder
        self.encoder = TransformerEncoder(
            img_size=config.IMAGE_SIZE,
            patch_size=config.PATCH_SIZE,
            embed_dim=model_config['hidden_size'],
            num_layers=model_config['num_layers'],
            num_heads=model_config['num_heads'],
            mlp_ratio=model_config['mlp_ratio']
        )
        
        # Initialize decoder based on model type
        if config.MODEL_TYPE == 'SETR-Naive':
            self.decoder = SETRNaiveDecoder(
                input_dim=model_config['hidden_size'],
                num_classes=config.NUM_CLASSES,
                image_size=config.IMAGE_SIZE,
                patch_size=config.PATCH_SIZE
            )
        elif config.MODEL_TYPE == 'SETR-PUP':
            self.decoder = SETPUPDecoder(
                input_dim=model_config['hidden_size'],
                num_classes=config.NUM_CLASSES,
                image_size=config.IMAGE_SIZE,
                patch_size=config.PATCH_SIZE
            )
        elif config.MODEL_TYPE == 'SETR-MLA':
            self.decoder = SETRMLADecoder(
                input_dim=model_config['hidden_size'],
                num_classes=config.NUM_CLASSES,
                image_size=config.IMAGE_SIZE,
                patch_size=config.PATCH_SIZE,
                num_layers=model_config['num_layers']
            )
        else:
            raise ValueError(f"Unknown model type: {config.MODEL_TYPE}")
        
        # Initialize auxiliary loss heads if needed
        self.aux_heads = None
        if config.MODEL_TYPE == 'SETR-Naive':
            self.aux_heads = nn.ModuleList([
                SETRNaiveDecoder(
                    input_dim=model_config['hidden_size'],
                    num_classes=config.NUM_CLASSES,
                    image_size=config.IMAGE_SIZE,
                    patch_size=config.PATCH_SIZE
                ) for _ in range(3)  # 3 auxiliary heads as mentioned in the paper
            ])
        elif config.MODEL_TYPE == 'SETR-PUP':
            self.aux_heads = nn.ModuleList([
                SETRNaiveDecoder(
                    input_dim=model_config['hidden_size'],
                    num_classes=config.NUM_CLASSES,
                    image_size=config.IMAGE_SIZE,
                    patch_size=config.PATCH_SIZE
                ) for _ in range(4)  # 4 auxiliary heads for PUP
            ])
        elif config.MODEL_TYPE == 'SETR-MLA':
            self.aux_heads = nn.ModuleList([
                SETRNaiveDecoder(
                    input_dim=model_config['hidden_size'],
                    num_classes=config.NUM_CLASSES,
                    image_size=config.IMAGE_SIZE,
                    patch_size=config.PATCH_SIZE
                ) for _ in range(4)  # 4 auxiliary heads for MLA
            ])
        
    def forward(self, x):
        # Get the encoder output and intermediate features
        encoded, features = self.encoder(x)
        
        # Apply decoder
        if isinstance(self.decoder, SETRMLADecoder):
            output = self.decoder(encoded, features)
        else:
            output = self.decoder(encoded)
        
        # Auxiliary heads output if training
        if self.training and self.aux_heads is not None:
            aux_outputs = []
            if config.MODEL_TYPE == 'SETR-Naive':
                aux_layers = [9, 14, 19]  # Z10, Z15, Z20 (0-indexed)
                 # Ensure aux_layers are within the range of features
                aux_layers = [layer for layer in aux_layers if layer < len(features)]
              
            elif config.MODEL_TYPE == 'SETR-PUP':
                aux_layers = [9, 14, 19, 23]  # Z10, Z15, Z20, Z24 (0-indexed)
                 # Ensure aux_layers are within the range of features
                aux_layers = [layer for layer in aux_layers if layer < len(features)]
            elif config.MODEL_TYPE == 'SETR-MLA':
                aux_layers = [5, 11, 17, 23]  # Z6, Z12, Z18, Z24 (0-indexed)
                 # Ensure aux_layers are within the range of features
                aux_layers = [layer for layer in aux_layers if layer < len(features)]
            
            for i, layer_idx in enumerate(aux_layers):
                aux_outputs.append(self.aux_heads[i](features[layer_idx]))
            
            return output, aux_outputs
        
        return output