import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


class VecInt(nn.Module):
    """
    Integrates a vector field via scaling and squaring.
    """

    def __init__(self, inshape, nsteps=7):
        super().__init__()

        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec

class ConvInsBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, alpha=0.01):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
    
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.instnorm = nn.InstanceNorm3d(out_channels)
        self.lrelu = nn.LeakyReLU(negative_slope=alpha, inplace=True)

    def forward(self, x):

        out = self.conv(x)
        out = self.instnorm(out)
        out = self.lrelu(out)
        return out

class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, alpha=0.1):
        super(UpConvBlock, self).__init__()

        self.upconv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1)

        self.actout = nn.Sequential(
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(alpha)
        )
    def forward(self, x):
        x = self.upconv(x)
        return self.actout(x)

class ResBlock(nn.Module):
    """
    VoxRes module
    """

    def __init__(self, channel, alpha=0.1):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.InstanceNorm3d(channel),
            nn.LeakyReLU(alpha),
            nn.Conv3d(channel, channel, kernel_size=3, padding=1)
        )
        self.actout = nn.Sequential(
            nn.InstanceNorm3d(channel),
            nn.LeakyReLU(alpha),
        )
    def forward(self, x):
        out = self.block(x) + x
        return self.actout(out)


class Encoder(nn.Module):
    """
    Main model
    """

    def __init__(self, in_channel=1, first_out_channel=16):
        super(Encoder, self).__init__()

        c = first_out_channel
        self.conv0 = ConvInsBlock(in_channel, c, 3, 1)

        self.conv1 = nn.Sequential(
            nn.Conv3d(c, 2*c, kernel_size=3, stride=2, padding=1),#80
            ResBlock(2*c)
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(2*c, 4*c, kernel_size=3, stride=2, padding=1),#40
            ResBlock(4*c)
        )

        self.conv3 = nn.Sequential(
            nn.Conv3d(4*c, 8*c, kernel_size=3, stride=2, padding=1),#20
            ResBlock(8*c)
        )

    def forward(self, x):
        out0 = self.conv0(x)  # 1
        out1 = self.conv1(out0)  # 1/2
        out2 = self.conv2(out1)  # 1/4
        out3 = self.conv3(out2)  # 1/8

        return [out0, out1, out2, out3]

class CConv(nn.Module):
    def __init__(self, channel):
        super(CConv, self).__init__()

        c = channel

        self.conv = nn.Sequential(
            ConvInsBlock(c, c, 3, 1),
            ConvInsBlock(c, c, 3, 1)
        )

    def forward(self, float_fm, fixed_fm, d_fm):
        concat_fm = torch.cat([float_fm, fixed_fm, d_fm], dim=1)
        x = self.conv(concat_fm)
        return x

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
     
        self.proj = nn.Linear(dim, dim)

    def forward(self, x1, x2):
        B, N1, C = x1.size()
        N2 = x2.size(1)

        q = self.q(x1).view(B, N1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x2).view(B, N2, self.num_heads, C // self.num_heads).permute(0, 2, 3, 1)
        v = self.v(x2).view(B, N2, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k) * self.scale
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N1, C)
        x = self.proj(x)
      
        return x


class PromptGenerator(nn.Module):
    def __init__(self, dim, num_tasks, c, inshape):
        super().__init__()
        self.dim = dim
        self.num_tasks = num_tasks
        self.c = c
        self.inshape = inshape

        self._init_static_prompts(c, inshape)
        self._init_dynamic_prompts(c, inshape)
        self.prompt_mlp = nn.Sequential(
            nn.Linear((inshape[0]//8)*(inshape[1]//8)*(inshape[2]//8), 16*c),
            nn.ReLU(inplace=True),
            nn.Linear(16*c, 8*c)
        )
        self.conv = nn.Sequential(
            nn.Conv3d(8*c, 8*c, kernel_size=3, padding=1),
            nn.InstanceNorm3d(8*c),
            nn.LeakyReLU(inplace=True)
        )
        self.fusion = nn.Sequential(
            nn.Conv3d(16*c, 8*c, kernel_size=3, padding=1),
            nn.InstanceNorm3d(8*c),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(8*c, 8*c, kernel_size=1, padding=0),
            nn.InstanceNorm3d(8*c),
            nn.LeakyReLU(inplace=True)
        )
        self.x_proj = nn.Conv3d(16*c, 8*c, kernel_size=1, padding=0)
        self.cross_attention = CrossAttention(dim=8*c)

        
    def _init_static_prompts(self, c, inshape):
        self.static_prompt = nn.Parameter(torch.zeros(self.num_tasks, 8*c, inshape[0]//8, inshape[1]//8, inshape[2]//8))
        nn.init.kaiming_normal_(self.static_prompt, a=1e-2, mode='fan_in', nonlinearity='leaky_relu')
    
    def _init_dynamic_prompts(self, c, inshape):
        self.prompt_template = nn.Parameter(torch.zeros(1, 8*c, inshape[0]//8, inshape[1]//8, inshape[2]//8))
        nn.init.kaiming_normal_(self.prompt_template, a=1e-2, mode='fan_in', nonlinearity='leaky_relu')
     
        self.dynamic_fusion = nn.Sequential(
            nn.Conv3d(16*c, 8*c, kernel_size=3, padding=1),
            nn.InstanceNorm3d(8*c),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(8*c, 8*c, kernel_size=3, padding=1),
            nn.InstanceNorm3d(8*c),
            nn.LeakyReLU(inplace=True),
        )
    
    

        
    def forward(self, x, task_id=None, is_training=False):
        c=16
        B, C, H, W, D = x.shape
                
        dynamic_prompt, dynamic_prompt_flat = self._generate_dynamic_prompts(x, B)
        
        similarity_distribution = self._compute_similarity_distribution(dynamic_prompt_flat, self.static_prompt)
        
       
        weighted_static_prompt = self._compute_weighted_static_prompt(self.static_prompt, similarity_distribution, B)
        weighted_static_prompt = self.conv(weighted_static_prompt)
        if is_training:
            orthogonal_loss = self._compute_orthogonal_loss(self.static_prompt)
            classification_loss = self._compute_classification_loss(weighted_static_prompt, self.static_prompt, task_id, B)
        else:
            orthogonal_loss = torch.tensor(0.0, device=x.device)
            classification_loss = torch.tensor(0.0, device=x.device)
        
        fused_features = torch.cat([dynamic_prompt, weighted_static_prompt], dim=1)
        prompt = self.fusion(fused_features)
        
       
        output_pooled = torch.mean(prompt, dim=(2, 3, 4), keepdim=True)  
        batch_size, channels, h, w, d = prompt.shape
        spatial_dims = (h, w, d)
        
        output_pooled_flat = output_pooled.view(batch_size, channels, 1)  
        
      
        kv_features = output_pooled_flat.permute(0, 2, 1)  
       
        x_flat = x.view(batch_size, 8*c, -1).permute(0, 2, 1) 
        
   
        enhanced_feat_flat = self.cross_attention(x_flat, kv_features) 
        
       
        enhanced_feat = enhanced_feat_flat.permute(0, 2, 1).view(batch_size, channels, *spatial_dims) 
        
        return enhanced_feat, orthogonal_loss, classification_loss
    
    def _generate_dynamic_prompts(self, x, B):    
        dynamic_prompt = self.dynamic_fusion(torch.cat([x, self.prompt_template], dim=1))
        dynamic_prompt_flat = dynamic_prompt.view(B, -1)
        
        return dynamic_prompt, dynamic_prompt_flat
    
    def _compute_similarity_distribution(self, dynamic_prompt_flat, static_prompt_out):
        similarities = []
        for i in range(self.num_tasks):
            static_prompt_flat = static_prompt_out[i:i+1].expand(dynamic_prompt_flat.shape[0], -1, -1, -1, -1).view(dynamic_prompt_flat.shape[0], -1)
            similarity = F.cosine_similarity(dynamic_prompt_flat, static_prompt_flat, dim=1)
            similarities.append(similarity)
        
        similarities = torch.stack(similarities, dim=1)
        temperature = 0.5
        similarities = similarities / temperature
        return F.softmax(similarities, dim=1)
    
    def _compute_weighted_static_prompt(self, static_prompt_out, similarity_distribution, B):
        weighted_static_prompt = None
        for i in range(self.num_tasks):
            weight = similarity_distribution[:, i:i+1].view(B, 1, 1, 1, 1)
            current_prompt = static_prompt_out[i:i+1]
            if weighted_static_prompt is None:
                weighted_static_prompt = weight * current_prompt
            else:
                weighted_static_prompt += weight * current_prompt
        return weighted_static_prompt
    
    def _compute_classification_loss(self, weighted_prompt, static_prompt_out, task_id, B):
        weighted_prompt_flat = F.normalize(weighted_prompt.view(B, -1), dim=1)
        target_prompt_flat = F.normalize(static_prompt_out[task_id:task_id+1].view(B, -1), dim=1)
        cosine_sim = torch.sum(weighted_prompt_flat * target_prompt_flat, dim=1)
        return torch.mean((1 - cosine_sim) ** 2)

    def _compute_orthogonal_loss(self, static_prompt_out):
        B_matrix = F.adaptive_avg_pool3d(static_prompt_out, (4, 4, 4))
        B_matrix = static_prompt_out.view(self.num_tasks, -1)
        B_matrix = F.normalize(B_matrix, dim=1)
        
        correlation_matrix = torch.mm(B_matrix, B_matrix.t())
        diagonal_matrix = torch.eye(self.num_tasks, device=correlation_matrix.device)
        return torch.norm(correlation_matrix - diagonal_matrix, p='fro')

class PromptReg(nn.Module):
    def __init__(self, inshape=(160,160,160), in_channel=1, channels=16, task_total_number=3):
        super(PromptReg, self).__init__()
        c = channels
        # Task prompts
        self.prompt_generator = PromptGenerator(16*c, task_total_number,c,inshape)
      
        self.prompt_fusion_layer = nn.Sequential(
            ConvInsBlock(16*c, 8*c, 3, 1), 
            ConvInsBlock(8*c, 8*c, 3, 1)
        )
        
        
        self.channels = channels
        self.step = 7
        self.inshape = inshape

        self.encoder_moving = Encoder(in_channel=in_channel, first_out_channel=c)
        self.encoder_fixed = Encoder(in_channel=in_channel, first_out_channel=c)

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample_trilin = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)#nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)


        self.warp = nn.ModuleList()
        self.diff = nn.ModuleList()
        for i in range(4):
            self.warp.append(SpatialTransformer([s // 2**i for s in inshape]))
            self.diff.append(VecInt([s // 2**i for s in inshape]))
            
        # bottleNeck
        self.cconv_4 = nn.Sequential(
            ConvInsBlock(16 * c, 8 * c, 3, 1),
            ConvInsBlock(8 * c, 8 * c, 3, 1)
        )
        # warp scale 2
        self.defconv4 = nn.Conv3d(8*c, 3, 3, 1, 1)
        self.defconv4.weight = nn.Parameter(Normal(0, 1e-5).sample(self.defconv4.weight.shape))
        self.defconv4.bias = nn.Parameter(torch.zeros(self.defconv4.bias.shape))
        self.dconv4 = nn.Sequential(
            ConvInsBlock(3*8*c, 8*c),
            ConvInsBlock(8*c, 8*c)
        )
        
        self.upconv3 = UpConvBlock(8*c, 4*c, 4, 2)
        self.cconv_3 = CConv(3*4*c)

        # warp scale 1
        self.defconv3 = nn.Conv3d(3*4*c, 3, 3, 1, 1)
        self.defconv3.weight = nn.Parameter(Normal(0, 1e-5).sample(self.defconv3.weight.shape))
        self.defconv3.bias = nn.Parameter(torch.zeros(self.defconv3.bias.shape))
        self.dconv3 = ConvInsBlock(3 * 4 * c, 4 * c)
        
        self.upconv2 = UpConvBlock(3*4*c, 2*c, 4, 2)
        self.cconv_2 = CConv(3*2*c)

        # warp scale 0
        self.defconv2 = nn.Conv3d(3*2*c, 3, 3, 1, 1)
        self.defconv2.weight = nn.Parameter(Normal(0, 1e-5).sample(self.defconv2.weight.shape))
        self.defconv2.bias = nn.Parameter(torch.zeros(self.defconv2.bias.shape))
        self.dconv2 = ConvInsBlock(3 * 2 * c, 2 * c)
        
        self.upconv1 = UpConvBlock(3*2*c, c, 4, 2)
        self.cconv_1 = CConv(3*c)

        # decoder layers
        self.defconv1 = nn.Conv3d(3*c, 3, 3, 1, 1)
        self.defconv1.weight = nn.Parameter(Normal(0, 1e-5).sample(self.defconv1.weight.shape))
        self.defconv1.bias = nn.Parameter(torch.zeros(self.defconv1.bias.shape))

 
   
    def forward(self, moving, fixed, task_id,is_training=False):

        # encode stage
        M1, M2, M3, M4 = self.encoder_moving(moving)
        F1, F2, F3, F4 = self.encoder_fixed(fixed)
        # c=16, 2c, 4c, 8c  # 160, 80, 40, 20

        # first dec layer
        C4 = torch.cat([F4, M4], dim=1)
        C4 = self.cconv_4(C4)  # (1,128,20,24,20)

        task_prompt, similarity_loss, orthogonal_loss = self.prompt_generator(C4, task_id, is_training)

        C4 = torch.cat([C4, task_prompt], dim=1)  # [bs, 16c, H/8, W/8, D/8]
        C4 = self.prompt_fusion_layer(C4)

        flow = self.defconv4(C4)  # (1,3,20,24,20)
        flow = self.diff[3](flow)
        warped = self.warp[3](M4, flow)
        C4 = self.dconv4(torch.cat([F4, warped, C4], dim=1))
        v = self.defconv4(C4)  # (1,3,20,24,20)
        w = self.diff[3](v)


        D3 = self.upconv3(C4)   # (1, 64, 40, 48, 40)
        flow = self.upsample_trilin(2*(self.warp[3](flow, w)+w))
        warped = self.warp[2](M3, flow)  # (1, 64, 40, 48, 40)
        C3 = self.cconv_3(F3, warped, D3)  #  (1, 3 * 64, 40, 48, 40)
        v = self.defconv3(C3)
        w = self.diff[2](v)
        flow = self.warp[2](flow, w)+w
        warped = self.warp[2](M3, flow)  # (1, 64, 40, 48, 40)
        D3 = self.dconv3(C3)
        C3 = self.cconv_3(F3, warped, D3)  #  (1, 3 * 64, 40, 48, 40)
        v = self.defconv3(C3)
        w = self.diff[2](v)

        D2 = self.upconv2(C3)
        flow = self.upsample_trilin(2*(self.warp[2](flow, w)+w))
        warped = self.warp[1](M2, flow)
        C2 = self.cconv_2(F2, warped, D2)
        v = self.defconv2(C2)  # (1,3,80,96,80)
        w = self.diff[1](v)
        flow = self.warp[1](flow, w)+w
        warped = self.warp[1](M2, flow)
        D2 = self.dconv2(C2)
        C2 = self.cconv_2(F2, warped, D2)
        v = self.defconv2(C2)  # (1,3,80,96,80)
        w = self.diff[1](v)
        flow = self.warp[1](flow, w)+w
        warped = self.warp[1](M2, flow)
        D2 = self.dconv2(C2)
        C2 = self.cconv_2(F2, warped, D2)
        v = self.defconv2(C2)  # (1,3,80,96,80)
        w = self.diff[1](v)

        D1 = self.upconv1(C2)  # (1,16,160,196,160)
        flow = self.upsample_trilin(2*(self.warp[1](flow, w)+w))  # （1,3,160,196,160)
        warped = self.warp[0](M1, flow)  # （1,16,160,196,160)
        C1 = self.cconv_1(F1, warped, D1)  # （1,48,160,196,160)
        v = self.defconv1(C1)
        w = self.diff[0](v)
        flow = self.warp[0](flow, w)+w  # （1,3,160,196,160)

        y_moved = self.warp[0](moving, flow)

        
        return y_moved, flow, similarity_loss, orthogonal_loss
