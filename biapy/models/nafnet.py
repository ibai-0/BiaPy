"""NAFNet model components and GAN discriminator builder utilities.

This module provides:

1. Lightweight building blocks (`SimpleGate`, `LayerNormND`, `NAFBlock`) used
    by NAFNet.
2. The `NAFNet` encoder-decoder model for image restoration / image-to-image
    workflows.
3. A discriminator builder function used by GAN-based training setups in BiaPy.

Compared with traditional restoration backbones, NAFNet simplifies nonlinear
design while preserving strong reconstruction quality via gated depthwise blocks
and residual scaling.

Reference
---------
`Simple Baselines for Image Restoration <https://arxiv.org/abs/2204.04676>`_.

Related Work
------------
The generator design is also inspired by the NAFSSR family:
`NAFSSR: Stereo Image Super-Resolution Using NAFNet
<https://openaccess.thecvf.com/content/CVPR2022W/NTIRE/html/Chu_NAFSSR_Stereo_Image_Super-Resolution_Using_NAFNet_CVPRW_2022_paper.html>.
Implementation adapted for this project from:
https://github.com/GolpedeRemo37/NafNet-in-AI4Life-Microscopy-Supervised-Denoising-Challenge
Citation
--------
Chu, Xiaojie and Chen, Liangyu and Yu, Wenqing. "NAFSSR: Stereo Image
Super-Resolution Using NAFNet." CVPR Workshops, 2022.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from yacs.config import CfgNode as CN
from torchinfo import summary

from biapy.models.patchgan import PatchGANDiscriminator

class SimpleGate(nn.Module):
    """Simple channel-gating operator used in NAF blocks.

    The input tensor is split into two equal channel groups and both parts are
    multiplied element-wise.
    """

    def forward(self, x):
        """Apply channel-wise gating.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape `(N, C, ...)` where `C` must be divisible
            by 2.

        Returns
        -------
        torch.Tensor
            Tensor with shape `(N, C/2, ...)` obtained by multiplying both
            channel chunks element-wise.
        """
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class PixelShuffle3D(nn.Module):
    """3D Pixel Shuffle operation using native PyTorch operations.
    
    Rearranges elements in a tensor of shape (B, C * r^3, D, H, W) 
    to a tensor of shape (B, C, D * r, H * r, W * r), where r is the upscale factor.
    """
    def __init__(self, upscale_factor=2):
        super().__init__()
        self.r = upscale_factor

    def forward(self, x):
        B, C_in, D, H, W = x.size()
        
        # Calculate the actual output channels
        C_out = C_in // (self.r ** 3)
        
        #  1: Unroll the channels into the upscaling blocks
        # Shape becomes: (B, C_out, r, r, r, D, H, W)
        x = x.view(B, C_out, self.r, self.r, self.r, D, H, W)
        
        #  2: Interleave the spatial dimensions
        # Permute from: (B, C_out, r_D, r_H, r_W, D, H, W)
        #           To: (B, C_out, D, r_D, H, r_H, W, r_W)
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4)
        
        #  3: Flatten the interleaved spatial dimensions together
        # Shape becomes: (B, C_out, D * r, H * r, W * r)
        x = x.reshape(B, C_out, D * self.r, H * self.r, W * self.r)
        
        return x
    
class LayerNormND(nn.Module):
    """Layer normalization over channel dimension for N-dimensional features (supports 2D, 3D, etc.).

    This normalization computes mean and variance across channels for each
    spatial position and applies learned affine parameters.
    """

    def __init__(self, channels, eps=1e-6):
        """Initialize layer normalization parameters.

        Parameters
        ----------
        channels : int
            Number of channels in the input tensor.
        eps : float, optional
            Numerical stability constant added to the variance.
        """
        super(LayerNormND, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        """Normalize each spatial position across channels.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape `(N, C, ...)`.

        Returns
        -------
        torch.Tensor
            Normalized tensor with same shape as input.
        """
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + self.eps).sqrt()
        view_shape = [1, -1] + [1] * (x.ndim - 2)
        y = self.weight.view(*view_shape) * y + self.bias.view(*view_shape)
        return y


class NAFBlock(nn.Module):
    """Core NAFNet residual block.

    The block combines:
    1. Layer normalization.
    2. Pointwise + depthwise convolutions.
    3. `SimpleGate` and simplified channel attention.
    4. A lightweight FFN branch.
    5. Two residual scaling parameters (`beta`, `gamma`).
    """

    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., ndim=2):
        """Initialize one NAF block.

        Parameters
        ----------
        c : int
            Number of input/output channels in the block.
        DW_Expand : int, optional
            Expansion ratio for the depthwise branch before gating.
        FFN_Expand : int, optional
            Expansion ratio for the feed-forward branch.
        drop_out_rate : float, optional
            Dropout probability used in both residual branches.
        ndim : int, optional
            Number of spatial dimensions (2 or 3). Default 2.
        """
        super().__init__()
        conv = nn.Conv3d if ndim == 3 else nn.Conv2d
        pool = nn.AdaptiveAvgPool3d if ndim == 3 else nn.AdaptiveAvgPool2d

        dw_channel = c * DW_Expand
        self.conv1 = conv(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = conv(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel, bias=True)
        self.conv3 = conv(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            pool(1),
            conv(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1, groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = conv(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = conv(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNormND(c)
        self.norm2 = LayerNormND(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        shape = [1, c] + [1] * ndim
        self.beta = nn.Parameter(torch.zeros(shape), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros(shape), requires_grad=True)

    def forward(self, inp):
        """Apply the NAF block transformation.

        Parameters
        ----------
        inp : torch.Tensor
            Input feature map with shape `(N, C, ...)`.

        Returns
        -------
        torch.Tensor
            Output feature map with the same shape as `inp`.
        """
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma

class NAFNet(nn.Module):
    """NAFNet encoder-decoder architecture for image restoration.

    The model follows a U-shaped design with:
    1. Intro and ending convolutions.
    2. Multiple encoder stages with downsampling.
    3. Bottleneck NAF blocks.
    4. Decoder stages with upsampling and skip connections.
    """

    def __init__(
        self, 
        img_channel=3, 
        width=16, 
        middle_blk_num=1, 
        enc_blk_nums=[], 
        dec_blk_nums=[],
        drop_out_rate=0.0,   
        dw_expand=2,         
        ffn_expand=2,
        discriminator_arch=None,
        patchgan_base_filters=64,
        patchgan_n_layers=4,     
        ndim=2,
    ):
        """Initialize a NAFNet model.

        Parameters
        ----------
        img_channel : int, optional
            Number of input/output image channels.
        width : int, optional
            Base number of channels.
        middle_blk_num : int, optional
            Number of NAF blocks in the bottleneck.
        enc_blk_nums : list[int], optional
            Number of NAF blocks per encoder stage.
        dec_blk_nums : list[int], optional
            Number of NAF blocks per decoder stage.
        drop_out_rate : float, optional
            Dropout probability used inside blocks.
        dw_expand : int, optional
            Expansion ratio for depthwise branch.
        ffn_expand : int, optional
            Expansion ratio for feed-forward branch.
        discriminator_arch : str or None, optional
            Discriminator architecture name (e.g. ``"patchgan"``). ``None``
            disables the discriminator.
        patchgan_base_filters : int, optional
            Number of filters in the first discriminator block.
        patchgan_n_layers : int, optional
            Number of convolutional downsampling blocks in the PatchGAN
            discriminator.
        ndim : int, optional
            Number of spatial dimensions (2 for 2D, 3 for 3D). Default 2.

        Notes
        -----
        Spatial padding is handled in `check_image_size` to ensure dimensions are
        divisible by the encoder downsampling factor.
        """
        super().__init__()
        self.ndim = ndim
        conv = nn.Conv3d if ndim == 3 else nn.Conv2d
        self.intro = conv(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
        self.ending = conv(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1, bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan, DW_Expand=dw_expand, FFN_Expand=ffn_expand, drop_out_rate=drop_out_rate, ndim=ndim) for _ in range(num)]
                )
            )
            self.downs.append(
                conv(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = nn.Sequential(
            *[NAFBlock(chan, DW_Expand=dw_expand, FFN_Expand=ffn_expand, drop_out_rate=drop_out_rate, ndim=ndim) for _ in range(middle_blk_num)]
        )

        for num in dec_blk_nums:
            if ndim == 3:
                self.ups.append(
                    nn.Sequential(
                        nn.Conv3d(chan, chan * 4, 1, bias=False),  # Conv3d and chan * 4
                        PixelShuffle3D(2)
                    )
                )
            else:
                self.ups.append(
                    nn.Sequential(
                        nn.Conv2d(chan, chan * 2, 1, bias=False),  # Conv2d and chan * 2
                        nn.PixelShuffle(2)
                    )
                )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan, DW_Expand=dw_expand, FFN_Expand=ffn_expand, drop_out_rate=drop_out_rate, ndim=ndim) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

        discriminator = None
        if discriminator_arch == "patchgan":
            discriminator = PatchGANDiscriminator(
                in_channels=img_channel,
                base_filters=patchgan_base_filters,
                n_layers=patchgan_n_layers,
                ndim=ndim,
            )

        self.discriminator = discriminator

    @property
    def param_groups(self):
        """Return parameter groups for separate optimizers.
        When a discriminator is present, returns ``[generator_params, discriminator_params]``
        so that :func:`prepare_optimizer` can assign a separate optimizer and learning
        rate to each group. Without a discriminator, returns a single group.
        """
        if self.discriminator is not None:
            gen_params = [p for n, p in self.named_parameters() if not n.startswith("discriminator.")]
            return [gen_params, list(self.discriminator.parameters())]
        return [list(self.parameters())]

    def forward(self, inp):
        """Run a forward pass through NAFNet.

        Parameters
        ----------
        inp : torch.Tensor
            Input image tensor with shape `(N, C, H, W)` or `(N, C, D, H, W)`.

        Notes
        -----
        The input is internally padded to satisfy the downsampling factor and
        then cropped back to original size at the end of the forward pass.

        Returns
        -------
        torch.Tensor or dict
            Restored image. If the discriminator is active, returns ``{"pred": tensor}``
            so that the output can be enriched with loss information by ``forward_loss``.
        """
        if self.ndim == 3:
            B, C, D, H, W = inp.shape
        else:
            B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp

        if self.ndim == 3:
            pred = x[:, :, :D, :H, :W]
        else:
            pred = x[:, :, :H, :W]

        if self.discriminator is not None:
            return {"pred": pred}
        return pred

    def forward_loss(self, pred, targets, loss_fn, noisy_input=None):
        """Compute GAN losses using the discriminator and the given loss function.

        Parameters
        ----------
        pred : torch.Tensor
            Generator prediction (restored image).
        targets : torch.Tensor
            Ground-truth clean image.
        loss_fn : nn.Module
            Loss module (e.g. ``CycleGanLoss``) providing
            ``forward_generator`` and ``forward_discriminator``.
        noisy_input : torch.Tensor, optional
            Original noisy/degraded input fed to the generator.
            Forwarded to ``forward_generator`` for debug logging.

        Returns
        -------
        tuple or None
            ``(loss_generator, loss_discriminator)`` if the discriminator is
            available, otherwise ``None``.
        """
        if self.discriminator is None:
            return None

        fake_img = torch.clamp(pred, 0, 1)

        for p in self.discriminator.parameters():
            p.requires_grad_(False)
        d_fake_for_g = self.discriminator(fake_img)
        loss_g = loss_fn.forward_generator(fake_img, targets, d_fake_for_g, noisy_input=noisy_input)
        for p in self.discriminator.parameters():
            p.requires_grad_(True)

        # Enable requires_grad on real images so R1 penalty can compute
        # gradients of D(real) w.r.t. the real input (torch.autograd.grad needs this)
        real_imgs = targets.detach().requires_grad_(True)
        d_real = self.discriminator(real_imgs)
        d_fake = self.discriminator(fake_img.detach())
        loss_d = loss_fn.forward_discriminator(d_real, d_fake, real_imgs)

        return (loss_g, loss_d)

    def check_image_size(self, x):
        """Pad image so height/width (and depth if 3D) are divisible by internal stride.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape `(N, C, H, W)` or `(N, C, D, H, W)`.

        Returns
        -------
        torch.Tensor
            Padded tensor compatible with encoder/decoder downsampling.
        """
        if self.ndim == 3:
            _, _, d, h, w = x.size()
            mod_pad_d = (self.padder_size - d % self.padder_size) % self.padder_size
            mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
            mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
            x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h, 0, mod_pad_d))
        else:
            _, _, h, w = x.size()
            mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
            mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
            x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x
