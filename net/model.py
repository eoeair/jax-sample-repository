import flax.nnx as nn
import jax.numpy as jnp


class AxialDW(nn.Module):
    """Axial dilated DW convolution"""

    def __init__(self, dim, mixer_kernel, dilation=1, seed=0):
        super().__init__()
        self.dim = dim
        self.mixer_kernel = mixer_kernel
        self.dilation = dilation
        self.seed = seed
        self.dtype = jnp.float16

    def __call__(self, x):
        h, w = self.mixer_kernel
        dw_h = nn.Conv(
            self.dim,
            self.dim,
            kernel_size=(h, 1),
            padding="same",
            feature_group_count=self.dim,
            kernel_dilation=self.dilation,
            dtype=self.dtype,
            rngs=nn.Rngs(self.seed),
        )
        dw_w = nn.Conv(
            self.dim,
            self.dim,
            kernel_size=(1, w),
            padding="same",
            feature_group_count=self.dim,
            kernel_dilation=self.dilation,
            dtype=self.dtype,
            rngs=nn.Rngs(self.seed),
        )
        return x + dw_h(x) + dw_w(x)


class EncoderBlock(nn.Module):
    """Encoding then downsampling"""

    def __init__(
        self,
        in_c,
        out_c,
        mixer_kernel=(7, 7),
        train: bool = True,
        seed=0,
    ):
        super().__init__()
        self.dw = AxialDW(in_c, mixer_kernel=mixer_kernel, seed=seed)
        self.bn = nn.BatchNorm(in_c, use_running_average=not train, rngs=nn.Rngs(seed))
        self.pw = nn.Conv(in_c, out_c, kernel_size=1, rngs=nn.Rngs(seed))

    def __call__(self, x):
        skip = self.bn(self.dw(x))
        x = nn.gelu(self.pw(skip))
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        return x, skip


class DecoderBlock(nn.Module):
    """Upsampling then decoding"""

    def __init__(
        self,
        in_c,
        out_c,
        mixer_kernel=(7, 7),
        train: bool = True,
        seed=0,
    ):
        super().__init__()
        self.up = nn.ConvTranspose(
            in_c, in_c, kernel_size=(2, 2), strides=(2, 2), rngs=nn.Rngs(seed)
        )
        self.pw = nn.Conv(in_c + out_c, out_c, kernel_size=1, rngs=nn.Rngs(seed))
        self.bn = nn.BatchNorm(out_c, use_running_average=not train, rngs=nn.Rngs(seed))
        self.dw = AxialDW(out_c, mixer_kernel=mixer_kernel, seed=seed)
        self.pw2 = nn.Conv(out_c, out_c, kernel_size=1, rngs=nn.Rngs(seed))

    def __call__(self, x, skip):
        x = self.up(x)
        x = jnp.concatenate([x, skip], axis=3)
        x = self.pw(x)
        x = self.bn(x)
        x = self.dw(x)
        x = self.pw2(x)
        x = nn.gelu(x)
        return x


class BottleNeckBlock(nn.Module):
    """Axial dilated DW convolution"""

    def __init__(self, dim, train: bool = True, seed=0):
        super().__init__()

        gc = dim // 4
        self.pw1 = nn.Conv(dim, gc, kernel_size=1, rngs=nn.Rngs(seed))
        self.dw1 = AxialDW(gc, mixer_kernel=(3, 3), dilation=1, seed=seed)
        self.dw2 = AxialDW(gc, mixer_kernel=(3, 3), dilation=2, seed=seed)
        self.dw3 = AxialDW(gc, mixer_kernel=(3, 3), dilation=3, seed=seed)

        self.bn = nn.BatchNorm(
            4 * gc, use_running_average=not train, rngs=nn.Rngs(seed)
        )
        self.pw2 = nn.Conv(4 * gc, dim, kernel_size=1, rngs=nn.Rngs(seed))

    def __call__(self, x):
        pw1 = self.pw1(x)
        dw1 = self.dw1(pw1)
        dw2 = self.dw2(dw1)
        dw3 = self.dw3(dw2)
        x = jnp.concatenate([pw1, dw1, dw2, dw3], axis=3)
        x = nn.gelu(self.pw2(self.bn(x)))
        return x


class ULite(nn.Module):
    def __init__(self, num_class=4, channel=3, train=True, seed=0):
        self.num_class = num_class
        self.channel = channel

        """Encoder"""
        self.conv_in = nn.Conv(
            self.channel, 16, kernel_size=7, padding="same", rngs=nn.Rngs(seed)
        )
        self.e1 = EncoderBlock(16, 32, train=train, seed=seed)
        self.e2 = EncoderBlock(32, 64, train=train, seed=seed)
        self.e3 = EncoderBlock(64, 128, train=train, seed=seed)

        """Bottle Neck"""
        self.b1 = BottleNeckBlock(128, train=train, seed=seed)

        # """Decoder"""
        self.d3 = DecoderBlock(128, 64, train=train, seed=seed)
        self.d2 = DecoderBlock(64, 32, train=train, seed=seed)
        self.d1 = DecoderBlock(32, 16, train=train, seed=seed)
        self.conv_out = nn.Conv(16, self.num_class, kernel_size=1, rngs=nn.Rngs(seed))

    def __call__(self, x):
        """Encoder"""
        x = self.conv_in(x)
        x, skip1 = self.e1(x)
        x, skip2 = self.e2(x)
        x, skip3 = self.e3(x)

        """BottleNeck"""
        x = self.b1(x)

        # """Decoder"""
        x = self.d3(x, skip3)
        x = self.d2(x, skip2)
        x = self.d1(x, skip1)
        x = self.conv_out(x)
        return x
