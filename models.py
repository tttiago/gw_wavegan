import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


class Transpose1dLayer(nn.Module):
    """1-dimensional transposed convolutional block, which may include dropout"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding=11,
        output_padding=1,
        use_batch_norm=False,
    ):
        """Initialize Transpose1dLayer block.

        Args:
            in_channels (int): Number of input channels.
            output_channels (int): Number of output channels.
            kernel_size (int): Size of the 1D kernel used in the convolution.
            stride (int, optional): Stride of the 1D convolution.
            padding (int, optional): Padding of the 1D convolution. Defaults to 11.
            output_padding (int, optional): Used to find the appropriate output dimension
                                            (see PytTorch docs). Defaults to 1.
            use_batch_norm (bool, optional): Whether to use batch normalization. Defaults to False.
        """

        super().__init__()

        Conv1dTrans = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride, padding, output_padding
        )
        batch_norm = nn.BatchNorm1d(out_channels)

        operation_list = [Conv1dTrans]
        if use_batch_norm:
            operation_list.append(batch_norm)
        self.transpose_ops = nn.Sequential(*operation_list)

    def forward(self, x):
        return self.transpose_ops(x)


class Conv1D(nn.Module):
    """1D convolutional block, which may add batch normalization, phase shuffle and/or dropout."""

    def __init__(
        self,
        in_channels,
        output_channels,
        kernel_size,
        stride=4,
        padding=11,
        alpha=0.2,
        shift_factor=2,
        use_batch_norm=False,
        drop_prob=0,
    ):
        """Initialize Conv1D block.

        Args:
            in_channels (int): Number of input channels.
            output_channels (int): Number of output channels.
            kernel_size (int): Size of the 1D kernel used in the convolution.
            stride (int, optional): Stride of the 1D convolution. Defaults to 4.
            padding (int, optional): Padding of the 1D convolution. Defaults to 11.
            alpha (float, optional): Slope of the negative part of the LeakyRELU. Defaults to 0.2.
            shift_factor (int, optional): Maximum amount of shifting used in PhaseShuffle.
                                          Defaults to 2.
            use_batch_norm (bool, optional): Whether to use batch normalization. Defaults to False.
            drop_prob (int, optional): Dropout probability. Defaults to 0.
        """
        super().__init__()
        self.conv1d = nn.Conv1d(
            in_channels, output_channels, kernel_size, stride=stride, padding=padding
        )
        self.use_batch_norm = use_batch_norm
        self.use_phase_shuffle = shift_factor == 0
        self.use_dropout = drop_prob > 0
        self.alpha = alpha

        self.batch_norm = nn.BatchNorm1d(output_channels)
        self.phase_shuffle = PhaseShuffle(shift_factor)
        self.dropout = nn.Dropout2d(drop_prob)

    def forward(self, x):
        x = self.conv1d(x)
        if self.use_batch_norm:
            x = self.batch_norm(x)
        x = F.leaky_relu(x, negative_slope=self.alpha)
        if self.use_phase_shuffle:
            x = self.phase_shuffle(x)
        if self.use_drop:
            x = self.dropout(x)
        return x


class PhaseShuffle(nn.Module):
    """
    Performs phase shuffling, i.e. shifting feature axis of a 3D tensor
    by a random integer in {-n, n} and performing reflection padding where
    necessary.
    """

    # Copied from https://github.com/jtcramer/wavegan/blob/master/wavegan.py#L8
    # with some modifications.

    def __init__(self, shift_factor):
        """Initialize PhaseShuffle block.

        Args:
            shift_factor (int): The maximum number of samples by which an axis
                                can be shifted in each direction.
        """
        super().__init__()
        self.shift_factor = shift_factor

    def forward(self, x):
        if self.shift_factor == 0:
            return x

        # Uniform in (-shift_factor, +shift_factor)
        k_list = torch.Tensor(x.shape[0]).random_(0, 2 * self.shift_factor + 1) - self.shift_factor
        k_list = k_list.numpy().astype(int)

        # Combine sample indices into lists so that less shuffle operations need to be performed.
        k_map = {}
        for idx, k in enumerate(k_list):
            if k not in k_map:
                k_map[k] = []
            k_map[k].append(idx)

        # Make a copy of x for our output
        x_shuffle = x.clone()

        # Apply shuffle to each sample
        for k, idxs in k_map.items():
            if k > 0:
                x_shuffle[idxs] = F.pad(x[idxs][..., :-k], (k, 0), mode="reflect")
            else:
                x_shuffle[idxs] = F.pad(x[idxs][..., -k:], (0, -k), mode="reflect")

        assert x_shuffle.shape == x.shape, "{}, {}".format(x_shuffle.shape, x.shape)
        return x_shuffle


class WaveGANGenerator(nn.Module):
    """Generator model of the WaveGAN."""

    def __init__(
        self,
        noise_latent_dim=100,
        num_channels=1,
        model_dim=64,
        output_size=16384,
        use_batch_norm=False,
        verbose=False,
    ):
        """Initialize the WaveGAN generator.

        Args:
            noise_latent_dim (int, optional): Dimension of the sampling noise. Defaults to 100.
            num_channels (int, optional): Number of channels. Defaults to 1.
            model_dim (int, optional): Model dimensionality (known as d in the WaveGAN paper).
                                       Defaults to 64.
            output_size (int, optional): Size of the output. Defaults to 16384.
            use_batch_norm (bool, optional): Whether to use batch normalization. Defaults to False.
            verbose (bool, optional): Whether to print tensor shapes. Defaults to False.
        """

        super().__init__()
        assert output_size == 16384, "Only output_size of 16384 is implemented."

        self.latent_dim = noise_latent_dim
        self.d = model_dim
        self.c = num_channels
        self.verbose = verbose
        self.use_batch_norm = use_batch_norm

        self.stride = 4

        self.fc1 = nn.Linear(self.latent_dim, 256 * self.d)
        self.bn1 = nn.BatchNorm1d(num_features=16 * self.d)

        deconv_layers = [
            Transpose1dLayer(
                16 * self.d, 8 * self.d, 25, self.stride, use_batch_norm=use_batch_norm
            ),
            Transpose1dLayer(
                8 * self.d, 4 * self.d, 25, self.stride, use_batch_norm=use_batch_norm
            ),
            Transpose1dLayer(
                4 * self.d, 2 * self.d, 25, self.stride, use_batch_norm=use_batch_norm
            ),
            Transpose1dLayer(
                2 * self.d, 1 * self.d, 25, self.stride, use_batch_norm=use_batch_norm
            ),
            Transpose1dLayer(
                1 * self.d, 1 * self.c, 25, self.stride, use_batch_norm=use_batch_norm
            ),
        ]

        self.deconv_list = nn.ModuleList(deconv_layers)
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)

    def forward(self, x):
        x = self.fc1(x).view(-1, self.dim_mul * self.d, 16)
        if self.use_batch_norm:
            x = self.bn1(x)
        x = F.relu(x)
        if self.verbose:
            print(x.shape)

        for deconv in self.deconv_list[:-1]:
            x = F.relu(deconv(x))
            if self.verbose:
                print(x.shape)
        output = torch.tanh(self.deconv_list[-1](x))
        return output


class WaveGANDiscriminator(nn.Module):
    """Discriminator model of the WaveGAN."""

    def __init__(
        self,
        model_size=64,
        ngpus=1,
        num_channels=1,
        shift_factor=2,
        alpha=0.2,
        verbose=False,
        slice_len=16384,
        use_batch_norm=False,
    ):
        super().__init__()
        assert slice_len in [16384, 32768, 65536]  # used to predict longer utterances

        self.model_size = model_size  # d
        self.ngpus = ngpus
        self.use_batch_norm = use_batch_norm
        self.num_channels = num_channels  # c
        self.shift_factor = shift_factor  # n
        self.alpha = alpha
        self.verbose = verbose

        conv_layers = [
            Conv1D(
                num_channels,
                model_size,
                25,
                stride=4,
                padding=11,
                use_batch_norm=use_batch_norm,
                alpha=alpha,
                shift_factor=shift_factor,
            ),
            Conv1D(
                model_size,
                2 * model_size,
                25,
                stride=4,
                padding=11,
                use_batch_norm=use_batch_norm,
                alpha=alpha,
                shift_factor=shift_factor,
            ),
            Conv1D(
                2 * model_size,
                4 * model_size,
                25,
                stride=4,
                padding=11,
                use_batch_norm=use_batch_norm,
                alpha=alpha,
                shift_factor=shift_factor,
            ),
            Conv1D(
                4 * model_size,
                8 * model_size,
                25,
                stride=4,
                padding=11,
                use_batch_norm=use_batch_norm,
                alpha=alpha,
                shift_factor=shift_factor,
            ),
            Conv1D(
                8 * model_size,
                16 * model_size,
                25,
                stride=4,
                padding=11,
                use_batch_norm=use_batch_norm,
                alpha=alpha,
                shift_factor=0 if slice_len == 16384 else shift_factor,
            ),
        ]
        self.fc_input_size = 256 * model_size
        if slice_len == 32768:
            conv_layers.append(
                Conv1D(
                    16 * model_size,
                    32 * model_size,
                    25,
                    stride=2,
                    padding=11,
                    use_batch_norm=use_batch_norm,
                    alpha=alpha,
                    shift_factor=0,
                )
            )
            self.fc_input_size = 480 * model_size
        elif slice_len == 65536:
            conv_layers.append(
                Conv1D(
                    16 * model_size,
                    32 * model_size,
                    25,
                    stride=4,
                    padding=11,
                    use_batch_norm=use_batch_norm,
                    alpha=alpha,
                    shift_factor=0,
                )
            )
            self.fc_input_size = 512 * model_size

        self.conv_layers = nn.ModuleList(conv_layers)

        self.fc1 = nn.Linear(self.fc_input_size, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)

    def forward(self, x):
        for conv in self.conv_layers:
            x = conv(x)
            if self.verbose:
                print(x.shape)
        x = x.view(-1, self.fc_input_size)
        if self.verbose:
            print(x.shape)

        return self.fc1(x)


if __name__ == "__main__":
    from torch.autograd import Variable

    output_size = 16384
    noise_latent_dim = 100

    G = WaveGANGenerator(verbose=True, use_batch_norm=True, output_size=output_size)
    out = G(Variable(torch.randn(10, noise_latent_dim)))
    print(out.shape)
    assert out.shape == (10, 1, output_size)
    print("==========================")

    D = WaveGANDiscriminator(verbose=True, use_batch_norm=True, slice_len=output_size)
    out2 = D(Variable(torch.randn(10, 1, output_size)))
    print(out2.shape)
    assert out2.shape == (10, 1)
    print("==========================")
