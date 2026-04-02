import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


class Model:
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        self.weight = jnp.zeros((in_channels, out_channels, kernel_size))
        self.bias = jnp.zeros(out_channels) if bias else None
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self._kernel_size = kernel_size

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        x = jnp.transpose(x, (0, 2, 1))  # NWC

        N, W, Cin = x.shape
        Cin_w, Cout, K = self.weight.shape

        effective_kernel_size = self.dilation * (K - 1) + 1
        pad_total = effective_kernel_size - 1 - self.padding
        W_out = (W - 1) * self.stride - 2 * self.padding + effective_kernel_size

        x_flat = x.reshape(N * W, Cin)
        weight = self.weight  # (Cin, Cout, K)

        M = N * W_out
        BM = 128
        BN = 128

        def kernel(x_ref, w_ref, o_ref):
            pid = pl.program_id(axis=0)

            m_start = pid * BM
            n_start = 0

            acc = jnp.zeros((BM, BN), dtype=jnp.float32)

            def body_k(k, acc):
                def body_c(c, acc_inner):
                    # global row indices
                    m_idx = jnp.arange(BM) + m_start

                    n_idx = jnp.arange(BN) + n_start

                    n = m_idx // W_out
                    ow = m_idx % W_out

                    # compute iw
                    num = ow + self.padding - k * self.dilation
                    valid = (num % self.stride) == 0
                    iw = num // self.stride

                    valid = valid & (iw >= 0) & (iw < W)

                    x_index = n * W + iw
                    x_val = jnp.where(valid[:, None], x_ref[x_index, c:c+1], 0.0)

                    w_val = w_ref[c, n_idx, k]

                    acc_inner = acc_inner + x_val * w_val
                    return acc_inner

                acc = jax.lax.fori_loop(0, Cin, body_c, acc)
                return acc

            acc = jax.lax.fori_loop(0, K, body_k, acc)

            o_ref[m_start:m_start+BM, n_start:n_start+BN] = acc

        grid = (M // BM,)

        out = pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((M, Cout), jnp.float32),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[
                    pl.BlockSpec((BM, Cin), lambda i: (i, 0)),
                    pl.BlockSpec((Cin, BN, K), lambda i: (0, 0, 0)),
                ],
                out_specs=pl.BlockSpec((BM, BN), lambda i: (i, 0)),
            ),
        )(x_flat, weight)

        out = out.reshape(N, W_out, Cout)

        if self.bias is not None:
            out = out + self.bias

        out = jnp.transpose(out, (0, 2, 1))
        return out

    @property
    def kernel_size(self):
        return self.weight.shape[2]


def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, (32, 32, 131072))
    return [x]


def get_init_inputs():
    return [32, 64, 5, 1, 0, 3]
