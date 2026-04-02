import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

class Model:
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, dilation: int = 1, bias: bool = False):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.use_bias = bias
        
        key = jax.random.PRNGKey(0)
        weight_shape = (kernel_size, in_channels, out_channels)
        self.weight = jax.random.normal(key, weight_shape) * 0.02
        
        if bias:
            self.bias = jnp.zeros((out_channels,))
        else:
            self.bias = None
            
    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            if name == 'conv1d.weight':
                value = jnp.transpose(jnp.array(value), (2, 1, 0))
                setattr(self, 'weight', value)
            elif name == 'conv1d.bias':
                value = jnp.array(value)
                setattr(self, 'bias', value)

    def forward(self, x):
        x = jnp.transpose(x, (0, 2, 1))  # NLC

        N, L, Cin = x.shape
        K = self.kernel_size
        stride = self.stride
        dilation = self.dilation
        Cout = self.out_channels

        L_out = (L - dilation * (K - 1) - 1) // stride + 1

        x_flat = jnp.reshape(x, (N * L, Cin))

        Bm = 128
        Bn = 128

        M = N * L_out

        grid_m = M // Bm
        grid_n = Cout // Bn

        def kernel(x_ref, w_ref, b_ref, o_ref):
            pid_m = pl.program_id(axis=0)
            pid_n = pl.program_id(axis=1)

            m_idx = pid_m * Bm + jnp.arange(Bm)
            n_idx = pid_n * Bn + jnp.arange(Bn)

            b = m_idx // L_out
            p = m_idx % L_out

            acc = jnp.zeros((Bm, Bn), dtype=jnp.float32)

            def body(k, acc):
                input_pos = p * stride + k * dilation
                flat_idx = b * L + input_pos

                x_vals = x_ref[flat_idx, :]  # (Bm, Cin)
                w_vals = w_ref[k, :, n_idx]  # (Cin, Bn)

                acc = acc + jnp.dot(x_vals, w_vals)
                return acc

            acc = jax.lax.fori_loop(0, K, body, acc)

            if self.use_bias:
                acc = acc + b_ref[n_idx][None, :]

            o_ref[...] = acc

        out = pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((M, Cout), jnp.float32),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=(grid_m, grid_n),
                in_specs=[
                    pl.BlockSpec((Bm, Cin), lambda i, j: (i, 0)),
                    pl.BlockSpec((K, Cin, Cout), lambda i, j: (0, 0, 0)),
                    pl.BlockSpec((Cout,), lambda i, j: (0,)),
                ],
                out_specs=pl.BlockSpec((Bm, Bn), lambda i, j: (i, j)),
            ),
        )(x_flat, self.weight, self.bias if self.bias is not None else jnp.zeros((Cout,), dtype=jnp.float32))

        out = jnp.reshape(out, (N, L_out, Cout))
        out = jnp.transpose(out, (0, 2, 1))
        return out


batch_size = 16
in_channels = 64  
out_channels = 128
kernel_size = 3
length = 65536
stride = 3
dilation = 4

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (batch_size, in_channels, length))
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, dilation]
