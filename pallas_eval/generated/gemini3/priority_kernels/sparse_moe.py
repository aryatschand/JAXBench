def elementwise_kernel(gate_ref, up_ref, out_ref):
    out_ref[...] = jax.nn.silu(gate_ref[...]) * up_ref[...]

def pallas_elementwise(gate, up):
    block = (min(gate.shape[0], 256), min(gate.shape[1], 256), min(gate.shape[2], 256))
    # wait, gate is (B, S, M) -> (1, 2048, 14336)
    block = (1, 256, 1024)
    grid = (1, gate.shape[1] // block[1], gate.shape[2] // block[2])
    return pl.pallas_call(
        elementwise_kernel,
        out_shape=jax.ShapeDtypeStruct(gate.shape, gate.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=[
                pl.BlockSpec(block, lambda i, j, k: (i, j, k)),
                pl.BlockSpec(block, lambda i, j, k: (i, j, k)),
            ],
            out_specs=pl.BlockSpec(block, lambda i, j, k: (i, j, k)),
        )
    )(gate, up)
