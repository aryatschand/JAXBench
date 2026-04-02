def workload(query, key, value):
    B, S, Hq, D = query.shape
    Hkv = key.shape[2]
    
    q = query.transpose(0, 2, 1, 3)
    k = key.transpose(0, 2, 1, 3)
    v = value.transpose(0, 2, 1, 3)
    
    Bq = 128
    G = Hq // Hkv
    grid = (B, Hq, S // Bq)
    
    out = pl.pallas_call(
        gqa_kernel,
        out_shape=jax.ShapeDtypeStruct(q.shape, q.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=[
                pl.BlockSpec((1, 1, Bq, D), lambda b, hq, sq: (b, hq, sq, 0)),
                pl.BlockSpec((1, 1, S, D), lambda b, hq, sq: (b, hq // G, 0, 0)),
                pl.BlockSpec((1, 1, S, D), lambda b, hq, sq: (b, hq // G, 0, 0)),
            ],
            out_specs=pl.BlockSpec((1, 1, Bq, D), lambda b, hq, sq: (b, hq, sq, 0)),
        )
    )(q, k, v)
    
    return out.transpose(0, 2, 1, 3)
