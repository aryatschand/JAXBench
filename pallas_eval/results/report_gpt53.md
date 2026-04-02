# Pallas Evaluation Report


## GPT-5.3

### Jaxkernelbench

| Metric | Count |
|--------|-------|
| Total | 200 |
| Ran successfully | 2 |
| Errors (crash/timeout) | 198 |
| Correct output | 2 |
| Faster than baseline | 0 |
| Correct AND faster | 0 |

**level1**: 2 ran, 2 correct, 0 faster

Speedup stats (over ran): median=0.33x, mean=0.33x, max=0.42x, min=0.25x

| Workload | Correct | Orig (ms) | Gen (ms) | Speedup |
|----------|---------|-----------|----------|---------|
| 10_3D_tensor_matrix_multiplication | yes | 0.24 | 0.57 | 0.42x |
| 88_MinGPTNewGelu | yes | 0.49 | 2.00 | 0.25x |
| 100_HingeLoss | ERROR | — | — | pallas_call() missing 1 required positio |
| 11_4D_tensor_matrix_multiplication | ERROR | — | — | Out of bound slice: start=128, dim=128. |
| 12_Matmul_with_diagonal_matrices_ | ERROR | — | — | Model.forward.<locals>.<lambda>() takes  |
| 13_Matmul_for_symmetric_matrices | ERROR | — | — | Invalid shape for `swap`. Ref shape: (12 |
| 14_Matmul_for_upper_triangular_matrices | ERROR | — | — | load() takes 2 positional arguments but  |
| 15_Matmul_for_lower_triangular_matrices | ERROR | — | — | pallas_call() got an unexpected keyword  |
| 16_Matmul_with_transposed_A | ERROR | — | — | load() takes 2 positional arguments but  |
| 17_Matmul_with_transposed_B | ERROR | — | — | Pytree for `in_specs` and `inputs` do no |
| 18_Matmul_with_transposed_both | ERROR | — | — | pallas_call() got an unexpected keyword  |
| 19_ReLU | ERROR | — | — | Pytree for `in_specs` and `inputs` do no |
| 1_Square_matrix_multiplication_ | ERROR | — | — | pallas_call() got multiple values for ar |
| 20_LeakyReLU | ERROR | — | — | Pytree for `in_specs` and `inputs` do no |
| 21_Sigmoid | ERROR | — | — | pallas_call() missing 1 required positio |
| 22_Tanh | ERROR | — | — |  |
| 23_Softmax | ERROR | — | — | module 'jax.experimental.pallas' has no  |
| 24_LogSoftmax | ERROR | — | — | load() takes 2 positional arguments but  |
| 25_Swish | ERROR | — | — | Pytree for `in_specs` and `inputs` do no |
| 26_GELU_ | ERROR | — | — | Cannot do int indexing on TPU |
| 27_SELU_ | ERROR | — | — | load() takes 2 positional arguments but  |
| 28_HardSigmoid | ERROR | — | — | module 'jax.experimental.pallas' has no  |
| 29_Softplus | ERROR | — | — | pallas_call() missing 1 required positio |
| 2_Standard_matrix_multiplication_ | ERROR | — | — | pallas_call() got an unexpected keyword  |
| 30_Softsign | ERROR | — | — | Attempted boolean conversion of traced a |
| 31_ELU | ERROR | — | — | pallas_call() got an unexpected keyword  |
| 32_HardTanh | ERROR | — | — | INTERNAL: Core halted unexpectedly: INTE |
| 33_BatchNorm | ERROR | — | — | The Pallas TPU lowering currently requir |
| 34_InstanceNorm | ERROR | — | — | If `grid_spec` is specified, then `in_sp |
| 35_GroupNorm_ | ERROR | — | — | `broadcast_to` is a Triton-specific prim |
| 36_RMSNorm_ | ERROR | — | — | Block shape for args[0] (= (Blocked(bloc |
| 37_FrobeniusNorm_ | ERROR | — | — | module 'jax.experimental.pallas' has no  |
| 38_L1Norm_ | ERROR | — | — | The Pallas TPU lowering currently requir |
| 39_L2Norm_ | ERROR | — | — | pallas_call() missing 1 required positio |
| 3_Batched_matrix_multiplication | ERROR | — | — | Attempted boolean conversion of traced a |
| 40_LayerNorm | ERROR | — | — | The Pallas TPU lowering currently requir |
| 41_Max_Pooling_1D | ERROR | — | — | `broadcast_to` is a Triton-specific prim |
| 42_Max_Pooling_2D | ERROR | — | — | Cannot store scalars to VMEM |
| 43_Max_Pooling_3D | ERROR | — | — | `broadcast_to` is a Triton-specific prim |
| 44_Average_Pooling_1D | ERROR | — | — | `broadcast_to` is a Triton-specific prim |
| 45_Average_Pooling_2D | ERROR | — | — | Cannot store scalars to VMEM |
| 46_Average_Pooling_3D | ERROR | — | — | Pytree for `in_specs` and `inputs` do no |
| 47_Sum_reduction_over_a_dimension | ERROR | — | — | Cannot store scalars to VMEM |
| 48_Mean_reduction_over_a_dimension | ERROR | — | — | `broadcast_to` is a Triton-specific prim |
| 49_Max_reduction_over_a_dimension | ERROR | — | — | Attempted boolean conversion of traced a |
| 4_Matrix_vector_multiplication_ | ERROR | — | — | pallas_call() missing 1 required positio |
| 50_conv_standard_2D__square_input__square_kernel | ERROR | — | — | transpose requires ndarray or scalar arg |
| 51_Argmax_over_a_dimension | ERROR | — | — | Cannot store scalars to VMEM |
| 52_Argmin_over_a_dimension | ERROR | — | — | `broadcast_to` is a Triton-specific prim |
| 53_Min_reduction_over_a_dimension | ERROR | — | — | `broadcast_to` is a Triton-specific prim |
| 54_conv_standard_3D__square_input__square_kernel | ERROR | — | — | Pallas encountered an internal verificat |
| 55_conv_standard_2D__asymmetric_input__square_kernel | ERROR | — | — | 'NoneType' object has no attribute 'shap |
| 56_conv_standard_2D__asymmetric_input__asymmetric_kernel | ERROR | — | — | pallas_call() got an unexpected keyword  |
| 57_conv_transposed_2D__square_input__square_kernel | ERROR | — | — | pallas_call() missing 1 required positio |
| 58_conv_transposed_3D__asymmetric_input__asymmetric_kernel | ERROR | — | — | pallas_call() got an unexpected keyword  |
| 59_conv_standard_3D__asymmetric_input__square_kernel | ERROR | — | — | 'function' object is not iterable |
| 5_Matrix_scalar_multiplication | ERROR | — | — | Pytree for `in_specs` and `inputs` do no |
| 60_conv_standard_3D__square_input__asymmetric_kernel | ERROR | — | — | If `grid_spec` is specified, then `in_sp |
| 61_conv_transposed_3D__square_input__square_kernel | ERROR | — | — | Model.forward.<locals>.<lambda>() takes  |
| 62_conv_standard_2D__square_input__asymmetric_kernel | ERROR | — | — | pallas_call() got an unexpected keyword  |
| 63_conv_standard_2D__square_input__square_kernel | ERROR | — | — | pallas_call() got an unexpected keyword  |
| 64_conv_transposed_1D | ERROR | — | — | Invalid out_shape type: <class 'int'> |
| 65_conv_transposed_2D__square_input__asymmetric_kernel | ERROR | — | — | Attempted boolean conversion of traced a |
| 66_conv_standard_3D__asymmetric_input__asymmetric_kernel | ERROR | — | — | Attempted boolean conversion of traced a |
| 67_conv_standard_1D | ERROR | — | — | Attempted boolean conversion of traced a |
| 68_conv_transposed_3D__square_input__asymmetric_kernel | ERROR | — | — | pallas_call() missing 1 required positio |
| 69_conv_transposed_2D__asymmetric_input__asymmetric_kernel | ERROR | — | — | pallas_call() missing 1 required positio |
| 6_Matmul_with_large_K_dimension_ | ERROR | — | — | Pytree for `in_specs` and `inputs` do no |
| 70_conv_transposed_3D__asymmetric_input__square_kernel | ERROR | — | — | pallas_call() got an unexpected keyword  |
| 71_conv_transposed_2D__asymmetric_input__square_kernel | ERROR | — | — | Attempted boolean conversion of traced a |
| 72_conv_transposed_3D_asymmetric_input_asymmetric_kernel___strided_padded_grouped_ | ERROR | — | — | Pytree for `in_specs` and `inputs` do no |
| 73_conv_transposed_3D_asymmetric_input_square_kernel__strided_padded__grouped | ERROR | — | — | Attempted boolean conversion of traced a |
| 74_conv_transposed_1D_dilated | ERROR | — | — | Cannot store scalars to VMEM |
| 75_conv_transposed_2D_asymmetric_input_asymmetric_kernel_strided__grouped____padded____dilated__ | ERROR | — | — | Attempted boolean conversion of traced a |
| 76_conv_standard_1D_dilated_strided__ | ERROR | — | — | 'tuple' object is not callable |
| 77_conv_transposed_3D_square_input_square_kernel___padded____dilated____strided__ | ERROR | — | — | 'function' object is not iterable |
| 78_conv_transposed_2D_asymmetric_input_asymmetric_kernel___padded__ | ERROR | — | — | conv_general_dilated lhs feature dimensi |
| 79_conv_transposed_1D_asymmetric_input_square_kernel___padded____strided____dilated__ | ERROR | — | — | Attempted boolean conversion of traced a |
| 7_Matmul_with_small_K_dimension_ | ERROR | — | — | matmul input operand 0 must have ndim at |
| 80_conv_standard_2D_square_input_asymmetric_kernel___dilated____padded__ | ERROR | — | — | pallas_call() got an unexpected keyword  |
| 81_conv_transposed_2D_asymmetric_input_square_kernel___dilated____padded____strided__ | ERROR | — | — | Attempted boolean conversion of traced a |
| 82_conv_depthwise_2D_square_input_square_kernel | ERROR | — | — | Attempted boolean conversion of traced a |
| 83_conv_depthwise_2D_square_input_asymmetric_kernel | ERROR | — | — | Attempted boolean conversion of traced a |
| 84_conv_depthwise_2D_asymmetric_input_square_kernel | ERROR | — | — | Cannot store scalars to VMEM |
| 85_conv_depthwise_2D_asymmetric_input_asymmetric_kernel | ERROR | — | — | The Pallas TPU lowering currently requir |
| 86_conv_depthwise_separable_2D | ERROR | — | — | Model.forward.<locals>.kernel() missing  |
| 87_conv_pointwise_2D | ERROR | — | — | module 'jax.experimental.pallas.tpu' has |
| 89_cumsum | ERROR | — | — | The Pallas TPU lowering currently requir |
| 8_Matmul_with_irregular_shapes_ | ERROR | — | — | Pytree for `in_specs` and `inputs` do no |
| 90_cumprod | ERROR | — | — | 'function' object is not iterable |
| 91_cumsum_reverse | ERROR | — | — | Pytree for `in_specs` and `inputs` do no |
| 92_cumsum_exclusive | ERROR | — | — | `broadcast_to` is a Triton-specific prim |
| 93_masked_cumsum | ERROR | — | — | pallas_call() missing 1 required positio |
| 94_MSELoss | ERROR | — | — | Model.forward.<locals>.<lambda>() takes  |
| 95_CrossEntropyLoss | ERROR | — | — | Model.forward.<locals>.<lambda>() missin |
| 96_HuberLoss | ERROR | — | — | pallas_call() got an unexpected keyword  |
| 97_ScaledDotProductAttention | ERROR | — | — | Invalid out_shape type: <class 'int'> |
| 98_KLDivLoss | ERROR | — | — | pallas_call() missing 1 required positio |
| 99_TripletMarginLoss | ERROR | — | — | pallas_call() missing 1 required positio |
| 9_Tall_skinny_matrix_multiplication_ | ERROR | — | — | module 'jax.experimental.pallas.tpu' has |
| 100_ConvTranspose3d_Clamp_Min_Divide | ERROR | — | — | pallas_call() got an unexpected keyword  |
| 10_ConvTranspose2d_MaxPool_Hardtanh_Mean_Tanh | ERROR | — | — | pallas_call() got an unexpected keyword  |
| 11_ConvTranspose2d_BatchNorm_Tanh_MaxPool_GroupNorm | ERROR | — | — | Invalid out_shape type: <class 'int'> |
| 12_Gemm_Multiply_LeakyReLU | ERROR | — | — | _pallas_call.<locals>.wrapped() got an u |
| 13_ConvTranspose3d_Mean_Add_Softmax_Tanh_Scaling | ERROR | — | — | The Pallas TPU lowering currently requir |
| 14_Gemm_Divide_Sum_Scaling | ERROR | — | — | pallas_call() missing 1 required positio |
| 15_ConvTranspose3d_BatchNorm_Subtract | ERROR | — | — | Cannot store scalars to VMEM |
| 16_ConvTranspose2d_Mish_Add_Hardtanh_Scaling | ERROR | — | — | GridSpec.__init__() got an unexpected ke |
| 17_Conv2d_InstanceNorm_Divide | ERROR | — | — | Cannot store scalars to VMEM |
| 18_Matmul_Sum_Max_AvgPool_LogSumExp_LogSumExp | ERROR | — | — | Model.forward.<locals>.<lambda>() missin |
| 19_ConvTranspose2d_GELU_GroupNorm | ERROR | — | — | Cannot store scalars to VMEM |
| 1_Conv2D_ReLU_BiasAdd | ERROR | — | — | pallas_call() got an unexpected keyword  |
| 20_ConvTranspose3d_Sum_ResidualAdd_Multiply_ResidualAdd | ERROR | — | — | If `grid_spec` is specified, then `in_sp |
| 21_Conv2d_Add_Scale_Sigmoid_GroupNorm | ERROR | — | — | Pytree for `in_specs` and `inputs` do no |
| 22_Matmul_Scale_ResidualAdd_Clamp_LogSumExp_Mish | ERROR | — | — | 'Model' object has no attribute 'matmul_ |
| 23_Conv3d_GroupNorm_Mean | ERROR | — | — | pallas_call() missing 1 required positio |
| 24_Conv3d_Min_Softmax | ERROR | — | — | lax.conv_general_dilated requires argume |
| 25_Conv2d_Min_Tanh_Tanh | ERROR | — | — | pallas_call() got an unexpected keyword  |
| 26_ConvTranspose3d_Add_HardSwish | ERROR | — | — | 'function' object is not iterable |
| 27_Conv3d_HardSwish_GroupNorm_Mean | ERROR | — | — | pallas_call() missing 1 required positio |
| 28_BMM_InstanceNorm_Sum_ResidualAdd_Multiply | ERROR | — | — | Model.forward.<locals>.<lambda>() takes  |
| 29_Matmul_Mish_Mish | ERROR | — | — | pallas_call() got an unexpected keyword  |
| 2_ConvTranspose2d_BiasAdd_Clamp_Scaling_Clamp_Divide | ERROR | — | — | pallas_call() missing 1 required positio |
| 30_Gemm_GroupNorm_Hardtanh | ERROR | — | — | pallas_call() missing 1 required positio |
| 31_Conv2d_Min_Add_Multiply | ERROR | — | — | pallas_call() got an unexpected keyword  |
| 32_Conv2d_Scaling_Min | ERROR | — | — | pallas_call() got an unexpected keyword  |
| 33_Gemm_Scale_BatchNorm | ERROR | — | — | pallas_call() got an unexpected keyword  |
| 34_ConvTranspose3d_LayerNorm_GELU_Scaling | ERROR | — | — | pallas_call() missing 1 required positio |
| 35_Conv2d_Subtract_HardSwish_MaxPool_Mish | ERROR | — | — | pallas_call() missing 1 required positio |
| 36_ConvTranspose2d_Min_Sum_GELU_Add | ERROR | — | — | pallas_call() missing 1 required positio |
| 37_Matmul_Swish_Sum_GroupNorm | ERROR | — | — | pallas_call() got an unexpected keyword  |
| 38_ConvTranspose3d_AvgPool_Clamp_Softmax_Multiply | ERROR | — | — | pallas_call() missing 1 required positio |
| 39_Gemm_Scale_BatchNorm | ERROR | — | — | load() takes 2 positional arguments but  |
| 3_ConvTranspose3d_Sum_LayerNorm_AvgPool_GELU | ERROR | — | — | Invalid out_shape type: <class 'int'> |
| 40_Matmul_Scaling_ResidualAdd | ERROR | — | — | Attempted boolean conversion of traced a |
| 41_Gemm_BatchNorm_GELU_ReLU | ERROR | — | — | load() takes 2 positional arguments but  |
| 42_ConvTranspose2d_GlobalAvgPool_BiasAdd_LogSumExp_Sum_Multiply | ERROR | — | — | module 'jax.experimental.pallas' has no  |
| 43_Conv3d_Max_LogSumExp_ReLU | ERROR | — | — | pallas_call() missing 1 required positio |
| 44_ConvTranspose2d_Multiply_GlobalAvgPool_GlobalAvgPool_Mean | ERROR | — | — | Model.forward.<locals>.<lambda>() takes  |
| 45_Gemm_Sigmoid_LogSumExp | ERROR | — | — | 'F32Type' object has no attribute 'eleme |
| 46_Conv2d_Subtract_Tanh_Subtract_AvgPool | ERROR | — | — | Cannot store scalars to VMEM |
| 47_Conv3d_Mish_Tanh | ERROR | — | — | Invalid out_shape type: <class 'int'> |
| 48_Conv3d_Scaling_Tanh_Multiply_Sigmoid | ERROR | — | — | Cannot store scalars to VMEM |
| 49_ConvTranspose3d_Softmax_Sigmoid | ERROR | — | — | module 'jax.experimental.pallas' has no  |
| 4_Conv2d_Mish_Mish | ERROR | — | — | pallas_call() missing 1 required positio |
| 50_ConvTranspose3d_Scaling_AvgPool_BiasAdd_Scaling | ERROR | — | — | pallas_call() missing 1 required positio |
| 51_Gemm_Subtract_GlobalAvgPool_LogSumExp_GELU_ResidualAdd | ERROR | — | — | pallas_call() missing 1 required positio |
| 52_Conv2d_Activation_BatchNorm | ERROR | — | — | Cannot store scalars to VMEM |
| 53_Gemm_Scaling_Hardtanh_GELU | ERROR | — | — | If `grid_spec` is specified, then `in_sp |
| 54_Conv2d_Multiply_LeakyReLU_GELU | ERROR | — | — | Cannot store scalars to VMEM |
| 55_Matmul_MaxPool_Sum_Scale | ERROR | — | — | pallas_call() missing 1 required positio |
| 56_Matmul_Sigmoid_Sum | ERROR | — | — | pallas_call() missing 1 required positio |
| 57_Conv2d_ReLU_HardSwish | ERROR | — | — | pallas_call() got an unexpected keyword  |
| 58_ConvTranspose3d_LogSumExp_HardSwish_Subtract_Clamp | ERROR | — | — | pallas_call() missing 1 required positio |
| 59_Matmul_Swish_Scaling | ERROR | — | — | Pytree for `in_specs` and `inputs` do no |
| 5_ConvTranspose2d_Subtract_Tanh | ERROR | — | — | Attempted boolean conversion of traced a |
| 60_ConvTranspose3d_Swish_GroupNorm_HardSwish | ERROR | — | — | Invalid out_shape type: <class 'int'> |
| 61_ConvTranspose3d_ReLU_GroupNorm | ERROR | — | — | Block shape for outputs (= (Blocked(bloc |
| 62_Matmul_GroupNorm_LeakyReLU_Sum | ERROR | — | — | Attempted boolean conversion of traced a |
| 63_Gemm_ReLU_Divide | ERROR | — | — | pallas_call() got an unexpected keyword  |
| 64_Gemm_LogSumExp_LeakyReLU_LeakyReLU_GELU_GELU | ERROR | — | — | pallas_call() missing 1 required positio |
| 65_Conv2d_AvgPool_Sigmoid_Sum | ERROR | — | — | ERROR: Command timed out |
| 66_Matmul_Dropout_Softmax | ERROR | — | — | The Pallas TPU lowering currently requir |
| 67_Conv2d_GELU_GlobalAvgPool | ERROR | — | — | Cannot store scalars to VMEM |
| 68_Matmul_Min_Subtract | ERROR | — | — | pallas_call() got an unexpected keyword  |
| 69_Conv2d_HardSwish_ReLU | ERROR | — | — | Cannot store scalars to VMEM |
| 6_Conv3d_Softmax_MaxPool_MaxPool | ERROR | — | — | pallas_call() got an unexpected keyword  |
| 70_Gemm_Sigmoid_Scaling_ResidualAdd | ERROR | — | — | pallas_call() missing 1 required positio |
| 71_Conv2d_Divide_LeakyReLU | ERROR | — | — | If `grid_spec` is specified, then `in_sp |
| 72_ConvTranspose3d_BatchNorm_AvgPool_AvgPool | ERROR | — | — | 'function' object is not iterable |
| 73_Conv2d_BatchNorm_Scaling | ERROR | — | — | Model.forward.<locals>.conv_kernel() tak |
| 74_ConvTranspose3d_LeakyReLU_Multiply_LeakyReLU_Max | ERROR | — | — | Attempted boolean conversion of traced a |
| 75_Gemm_GroupNorm_Min_BiasAdd | ERROR | — | — | Model.forward.<locals>.row_kernel() miss |
| 76_Gemm_Add_ReLU | ERROR | — | — | Out of bound slice: start=128, dim=128. |
| 77_ConvTranspose3d_Scale_BatchNorm_GlobalAvgPool | ERROR | — | — | pallas_call() missing 1 required positio |
| 78_ConvTranspose3d_Max_Max_Sum | ERROR | — | — | No output |
| 79_Conv3d_Multiply_InstanceNorm_Clamp_Multiply_Max | ERROR | — | — | 'tuple' object has no attribute 'grid' |
| 7_Conv3d_ReLU_LeakyReLU_GELU_Sigmoid_BiasAdd | ERROR | — | — | No output |
| 80_Gemm_Max_Subtract_GELU | ERROR | — | — | No output |
| 81_Gemm_Swish_Divide_Clamp_Tanh_Clamp | ERROR | — | — | No output |
| 82_Conv2d_Tanh_Scaling_BiasAdd_Max | ERROR | — | — | No output |
| 83_Conv3d_GroupNorm_Min_Clamp_Dropout | ERROR | — | — | No output |
| 84_Gemm_BatchNorm_Scaling_Softmax | ERROR | — | — | No output |
| 85_Conv2d_GroupNorm_Scale_MaxPool_Clamp | ERROR | — | — | No output |
| 86_Matmul_Divide_GELU | ERROR | — | — | No output |
| 87_Conv2d_Subtract_Subtract_Mish | ERROR | — | — | No output |
| 88_Gemm_GroupNorm_Swish_Multiply_Swish | ERROR | — | — | No output |
| 89_ConvTranspose3d_MaxPool_Softmax_Subtract_Swish_Max | ERROR | — | — | Model.forward.<locals>.<lambda>() takes  |
| 8_Conv3d_Divide_Max_GlobalAvgPool_BiasAdd_Sum | ERROR | — | — |  Attempting to pass a Ref MemRef<None>{f |
| 90_Conv3d_LeakyReLU_Sum_Clamp_GELU | ERROR | — | — | pallas_call() missing 1 required positio |
| 91_ConvTranspose2d_Softmax_BiasAdd_Scaling_Sigmoid | ERROR | — | — | 'function' object is not subscriptable |
| 92_Conv2d_GroupNorm_Tanh_HardSwish_ResidualAdd_LogSumExp | ERROR | — | — | lax.conv_general_dilated requires argume |
| 93_ConvTranspose2d_Add_Min_GELU_Multiply | ERROR | — | — | Attempted boolean conversion of traced a |
| 94_Gemm_BiasAdd_Hardtanh_Mish_GroupNorm | ERROR | — | — | Invalid out_shape type: <class 'int'> |
| 95_Matmul_Add_Swish_Tanh_GELU_Hardtanh | ERROR | — | — | pallas_call() got an unexpected keyword  |
| 96_ConvTranspose3d_Multiply_Max_GlobalAvgPool_Clamp | ERROR | — | — | ERROR: Command timed out |
| 97_Matmul_BatchNorm_BiasAdd_Divide_Swish | ERROR | — | — | pallas_call() got an unexpected keyword  |
| 98_Matmul_AvgPool_GELU_Scale_Max | ERROR | — | — | pallas_call() got an unexpected keyword  |
| 99_Matmul_GELU_Softmax | ERROR | — | — | pallas_call() got an unexpected keyword  |
| 9_Matmul_Subtract_Multiply_ReLU | ERROR | — | — | pallas_call() got an unexpected keyword  |

### Priority Kernels

| Metric | Count |
|--------|-------|
| Total | 17 |
| Ran successfully | 0 |
| Errors (crash/timeout) | 17 |
| Correct output | 0 |
| Faster than baseline | 0 |
| Correct AND faster | 0 |

| Workload | Correct | Orig (ms) | Gen (ms) | Speedup |
|----------|---------|-----------|----------|---------|
| cross_entropy | ERROR | — | — | Attempted boolean conversion of traced a |
| flash_attention | ERROR | — | — | Invalid out_shape type: <class 'int'> |
| flex_attention | ERROR | — | — | Invalid out_shape type: <class 'int'> |
| gemm | ERROR | — | — | load() takes 2 positional arguments but  |
| gqa_attention | ERROR | — | — | The Pallas TPU lowering currently requir |
| mamba2_ssd | ERROR | — | — | 'F32Type' object has no attribute 'eleme |
| megablox_gmm | ERROR | — | — | Array slice indices must have static sta |
| mla_attention | ERROR | — | — | Invalid out_shape type: <class 'int'> |
| paged_attention | ERROR | — | — | pallas_call() missing 1 required positio |
| ragged_dot | ERROR | — | — | module 'jax.experimental.pallas' has no  |
| ragged_paged_attention | ERROR | — | — | The __index__() method was called on tra |
| retnet_retention | ERROR | — | — | pallas_call() missing 1 required positio |
| rms_norm | ERROR | — | — | workload.<locals>.kernel() takes 3 posit |
| sparse_attention | ERROR | — | — | pallas_call() missing 1 required positio |
| sparse_moe | ERROR | — | — | Invalid out_shape type: <class 'int'> |
| swiglu_mlp | ERROR | — | — | RESOURCE_EXHAUSTED: Allocation (size=469 |
| triangle_multiplication | ERROR | — | — | Attempted boolean conversion of traced a |
