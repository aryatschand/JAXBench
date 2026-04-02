out_nhwc = jax.lax.conv_general_dilated(
            x_nhwc, 
            w_final,
            window_strides=(1, 1),
            padding=padding_jax,
            lhs_dilation=self.stride,
            rhs_dilation=self.dilation,
            dimension_numbers=('NHWC', 'HWOI', 'NHWC'),
            feature_group_count=self.groups
        )
