inline int sparse_conv_weight_offset(
    int kernel_id,
    int in_channel,
    int out_channel,
    int weight_layout,
    int kernel_x,
    int kernel_y,
    int kernel_z,
    int weight_s0,
    int weight_s1,
    int weight_s2,
    int weight_s3,
    int weight_s4
) {
    if (weight_layout == 0) {
        return kernel_id * weight_s0 + in_channel * weight_s1 +
               out_channel * weight_s2;
    }

    int xy = kernel_y * kernel_z;
    int kx = kernel_id / xy;
    int rem = kernel_id % xy;
    int ky = rem / kernel_z;
    int kz = rem % kernel_z;
    (void)kernel_x;
    return out_channel * weight_s0 + kx * weight_s1 + ky * weight_s2 +
           kz * weight_s3 + in_channel * weight_s4;
}

inline int sparse_conv_dense_weight_offset(
    int kernel_id,
    int in_channel,
    int out_channel,
    int weight_layout,
    int kernel_x,
    int kernel_y,
    int kernel_z,
    int in_channels,
    int out_channels
) {
    if (weight_layout == 0) {
        return (kernel_id * in_channels + in_channel) * out_channels +
               out_channel;
    }

    int xy = kernel_y * kernel_z;
    int kx = kernel_id / xy;
    int rem = kernel_id % xy;
    int ky = rem / kernel_z;
    int kz = rem % kernel_z;
    return (((out_channel * kernel_x + kx) * kernel_y + ky) * kernel_z + kz) *
               in_channels +
           in_channel;
}
