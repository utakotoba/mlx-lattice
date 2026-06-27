template <typename T>
inline float4 load_contiguous4(device const T* values, int base) {
    return float4(
        float(values[base + 0]),
        float(values[base + 1]),
        float(values[base + 2]),
        float(values[base + 3])
    );
}

inline float4 load_contiguous4(device const half* values, int base) {
    const half4 packed = *reinterpret_cast<device const half4*>(values + base);
    return float4(
        float(packed.x), float(packed.y), float(packed.z), float(packed.w)
    );
}

template <typename T>
inline float4 load_strided4(device const T* values, int base, int stride) {
    return float4(
        float(values[base + 0 * stride]),
        float(values[base + 1 * stride]),
        float(values[base + 2 * stride]),
        float(values[base + 3 * stride])
    );
}

inline float4 load_strided4(device const half* values, int base, int stride) {
    if (stride == 1) {
        return load_contiguous4(values, base);
    }
    return float4(
        float(values[base + 0 * stride]),
        float(values[base + 1 * stride]),
        float(values[base + 2 * stride]),
        float(values[base + 3 * stride])
    );
}

inline void store_contiguous4(device float* out, int base, float4 value) {
    out[base + 0] = value.x;
    out[base + 1] = value.y;
    out[base + 2] = value.z;
    out[base + 3] = value.w;
}

inline void store_contiguous4(device half* out, int base, float4 value) {
    *reinterpret_cast<device half4*>(out + base) = half4(value);
}

template <typename T>
inline float4 load_dense_weight4(
    device const T* weights,
    int weight_s0,
    int channels,
    int kernel_id,
    int ci,
    int co_base
) {
    return float4(
        float(weights[(co_base + 0) * weight_s0 + kernel_id * channels + ci]),
        float(weights[(co_base + 1) * weight_s0 + kernel_id * channels + ci]),
        float(weights[(co_base + 2) * weight_s0 + kernel_id * channels + ci]),
        float(weights[(co_base + 3) * weight_s0 + kernel_id * channels + ci])
    );
}

template <typename T>
inline float4 load_dense_weight_ci4(
    device const T* weights,
    int weight_s0,
    int channels,
    int kernel_id,
    int co,
    int ci_base
) {
    return load_contiguous4(
        weights, co * weight_s0 + kernel_id * channels + ci_base
    );
}

inline void store4(device float* out, int base, float4 value) {
    store_contiguous4(out, base, value);
}

inline void store4(device half* out, int base, float4 value) {
    store_contiguous4(out, base, value);
}

struct DenseForwardParams {
    int edge_capacity;
    int out_capacity;
    int feat_s0;
    int feat_s1;
    int weight_s0;
};

struct DenseInputGradParams {
    int edge_capacity;
    int out_capacity;
    int in_capacity;
    int cotangent_s0;
    int cotangent_s1;
    int weight_s0;
};
