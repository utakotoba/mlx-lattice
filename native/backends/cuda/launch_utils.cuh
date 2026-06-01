#pragma once

namespace {

dim3 grid_for(size_t elements, int block = 256) {
    return dim3(static_cast<unsigned int>((elements + block - 1) / block));
}

template <typename Kernel, typename... Args>
void launch(
    mx::cu::CommandEncoder& encoder,
    Kernel kernel,
    size_t elements,
    Args... args
) {
    if (elements == 0) {
        return;
    }
    constexpr int block = 256;
    encoder.add_kernel_node(
        kernel, grid_for(elements, block), dim3(block), args...
    );
}

} // namespace
