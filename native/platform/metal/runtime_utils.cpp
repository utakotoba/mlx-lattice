#include "platform/metal/runtime_utils.h"

#include <dlfcn.h>

#include <filesystem>
#include <stdexcept>

namespace mlx_lattice::metal {

std::string binary_dir() {
    static std::string dir = [] {
        Dl_info info;
        if (!dladdr(reinterpret_cast<void*>(&binary_dir), &info)) {
            throw std::runtime_error("Unable to resolve native module path.");
        }
        return std::filesystem::path(info.dli_fname).parent_path().string();
    }();
    return dir;
}

} // namespace mlx_lattice::metal
