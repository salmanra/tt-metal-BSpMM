#pragma once

// SpGEMM zone configuration stub.
// Will be filled in when device-side algorithms are implemented.

#include <map>
#include <string>

namespace spgemm_zone_config {

// Placeholder: returns an empty defines map for now
inline std::map<std::string, std::string> get_zone_defines() {
    return {};
}

} // namespace spgemm_zone_config
