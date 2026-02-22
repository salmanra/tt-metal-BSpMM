#pragma once
#include <stdint.h>
#include <tools/profiler/kernel_profiler.hpp>

// Macro wrappers that preserve string literals for DeviceZoneScopedN.
// `fn` should be a callable (e.g. a lambda). The macro ensures the zone name
// reaches DeviceZoneScopedN as a literal so compile-time hashing works.

#define SPMM_PROFILE_ZONE(enabled, name, fn) \
    do { \
        if constexpr (enabled) { \
            DeviceZoneScopedN(name); \
            (fn)(); \
        } else { \
            (fn)(); \
        } \
    } while(0)

#define SPMM_PROFILE_ZONE_COMPUTE(enabled, name, fn) \
    do { \
        if constexpr (enabled) { \
            UNPACK(DeviceZoneScopedN(name)); \
            (fn)(); \
        } else { \
            (fn)(); \
        } \
    } while(0)
