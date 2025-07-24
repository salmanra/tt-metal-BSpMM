#pragma once

#include <cstdint>
#include <memory>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/test_tiles.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/command_queue.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <../matmul_common/bmm_op.hpp>
#include <tt-metalium/tilize_untilize.hpp>


#include <unordered_map>
#include <variant>
#include <vector>
#include <tuple>
#include <filesystem> // for emitting test output

#include "bsr_matrix.hpp"
#include "tt-metalium/core_coord.hpp"