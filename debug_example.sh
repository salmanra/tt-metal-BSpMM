if [ "$2" = "print" ]; then
    export TT_METAL_DPRINT_CORES=worker
fi

if [ "$2" = "watcher" ]; then

    export TT_METAL_WATCHER=120        # the number of seconds between Watcher updates (longer is less invasive)
    export TT_METAL_WATCHER_APPEND=1   # optional: append to the end of the existing log file (vs creating a new file)
    export TT_METAL_WATCHER_DUMP_ALL=0 # optional: dump all state including unsafe state

    # TODO: what's the minimal set of things to disable which allows watcher to run?
    export TT_METAL_WATCHER_DISABLE_ASSERT=1
    export TT_METAL_WATCHER_DISABLE_PAUSE=1
    export TT_METAL_WATCHER_DISABLE_RING_BUFFER=1
    export TT_METAL_WATCHER_DISABLE_NOC_SANITIZE=1
    export TT_METAL_WATCHER_DISABLE_WAYPOINT=1
    export TT_METAL_WATCHER_DISABLE_STACK_USAGE=1

    # In certain cases enabling watcher can cause the binary to be too large. In this case, disable inlining.
    export TT_METAL_WATCHER_NOINLINE=1

    # If the above doesn't work, and dispatch kernels (cq_prefetch.cpp, cq_dispatch.cpp) are still too large, compile out
    # debug tools on dispatch kernels.
    export TT_METAL_WATCHER_DISABLE_DISPATCH=1

    # If you need to see the physical coordinates in the watcher log (note that physical coordinates are not expected
    # to be used in host-side code).
    export TT_METAL_WATCHER_PHYS_COORDS=1
fi

example_name="$1"
executable_name="./build/programming_examples/rahmy/$example_name"

./$executable_name 41 0

