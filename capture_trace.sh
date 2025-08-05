#!/usr/bin/bash

# TODO: this file is practically useless now

function capture_all_one_host {
    num_tests=17 # size of ProfileCaseRegistry
    host_program_index=0
    if [[ "$#" -eq 1 ]]; then
        host_program_index=$1
    fi
    
    for ((test_case = 0 ; test_case < num_tests ; test_case++)); do
        capture_trace $test_case $host_program_index
    done
}


function capture_trace {
    latest_host_program_index=0
    default_test_case=0
    if [[ "$#" -eq "0" ]]; then
        # run test 30
        ./build/programming_examples/rahmy/profile_block $default_test_case $latest_host_program_index
    elif [[ "$#" -eq "1" ]]; then
        # run test given test with default func
        ./build/programming_examples/rahmy/profile_block $1 $latest_host_program_index
    elif [[ "$#" -eq "2" ]]; then
        # run the given test with the given program
        ./build/programming_examples/rahmy/profile_block $1 $2
    fi
}


# function capture_trace {
#     # $1 index into HostCodeRegistry (see /block/include/host_code.hpp)
#     # $2 index into TestCaseRegistry (see /block/include/test_suite.hpp)
#     host_program_names=("bsr_spmm_multicore_reuse_many_blocks_per_core" "bsr_spmm_multicore_reuse" "bsr_spmm_multicore_reuse_naive")
#     latest_host_program_index=0
#     default_test_case=0
#     if [[ "$#" -eq "0" ]]; then
#         # capture-release with latest func
#         # run test 30
#         mkdir -p "profiles/${host_program_names[$latest_host_program_index]}"
#         ./capture-release -f -o "profiles/${host_program_names[$latest_host_program_index]}/$default_test_case.tracy" &
#         ./build/programming_examples/rahmy/profile_dense $default_test_case $latest_host_program_index
#     elif [[ "$#" -eq "1" ]]; then
#         # capture-release with given func
#         # run test 30
#         mkdir -p "profiles/${host_program_names[$1]}"
#         ./capture-release -f -o "profiles/${host_program_names[$1]}/$default_test_case.tracy" &
#         ./build/programming_examples/rahmy/profile_block $default_test_case $1
#     elif [[ "$#" -eq "2" ]]; then
#         # capture-release with given func
#         # run the given test
#         mkdir -p "profiles/${host_program_names[$1]}"
#         ./capture-release -f -o "profiles/${host_program_names[$1]}/$2.tracy" &
#         ./build/programming_examples/rahmy/profile_block $2 $1
#     fi
# }

num_host_programs=3

./build_metal.sh --enable-profiler --build-programming-examples
if [[ $? -ne 0 ]]; then
    echo Build failed
    return
else 
    echo Build Suceeded!
fi

if [[ "$#" -eq "0" ]]; then
    capture_trace
elif [[ "$#" -eq "1" ]]; then
    if [[ "$1" == "all" ]]; then
        for ((host_program = 0 ; host_program < $num_host_programs ; host_program++)); do
            for ((test_case = 0 ; test_case < num_tests ; test_case++)); do
                capture_trace $host_program $test_case
            done
        done
    fi
elif [[ "$#" -eq "2" ]]; then
    if [[ "$1" == "all" ]]; then
        capture_all_one_host $2
    fi
fi