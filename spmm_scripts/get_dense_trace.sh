#!/usr/bin/bash

function capture_all_one_host {
    num_tests=14 # size of ProfileCaseRegistry
    host_program_index=0
    profile_case_registry_index=0
    if [[ "$#" -eq 1 ]]; then
        host_program_index=$1
    fi
    if [[ "$#" -eq 2 ]]; then
        host_program_index=$1
        profile_case_registry_index=$2
    fi
    
    for ((test_case = 0 ; test_case < num_tests ; test_case++)); do
        capture_trace $test_case $host_program_index $profile_case_registry_index
    done
}


function capture_trace {
    latest_host_program_index=0
    default_test_case=0
    default_registry=0
    if [[ "$#" -eq "0" ]]; then
        # run test 30
        ./build/programming_examples/rahmy/profile_dense $default_test_case $latest_host_program_index
    elif [[ "$#" -eq "1" ]]; then
        # run test given test with default func
        ./build/programming_examples/rahmy/profile_dense $1 $latest_host_program_index
    elif [[ "$#" -eq "2" ]]; then
        # run the given test with the given program
        ./build/programming_examples/rahmy/profile_dense $1 $2
    elif [[ "$#" -eq "3" ]]; then
        ./build/programming_examples/rahmy/profile_dense $1 $2 $3
    fi
}

num_host_programs=2

./build_metal.sh --enable-profiler --build-programming-examples
if [[ $? -ne 0 ]]; then
    echo Build failed
    return
else 
    echo Build Suceeded!
fi

if [[ "$#" -eq "0" ]]; then
    echo capturing a single trace
    capture_trace
elif [[ "$#" -eq "1" ]]; then
    if [[ "$1" == "all" ]]; then
        echo Capturing all traces for all $num_host_programs hosts 
        for ((host_program = 0 ; host_program < $num_host_programs ; host_program++)); do
            capture_all_one_host $host_program 0
        done
    elif [[ "$1" == "reduction" ]]; then
        echo Capturing reduction ablation traces for all $num_host_programs hosts
        for ((host_program = 0 ; host_program < $num_host_programs ; host_program++)); do
            capture_all_one_host $host_program 1
        done
    fi
elif [[ "$#" -eq "2" ]]; then
    if [[ "$1" == "all" ]]; then
        echo Capturing all traces for host $2
        capture_all_one_host $2
    fi
fi