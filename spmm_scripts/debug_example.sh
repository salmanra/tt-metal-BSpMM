#!/bin/bash

export TT_METAL_DPRINT_CORES=worker

program_name=$1
test_case=$2
host_code_index=0
executable_name="./build/programming_examples/rahmy/$program_name"

./$executable_name $test_case $host_code_index