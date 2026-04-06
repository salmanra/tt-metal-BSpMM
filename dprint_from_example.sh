export TT_METAL_DPRINT_CORES=worker

example_name="$1"
executable_name="./build/programming_examples/rahmy/$example_name"

./$executable_name $2 $3 $4 | grep -Ei '(:BR:)|(:NC:)|(:TR)'