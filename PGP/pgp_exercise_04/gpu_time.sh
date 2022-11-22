nvcc --std=c++11 -Werror cross-execution-space-call -lm gpu.cu -o build/gpu
for ((i=0; i < $1; i++))
do
build/gpu < time_tests/test${i}.txt
done