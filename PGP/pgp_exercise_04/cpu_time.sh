nvcc --std=c++11 -Werror cross-execution-space-call -lm cpu.cu -o build/cpu
for ((i=0; i < $1; i++))
do
build/cpu < time_tests/test${i}.txt
done