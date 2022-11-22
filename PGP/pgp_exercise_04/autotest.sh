nvcc --std=c++11 -Werror cross-execution-space-call -lm gpu.cu -o build/gpu
mkdir results tests
python3 generate_tests.py -tc $1 -nmin $2 -nmax $3 -elmin $4 -elmax $5
for ((i=0; i < $1; i++))
do
build/gpu < tests/test${i}.txt > results/test${i}.txt
done
python3 check_tests.py
rm -rf results tests