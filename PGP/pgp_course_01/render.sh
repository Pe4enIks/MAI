nvcc --std=c++11 -Werror cross-execution-space-call -lm render.cu -o build/render
build/render --default > logs/parameters.txt
build/render < gpu.txt > logs/gpu.txt
build/render --cpu < cpu.txt > logs/cpu.txt