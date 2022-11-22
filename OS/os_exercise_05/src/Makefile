all: main_0 main_1 
	rm -f *.o

main_0: main_0.o libimp0.so
	gcc -o main_0 main_0.o -L. -limp0 -lm -Wl,-rpath,.

main_0.o: ./src/main_0.c
	gcc -c ./src/main_0.c -lm

libimp0.so: sinintegral.o square.o
	gcc -shared -o libimp0.so sinintegral.o square.o -lm

sinintegral.o: ./src/lib/imp_0/src/sinintegral.c
	gcc -c -fPIC ./src/lib/imp_0/src/sinintegral.c -lm

square.o: ./src/lib/imp_0/src/square.c
	gcc -c -fPIC ./src/lib/imp_0/src/square.c -lm

main_1: main_1.o libimp0.so libimp1.so
	gcc -o main_1 main_1.o -L. -limp0 -limp1 -lm -ldl -Wl,-rpath,.

main_1.o: ./src/main_1.c
	gcc -c ./src/main_1.c -lm

libimp1.so: sinintegral1.o square1.o
	gcc -shared -o libimp1.so sinintegral1.o square1.o -lm

sinintegral1.o: ./src/lib/imp_1/src/sinintegral.c
	gcc -c -fPIC ./src/lib/imp_1/src/sinintegral.c -o sinintegral1.o -lm

square1.o: ./src/lib/imp_1/src/square.c
	gcc -c -fPIC ./src/lib/imp_1/src/square.c -o square1.o -lm

clean:
	rm -f *.o *.so main_0 main_1