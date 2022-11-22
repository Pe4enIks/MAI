GPP = g++
GCC = gcc
GPPFLAGS = -std=c++11
GCCFLAGS =
SERVNAME = server
CLIENNAME = client 

all: server.o client.o help_functions.o crypt.o history.o sqlite.o
	$(GPP) $(GPPFLAGS) server.o crypt.o help_functions.o \
history.o sqlite.o -o $(SERVNAME) -ldl -lpthread
	$(GPP) $(GPPFLAGS) client.o crypt.o help_functions.o \
history.o sqlite.o -o $(CLIENNAME) -ldl -lpthread

client.o: crypt.hpp help_functions.hpp
	$(GPP) $(GPPFLAGS) -c client.cpp -o client.o

server.o: crypt.hpp history.hpp help_functions.hpp sqlite3.h
	$(GPP) $(GPPFLAGS) -c server.cpp -o server.o

help_functions.o: help_functions.hpp
	$(GPP) $(GPPFLAGS) -c help_functions.cpp -o help_functions.o

crypt.o: crypt.hpp
	$(GPP) $(GPPFLAGS) -c crypt.cpp -o crypt.o

history.o: history.hpp
	$(GPP) $(GPPFLAGS) -c history.cpp -o history.o

sqlite.o: sqlite3.h
	$(GCC) $(GCCFLAGS) -c sqlite3.c -o sqlite.o

clean:
	rm *.o $(SERVNAME) $(CLIENNAME) history.db