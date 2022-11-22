#include <iostream>
#include <unistd.h> 
#include <stdio.h>
#include <stdlib.h>
#include <sys/socket.h> 
#include <netinet/in.h> 
#include <arpa/inet.h> 
#include <string.h>
#include <netdb.h> 
#include "help_functions.hpp"

void* my_malloc(size_t size)
{
	void* mal = malloc(size);
	if(mal == NULL)
	{
		perror("malloc error\n");
		exit(-1);
	}
	return mal;
}

int my_socket(int domain, int type, int protocol)
{
	int fd = socket(domain, type, protocol);
	if(fd == 0)
	{
		perror("socket failed\n"); 
		exit(-1);
	}
	return fd;
}

int my_inet_pton(int af, const char* src, void* dst)
{
	int ip = inet_pton(af, src, dst);
	if(ip <= 0)
	{
		perror("invalid address / address not supported\n"); 
		exit(-1);
	}
	return ip;
}

ssize_t my_send(int s, const void* msg, size_t len, int flags)
{
	ssize_t var = send(s, msg, len, flags);
	if(var == -1)
	{
		perror("send error\n");
		exit(-1);
	}
	return var;
}

int my_connect(int sockfd,struct sockaddr* serv_addr, socklen_t addrlen)
{
	int var = connect(sockfd, serv_addr, addrlen);
	if(var == -1)
	{
		perror("connection failed\n"); 
		exit(-1);
	}
	return var;
}

int my_read(int fd, void* buff, size_t count)
{
	int var = read(fd, buff, count);
	if(var != count)
	{
		perror("read error\n");
		exit(-1);
	}
	return var;
}

int my_listen(int s, int backlog)
{
	int var = listen(s, backlog);
	if(var == -1)
	{
		perror("listen error\n"); 
		exit(-1);
	}
	return var;
}

int my_bind(int sockfd, struct sockaddr* my_addr, socklen_t addrlen)
{
	int var = bind(sockfd, my_addr, addrlen);
	if(var == -1)
	{
		perror("bind failed\n"); 
		exit(-1); 
	}
	return var;
}

int my_setsockopt(int sockfd, int lev, int optname, const void* optval, socklen_t optlen)
{
	int var = setsockopt(sockfd, lev, optname, optval, optlen);
	if(var == -1)
	{
		perror("setsockopt error\n"); 
		exit(-1); 
	}
	return var;
}

int my_accept(int s, struct sockaddr* addr, socklen_t* addrlen)
{
	int var = accept(s, addr, addrlen);
	if(var == -1)
	{
		perror("accept error\n"); 
		exit(-1); 
	}
	return var;
}