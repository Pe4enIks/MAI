#pragma once

#ifndef PORT
#define PORT 8080
#endif 

#ifndef MAXMSGLEN
#define MAXMSGLEN 300
#endif

void* my_malloc(size_t);
int my_socket(int,int,int);
int my_inet_pton(int,const char*,void*);
ssize_t my_send(int,const void*,size_t,int);
int my_connect(int,struct sockaddr*,socklen_t);
int my_read(int,void*,size_t);
int my_listen(int,int);
int my_bind(int,struct sockaddr*,socklen_t);
int my_setsockopt(int,int,int,const void*,socklen_t);
int my_accept(int,struct sockaddr*,socklen_t*);
void get_host(char*,char*);