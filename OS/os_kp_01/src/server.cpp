#include <unistd.h> 
#include <stdio.h> 
#include <sys/socket.h> 
#include <stdlib.h> 
#include <netinet/in.h> 
#include <netdb.h> 
#include <arpa/inet.h> 
#include <string.h>
#include <iostream>
#include "crypt.hpp"
#include "help_functions.hpp"
#include "history.hpp"
#include "sqlite3.h"


ll* encrypted(public_key key, bool& flag, ll& pid)
{
	std::string str;
	ll len = 0;
	size_t j = 0;
	getline(std::cin, str, '\n');
	if(str == "exit") { flag = true; }
	else { flag = false; }
	char* my_str = (char*)str.c_str();
	len = str.size();
	ll* enc = (ll*)my_malloc((len+2)*sizeof(ll));
	enc[0] = (ll)len;
	enc[1] = (ll)pid;
	for(size_t i = 2; i < len+2; ++i) { enc[i] = (ll)0; }
	for (size_t i = 2; i < len+2; ++i) { enc[i] = (ll)encrypt(key, my_str[i-2]); }
	return enc;
}

void get_host(char* name, char* ip)
{
	char namebuff[MAXMSGLEN];
	char* ipbuff;
	struct hostent *host;
	if(gethostname(namebuff, sizeof(namebuff)) == -1)
	{
		perror("get host name error\n");
		exit(-1);
	}
	host = gethostbyname(namebuff);
	if(host == NULL)
	{
		perror("get host by name error\n");
		exit(-1);
	}
	ipbuff = inet_ntoa(*((struct in_addr*)host->h_addr_list[0]));
	std::cout << "Host name " << namebuff << "\n";
	std::cout << "Host ip " << ipbuff << "\n\n";
	strcpy(name, namebuff);
	strcpy(ip, ipbuff);
}

std::pair<std::string,std::string> decrypted(private_key key, int sock)
{
	std::string str;
	std::string name_str;
	ll len = 0;
	ll len_name = 0;
	read(sock, &len, sizeof(ll));
	ll* dec = (ll*)my_malloc(len*sizeof(ll));
	read(sock, &len_name, sizeof(ll));
	ll* dec_name = (ll*)my_malloc(len_name*sizeof(ll));
	read(sock, dec, len*sizeof(ll));
	read(sock, dec_name, len_name*(sizeof(ll)));
	for(ll i = 0; i < len; ++i)  {	str += (char)decrypt(key, dec[i]); }
	for(ll i = 0; i < len_name; ++i) { name_str += (char)decrypt(key, dec_name[i]); }
	free(dec);
	free(dec_name);
	return std::make_pair(str,name_str); 
}

ll* fill_buff(keys key)
{
	ll* buff = (ll*)my_malloc(2*sizeof(ll));
	buff[0] = key.publ.first;
	buff[1] = key.publ.second;
	return buff;
}

int main() 
{
	History hist;
	struct sockaddr_in address; 
	int opt = 1; 
	int addrlen = sizeof(address);
	char* ip;
	char hostname[MAXMSGLEN];

	srand(time(NULL));
    keys key = generate_keys();
    ll* buff = fill_buff(key);

	int server_fd = my_socket(AF_INET, SOCK_STREAM, 0);
	my_setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt)); 

	address.sin_family = AF_INET;
	inet_aton(ip, &address.sin_addr);
	address.sin_addr.s_addr = htonl(INADDR_ANY);  
	address.sin_port = htons(PORT);

	get_host(hostname, ip);
	
	my_bind(server_fd, (struct sockaddr*)&address, sizeof(address));
	my_listen(server_fd, 10);

	while(true)
	{
		int new_socket = my_accept(server_fd,(struct sockaddr*)&address,(socklen_t*)&addrlen);
		my_send(new_socket, buff, 2*sizeof(ll), 0);
		std::pair<std::string,std::string> msg = decrypted(key.priv, new_socket);
		hist.add(msg.first, msg.second);
		hist.print();
		std::cout << "\n";
	}

	free(buff);
	return 0; 
} 
