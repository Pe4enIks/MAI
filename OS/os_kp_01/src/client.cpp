#include <iostream>
#include <stdio.h> 
#include <stdlib.h>
#include <sys/socket.h> 
#include <arpa/inet.h> 
#include <unistd.h> 
#include <string.h>
#include "crypt.hpp"
#include "history.hpp"
#include "sqlite3.h" 
#include "help_functions.hpp"

ll* encrypted(public_key key, bool& flag, std::string& name_str, History& hist)
{
	std::string str;
	ll len = 0;
	ll len_name = 0;
	size_t j = 0;
	getline(std::cin, str, '\n');
	if(str == "exit") { flag = true; }
	else { flag = false; }
	if(str == "history")
	{
		std::cout << "\nMessages history\n";
		hist.print();
		std::cout << "\n";
	}

	char* my_str = (char*)str.c_str();
	len = str.size();
	len_name = name_str.size();
	ll* enc = (ll*)my_malloc((len+len_name+2)*sizeof(ll));
	enc[0] = (ll)len;
	enc[1] = (ll)len_name;
	for(size_t i = 2; i < len+len_name+2; ++i) { enc[i] = (ll)0; }
	for (size_t i = 2; i < len+2; ++i) { enc[i] = (ll)encrypt(key, my_str[i-2]); }
	for(size_t i = len+2; i < len+len_name+2; ++i) { enc[i] = (ll)encrypt(key, name_str[i-len-2]); }
	return enc;
}

public_key publ_key(int sock)
{
	ll* buff = (ll*)my_malloc(2*sizeof(ll));
	my_read(sock, buff, 2*sizeof(ll));
	public_key new_k(buff[0], buff[1]);
	free(buff);
	return new_k;
}


int main(int argc, const char* argv[]) 
{   
	History hist;
	std::string name;
	if(argc != 2)
	{
		perror("Invalid count of arguments\n");
		exit(-1);
	}
	struct sockaddr_in serv_addr; 
	serv_addr.sin_family = AF_INET; 
	serv_addr.sin_addr.s_addr = htonl(INADDR_ANY);
	serv_addr.sin_port = htons(PORT); 
	bool flag = false;
	std::cout << "Please enter your nick\n";
	std::cin >> name;
	while(true)
	{
		int sock = my_socket(AF_INET, SOCK_STREAM, 0);
		my_inet_pton(AF_INET, argv[1], &serv_addr.sin_addr);
		my_connect(sock, (struct sockaddr*)&serv_addr, sizeof(serv_addr));

		public_key new_k = publ_key(sock);
		ll* enc = encrypted(new_k, flag, name, hist);
		my_send(sock, enc, (enc[0]+enc[1]+2)*sizeof(ll), 0);
		free(enc);
		if(flag) { break; }
	}
	return 0; 
} 
