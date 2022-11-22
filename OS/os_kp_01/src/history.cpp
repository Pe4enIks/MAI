#include <iostream>
#include <string>
#include "history.hpp"

History::History()
{
	if(sqlite3_open_v2(DATABASE, &db, 
		SQLITE_OPEN_NOMUTEX | SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE, NULL))
	{
		perror("open/create database error\n");
		sqlite3_close(db);
		exit(-1);
	}
	std::string sql_request = "CREATE TABLE IF NOT EXISTS HISTORY";
	sql_request += "(MSG	TEXT NOT NULL, NICKNAME		TEXT NOT NULL);";
	if(sqlite3_exec(db, sql_request.c_str(), NULL, NULL, NULL))
	{
		perror("table create error\n");
	}
}

int callback(void* data, int argc, char** argv, char** azcolname)
{
	for(int i = 0; i < argc; ++i)
	{
		std::cout << azcolname[i] << ": " << argv[i] << "\n";
	}
	return 0;
}

void History::add(std::string& msg, std::string& nickname)
{

	std::string sql_request = "INSERT INTO HISTORY(\'MSG\',\'NICKNAME\') VALUES(";
	sql_request += "\'" + msg + "\'" + ",\'" + nickname + "\'" + ");";
	if(sqlite3_exec(db, sql_request.c_str(), NULL, NULL, NULL))
	{
		perror("insert error\n");
	}
}	

void History::print()
{
	std::string sql_request = "SELECT * FROM HISTORY;";
	if(sqlite3_exec(db, sql_request.c_str(), callback, NULL, NULL))
	{
		perror("select error\n");
	}
}

History::~History()
{
	sqlite3_close(db);
}