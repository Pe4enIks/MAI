#pragma once

#include "sqlite3.h"

#ifndef DATABASE
#define DATABASE "history.db"
#endif

class History
{
private:
	sqlite3* db;
public:
	void add(std::string&, std::string&);
	void print();
	~History();
	History();
};