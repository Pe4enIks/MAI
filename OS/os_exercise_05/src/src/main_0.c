#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "./lib/imp_0/sinintegral.h"
#include "./lib/imp_0/square.h"

int main()
{
	char cmd = '0';
	float a = 0.0;
	float b = 0.0;
	float e = 0.0;
	printf("Enter command:\n");
	printf("1 arg1 arg2 -> return value - float; arg1,arg2 - float\n");
	printf("2 arg1 arg2 arg3 -> return value - float; arg1,arg2,arg3 - float\n\n");
	cmd = getchar();
	if (cmd == '1')
	{
		if(scanf("%f%f", &a, &b) != 2)
		{
			printf("SCANF ERROR\n");
			exit(-1);
		}
		printf("square = %f\n", square(a, b));
	}
	else if(cmd == '2')
	{
		if(scanf("%f%f%f", &a, &b, &e) != 3)
		{
			printf("SCANF ERROR\n");
			exit(-1);
		}
		printf("integral = %f\n", sinintegral(a, b, e));
	}
	else
	{
		printf("INVALID COMMAND\n");
		exit(-1);
	}
}