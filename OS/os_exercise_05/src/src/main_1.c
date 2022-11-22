#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <dlfcn.h>

int main()
{
	char cmd = 'a';
	char changer = '0';
	float a = 0.0;
	float b = 0.0;
	float e = 0.0;
	void* library_handler_0 = NULL;
	void* library_handler_1 = NULL;
	float (*squarefunc)(float,float);
	float (*sinintfunc)(float,float,float);


	if((library_handler_0 = dlopen("libimp0.so", RTLD_LAZY)) == 0)
	{
		printf("OPEN LIBRARY ERROR\n");
		exit(-1);
	}
	if((library_handler_1 = dlopen("libimp1.so", RTLD_LAZY)) == 0)
	{
		printf("OPEN LIBRARY ERROR\n");
		exit(-1);
	}

	printf("Enter command:\n");
	printf("0 -> change implementation\n");
	printf("1 arg1 arg2 -> return value - float; arg1,arg2 - float\n");
	printf("2 arg1 arg2 arg3 -> return value - float; arg1,arg2,arg3 - float\n");
	printf("3 -> exit\n\n");

	while(cmd != '3')
	{
		if(scanf("%c", &cmd) != 1)
		{
			printf("SCANF ERROR\n");
			exit(-1);
		}
		if(cmd == '0')
		{
			if(changer == '0') { changer = '1';	}
			else { changer = '0'; }
		}
		else if (cmd == '1')
		{
			if(scanf("%f%f", &a, &b) != 2)
			{
				printf("SCANF ERROR\n");
				exit(-1);
			}
			if(changer == '0') { squarefunc = dlsym(library_handler_0, "square"); }
			else { squarefunc = dlsym(library_handler_1, "square"); }
			printf("square = %f\n", (*squarefunc)(a, b));
		}
		else if(cmd == '2')
		{
			if(scanf("%f%f%f", &a, &b, &e) != 3)
			{
				printf("SCANF ERROR\n");
				exit(-1);
			}
			if(changer == '0') { sinintfunc = dlsym(library_handler_0, "sinintegral"); }
			else { sinintfunc = dlsym(library_handler_1, "sinintegral"); }
			printf("integral = %f\n", (*sinintfunc)(a, b, e));
		}
	}
	dlclose(library_handler_0);
	dlclose(library_handler_1);
	return 0;
}