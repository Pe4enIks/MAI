#include "unistd.h"
#include "stdio.h"
#include "stdlib.h"
#include "ctype.h"

int main()
{
	int fd_0[2];
	int fd_1[2];
	int fd_2[2];
	pipe(fd_0);
	pipe(fd_1);
	pipe(fd_2);
	int pid_0 = 0;
	int pid_1 = 0;
	if((pid_0 = fork()) > 0)
	{
		if((pid_1 = fork()) > 0)
		{
			fflush(stdout);
			fflush(stdin);
			char* in = (char*)malloc(2*sizeof(char));
			in[0] = 0;
			char c;
			while((c = getchar()) != EOF)
			{
				in[0] += 1;
				in[in[0]] = c;
				in = (char*)realloc(in, (in[0]+2)*sizeof(char));
			}
			in[in[0]+1] = '\0';
			write(fd_0[1], in, (in[0]+2)*sizeof(char));
			char* out = (char*)malloc((in[0]+2)*sizeof(char));
			for(int i = 0; i <= in[0]+1; ++i)
			{
				read(fd_2[0], &out[i], sizeof(char));
			}
			for(int i = 1; i <= out[0]+1; ++i)
			{
				printf("%c", out[i]);
			}
			printf("\n");
			fflush(stdout);
			close(fd_2[0]);
			close(fd_0[1]);
			free(in);
			free(out);
		}
		else if(pid_1 == 0)
		{
			fflush(stdout);
			fflush(stdin);
			char* in = (char*)malloc(sizeof(char));
			read(fd_1[0], &in[0], sizeof(char));
			in = (char*)realloc(in, (in[0]+2)*sizeof(char));
			char* out = (char*)malloc((in[0]+2)*sizeof(char));
			for(int i = 1; i <= in[0]+1; ++i)
			{
				read(fd_1[0], &in[i], sizeof(char));
			}
			for(int i = 1; i <= in[0]+1; ++i)
			{
				if(in[i] == ' ')
				{
					out[i] = in[i];
					for(int j = i+1; j <= in[0]+1; ++j)
					{
						if(in[j] == ' ')
						{
							out[j] = '\0';
				            ++i;
						}
						else
						{
							out[j] = in[j];
							++i;
							break;
						}
					}
				}
				else
				{
					out[i] = in[i];
				}
			}
			out[0] = in[0];
			write(fd_2[1], out, (out[0]+2)*sizeof(char));
			fflush(stdout);
			close(fd_2[1]);
			close(fd_1[0]);
			free(in);
			free(out);
		}
		else
		{
			perror("fork error\n");
			exit(-1);
		}
	}
	else if (pid_0 == 0)
	{
		fflush(stdout);
		fflush(stdin);
		char* in = (char*)malloc(sizeof(char));
		read(fd_0[0], &in[0], sizeof(char));
		in = (char*)realloc(in, (in[0]+2)*sizeof(char));
		char* out = (char*)malloc((in[0]+2)*sizeof(char));
		for(int i = 1; i <= in[0]+1; ++i)
		{
			read(fd_0[0], &in[i], sizeof(char));
		}
		for(int i = 1; i <= in[0]+1; ++i)
		{
			out[i] = toupper(in[i]);
		}
		out[0] = in[0];
		write(fd_1[1], out, (out[0]+2)*sizeof(char));
		fflush(stdout);
		close(fd_0[0]);
		close(fd_1[1]);
		free(in);
		free(out);
	}
	else
	{
		perror("fork error\n");
		exit(-1);
	}
	return 0;
}