//gcc -pthread main.c bitonic_sort.c
//Пищик Е.С. М8О-206Б-19
//Лабораторная работа №3
//Вариант 1
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <limits.h>
#include "sys/time.h"
#include "bitonic_sort.h"

//константа
#define MYMAX 1000000

//глобальные переменные
int thread_count = 0;
int bar_count = 0;
int n = 0;
int *list1 = NULL; 
int *list2 = NULL;
int *l_a = NULL;
int *l_b = NULL;
pthread_mutex_t bar_mutex;
pthread_cond_t bar_cond;

//функция вычисления текущего времени
long long current_timestamp() 
{
   struct timeval te;
   if(gettimeofday(&te, NULL) != 0)
   {
      fprintf(stderr, "GETTIME ERROR\n");
      exit(-1);
   }

   long long milliseconds = te.tv_sec * 1000LL + te.tv_usec / 1000;
   return milliseconds;
}

//функция описания использования при неправильном кол-ве аргументов
void usage(char* prog_name) 
{
   fprintf(stderr, "usage: %s <thread count> <n> [g] [o]\n", prog_name);
   fprintf(stderr, "thread_count should be nonzero\n");
   fprintf(stderr, "thread_count should be pow of 2\n");
   fprintf(stderr, "n = number of elements in list\n");
   fprintf(stderr, "n should be evenly divisible by thread count\n");
   fprintf(stderr, "'g':  program should generate the list\n");
   fprintf(stderr, "'o':  program should output original and sorted lists\n");
   exit(-1);
}

void get_args(int argc, char *argv[], int* gen_list_p, int* output_list_p) 
{
   if (argc < 3 || argc > 5) { usage(argv[0]); }

   //Количество потоков и количество чисел в массиве
   if((thread_count = strtol(argv[1], NULL, 10)) == LONG_MIN || thread_count == LONG_MAX)
   {
      fprintf(stderr, "STRTOL ERROR\n");
      exit(-1);  
   }
   if((n = strtol(argv[2], NULL, 10)) == LONG_MIN || n == LONG_MAX)
   {
      fprintf(stderr, "STRTOL ERROR\n");
      exit(-1);
   }

   //Если количество эл-тов массива не делится нацело на количество потоков
   if(thread_count == 0) { usage(argv[0]); }
   if (n % thread_count != 0) { usage(argv[0]); }
   if ((thread_count & (thread_count-1)) != 0) { usage(argv[0]); }

   *gen_list_p = 0;
   *output_list_p = 0;

   if (argc == 4) 
   {
      char c1 = argv[3][0];
      if (c1 == 'g') { *gen_list_p = 1; }
      else { *output_list_p = 1; }
   } 
   else if (argc == 5) 
   {
      *gen_list_p = 1;
      *output_list_p = 1;
   }
}

//функция генрации массива
void gen_list(int list[], int n) 
{
   srandom(1);
   for (int i = 0; i < n; ++i) { list[i] = random() % MYMAX; }
}

//функция чтения массива
void read_list(char prompt[], int list[], int n) 
{
   printf("%s\n", prompt);
   for (int i = 0; i < n; ++i) { scanf("%d", &list[i]); }
}

//функция печати на экран массива
void print_list(char title[], int list[], int n) 
{
   printf("%s:\n", title);
   for (int i = 0; i < n; ++i) { printf("%d\n", list[i]); }
}


int main(int argc, char* argv[]) 
{
   pthread_t* thread_handles = NULL;
   int g_list = 0; 
   int output_list = 0;
   long long start = 0;
   long long finish = 0;
   get_args(argc, argv, &g_list, &output_list);
   if((thread_handles = malloc(thread_count*sizeof(pthread_t))) == NULL)
   {
      fprintf(stderr, "MALLOC ERROR\n");
      exit(-1);
   }
   if(pthread_mutex_init(&bar_mutex, NULL) != 0)
   {
      fprintf(stderr, "MUTEX INIT ERROR\n");
      exit(-1);
   }
   if(pthread_cond_init(&bar_cond, NULL) != 0)
   {
      fprintf(stderr, "COND INIT ERROR\n");
      exit(-1);
   }
   if((list1 = malloc(n*sizeof(int))) == NULL)
   {
      fprintf(stderr, "MALLOC ERROR\n");
      exit(-1);  
   }
   if((list2 = malloc(n*sizeof(int))) == NULL)
   {
      fprintf(stderr, "MALLOC ERROR\n");
      exit(-1);
   }
   l_a = list1;
   l_b = list2;

   if (g_list) { gen_list(list1, n); }
   else { read_list("Enter the list", list1, n); }

   if (output_list) { print_list("The input list is", list1, n); }

   start = current_timestamp();
   for (long thread = 0; thread < thread_count; ++thread)
   {
      if(pthread_create(&thread_handles[thread], NULL, bitonic_sort, (void*)thread) != 0)
      {
         fprintf(stderr, "PTHREAD CREATE ERROR\n");
         exit(-1);
      }
   }

   for (long thread = 0; thread < thread_count; ++thread)
   {
      if(pthread_join(thread_handles[thread], NULL) != 0)
      {
         fprintf(stderr, "PTHREAD JOIN ERROR\n");
         exit(-1);
      }
   }
   finish = current_timestamp();

   if (output_list) { print_list("The sorted list is", l_a, n); }

   printf("Elapsed time = %lld milliseconds\n", finish - start);

   free(list1);
   free(list2);
   if(pthread_mutex_destroy(&bar_mutex) != 0)
   {
      fprintf(stderr, "MUTEX DESTROY ERROR\n");
      exit(-1);
   }
   if(pthread_cond_destroy(&bar_cond) != 0)
   {
      fprintf(stderr, "COND DESTROY ERROR\n");
      exit(-1);
   }
   free(thread_handles);
   return 0;
}
