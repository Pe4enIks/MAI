#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <limits.h>
#include "sys/time.h"
#include "bitonic_sort.h"

//Глобальные переменные
extern int thread_count;
extern int bar_count;
extern pthread_mutex_t bar_mutex;
extern pthread_cond_t bar_cond;
extern int n;
extern int* list1;
extern int* list2;
extern int* l_a;
extern int* l_b;

//Функция сортировки
void* bitonic_sort(void* rank) 
{
   long tmp = (long)rank;
   int my_rank = (int)tmp;
   int local_n = n/thread_count;
   int my_first = my_rank*local_n;
   unsigned th_count, and_bit, dim;

   qsort(list1 + my_first, local_n, sizeof(int), compare);
   barrier();

   for (th_count = 2, and_bit = 2, dim = 1; th_count <= thread_count;
      th_count <<= 1, and_bit <<= 1, ++dim) 
   {
      if ((my_rank & and_bit) == 0)
         { bitonic_sort_incr(th_count, dim, my_first, local_n, my_rank); }
      else
         { bitonic_sort_decr(th_count, dim, my_first, local_n, my_rank); }
   }
   return NULL;
}

void bitonic_sort_incr(int th_count, int dim, int my_first, int local_n, int my_rank)
 {
   int partner = 0;;
   int* tmp = NULL;
   unsigned eor_bit = 1 << (dim - 1);

   for (int stage = 0; stage < dim; ++stage) 
   {
      partner = my_rank ^ eor_bit;
      if (my_rank < partner)
         { merge_split_lo(my_rank, my_first, local_n, partner); }
      else
         { merge_split_hi(my_rank, my_first, local_n, partner); }
      eor_bit >>= 1;
      barrier();
      if (my_rank == 0) 
      {
         tmp = l_a;
         l_a = l_b;
         l_b = tmp;
      }
      barrier();
   }

}
void bitonic_sort_decr(int th_count, int dim, int my_first, int local_n, int my_rank)
{
   int partner = 0;
   int* tmp = NULL;
   unsigned eor_bit = 1 << (dim - 1);

   for (int stage = 0; stage < dim; ++stage) 
   {
      partner = my_rank ^ eor_bit;
      if (my_rank > partner)
         { merge_split_lo(my_rank, my_first, local_n, partner); }
      else
         { merge_split_hi(my_rank, my_first, local_n, partner); }
      eor_bit >>= 1;
      barrier();
      if (my_rank == 0) 
      {
         tmp = l_a;
         l_a = l_b;
         l_b = tmp;
      }
      barrier();
   }

}

//Функция верхнего слияния
void merge_split_lo(int my_rank, int my_first, int local_n, int partner)
 {
   int ai = my_first;
   int bi = my_first;
   int xi = partner*local_n;

   for (int i = 0; i < local_n; ++i)
   {
      if (l_a[ai] <= l_a[xi]) { l_b[bi++] = l_a[ai++]; }
      else { l_b[bi++] = l_a[xi++]; }
   }
}

//Функция нижнего слияния
void merge_split_hi(int my_rank, int my_first, int local_n, int partner)
 {
   int ai = my_first + local_n - 1;
   int bi = my_first + local_n - 1;
   int xi = (partner+1)*local_n - 1;

   for (int i = 0; i < local_n; ++i)
   {
      if (l_a[ai] >= l_a[xi]) { l_b[bi--] = l_a[ai--]; }
      else { l_b[bi--] = l_a[xi--]; }
   }
}

//Барьер
void barrier(void) 
{
   if(pthread_mutex_lock(&bar_mutex) != 0)
   {
      fprintf(stderr, "MUTEX LOCK ERROR\n");
      exit(-1);
   }
   ++bar_count;
   if (bar_count == thread_count) 
   {
      bar_count = 0;
      if(pthread_cond_broadcast(&bar_cond) != 0)
      {
         fprintf(stderr, "COND BROADCAST ERROR\n");
         exit(-1);
      }
   } 
   else { while (pthread_cond_wait(&bar_cond, &bar_mutex) != 0); }
   if(pthread_mutex_unlock(&bar_mutex) != 0)
   {
      fprintf(stderr, "MUTEX UNLOCK ERROR\n");
      exit(-1);
   }
}

//Функция сравнения
int compare(void const* x_p, void const* y_p) 
{
   int x = *((int*)x_p);
   int y = *((int*)y_p);
   if (x < y) { return -1; }
   else if (x == y) { return 0; }
   else { return 1; }
}