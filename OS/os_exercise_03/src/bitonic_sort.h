#ifndef __BITONICSORT_
#define __BITONICSORT_
void* bitonic_sort(void*);
void bitonic_sort_incr(int,int,int,int,int);
void bitonic_sort_decr(int,int,int,int,int);
void merge_split_lo(int,int,int,int);
void merge_split_hi(int,int,int,int);
void barrier(void);
int compare(void const*, void const*);
#endif
