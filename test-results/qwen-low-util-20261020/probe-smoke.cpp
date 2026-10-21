#include <cuda_runtime_api.h>
#include <unistd.h>
int main(){int d=-1;for(int i=0;i<100;++i)if(cudaGetDevice(&d)!=cudaSuccess)return 2;usleep(1100000);if(cudaGetDevice(&d)!=cudaSuccess)return 3;}
