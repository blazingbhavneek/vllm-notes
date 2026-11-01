#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#define CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(e));exit(1);} } while(0)
__global__ void init(float* a,float* b,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){a[i]=(i%251)/256.f;b[i]=(i%127)/128.f;}}
__global__ void probe(const float* a,const float* b,float* c,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)c[i]=a[i]+b[i];}
int main(){
 int count;CHECK(cudaGetDeviceCount(&count));CHECK(cudaSetDevice(0));
 cudaDeviceProp p;CHECK(cudaGetDeviceProperties(&p,0));char bus[32];CHECK(cudaDeviceGetPCIBusId(bus,sizeof(bus),0));
 printf("device=%s pci=%s visible_devices=%d cc=%d.%d sms=%d warp=%d max_threads_sm=%d regs_sm=%d shared_optin=%zu\n",p.name,bus,count,p.major,p.minor,p.multiProcessorCount,p.warpSize,p.maxThreadsPerMultiProcessor,p.regsPerMultiprocessor,p.sharedMemPerBlockOptin);
 int blocks;CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,probe,256,0));
 printf("probe_theoretical_occupancy=%.1f%% blocks_per_sm=%d (not model achieved occupancy)\n",100.*blocks*256/p.maxThreadsPerMultiProcessor,blocks);
 const int n=1<<25;float *a,*b,*c;CHECK(cudaMalloc(&a,n*sizeof(float)));CHECK(cudaMalloc(&b,n*sizeof(float)));CHECK(cudaMalloc(&c,n*sizeof(float)));
 init<<<(n+255)/256,256>>>(a,b,n);CHECK(cudaGetLastError());
 for(int i=0;i<4;i++)probe<<<(n+255)/256,256>>>(a,b,c,n);CHECK(cudaDeviceSynchronize());
 cudaEvent_t start,end;CHECK(cudaEventCreate(&start));CHECK(cudaEventCreate(&end));
 CHECK(cudaEventRecord(start));for(int i=0;i<20;i++)probe<<<(n+255)/256,256>>>(a,b,c,n);CHECK(cudaEventRecord(end));CHECK(cudaEventSynchronize(end));float ms;CHECK(cudaEventElapsedTime(&ms,start,end));
 float h[256];CHECK(cudaMemcpy(h,c,sizeof(h),cudaMemcpyDeviceToHost));for(int i=0;i<256;i++){float expected=(i%251)/256.f+(i%127)/128.f;if(h[i]!=expected){fprintf(stderr,"mismatch %d\n",i);return 2;}}
 printf("validation=PASS ms_per_launch=%.6f effective_GB_s=%.2f bytes_per_launch=%zu\n",ms/20,3.*n*sizeof(float)/(ms/20*1e6),size_t(3)*n*sizeof(float));
 CHECK(cudaFree(a));CHECK(cudaFree(b));CHECK(cudaFree(c));CHECK(cudaEventDestroy(start));CHECK(cudaEventDestroy(end));return 0;
}
