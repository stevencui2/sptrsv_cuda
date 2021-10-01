#ifndef _SPTRSV_SYNCFREE_CUDA_
#define _SPTRSV_SYNCFREE_CUDA_

#include "common.h"
#include "utils.h"
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <cuda_fp16.h>

__global__ void sptrsv_syncfree_cuda_analyser(const int *d_cscRowIdx,
                                              const int m,
                                              const int nnz,
                                              int *d_graphInDegree)
{
    const int global_id = blockIdx.x * blockDim.x + threadIdx.x; //get_global_id(0);
    if (global_id < nnz)
    {
        atomicAdd(&d_graphInDegree[d_cscRowIdx[global_id]], 1);
    }
}
__global__ void sptrsvCSRAnalyser(const int *d_csrRowPtr,
                                  const int m,
                                  int *d_graphInDegree)
{
    const int gidx = blockIdx.x * blockDim.x + threadIdx.x; //get_global_id(0);
    if (gidx < m)
    {
        d_graphInDegree[gidx] = d_csrRowPtr[gidx + 1] - d_csrRowPtr[gidx];
    }
}
template <typename T>
__global__ void sptrsvCSRSyncfree(
    const int *__restrict__ d_csrRowPtr,
    const int *__restrict__ d_csrColInd,
    const T *__restrict__ d_csrVal,
    const T *__restrict__ d_RHS,
    int *d_graphInDegree,
    const int substitution,
    const int m,
    T *d_y)
{
    const int gIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int gXIdx = gIdx / WARP_SIZE;
    if (gXIdx >= m)
        return;

    // substitution is forward or backward
    gXIdx = substitution == SUBSTITUTION_FORWARD ? gXIdx : m - 1 - gXIdx;

    volatile __shared__ int s_graphInDegree[WARP_PER_BLOCK];
    volatile __shared__ T s_left_sum[WARP_PER_BLOCK];

    // Initialize
    const int localWarpIdx = threadIdx.x / WARP_SIZE;
    const int laneIdx = (WARP_SIZE - 1) & threadIdx.x;

    int starting_x = (gIdx / (WARP_PER_BLOCK * WARP_SIZE)) * WARP_PER_BLOCK;
    starting_x = substitution == SUBSTITUTION_FORWARD ? starting_x : m - 1 - starting_x;

    // Prefetch
    //const int pos = substitution == SUBSTITUTION_FORWARD ? d_cscColPtr[gXIdx] : d_cscColPtr[gXIdx + 1] - 1;
    const int pos = substitution == SUBSTITUTION_FORWARD ? d_csrRowPtr[gXIdx + 1] - 1 : d_csrRowPtr[gXIdx];
    const T coef = (T)1 / d_csrVal[pos];

    if (threadIdx.x < WARP_PER_BLOCK)
    {
        s_graphInDegree[threadIdx.x] = 1;
        s_left_sum[threadIdx.x] = 0;
    }
    __syncthreads();

    clock_t start;
    // Consumer
    do
    {
        start = clock();
    } while (s_graphInDegree[localWarpIdx] != d_graphInDegree[gXIdx]);

    T xi = s_left_sum[localWarpIdx];
    xi = (d_RHS[gXIdx] - xi) * coef;

    // Producer
    //const int start_ptr = substitution == SUBSTITUTION_FORWARD ? d_cscColPtr[gXIdx] + 1 : d_cscColPtr[gXIdx];
    const int start_ptr = substitution == SUBSTITUTION_FORWARD ? d_csrRowPtr[gXIdx] : d_csrRowPtr[gXIdx] + 1;
    //const int stop_ptr = substitution == SUBSTITUTION_FORWARD ? d_cscColPtr[gXIdx + 1] : d_cscColPtr[gXIdx + 1] - 1;
    const int stop_ptr = substitution == SUBSTITUTION_FORWARD ? d_csrRowPtr[gXIdx + 1] - 1 : d_csrRowPtr[gXIdx + 1];
    for (int jj = start_ptr + laneIdx; jj < stop_ptr; jj += WARP_SIZE)
    {
        const int j = substitution == SUBSTITUTION_FORWARD ? jj : stop_ptr - 1 - (jj - start_ptr);
        const int colIdx = d_csrColInd[j];
        const int pos = substitution == SUBSTITUTION_FORWARD ? colIdx - starting_x : starting_x - colIdx;
        atomicAdd((T *)&s_left_sum[pos], xi * d_csrVal[j]);
        __threadfence_block();
        atomicAdd((int *)&s_graphInDegree[pos], 1);
    }

    //finish
    if (!laneIdx)
        d_y[gXIdx] = xi;
}

template <typename T>
__global__ void ELMR(int *__restrict__ rowptr, int *__restrict__ colind, T *__restrict__ val, T *__restrict__ f, int n, volatile T *x,
                     volatile int *ready)
{
    int wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & (WARP_SIZE - 1);
    if (wid >= n)
        return;
    int p = 0, q = 0, i = -1;
    T sum = 0.0, diag;
    if (lane < 2)
    {
        i = wid;
        p = rowptr[i + lane];
    }
    q = __shfl_sync(-1, p, 1);
    p = __shfl_sync(-1, p, 0);
    if (lane == 0)
    {
        sum = f[i];
        diag = val[rowptr[wid + 1] - 1];
    }
    for (p += lane; p < q; p += WARP_SIZE)
    {
        while (ready[colind[p]] == 0)
            ;
        sum -= val[p] * x[colind[p]];
    }
#pragma unroll // parallel reduction
    for (int d = WARP_SIZE / 2; d > 0; d >>= 1)
        sum += __shfl_down_sync(-1, sum, d);
    if (lane == 0)
    {
        x[i] = sum / diag;
        __threadfence();
        ready[i] = 1;
    }
}


template <typename T>
__global__ void sptrsvElementschedulingCSCV2(
	const int* __restrict__ d_cscColPtr,
	const int* __restrict__ d_cscRowInd,
	const T* __restrict__ d_cscVal,
	const T* __restrict__ d_RHS, 
	int* d_graphInDegree,
	const int substitution,
	const int m, T* d_y, int* jlev)
{
	int warpIdx = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
	int laneIdx = threadIdx.x & (WARP_SIZE - 1);
	if (warpIdx >= m)
		return;

	// substitution is forward or backward
	warpIdx = substitution == SUBSTITUTION_FORWARD ? warpIdx : m - 1 - warpIdx;

	int start_ptr = 0, stop_ptr = 0, pos = -1;
	T coef = 0.0;

	volatile __shared__ int s_graphInDegree[WARP_PER_BLOCK];
	if (threadIdx.x < WARP_PER_BLOCK) { s_graphInDegree[threadIdx.x] = 1; }
	__syncthreads();

    pos = substitution == SUBSTITUTION_FORWARD ? jlev[warpIdx] : jlev[warpIdx + 1] - 1;
    start_ptr = substitution == SUBSTITUTION_FORWARD ? d_cscColPtr[pos] + 1 : d_cscColPtr[pos];
	stop_ptr = substitution == SUBSTITUTION_FORWARD ? d_cscColPtr[pos + 1] : d_cscColPtr[pos + 1] - 1;

	if (laneIdx == 0)
	{
		coef = 1.0 / d_cscVal[pos];
		d_y[pos] = d_RHS[pos];
		while (s_graphInDegree[pos] != d_graphInDegree[pos]);
		d_y[pos] = coef = d_y[pos] * coef;
	}
	coef = __shfl_sync(-1, coef, 0);
	for (int jj =start_ptr+ laneIdx; jj < stop_ptr; jj += WARP_SIZE)
	{
		jj = substitution == SUBSTITUTION_FORWARD ? jj : stop_ptr - 1 - (jj - start_ptr);
		int rowIdx = d_cscRowInd[jj];
		atomicAdd(&d_y[rowIdx], -coef * d_cscVal[jj]);
		__threadfence();
		atomicAdd((int*)&s_graphInDegree[rowIdx], 1);
	}
}

__global__ void sptrsv_syncfree_cuda_executor(const int *__restrict__ d_cscColPtr,
                                              const int *__restrict__ d_cscRowIdx,
                                              const VALUE_TYPE *__restrict__ d_cscVal,
                                              int *d_graphInDegree,
                                              VALUE_TYPE *d_left_sum,
                                              const int m,
                                              const int substitution,
                                              const VALUE_TYPE *__restrict__ d_b,
                                              VALUE_TYPE *d_x,
                                              int *d_while_profiler)
{
    const int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int global_x_id = global_id / WARP_SIZE;
    if (global_x_id >= m)
        return;

    // substitution is forward or backward
    global_x_id = substitution == SUBSTITUTION_FORWARD ? global_x_id : m - 1 - global_x_id;

    volatile __shared__ int s_graphInDegree[WARP_PER_BLOCK];
    volatile __shared__ VALUE_TYPE s_left_sum[WARP_PER_BLOCK];

    // Initialize
    const int local_warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = (WARP_SIZE - 1) & threadIdx.x;
    int starting_x = (global_id / (WARP_PER_BLOCK * WARP_SIZE)) * WARP_PER_BLOCK;
    starting_x = substitution == SUBSTITUTION_FORWARD ? starting_x : m - 1 - starting_x;

    // Prefetch
    const int pos = substitution == SUBSTITUTION_FORWARD ? d_cscColPtr[global_x_id] : d_cscColPtr[global_x_id + 1] - 1;
    const VALUE_TYPE coef = (VALUE_TYPE)1 / d_cscVal[pos];
    //asm("prefetch.global.L2 [%0];"::"d"(d_cscVal[d_cscColPtr[global_x_id] + 1 + lane_id]));
    //asm("prefetch.global.L2 [%0];"::"r"(d_cscRowIdx[d_cscColPtr[global_x_id] + 1 + lane_id]));

    if (threadIdx.x < WARP_PER_BLOCK)
    {
        s_graphInDegree[threadIdx.x] = 1;
        s_left_sum[threadIdx.x] = 0;
    }
    __syncthreads();

    clock_t start;
    // Consumer
    do
    {
        start = clock();
    } while (s_graphInDegree[local_warp_id] != d_graphInDegree[global_x_id]);

    //// Consumer
    //int graphInDegree;
    //do {
    //    //bypass Tex cache and avoid other mem optimization by nvcc/ptxas
    //    asm("ld.global.u32 %0, [%1];" : "=r"(graphInDegree),"=r"(d_graphInDegree[global_x_id]) :: "memory");
    //}
    //while (s_graphInDegree[local_warp_id] != graphInDegree );

    VALUE_TYPE xi = d_left_sum[global_x_id] + s_left_sum[local_warp_id];
    xi = (d_b[global_x_id] - xi) * coef;

    // Producer
    const int start_ptr = substitution == SUBSTITUTION_FORWARD ? d_cscColPtr[global_x_id] + 1 : d_cscColPtr[global_x_id];
    const int stop_ptr = substitution == SUBSTITUTION_FORWARD ? d_cscColPtr[global_x_id + 1] : d_cscColPtr[global_x_id + 1] - 1;
    for (int jj = start_ptr + lane_id; jj < stop_ptr; jj += WARP_SIZE)
    {
        const int j = substitution == SUBSTITUTION_FORWARD ? jj : stop_ptr - 1 - (jj - start_ptr);
        const int rowIdx = d_cscRowIdx[j];
        const bool cond = substitution == SUBSTITUTION_FORWARD ? (rowIdx < starting_x + WARP_PER_BLOCK) : (rowIdx > starting_x - WARP_PER_BLOCK);
        if (cond)
        {
            const int pos = substitution == SUBSTITUTION_FORWARD ? rowIdx - starting_x : starting_x - rowIdx;
            atomicAdd((VALUE_TYPE *)&s_left_sum[pos], xi * d_cscVal[j]);
            __threadfence_block();
            atomicAdd((int *)&s_graphInDegree[pos], 1);
        }
        else
        {
            atomicAdd(&d_left_sum[rowIdx], xi * d_cscVal[j]);
            __threadfence();
            atomicSub(&d_graphInDegree[rowIdx], 1);
        }
    }

    //finish
    if (!lane_id)
        d_x[global_x_id] = xi;
}

__global__ void sptrsv_syncfree_cuda_shared(const int *__restrict__ d_cscColPtr,
                                            const int *__restrict__ d_cscRowIdx,
                                            const VALUE_TYPE *__restrict__ d_cscVal,
                                            const VALUE_TYPE *__restrict__ d_b,
                                            int *d_graphInDegree,
                                            const int substitution,
                                            const int m,
                                            VALUE_TYPE *d_x)
                                            {

    const int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int global_x_id = global_id / WARP_SIZE;
    if (global_x_id >= m)
        return;

    // substitution is forward or backward
    global_x_id = substitution == SUBSTITUTION_FORWARD ? global_x_id : m - 1 - global_x_id;

    volatile __shared__ int s_graphInDegree[WARP_PER_BLOCK];
    volatile __shared__ VALUE_TYPE s_left_sum[WARP_PER_BLOCK];

    // Initialize
    const int local_warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = (WARP_SIZE - 1) & threadIdx.x;

    int starting_x = (global_id / (WARP_PER_BLOCK * WARP_SIZE)) * WARP_PER_BLOCK;
    starting_x = substitution == SUBSTITUTION_FORWARD ? starting_x : m - 1 - starting_x;

    // Prefetch
    const int pos = substitution == SUBSTITUTION_FORWARD ? d_cscColPtr[global_x_id] : d_cscColPtr[global_x_id + 1] - 1;
    const VALUE_TYPE coef = (VALUE_TYPE)1 / d_cscVal[pos];
    //asm("prefetch.global.L2 [%0];"::"d"(d_cscVal[d_cscColPtr[global_x_id] + 1 + lane_id]));
    //asm("prefetch.global.L2 [%0];"::"r"(d_cscRowIdx[d_cscColPtr[global_x_id] + 1 + lane_id]));

    if (threadIdx.x < WARP_PER_BLOCK)
    {
        s_graphInDegree[threadIdx.x] = 1;
        s_left_sum[threadIdx.x] = 0;
    }
    __syncthreads();

    clock_t start;
    // Consumer
    do
    {
        start = clock();
    } while (s_graphInDegree[local_warp_id]= d_graphInDegree[global_x_id]);

    //// Consumer
    //int graphInDegree;
    //do {
    //    //bypass Tex cache and avoid other mem optimization by nvcc/ptxas
    //    asm("ld.global.u32 %0, [%1];" : "=r"(graphInDegree),"=r"(d_graphInDegree[global_x_id]) :: "memory");
    //}
    //while (s_graphInDegree[local_warp_id] != graphInDegree );

    VALUE_TYPE xi = s_left_sum[local_warp_id];
    xi = (d_b[global_x_id] - xi) * coef;

    // Producer
    const int start_ptr = substitution == SUBSTITUTION_FORWARD ? d_cscColPtr[global_x_id] + 1 : d_cscColPtr[global_x_id];
    const int stop_ptr = substitution == SUBSTITUTION_FORWARD ? d_cscColPtr[global_x_id + 1] : d_cscColPtr[global_x_id + 1] - 1;
    for (int jj = start_ptr + lane_id; jj < stop_ptr; jj += WARP_SIZE)
    {
        const int j = substitution == SUBSTITUTION_FORWARD ? jj : stop_ptr - 1 - (jj - start_ptr);
        const int rowIdx = d_cscRowIdx[j];
        const int pos = substitution == SUBSTITUTION_FORWARD ? rowIdx - starting_x : starting_x - rowIdx;
        atomicAdd((VALUE_TYPE *)&s_left_sum[pos], xi * d_cscVal[j]);
        __threadfence_block();
        atomicAdd((int *)&s_graphInDegree[pos], 1);
    }

    //finish
    if (!lane_id)
        d_x[global_x_id] = xi;
}
__global__ void sptrsv_syncfree_cuda_executor_update(const int *d_cscColPtr,
                                                     const int *d_cscRowIdx,
                                                     const VALUE_TYPE *d_cscVal,
                                                     int *d_graphInDegree,
                                                     VALUE_TYPE *d_left_sum,
                                                     const int m,
                                                     const int substitution,
                                                     const VALUE_TYPE *d_b,
                                                     VALUE_TYPE *d_x,
                                                     int *d_while_profiler,
                                                     int *d_id_extractor)
{
    const int global_id = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize
    const int local_warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = (WARP_SIZE - 1) & threadIdx.x;
    int global_x_id = 0;
    if (!lane_id)
        global_x_id = atomicAdd(d_id_extractor, 1);
    global_x_id = __shfl(global_x_id, 0);

    if (global_x_id >= m)
        return;

    // substitution is forward or backward
    global_x_id = substitution == SUBSTITUTION_FORWARD ? global_x_id : m - 1 - global_x_id;

    // Prefetch
    const int pos = substitution == SUBSTITUTION_FORWARD ? d_cscColPtr[global_x_id] : d_cscColPtr[global_x_id + 1] - 1;
    const VALUE_TYPE coef = (VALUE_TYPE)1 / d_cscVal[pos];
    //asm("prefetch.global.L2 [%0];"::"d"(d_cscVal[d_cscColPtr[global_x_id] + 1 + lane_id]));
    //asm("prefetch.global.L2 [%0];"::"r"(d_cscRowIdx[d_cscColPtr[global_x_id] + 1 + lane_id]));

    // Consumer
    do
    {
        __threadfence_block();
    } while (d_graphInDegree[global_x_id] != 1);

    VALUE_TYPE xi = d_left_sum[global_x_id];
    xi = (d_b[global_x_id] - xi) * coef;

    // Producer
    const int start_ptr = substitution == SUBSTITUTION_FORWARD ? d_cscColPtr[global_x_id] + 1 : d_cscColPtr[global_x_id];
    const int stop_ptr = substitution == SUBSTITUTION_FORWARD ? d_cscColPtr[global_x_id + 1] : d_cscColPtr[global_x_id + 1] - 1;
    for (int jj = start_ptr + lane_id; jj < stop_ptr; jj += WARP_SIZE)
    {
        const int j = substitution == SUBSTITUTION_FORWARD ? jj : stop_ptr - 1 - (jj - start_ptr);
        const int rowIdx = d_cscRowIdx[j];

        atomicAdd(&d_left_sum[rowIdx], xi * d_cscVal[j]);
        __threadfence();
        atomicSub(&d_graphInDegree[rowIdx], 1);
    }

    //finish
    if (!lane_id)
        d_x[global_x_id] = xi;
}

__global__ void sptrsm_syncfree_cuda_executor(const int *__restrict__ d_cscColPtr,
                                              const int *__restrict__ d_cscRowIdx,
                                              const VALUE_TYPE *__restrict__ d_cscVal,
                                              int *d_graphInDegree,
                                              VALUE_TYPE *d_left_sum,
                                              const int m,
                                              const int substitution,
                                              const int rhs,
                                              const int opt,
                                              const VALUE_TYPE *__restrict__ d_b,
                                              VALUE_TYPE *d_x,
                                              int *d_while_profiler)
{
    const int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int global_x_id = global_id / WARP_SIZE;
    if (global_x_id >= m)
        return;

    // substitution is forward or backward
    global_x_id = substitution == SUBSTITUTION_FORWARD ? global_x_id : m - 1 - global_x_id;

    // Initialize
    const int lane_id = (WARP_SIZE - 1) & threadIdx.x;

    // Prefetch
    const int pos = substitution == SUBSTITUTION_FORWARD ? d_cscColPtr[global_x_id] : d_cscColPtr[global_x_id + 1] - 1;
    const VALUE_TYPE coef = (VALUE_TYPE)1 / d_cscVal[pos];
    //asm("prefetch.global.L2 [%0];"::"d"(d_cscVal[d_cscColPtr[global_x_id] + 1 + lane_id]));
    //asm("prefetch.global.L2 [%0];"::"r"(d_cscRowIdx[d_cscColPtr[global_x_id] + 1 + lane_id]));

    clock_t start;
    // Consumer
    do
    {
        start = clock();
    } while (1 != d_graphInDegree[global_x_id]);

    //// Consumer
    //int graphInDegree;
    //do {
    //    //bypass Tex cache and avoid other mem optimization by nvcc/ptxas
    //    asm("ld.global.u32 %0, [%1];" : "=r"(graphInDegree),"=r"(d_graphInDegree[global_x_id]) :: "memory");
    //}
    //while (1 != graphInDegree );

    for (int k = lane_id; k < rhs; k += WARP_SIZE)
    {
        const int pos = global_x_id * rhs + k;
        d_x[pos] = (d_b[pos] - d_left_sum[pos]) * coef;
    }

    // Producer
    const int start_ptr = substitution == SUBSTITUTION_FORWARD ? d_cscColPtr[global_x_id] + 1 : d_cscColPtr[global_x_id];
    const int stop_ptr = substitution == SUBSTITUTION_FORWARD ? d_cscColPtr[global_x_id + 1] : d_cscColPtr[global_x_id + 1] - 1;

    if (opt == OPT_WARP_NNZ)
    {
        for (int jj = start_ptr + lane_id; jj < stop_ptr; jj += WARP_SIZE)
        {
            const int j = substitution == SUBSTITUTION_FORWARD ? jj : stop_ptr - 1 - (jj - start_ptr);
            const int rowIdx = d_cscRowIdx[j];
            for (int k = 0; k < rhs; k++)
                atomicAdd(&d_left_sum[rowIdx * rhs + k], d_x[global_x_id * rhs + k] * d_cscVal[j]);
            __threadfence();
            atomicSub(&d_graphInDegree[rowIdx], 1);
        }
    }
    else if (opt == OPT_WARP_RHS)
    {
        for (int jj = start_ptr; jj < stop_ptr; jj++)
        {
            const int j = substitution == SUBSTITUTION_FORWARD ? jj : stop_ptr - 1 - (jj - start_ptr);
            const int rowIdx = d_cscRowIdx[j];
            for (int k = lane_id; k < rhs; k += WARP_SIZE)
                atomicAdd(&d_left_sum[rowIdx * rhs + k], d_x[global_x_id * rhs + k] * d_cscVal[j]);
            __threadfence();
            if (!lane_id)
                atomicSub(&d_graphInDegree[rowIdx], 1);
        }
    }
    else if (opt == OPT_WARP_AUTO)
    {
        const int len = stop_ptr - start_ptr;

        if ((len <= rhs || rhs > 16) && len < 2048)
        {
            for (int jj = start_ptr; jj < stop_ptr; jj++)
            {
                const int j = substitution == SUBSTITUTION_FORWARD ? jj : stop_ptr - 1 - (jj - start_ptr);
                const int rowIdx = d_cscRowIdx[j];
                for (int k = lane_id; k < rhs; k += WARP_SIZE)
                    atomicAdd(&d_left_sum[rowIdx * rhs + k], d_x[global_x_id * rhs + k] * d_cscVal[j]);
                __threadfence();
                if (!lane_id)
                    atomicSub(&d_graphInDegree[rowIdx], 1);
            }
        }
        else
        {
            for (int jj = start_ptr + lane_id; jj < stop_ptr; jj += WARP_SIZE)
            {
                const int j = substitution == SUBSTITUTION_FORWARD ? jj : stop_ptr - 1 - (jj - start_ptr);
                const int rowIdx = d_cscRowIdx[j];
                for (int k = 0; k < rhs; k++)
                    atomicAdd(&d_left_sum[rowIdx * rhs + k], d_x[global_x_id * rhs + k] * d_cscVal[j]);
                __threadfence();
                atomicSub(&d_graphInDegree[rowIdx], 1);
            }
        }
    }
}

__global__ void sptrsm_syncfree_cuda_executor_update(const int *__restrict__ d_cscColPtr,
                                                     const int *__restrict__ d_cscRowIdx,
                                                     const VALUE_TYPE *__restrict__ d_cscVal,
                                                     int *d_graphInDegree,
                                                     VALUE_TYPE *d_left_sum,
                                                     const int m,
                                                     const int substitution,
                                                     const int rhs,
                                                     const int opt,
                                                     const VALUE_TYPE *__restrict__ d_b,
                                                     VALUE_TYPE *d_x,
                                                     int *d_while_profiler,
                                                     int *d_id_extractor)
{
    const int global_id = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize
    const int local_warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = (WARP_SIZE - 1) & threadIdx.x;
    int global_x_id = 0;
    if (!lane_id)
        global_x_id = atomicAdd(d_id_extractor, 1);
    global_x_id = __shfl(global_x_id, 0);

    if (global_x_id >= m)
        return;

    // substitution is forward or backward
    global_x_id = substitution == SUBSTITUTION_FORWARD ? global_x_id : m - 1 - global_x_id;

    // Prefetch
    const int pos = substitution == SUBSTITUTION_FORWARD ? d_cscColPtr[global_x_id] : d_cscColPtr[global_x_id + 1] - 1;
    const VALUE_TYPE coef = (VALUE_TYPE)1 / d_cscVal[pos];
    //asm("prefetch.global.L2 [%0];"::"d"(d_cscVal[d_cscColPtr[global_x_id] + 1 + lane_id]));
    //asm("prefetch.global.L2 [%0];"::"r"(d_cscRowIdx[d_cscColPtr[global_x_id] + 1 + lane_id]));

    // Consumer
    do
    {
        __threadfence_block();
    } while (1 != d_graphInDegree[global_x_id]);

    //// Consumer
    //int graphInDegree;
    //do {
    //    //bypass Tex cache and avoid other mem optimization by nvcc/ptxas
    //    asm("ld.global.u32 %0, [%1];" : "=r"(graphInDegree),"=r"(d_graphInDegree[global_x_id]) :: "memory");
    //}
    //while (1 != graphInDegree );

    for (int k = lane_id; k < rhs; k += WARP_SIZE)
    {
        const int pos = global_x_id * rhs + k;
        d_x[pos] = (d_b[pos] - d_left_sum[pos]) * coef;
    }

    // Producer
    const int start_ptr = substitution == SUBSTITUTION_FORWARD ? d_cscColPtr[global_x_id] + 1 : d_cscColPtr[global_x_id];
    const int stop_ptr = substitution == SUBSTITUTION_FORWARD ? d_cscColPtr[global_x_id + 1] : d_cscColPtr[global_x_id + 1] - 1;

    if (opt == OPT_WARP_NNZ)
    {
        for (int jj = start_ptr + lane_id; jj < stop_ptr; jj += WARP_SIZE)
        {
            const int j = substitution == SUBSTITUTION_FORWARD ? jj : stop_ptr - 1 - (jj - start_ptr);
            const int rowIdx = d_cscRowIdx[j];
            for (int k = 0; k < rhs; k++)
                atomicAdd(&d_left_sum[rowIdx * rhs + k], d_x[global_x_id * rhs + k] * d_cscVal[j]);
            __threadfence();
            atomicSub(&d_graphInDegree[rowIdx], 1);
        }
    }
    else if (opt == OPT_WARP_RHS)
    {
        for (int jj = start_ptr; jj < stop_ptr; jj++)
        {
            const int j = substitution == SUBSTITUTION_FORWARD ? jj : stop_ptr - 1 - (jj - start_ptr);
            const int rowIdx = d_cscRowIdx[j];
            for (int k = lane_id; k < rhs; k += WARP_SIZE)
                atomicAdd(&d_left_sum[rowIdx * rhs + k], d_x[global_x_id * rhs + k] * d_cscVal[j]);
            __threadfence();
            if (!lane_id)
                atomicSub(&d_graphInDegree[rowIdx], 1);
        }
    }
    else if (opt == OPT_WARP_AUTO)
    {
        const int len = stop_ptr - start_ptr;

        if ((len <= rhs || rhs > 16) && len < 2048)
        {
            for (int jj = start_ptr; jj < stop_ptr; jj++)
            {
                const int j = substitution == SUBSTITUTION_FORWARD ? jj : stop_ptr - 1 - (jj - start_ptr);
                const int rowIdx = d_cscRowIdx[j];
                for (int k = lane_id; k < rhs; k += WARP_SIZE)
                    atomicAdd(&d_left_sum[rowIdx * rhs + k], d_x[global_x_id * rhs + k] * d_cscVal[j]);
                __threadfence();
                if (!lane_id)
                    atomicSub(&d_graphInDegree[rowIdx], 1);
            }
        }
        else
        {
            for (int jj = start_ptr + lane_id; jj < stop_ptr; jj += WARP_SIZE)
            {
                const int j = substitution == SUBSTITUTION_FORWARD ? jj : stop_ptr - 1 - (jj - start_ptr);
                const int rowIdx = d_cscRowIdx[j];
                for (int k = 0; k < rhs; k++)
                    atomicAdd(&d_left_sum[rowIdx * rhs + k], d_x[global_x_id * rhs + k] * d_cscVal[j]);
                __threadfence();
                atomicSub(&d_graphInDegree[rowIdx], 1);
            }
        }
    }
}

int sptrsvCSC_syncfree_cuda(const int *cscColPtrTR,
                            const int *cscRowIdxTR,
                            const VALUE_TYPE *cscValTR,
                            const int m,
                            const int n,
                            const int nnzTR,
                            const int substitution,
                            const int rhs,
                            const int opt,
                            VALUE_TYPE *x,
                            const VALUE_TYPE *b,
                            const VALUE_TYPE *x_ref,
                            double *gflops,
                            int * levelItem)
{
    if (m != n)
    {
        printf("This is not a square matrix, return.\n");
        return -1;
    }

    // transfer host mem to device mem
    int *d_cscColPtrTR;
    int *d_cscRowIdxTR;
    int *jlev;
    VALUE_TYPE *d_cscValTR;
    VALUE_TYPE *d_b;
    VALUE_TYPE *d_x;

    // Matrix L
    cudaMalloc((void **)&d_cscColPtrTR, (n + 1) * sizeof(int));
    cudaMalloc((void **)&d_cscRowIdxTR, nnzTR * sizeof(int));
    cudaMalloc((void **)&d_cscValTR, nnzTR * sizeof(VALUE_TYPE));
     cudaMalloc((void **)&jlev, n * sizeof(int));

    cudaMemcpy(d_cscColPtrTR, cscColPtrTR, (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cscRowIdxTR, cscRowIdxTR, nnzTR * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cscValTR, cscValTR, nnzTR * sizeof(VALUE_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(jlev,levelItem,sizeof(int)*n,cudaMemcpyHostToDevice);

    // Vector b
    cudaMalloc((void **)&d_b, m * rhs * sizeof(VALUE_TYPE));
    cudaMemcpy(d_b, b, m * rhs * sizeof(VALUE_TYPE), cudaMemcpyHostToDevice);

    // Vector x
    cudaMalloc((void **)&d_x, n * rhs * sizeof(VALUE_TYPE));
    cudaMemset(d_x, 0, n * rhs * sizeof(VALUE_TYPE));

    //  - cuda syncfree SpTRSV analysis start!
    printf(" - cuda syncfree SpTRSV CSC analysis start!\n");

    struct timeval t1, t2;
    gettimeofday(&t1, NULL);

    // malloc tmp memory to generate in-degree
    int *d_graphInDegree;
    int *d_graphInDegree_backup;
    cudaMalloc((void **)&d_graphInDegree, m * sizeof(int));
    cudaMalloc((void **)&d_graphInDegree_backup, m * sizeof(int));

    int *d_id_extractor;
    cudaMalloc((void **)&d_id_extractor, sizeof(int));

    int num_threads = 128;
    int num_blocks = ceil((double)nnzTR / (double)num_threads);

    for (int i = 0; i < BENCH_REPEAT; i++)
    {
        cudaMemset(d_graphInDegree, 0, m * sizeof(int));
        sptrsv_syncfree_cuda_analyser<<<num_blocks, num_threads>>>(d_cscRowIdxTR, m, nnzTR, d_graphInDegree);
    }
    cudaDeviceSynchronize();

    gettimeofday(&t2, NULL);
    double time_cuda_analysis = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    time_cuda_analysis /= BENCH_REPEAT;

    printf("cuda syncfree SpTRSV CSC analysis on L used %4.9f ms\n", time_cuda_analysis);

    //  - cuda syncfree SpTRSV solve start!
    printf(" - cuda syncfree SpTRSV CSC solve start!\n");

    // malloc tmp memory to collect a partial sum of each row
    VALUE_TYPE *d_left_sum;
    cudaMalloc((void **)&d_left_sum, sizeof(VALUE_TYPE) * m * rhs);

    // backup in-degree array, only used for benchmarking multiple runs
    cudaMemcpy(d_graphInDegree_backup, d_graphInDegree, m * sizeof(int), cudaMemcpyDeviceToDevice);

    // this is for profiling while loop only
    int *d_while_profiler;
    cudaMalloc((void **)&d_while_profiler, sizeof(int) * n);
    cudaMemset(d_while_profiler, 0, sizeof(int) * n);
    int *while_profiler = (int *)malloc(sizeof(int) * n);

    // step 5: solve L*y = x
    double time_cuda_solve = 0;

    for (int i = 0; i < BENCH_REPEAT; i++)
    {
        // get a unmodified in-degree array, only for benchmarking use
        cudaMemcpy(d_graphInDegree, d_graphInDegree_backup, m * sizeof(int), cudaMemcpyDeviceToDevice);
        //cudaMemset(d_graphInDegree, 0, sizeof(int) * m);

        // clear left_sum array, only for benchmarking use
        cudaMemset(d_left_sum, 0, sizeof(VALUE_TYPE) * m * rhs);
        cudaMemset(d_x, 0, sizeof(VALUE_TYPE) * n * rhs);
        cudaMemset(d_id_extractor, 0, sizeof(int));

        gettimeofday(&t1, NULL);

        if (rhs == 1)
        {
            num_threads = WARP_PER_BLOCK * WARP_SIZE;
            //num_threads = 1 * WARP_SIZE;
            num_blocks = ceil((double)m / (double)(num_threads / WARP_SIZE));
                        sptrsvElementschedulingCSCV2<VALUE_TYPE><<<num_blocks, num_threads>>>
                (d_cscColPtrTR, d_cscRowIdxTR, d_cscValTR,d_b,
                 d_graphInDegree,substitution,m,  d_x, jlev);
            // sptrsv_syncfree_cuda_shared<<<num_blocks, num_threads>>>
            //     (d_cscColPtrTR, d_cscRowIdxTR, d_cscValTR,
            //      d_graphInDegree, d_left_sum,
            //      m, substitution, d_b, d_x, d_while_profiler);
        }
        else
        {
            num_threads = 4 * WARP_SIZE;
            num_blocks = ceil((double)m / (double)(num_threads / WARP_SIZE));
            sptrsm_syncfree_cuda_executor_update<<<num_blocks, num_threads>>>(d_cscColPtrTR, d_cscRowIdxTR, d_cscValTR,
                                                                              d_graphInDegree, d_left_sum,
                                                                              m, substitution, rhs, opt,
                                                                              d_b, d_x, d_while_profiler, d_id_extractor);
        }

        cudaDeviceSynchronize();
        gettimeofday(&t2, NULL);

        time_cuda_solve += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    }

    time_cuda_solve /= BENCH_REPEAT;
    double flop = 2 * (double)rhs * (double)nnzTR;

    printf("cuda syncfree SpTRSV CSC solve used %4.9f ms, throughput is %4.2f gflops\n",
           time_cuda_solve, flop / (1e6 * time_cuda_solve));
    *gflops = flop / (1e6 * time_cuda_solve);

    cudaMemcpy(x, d_x, n * rhs * sizeof(VALUE_TYPE), cudaMemcpyDeviceToHost);

    // validate x
    double accuracy = 1e-6;
    double ref = 0.0;
    double res = 0.0;

    for (int i = 0; i < n * rhs; i++)
    {
        ref += abs(x_ref[i]);
        res += abs(x[i] - x_ref[i]);
        //if (x_ref[i] != x[i]) printf ("[%i, %i] x_ref = %f, x = %f\n", i/rhs, i%rhs, x_ref[i], x[i]);
    }
    res = ref == 0 ? res : res / ref;

    if (res < accuracy)
        printf("cuda syncfree SpTRSV CSC executor passed! |x-xref|/|xref| = %8.9e\n", res);
    else
        printf("cuda syncfree SpTRSV CSC executor _NOT_ passed! |x-xref|/|xref| = %8.9e\n", res);

    // profile while loop
    cudaMemcpy(while_profiler, d_while_profiler, n * sizeof(int), cudaMemcpyDeviceToHost);
    long long unsigned int while_count = 0;
    for (int i = 0; i < n; i++)
    {
        while_count += while_profiler[i];
        //printf("while_profiler[%i] = %i\n", i, while_profiler[i]);
    }
    //printf("\nwhile_count= %llu in total, %llu per row/column\n", while_count, while_count/m);

    // step 6: free resources
    free(while_profiler);

cudaFree(jlev);
    cudaFree(d_graphInDegree);
    cudaFree(d_graphInDegree_backup);
    cudaFree(d_id_extractor);
    cudaFree(d_left_sum);
    cudaFree(d_while_profiler);

    cudaFree(d_cscColPtrTR);
    cudaFree(d_cscRowIdxTR);
    cudaFree(d_cscValTR);
    cudaFree(d_b);
    cudaFree(d_x);

    return 0;
}

int sptrsvCSR_syncfree_cuda(const int *csrRowPtrTR,
                            const int *cscColIdxTR,
                            const VALUE_TYPE *csrValTR,
                            const int m,
                            const int n,
                            const int nnzTR,
                            const int substitution,
                            const int rhs,
                            const int opt,
                            VALUE_TYPE *x,
                            const VALUE_TYPE *b,
                            const VALUE_TYPE *x_ref,
                            double *gflops)
{
    if (m != n)
    {
        printf("This is not a square matrix, return.\n");
        return -1;
    }

    // transfer host mem to device mem
    int *d_csrRowPtrTR;
    int *d_csrColIdxTR;
    int *ready;
    int *jlev;
    VALUE_TYPE *d_csrValTR;
    VALUE_TYPE *d_b;
    VALUE_TYPE *d_x;

    // Matrix L
    cudaMalloc((void **)&d_csrRowPtrTR, (n + 1) * sizeof(int));
    cudaMalloc((void **)&d_csrColIdxTR, nnzTR * sizeof(int));
    cudaMalloc((void **)&d_csrValTR, nnzTR * sizeof(VALUE_TYPE));

    cudaMalloc((void **)&ready, nnzTR * sizeof(int));
    cudaMalloc((void **)&jlev, (n + 1) * sizeof(int));
    cudaMemset(ready, 0, nnzTR * sizeof(int));

    cudaMemcpy(d_csrRowPtrTR, csrRowPtrTR, (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrColIdxTR, cscColIdxTR, nnzTR * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrValTR, csrValTR, nnzTR * sizeof(VALUE_TYPE), cudaMemcpyHostToDevice);

    // Vector b
    cudaMalloc((void **)&d_b, m * rhs * sizeof(VALUE_TYPE));
    cudaMemcpy(d_b, b, m * rhs * sizeof(VALUE_TYPE), cudaMemcpyHostToDevice);

    // Vector x
    cudaMalloc((void **)&d_x, n * rhs * sizeof(VALUE_TYPE));
    cudaMemset(d_x, 0, n * rhs * sizeof(VALUE_TYPE));

    //  - cuda syncfree SpTRSV analysis start!
    printf(" - cuda syncfree SpTRSV analysis start!\n");

    struct timeval t1, t2;
    gettimeofday(&t1, NULL);

    // malloc tmp memory to generate in-degree
    int *d_graphInDegree;
    int *d_graphInDegree_backup;
    cudaMalloc((void **)&d_graphInDegree, m * sizeof(int));
    cudaMalloc((void **)&d_graphInDegree_backup, m * sizeof(int));

    int *d_id_extractor;
    cudaMalloc((void **)&d_id_extractor, sizeof(int));

    int num_threads = 128;
    int num_blocks = ceil((double)m / (double)num_threads);

    for (int i = 0; i < BENCH_REPEAT; i++)
    {
        cudaMemset(d_graphInDegree, 0, m * sizeof(int));
        sptrsvCSRAnalyser<<<num_blocks, num_threads>>>(d_csrColIdxTR, m, d_graphInDegree);
    }
    cudaDeviceSynchronize();

    gettimeofday(&t2, NULL);
    double time_cuda_analysis = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    time_cuda_analysis /= BENCH_REPEAT;

    printf("cuda syncfree SpTRSV CSR analysis on L used %4.9f ms\n", time_cuda_analysis);

    //  - cuda syncfree SpTRSV solve start!
    printf(" - cuda syncfree SpTRSV CSR solve start!\n");

    // malloc tmp memory to collect a partial sum of each row
    VALUE_TYPE *d_left_sum;
    cudaMalloc((void **)&d_left_sum, sizeof(VALUE_TYPE) * m * rhs);

    // backup in-degree array, only used for benchmarking multiple runs
    cudaMemcpy(d_graphInDegree_backup, d_graphInDegree, m * sizeof(int), cudaMemcpyDeviceToDevice);

    // this is for profiling while loop only
    int *d_while_profiler;
    cudaMalloc((void **)&d_while_profiler, sizeof(int) * n);
    cudaMemset(d_while_profiler, 0, sizeof(int) * n);
    int *while_profiler = (int *)malloc(sizeof(int) * n);

    // step 5: solve L*y = x
    double time_cuda_solve = 0;

    for (int i = 0; i < BENCH_REPEAT; i++)
    {
        // get a unmodified in-degree array, only for benchmarking use
        cudaMemcpy(d_graphInDegree, d_graphInDegree_backup, m * sizeof(int), cudaMemcpyDeviceToDevice);
        //cudaMemset(d_graphInDegree, 0, sizeof(int) * m);

        // clear left_sum array, only for benchmarking use
        cudaMemset(d_x, 0, sizeof(VALUE_TYPE) * n * rhs);

        gettimeofday(&t1, NULL);

        if (rhs == 1)
        {
            num_threads = WARP_PER_BLOCK * WARP_SIZE;
            //num_threads = 1 * WARP_SIZE;
            num_blocks = ceil((double)m / (double)(num_threads / WARP_SIZE));
            sptrsv_syncfree_cuda_shared<<<num_blocks, num_threads>>>
                (d_csrRowPtrTR, d_csrColIdxTR, d_csrValTR, d_b,
                 d_graphInDegree, 1,
                 m, d_x);
            //ELMR<VALUE_TYPE><<<num_blocks, num_threads>>>(d_csrRowPtrTR, d_csrColIdxTR, d_csrValTR, d_b, m, d_x, ready);
        }
        else
        {
            printf("no more than 1 rhs.\n");
        }

        cudaDeviceSynchronize();
        gettimeofday(&t2, NULL);

        time_cuda_solve += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    }

    time_cuda_solve /= BENCH_REPEAT;
    double flop = 2 * (double)rhs * (double)nnzTR;

    printf("cuda syncfree SpTRSV CSR solve used %4.9f ms, throughput is %4.2f gflops\n",
           time_cuda_solve, flop / (1e6 * time_cuda_solve));
    *gflops = flop / (1e6 * time_cuda_solve);

    cudaMemcpy(x, d_x, n * rhs * sizeof(VALUE_TYPE), cudaMemcpyDeviceToHost);

    // validate x
    double accuracy = 1e-6;
    double ref = 0.0;
    double res = 0.0;

    for (int i = 0; i < n * rhs; i++)
    {
        ref += abs(x_ref[i]);
        res += abs(x[i] - x_ref[i]);
        //if (x_ref[i] != x[i]) printf ("[%i, %i] x_ref = %f, x = %f\n", i/rhs, i%rhs, x_ref[i], x[i]);
    }
    res = ref == 0 ? res : res / ref;

    if (res < accuracy)
        printf("cuda syncfree SpTRSV CSR executor passed! |x-xref|/|xref| = %8.9e\n", res);
    else
        printf("cuda syncfree SpTRSV CSR executor _NOT_ passed! |x-xref|/|xref| = %8.9e\n", res);

    // profile while loop
    cudaMemcpy(while_profiler, d_while_profiler, n * sizeof(int), cudaMemcpyDeviceToHost);
    long long unsigned int while_count = 0;
    for (int i = 0; i < n; i++)
    {
        while_count += while_profiler[i];
        //printf("while_profiler[%i] = %i\n", i, while_profiler[i]);
    }
    //printf("\nwhile_count= %llu in total, %llu per row/column\n", while_count, while_count/m);

    // step 6: free resources
    free(while_profiler);

    cudaFree(d_graphInDegree);
    cudaFree(d_graphInDegree_backup);
    cudaFree(ready);
    cudaFree(d_id_extractor);
    cudaFree(d_left_sum);
    cudaFree(d_while_profiler);

    cudaFree(d_csrRowPtrTR);
    cudaFree(d_csrColIdxTR);
    cudaFree(d_csrValTR);
    cudaFree(d_b);
    cudaFree(d_x);

    return 0;
}

#endif
