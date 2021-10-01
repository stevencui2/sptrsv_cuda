#include "common.h"
#include "mmio_highlevel.h"
#include "utils.h"
#include "tranpose.h"
#include "findlevel.h"

#include "sptrsv_syncfree_serialref.h"
#include "sptrsv_syncfree_cuda.cuh"

int main(int argc, char ** argv)
{
    // report precision of floating-point
    printf("---------------------------------------------------------------------------------------------\n");
    char  *precision;
    if (sizeof(VALUE_TYPE) == 4)
    {
        precision = (char *)"32-bit Single Precision";
    }
    else if (sizeof(VALUE_TYPE) == 8)
    {
        precision = (char *)"64-bit Double Precision";
    }
    else
    {
        printf("Wrong precision. Program exit!\n");
        return 0;
    }

    printf("PRECISION = %s\n", precision);
    printf("Benchmark REPEAT = %i\n", BENCH_REPEAT);
    printf("---------------------------------------------------------------------------------------------\n");

    int m, n, nnzA, isSymmetricA;
    int *csrRowPtrA;
    int *csrColIdxA;
    VALUE_TYPE *csrValA;

    int *csrRowPtrTR;
    int *csrColIdxTR;
    VALUE_TYPE *csrValTR;

    int nnzTR;
    int *cscRowIdxTR;
    int *cscColPtrTR;
    VALUE_TYPE *cscValTR;

    int device_id = 0;
    int rhs = 0;
    int substitution = SUBSTITUTION_FORWARD;

    // "Usage: ``./sptrsv -d 0 -rhs 1 -forward -mtx A.mtx'' for LX=B on device 0"
    int argi = 1;

    // load device id
    char *devstr;
    if(argc > argi)
    {
        devstr = argv[argi];
        argi++;
    }

    if (strcmp(devstr, "-d") != 0) return 0;

    if(argc > argi)
    {
        device_id = atoi(argv[argi]);
        argi++;
    }
    printf("device_id = %i\n", device_id);

    // load the number of right-hand-side
    char *rhsstr;
    if(argc > argi)
    {
        rhsstr = argv[argi];
        argi++;
    }

    if (strcmp(rhsstr, "-rhs") != 0) return 0;

    if(argc > argi)
    {
        rhs = atoi(argv[argi]);
        argi++;
    }
    printf("rhs = %i\n", rhs);

    // load substitution, forward or backward
    char *substitutionstr;
    if(argc > argi)
    {
        substitutionstr = argv[argi];
        argi++;
    }

    if (strcmp(substitutionstr, "-forward") == 0)
        substitution = SUBSTITUTION_FORWARD;
    else if (strcmp(substitutionstr, "-backward") == 0)
        substitution = SUBSTITUTION_BACKWARD;
    printf("substitutionstr = %s\n", substitutionstr);
    printf("substitution = %i\n", substitution);

    // load matrix file type, mtx, cscl, or cscu
    char *matstr;
    if(argc > argi)
    {
        matstr = argv[argi];
        argi++;
    }
    printf("matstr = %s\n", matstr);

    // load matrix data from file
    char  *filename;
    if(argc > argi)
    {
        filename = argv[argi];
        argi++;
    }
    printf("-------------- %s --------------\n", filename);

    srand(time(NULL));
    if (strcmp(matstr, "-mtx") == 0)
    {
        // load mtx data to the csr format
        mmio_info(&m, &n, &nnzA, &isSymmetricA, filename);
        csrRowPtrA = (int *)malloc((m+1) * sizeof(int));
        csrColIdxA = (int *)malloc(nnzA * sizeof(int));
        csrValA    = (VALUE_TYPE *)malloc(nnzA * sizeof(VALUE_TYPE));
        mmio_data(csrRowPtrA, csrColIdxA, csrValA, filename);
        printf("input matrix A: ( %i, %i ) nnz = %i\n", m, n, nnzA);

        // extract L or U with a unit diagonal of A
        csrRowPtrTR = (int *)malloc((m+1) * sizeof(int));
        csrColIdxTR = (int *)malloc((m+nnzA) * sizeof(int));
        csrValTR    = (VALUE_TYPE *)malloc((m+nnzA) * sizeof(VALUE_TYPE));

        int nnz_pointer = 0;
        csrRowPtrTR[0] = 0;
        for (int i = 0; i < m; i++)
        {
            for (int j = csrRowPtrA[i]; j < csrRowPtrA[i+1]; j++)
            {   
                if (substitution == SUBSTITUTION_FORWARD)
                {
                    if (csrColIdxA[j] < i)
                    {
                        csrColIdxTR[nnz_pointer] = csrColIdxA[j];
                        //csrValTR[nnz_pointer] = rand() % 10 + 1; 
                        csrValTR[nnz_pointer] = csrValA[j]; 
                        nnz_pointer++;
                    }
                }
                else if (substitution == SUBSTITUTION_BACKWARD)
                {
                    if (csrColIdxA[j] > i)
                    {
                        csrColIdxTR[nnz_pointer] = csrColIdxA[j];
                        //csrValTR[nnz_pointer] = rand() % 10 + 1; 
                        csrValTR[nnz_pointer] = csrValA[j];
                        nnz_pointer++;
                    }
                }
            }

            // add dia nonzero
            csrColIdxTR[nnz_pointer] = i;
            csrValTR[nnz_pointer] = 1.0;
            nnz_pointer++;

            csrRowPtrTR[i+1] = nnz_pointer;
        }

        int nnz_tmp = csrRowPtrTR[m];
        nnzTR = nnz_tmp;

        if (substitution == SUBSTITUTION_FORWARD)
            printf("A's unit-lower triangular L: ( %i, %i ) nnz = %i\n", m, n, nnzTR);
        else if (substitution == SUBSTITUTION_BACKWARD)
            printf("A's unit-upper triangular U: ( %i, %i ) nnz = %i\n", m, n, nnzTR);

        csrColIdxTR = (int *)realloc(csrColIdxTR, sizeof(int) * nnzTR);
        csrValTR = (VALUE_TYPE *)realloc(csrValTR, sizeof(VALUE_TYPE) * nnzTR);

        cscRowIdxTR = (int *)malloc(nnzTR * sizeof(int));
        cscColPtrTR = (int *)malloc((n+1) * sizeof(int));
        memset(cscColPtrTR, 0, (n+1) * sizeof(int));
        cscValTR    = (VALUE_TYPE *)malloc(nnzTR * sizeof(VALUE_TYPE));

        // transpose from csr to csc
        matrix_transposition(m, n, nnzTR,
                             csrRowPtrTR, csrColIdxTR, csrValTR,
                             cscRowIdxTR, cscColPtrTR, cscValTR);

        // keep each column sort 
        for (int i = 0; i < n; i++)
        {
            quick_sort_key_val_pair<int, int>(&cscRowIdxTR[cscColPtrTR[i]],
                                              &cscRowIdxTR[cscColPtrTR[i]],
                                              cscColPtrTR[i+1]-cscColPtrTR[i]);
        }

        // check unit diagonal
        int dia_miss = 0;
        for (int i = 0; i < n; i++)
        {
            bool miss;
            if (substitution == SUBSTITUTION_FORWARD)
                miss = cscRowIdxTR[cscColPtrTR[i]] != i;
            else if (substitution == SUBSTITUTION_BACKWARD)
                cscRowIdxTR[cscColPtrTR[i+1] - 1] != i;

            if (miss) dia_miss++;
        }
        //printf("dia miss = %i\n", dia_miss);
        if (dia_miss != 0) 
        {
            printf("This matrix has incomplete diagonal, #missed dia nnz = %i\n", dia_miss); 
            return;
        }

        free(csrColIdxA);
        free(csrValA);
        free(csrRowPtrA);
    }
    else if (strcmp(matstr, "-csc") == 0)
    {
        FILE *f;
        int returnvalue;

        if ((f = fopen(filename, "r")) == NULL)
            return -1;

        returnvalue = fscanf(f, "%d", &m);
        returnvalue = fscanf(f, "%d", &n);
        returnvalue = fscanf(f, "%d", &nnzTR);

        cscColPtrTR = (int *)malloc((n+1) * sizeof(int));
        memset(cscColPtrTR, 0, (n+1) * sizeof(int));
        cscRowIdxTR = (int *)malloc(nnzTR * sizeof(int));
        cscValTR    = (VALUE_TYPE *)malloc(nnzTR * sizeof(VALUE_TYPE));

        // read row idx
        for (int i = 0; i < n+1; i++)
        {
            returnvalue = fscanf(f, "%d", &cscColPtrTR[i]);
            cscColPtrTR[i]--; // from 1-based to 0-based
        }

        // read col idx
        for (int i = 0; i < nnzTR; i++)
        {
            returnvalue = fscanf(f, "%d", &cscRowIdxTR[i]);
            cscRowIdxTR[i]--; // from 1-based to 0-based
        }

        // read val
        for (int i = 0; i < nnzTR; i++)
        {
            cscValTR[i] = rand() % 10 + 1;
            //returnvalue = fscanf(f, "%lg", &cscValTR[i]);
        }

        if (f != stdin)
            fclose(f);

        // keep each column sort 
        for (int i = 0; i < n; i++)
        {
            quick_sort_key_val_pair<int, int>(&cscRowIdxTR[cscColPtrTR[i]],
                                              &cscRowIdxTR[cscColPtrTR[i]],
                                              cscColPtrTR[i+1]-cscColPtrTR[i]);
        }

        if (substitution == SUBSTITUTION_FORWARD)
            printf("Input csc unit-lower triangular L: ( %i, %i ) nnz = %i\n", m, n, nnzTR);
        else if (substitution == SUBSTITUTION_BACKWARD)
            printf("Input csc unit-upper triangular U: ( %i, %i ) nnz = %i\n", m, n, nnzTR);
       
        // check unit diagonal
        int dia_miss = 0;
        for (int i = 0; i < n; i++)
        {
            bool miss;
            if (substitution == SUBSTITUTION_FORWARD)
                miss = cscRowIdxTR[cscColPtrTR[i]] != i;
            else if (substitution == SUBSTITUTION_BACKWARD)
                cscRowIdxTR[cscColPtrTR[i+1] - 1] != i;

            if (miss) dia_miss++;
        }
        //printf("dia miss = %i\n", dia_miss);
        if (dia_miss != 0) 
        {
            printf("This matrix has incomplete diagonal, #missed dia nnz = %i\n", dia_miss); 
            return;
        }
    }

    // find level sets
    int nlevel = 0;
    int parallelism_min = 0;
    int parallelism_avg = 0;
    int parallelism_max = 0;
    int  *levelPtr  = (int *)malloc((m+1) * sizeof(int));
    int  *levelItem = (int *)malloc(m * sizeof(int));
    findlevel_csc(cscColPtrTR, cscRowIdxTR, cscValTR, m, n, nnzTR, &nlevel,
                  &parallelism_min, &parallelism_avg, &parallelism_max,levelPtr,levelItem);
    double fparallelism = (double)m/(double)nlevel;
    printf("This matrix/graph has %i levels, its parallelism is %4.2f (min: %i ; avg: %i ; max: %i )\n", 
           nlevel, fparallelism, parallelism_min, parallelism_avg, parallelism_max);

    // x and b are all row-major
    VALUE_TYPE *x_ref = (VALUE_TYPE *)malloc(sizeof(VALUE_TYPE) * n * rhs);
    for ( int i = 0; i < n; i++)
        for (int j = 0; j < rhs; j++)
            x_ref[i * rhs + j] = rand() % 10 + 1; //j + 1;

    VALUE_TYPE *b = (VALUE_TYPE *)malloc(sizeof(VALUE_TYPE) * m * rhs);
    VALUE_TYPE *x = (VALUE_TYPE *)malloc(sizeof(VALUE_TYPE) * n * rhs);

    for (int i = 0; i < m * rhs; i++)
        b[i] = 0;

    for (int i = 0; i < n * rhs; i++)
        x[i] = 0;

    // run csc spmv to generate b
    for (int i = 0; i < n; i++)
    {
        for (int j = cscColPtrTR[i]; j < cscColPtrTR[i+1]; j++)
        {
            int rowid = cscRowIdxTR[j]; //printf("rowid = %i\n", rowid);
            for (int k = 0; k < rhs; k++)
            {
                b[rowid * rhs + k] += cscValTR[j] * x_ref[i * rhs + k];
            }
        }
    }

    // run serial syncfree SpTRSV as a reference
    printf("---------------------------------------------------------------------------------------------\n");
    sptrsv_syncfree_serialref(cscColPtrTR, cscRowIdxTR, cscValTR, m, n, nnzTR,
                              substitution, rhs, x, b, x_ref);

    // set device
    cudaSetDevice(device_id);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device_id);

    printf("---------------------------------------------------------------------------------------------\n");
    printf("Device [ %i ] %s @ %4.2f MHz\n", device_id, deviceProp.name, deviceProp.clockRate * 1e-3f);

    // run cuda syncfree SpTRSV or SpTRSM

    printf("---------------------------------------------------------------------------------------------\n");
    double gflops_autotuned = 0;
    sptrsvCSC_syncfree_cuda(cscColPtrTR, cscRowIdxTR, cscValTR, m, n, nnzTR,
                         substitution, rhs, OPT_WARP_AUTO, x, b, x_ref, &gflops_autotuned,levelItem);

    printf("---------------------------------------------------------------------------------------------\n");


    printf("---------------------------------------------------------------------------------------------\n");
     gflops_autotuned = 0;
    sptrsvCSR_syncfree_cuda(csrRowPtrTR, csrColIdxTR, csrValTR, m, n, nnzTR,
                         substitution, rhs, OPT_WARP_AUTO, x, b, x_ref, &gflops_autotuned);

    printf("---------------------------------------------------------------------------------------------\n");

    // done!

    free(csrColIdxTR);
    free(csrValTR);
    free(csrRowPtrTR);

    free(cscRowIdxTR);
    free(cscColPtrTR);
    free(cscValTR);

    free(levelPtr);
    free(levelItem);

    free(x);
    free(x_ref);
    free(b);

    return 0;
}
