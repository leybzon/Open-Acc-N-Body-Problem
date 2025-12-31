#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h> 

#define SOFTENING 1e-9f

int main(int argc, char *argv[]) {
    // 1. Read N from Command Line
    if (argc != 2) {
        printf("Usage: %s <N>\n", argv[0]);
        return 1;
    }
    int N = atoi(argv[1]);
    
    // 2. Allocate Memory Dynamically
    // (We use pointers now, which works perfectly with Unified Memory)
    size_t bytes = N * sizeof(float);
    float *px = (float*)malloc(bytes);
    float *py = (float*)malloc(bytes);
    float *pz = (float*)malloc(bytes);
    float *vx = (float*)malloc(bytes);
    float *vy = (float*)malloc(bytes);
    float *vz = (float*)malloc(bytes);

    // 3. Initialize
    for (int i = 0; i < N; i++) {
        px[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
        py[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
        pz[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
        vx[i] = 0.0f; vy[i] = 0.0f; vz[i] = 0.0f;
    }

    double start_time = omp_get_wtime();

    // ---------------------------------------------------------
    // COMPUTE KERNEL
    // Notice: We must explicitly specify the array size [0:N]
    // for the copy clause when using pointers.
    // ---------------------------------------------------------
    #pragma acc parallel loop gang vector copy(px[0:N], py[0:N], pz[0:N], vx[0:N], vy[0:N], vz[0:N])
    for (int i = 0; i < N; ++i) {
        float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f;
        
        #pragma acc loop vector reduction(+:Fx,Fy,Fz)
        for (int j = 0; j < N; j++) {
            float dx = px[j] - px[i];
            float dy = py[j] - py[i];
            float dz = pz[j] - pz[i];
            float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
            float invDist = 1.0f / sqrtf(distSqr); 
            float invDist3 = invDist * invDist * invDist;

            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }
        vx[i] += Fx; vy[i] += Fy; vz[i] += Fz;
    }
    // ---------------------------------------------------------

    #pragma acc wait
    double end_time = omp_get_wtime();
    double total_time = end_time - start_time;
    double total_ops = (double)N * (double)N;
    double giga_interactions = (total_ops * 1e-9) / total_time;

    // Output strictly in CSV format for easier parsing logic
    // Format: N, Time, GigaInteractions
    printf("%d,%.4f,%.2f\n", N, total_time, giga_interactions);

    // Cleanup
    free(px); free(py); free(pz);
    free(vx); free(vy); free(vz);

    return 0;
}
