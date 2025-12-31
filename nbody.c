#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h> // Using OpenMP just for the accurate timer function

#define N  131072
// Softening parameter to prevent division by zero if stars collide
#define SOFTENING 1e-9f

// Data structures for Position (p) and Velocity (v)
// Using "Struct of Arrays" pattern for better GPU memory access
float px[N], py[N], pz[N];
float vx[N], vy[N], vz[N];

void randomizeBodies() {
    for (int i = 0; i < N; i++) {
        px[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
        py[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
        pz[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
        vx[i] = 0.0f; vy[i] = 0.0f; vz[i] = 0.0f;
    }
}

int main() {
    randomizeBodies();
    printf("Running N-Body Simulation with N=%d bodies.\n", N);
    printf("Total interactions per step: %.2f Billion\n\n", (double)N*N/1e9);

    double start_time = omp_get_wtime();

    // ---------------------------------------------------------
    // THE COMPUTE KERNEL
    // FIX APPLIED: Changed 'present' to 'copy'.
    // This instructs the compiler to handle the data movement to/from GPU
    // automatically for these arrays.
    // ---------------------------------------------------------
    #pragma acc parallel loop gang vector copy(px[0:N], py[0:N], pz[0:N], vx[0:N], vy[0:N], vz[0:N])
    for (int i = 0; i < N; ++i) {
        float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f;
        
        // Inner loop: calculate force from all other bodies 'j'
        // This runs sequentially within one GPU thread.
        for (int j = 0; j < N; j++) {
            float dx = px[j] - px[i];
            float dy = py[j] - py[i];
            float dz = pz[j] - pz[i];
            float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
            float invDist = 1.0f / sqrtf(distSqr); // Heavy math operation
            float invDist3 = invDist * invDist * invDist;

            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }
        // Update velocity (simple Euler integration with dt=1)
        vx[i] += Fx; vy[i] += Fy; vz[i] += Fz;
    }
    // ---------------------------------------------------------

    // Force synchronization to ensure GPU is done before stopping timer
    #pragma acc wait

    double end_time = omp_get_wtime();
    double total_time = end_time - start_time;

    printf("Simulation Complete.\n");
    printf("Compute Time: %.4f seconds\n", total_time);
    
    // Calculate Giga-Interactions Per Second
    // Formula: (N * N) interactions / Time in seconds / 1 Billion
    double giga_interactions = (1e-9 * (double)N * (double)N) / total_time;
    printf("Performance:  %.2f Giga-Interactions/sec\n", giga_interactions);

    // Print a sanity check value to ensure the compiler didn't optimize everything away
    printf("Sanity Check (Velocity of star 0): %f\n", vx[0]);

    return 0;
}
