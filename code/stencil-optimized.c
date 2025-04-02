#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>
#include <numa.h>
#include <sched.h>
#include <immintrin.h>

const char* version_name = "Optimized version ";
#include "common.h"

#define max(a, b) (a > b) ? a : b
#define min(a, b) (a < b) ? a : b

void create_dist_grid(dist_grid_info_t *grid_info, int stencil_type) {
    if(grid_info->p_id == 0) {
        grid_info->local_size_x = grid_info->global_size_x;
        grid_info->local_size_y = grid_info->global_size_y;
        grid_info->local_size_z = grid_info->global_size_z;
    } else {
        grid_info->local_size_x = 0;
        grid_info->local_size_y = 0;
        grid_info->local_size_z = 0;
    }
    grid_info->offset_x = 0;
    grid_info->offset_y = 0;
    grid_info->offset_z = 0;
    grid_info->halo_size_x = 1;
    grid_info->halo_size_y = 1;
    grid_info->halo_size_z = 1;
}

void destroy_dist_grid(dist_grid_info_t *grid_info) {
    
}

void do_cal27(cptr_t a0, ptr_t a1, int x, int y, int z, int ldx, int ldy){
    __m256d res = _mm256_setzero_pd();
    __m256d in1, in2, in3, in4, in5, in6, in7;
    __m256d al1, al2, al3, al4, al5, al6, al7;
    in1 = _mm256_loadu_pd(a0 + INDEX(x, y, z, ldx, ldy));
    in2 = _mm256_loadu_pd(a0 + INDEX(x - 1, y, z, ldx, ldy));
    in3 = _mm256_loadu_pd(a0 + INDEX(x + 1, y, z, ldx, ldy));
    in4 = _mm256_loadu_pd(a0 + INDEX(x, y - 1, z, ldx, ldy));
    in5 = _mm256_loadu_pd(a0 + INDEX(x, y + 1, z, ldx, ldy));
    in6 = _mm256_loadu_pd(a0 + INDEX(x, y, z - 1, ldx, ldy));
    in7 = _mm256_loadu_pd(a0 + INDEX(x, y, z + 1, ldx, ldy));
    al1 = _mm256_set1_pd((double)ALPHA_ZZZ);
    al2 = _mm256_set1_pd((double)ALPHA_NZZ);
    al3 = _mm256_set1_pd((double)ALPHA_PZZ);
    al4 = _mm256_set1_pd((double)ALPHA_ZNZ);
    al5 = _mm256_set1_pd((double)ALPHA_ZPZ);
    al6 = _mm256_set1_pd((double)ALPHA_ZZN);
    al7 = _mm256_set1_pd((double)ALPHA_ZZP);
    res = _mm256_fmadd_pd(al1, in1, res);
    res = _mm256_fmadd_pd(al2, in2, res);
    res = _mm256_fmadd_pd(al3, in3, res);
    res = _mm256_fmadd_pd(al4, in4, res);
    res = _mm256_fmadd_pd(al5, in5, res);
    res = _mm256_fmadd_pd(al6, in6, res);
    res = _mm256_fmadd_pd(al7, in7, res);

    in1 = _mm256_loadu_pd(a0 + INDEX(x - 1, y - 1, z, ldx, ldy));
    in2 = _mm256_loadu_pd(a0 + INDEX(x + 1, y - 1, z, ldx, ldy));
    in3 = _mm256_loadu_pd(a0 + INDEX(x - 1, y + 1, z, ldx, ldy));
    in4 = _mm256_loadu_pd(a0 + INDEX(x + 1, y + 1, z, ldx, ldy));
    in5 = _mm256_loadu_pd(a0 + INDEX(x - 1, y, z - 1, ldx, ldy));
    in6 = _mm256_loadu_pd(a0 + INDEX(x + 1, y, z - 1, ldx, ldy));
    in7 = _mm256_loadu_pd(a0 + INDEX(x - 1, y, z + 1, ldx, ldy));
    al1 = _mm256_set1_pd((double)ALPHA_NNZ);
    al2 = _mm256_set1_pd((double)ALPHA_PNZ);
    al3 = _mm256_set1_pd((double)ALPHA_NPZ);
    al4 = _mm256_set1_pd((double)ALPHA_PPZ);
    al5 = _mm256_set1_pd((double)ALPHA_NZN);
    al6 = _mm256_set1_pd((double)ALPHA_PZN);
    al7 = _mm256_set1_pd((double)ALPHA_NZP);
    res = _mm256_fmadd_pd(al1, in1, res);
    res = _mm256_fmadd_pd(al2, in2, res);
    res = _mm256_fmadd_pd(al3, in3, res);
    res = _mm256_fmadd_pd(al4, in4, res);
    res = _mm256_fmadd_pd(al5, in5, res);
    res = _mm256_fmadd_pd(al6, in6, res);
    res = _mm256_fmadd_pd(al7, in7, res);

    in1 = _mm256_loadu_pd(a0 + INDEX(x + 1, y, z + 1, ldx, ldy));
    in2 = _mm256_loadu_pd(a0 + INDEX(x, y - 1, z - 1, ldx, ldy));
    in3 = _mm256_loadu_pd(a0 + INDEX(x, y + 1, z - 1, ldx, ldy));
    in4 = _mm256_loadu_pd(a0 + INDEX(x, y - 1, z + 1, ldx, ldy));
    in5 = _mm256_loadu_pd(a0 + INDEX(x, y + 1, z + 1, ldx, ldy));
    in6 = _mm256_loadu_pd(a0 + INDEX(x - 1, y - 1, z - 1, ldx, ldy));
    in7 = _mm256_loadu_pd(a0 + INDEX(x + 1, y - 1, z - 1, ldx, ldy));
    al1 = _mm256_set1_pd((double)ALPHA_PZP);
    al2 = _mm256_set1_pd((double)ALPHA_ZNN);
    al3 = _mm256_set1_pd((double)ALPHA_ZPN);
    al4 = _mm256_set1_pd((double)ALPHA_ZNP);
    al5 = _mm256_set1_pd((double)ALPHA_ZPP);
    al6 = _mm256_set1_pd((double)ALPHA_NNN);
    al7 = _mm256_set1_pd((double)ALPHA_PNN);
    res = _mm256_fmadd_pd(al1, in1, res);
    res = _mm256_fmadd_pd(al2, in2, res);
    res = _mm256_fmadd_pd(al3, in3, res);
    res = _mm256_fmadd_pd(al4, in4, res);
    res = _mm256_fmadd_pd(al5, in5, res);
    res = _mm256_fmadd_pd(al6, in6, res);
    res = _mm256_fmadd_pd(al7, in7, res);

    in1 = _mm256_loadu_pd(a0 + INDEX(x - 1, y + 1, z - 1, ldx, ldy));
    in2 = _mm256_loadu_pd(a0 + INDEX(x + 1, y + 1, z - 1, ldx, ldy));
    in3 = _mm256_loadu_pd(a0 + INDEX(x - 1, y - 1, z + 1, ldx, ldy));
    in4 = _mm256_loadu_pd(a0 + INDEX(x + 1, y - 1, z + 1, ldx, ldy));
    in5 = _mm256_loadu_pd(a0 + INDEX(x - 1, y + 1, z + 1, ldx, ldy));
    in6 = _mm256_loadu_pd(a0 + INDEX(x + 1, y + 1, z + 1, ldx, ldy));
    al1 = _mm256_set1_pd((double)ALPHA_NPN);
    al2 = _mm256_set1_pd((double)ALPHA_PPN);
    al3 = _mm256_set1_pd((double)ALPHA_NNP);
    al4 = _mm256_set1_pd((double)ALPHA_PNP);
    al5 = _mm256_set1_pd((double)ALPHA_NPP);
    al6 = _mm256_set1_pd((double)ALPHA_PPP);
    res = _mm256_fmadd_pd(al1, in1, res);
    res = _mm256_fmadd_pd(al2, in2, res);
    res = _mm256_fmadd_pd(al3, in3, res);
    res = _mm256_fmadd_pd(al4, in4, res);
    res = _mm256_fmadd_pd(al5, in5, res);
    res = _mm256_fmadd_pd(al6, in6, res);

    _mm256_storeu_pd(a1 + INDEX(x, y, z, ldx, ldy), res);
}

ptr_t stencil_27(ptr_t grid, ptr_t aux, const dist_grid_info_t *grid_info, int nt) {
    ptr_t buffer[2] = {grid, aux};
    int x_start = grid_info->halo_size_x, x_end = grid_info->local_size_x + grid_info->halo_size_x;
    int y_start = grid_info->halo_size_y, y_end = grid_info->local_size_y + grid_info->halo_size_y;
    int z_start = grid_info->halo_size_z, z_end = grid_info->local_size_z + grid_info->halo_size_z;
    int ldx = grid_info->local_size_x + 2 * grid_info->halo_size_x;
    int ldy = grid_info->local_size_y + 2 * grid_info->halo_size_y;
    

    int num_threads = omp_get_max_threads();
    //printf("num = %d\n",num_threads);
    omp_set_num_threads(num_threads);
    
    int bs_z = 256;
    if(z_end - z_start == 384)  bs_z=384;
    if(z_end - z_start == 512)  bs_z=512;
    for(int zz = z_start; zz < z_end; zz += bs_z){
        int neg_z = 1;
        int pos_z = -1;
        if(zz == z_start) neg_z = 0;
        if(zz == z_end - bs_z) pos_z = 0;

        int bs_y = 16;
        if(y_end - y_start == 384) bs_y = 12;
        if(y_end - y_start == 512) bs_y = 8;
        for(int yy = y_start; yy < y_end; yy += bs_y){
            int neg_y = 1;
            int pos_y = -1;
            if(yy == y_start) neg_y = 0;
            if(yy == y_end - bs_y) pos_y = 0;

            for(int xx = x_start; xx < x_end; xx += (x_end - x_start)){
                int neg_x = 1;
                int pos_x = -1;
                if(xx == x_start) neg_x = 0;
                if(xx == x_end - (x_end - x_start)) pos_x = 0;

                for(int t = 0; t < nt; t++){
                    cptr_t a0 = buffer[t % 2];
                    ptr_t a1 = buffer[(t + 1) % 2];

                    int blockmin_x = max(x_start, xx - t * neg_x);
                    int blockmin_y = max(y_start, yy - t * neg_y);
                    int blockmin_z = max(z_start, zz - t * neg_z);

                    int blockmax_x = max(x_start, xx + (x_end-x_start) + t * pos_x);
                    int blockmax_y = max(y_start, yy + bs_y + t * pos_y);
                    int blockmax_z = max(z_start, zz + bs_z + t * pos_z);

                    #pragma omp parallel for
                    for(int z = blockmin_z; z < blockmax_z; z++){
                        for(int y = blockmin_y; y < blockmax_y; y++){
                            for(int x = blockmin_x; x < blockmin_x + (blockmax_x - blockmin_x) / 4 * 4; x += 4){
                                do_cal27(a0, a1, x, y, z, ldx, ldy);
                            }
                            for(int x = blockmin_x + (blockmax_x - blockmin_x) / 4 * 4; x < blockmax_x; x++){
                                a1[INDEX(x, y, z, ldx, ldy)] \
                                = ALPHA_ZZZ * a0[INDEX(x, y, z, ldx, ldy)] \
                                + ALPHA_NZZ * a0[INDEX(x-1, y, z, ldx, ldy)] \
                                + ALPHA_PZZ * a0[INDEX(x+1, y, z, ldx, ldy)] \
                                + ALPHA_ZNZ * a0[INDEX(x, y-1, z, ldx, ldy)] \
                                + ALPHA_ZPZ * a0[INDEX(x, y+1, z, ldx, ldy)] \
                                + ALPHA_ZZN * a0[INDEX(x, y, z-1, ldx, ldy)] \
                                + ALPHA_ZZP * a0[INDEX(x, y, z+1, ldx, ldy)] \
                                + ALPHA_NNZ * a0[INDEX(x-1, y-1, z, ldx, ldy)] \
                                + ALPHA_PNZ * a0[INDEX(x+1, y-1, z, ldx, ldy)] \
                                + ALPHA_NPZ * a0[INDEX(x-1, y+1, z, ldx, ldy)] \
                                + ALPHA_PPZ * a0[INDEX(x+1, y+1, z, ldx, ldy)] \
                                + ALPHA_NZN * a0[INDEX(x-1, y, z-1, ldx, ldy)] \
                                + ALPHA_PZN * a0[INDEX(x+1, y, z-1, ldx, ldy)] \
                                + ALPHA_NZP * a0[INDEX(x-1, y, z+1, ldx, ldy)] \
                                + ALPHA_PZP * a0[INDEX(x+1, y, z+1, ldx, ldy)] \
                                + ALPHA_ZNN * a0[INDEX(x, y-1, z-1, ldx, ldy)] \
                                + ALPHA_ZPN * a0[INDEX(x, y+1, z-1, ldx, ldy)] \
                                + ALPHA_ZNP * a0[INDEX(x, y-1, z+1, ldx, ldy)] \
                                + ALPHA_ZPP * a0[INDEX(x, y+1, z+1, ldx, ldy)] \
                                + ALPHA_NNN * a0[INDEX(x-1, y-1, z-1, ldx, ldy)] \
                                + ALPHA_PNN * a0[INDEX(x+1, y-1, z-1, ldx, ldy)] \
                                + ALPHA_NPN * a0[INDEX(x-1, y+1, z-1, ldx, ldy)] \
                                + ALPHA_PPN * a0[INDEX(x+1, y+1, z-1, ldx, ldy)] \
                                + ALPHA_NNP * a0[INDEX(x-1, y-1, z+1, ldx, ldy)] \
                                + ALPHA_PNP * a0[INDEX(x+1, y-1, z+1, ldx, ldy)] \
                                + ALPHA_NPP * a0[INDEX(x-1, y+1, z+1, ldx, ldy)] \
                                + ALPHA_PPP * a0[INDEX(x+1, y+1, z+1, ldx, ldy)];
                            }
                        }
                    }
                }
            }
        }
    }
    return buffer[nt % 2];
}