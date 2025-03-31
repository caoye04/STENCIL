#include "common.h"

const char* version_name = "A naive base-line";

void create_dist_grid(dist_grid_info_t *grid_info, int stencil_type) {
    /* Naive implementation uses Process 0 to do all computations */
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

ptr_t stencil_27(ptr_t grid, ptr_t aux, const dist_grid_info_t *grid_info, int nt) {
    ptr_t buffer[2] = {grid, aux};
    int x_start = grid_info->halo_size_x, x_end = grid_info->local_size_x + grid_info->halo_size_x;
    int y_start = grid_info->halo_size_y, y_end = grid_info->local_size_y + grid_info->halo_size_y;
    int z_start = grid_info->halo_size_z, z_end = grid_info->local_size_z + grid_info->halo_size_z;
    int ldx = grid_info->local_size_x + 2 * grid_info->halo_size_x;
    int ldy = grid_info->local_size_y + 2 * grid_info->halo_size_y;
    int ldz = grid_info->local_size_z + 2 * grid_info->halo_size_z;
    (void)(ldz); // unused variable
    for(int t = 0; t < nt; ++t) {
        cptr_t a0 = buffer[t % 2];
        ptr_t a1 = buffer[(t + 1) % 2];
        #pragma omp parallel for schedule (dynamic)
        for(int z = z_start; z < z_end; ++z) {
            for(int y = y_start; y < y_end; ++y) {
                for(int x = x_start; x < x_end; ++x) {
                    a1[INDEX(x, y, z, ldx, ldy)]  
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
    return buffer[nt % 2];
}
