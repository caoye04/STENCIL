#include <math.h>
#include <mpi.h>
#include "common.h"

typedef struct {
    data_t norm_1, norm_2, norm_inf;
} check_result_t;

check_result_t check_answer(cptr_t ans0, cptr_t ans1, const dist_grid_info_t *info, MPI_Comm active) {
    check_result_t result;
    int x_start = info->halo_size_x, x_end = info->local_size_x + info->halo_size_x;
    int y_start = info->halo_size_y, y_end = info->local_size_y + info->halo_size_y;
    int z_start = info->halo_size_z, z_end = info->local_size_z + info->halo_size_z;
	int ldx = info->local_size_x + 2 * info->halo_size_x;
    int ldy = info->local_size_y + 2 * info->halo_size_y;
    int ldz = info->local_size_z + 2 * info->halo_size_z;
    (void)(ldz); // unused variable
    double norm_1, norm_2, norm_inf;
    double norm_1_tmp_local = 0.0, norm_2_tmp_local = 0.0, norm_inf_tmp_local = 0.0;
#ifdef _OPENMP /* when compiled with `icc -qopenmp` */
#pragma omp parallel for reduction(+:norm_1_tmp_local,norm_2_tmp_local) \
            reduction(max:norm_inf_tmp_local)
#endif
    for(int z = z_start; z < z_end; ++z) {
        double norm_1_tmp_z = 0.0, norm_2_tmp_z = 0.0, norm_inf_tmp_z = 0.0;
        for(int y = y_start; y < y_end; ++y) {
            double norm_1_tmp_y = 0.0, norm_2_tmp_y = 0.0, norm_inf_tmp_y = 0.0;
#ifdef _OPENMP
#pragma omp simd
#endif
            for(int x = x_start; x < x_end; ++x) {
                double err = fabs(ans0[INDEX(x, y, z, ldx, ldy)] - ans1[INDEX(x, y, z, ldx, ldy)]);
                norm_1_tmp_y += err;
                norm_2_tmp_y += err * err;
                if(norm_inf_tmp_y < err) {
                    norm_inf_tmp_y = err;
                }
            }
            norm_1_tmp_z += norm_1_tmp_y;
            norm_2_tmp_z += norm_2_tmp_y;
            if(norm_inf_tmp_z < norm_inf_tmp_y) {
                norm_inf_tmp_z = norm_inf_tmp_y;
            }
        }
        norm_1_tmp_local += norm_1_tmp_z;
        norm_2_tmp_local += norm_2_tmp_z;
        if(norm_inf_tmp_local < norm_inf_tmp_z) {
            norm_inf_tmp_local = norm_inf_tmp_z;
        }
    }
    MPI_Reduce(&norm_1_tmp_local, &norm_1, 1, MPI_DOUBLE, MPI_SUM, 0, active);
    MPI_Reduce(&norm_2_tmp_local, &norm_2, 1, MPI_DOUBLE, MPI_SUM, 0, active);
    MPI_Reduce(&norm_inf_tmp_local, &norm_inf, 1, MPI_DOUBLE, MPI_MAX, 0, active);
    result.norm_1 = norm_1 / (info->global_size_x * info->global_size_y * info->global_size_z);
    result.norm_2 = sqrt(norm_2 / (info->global_size_x * info->global_size_y * info->global_size_z));
    result.norm_inf = norm_inf;
    return result;
}
