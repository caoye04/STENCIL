#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <stdbool.h>
#include "common.h"

extern const char* version_name;

typedef struct {
    MPI_Datatype file_view, array_center;
} file_helper_t;

int parse_args(dist_grid_info_t *info, int *nsteps, int* stencil_type, int argc, char **argv);
int find_active_procs(const dist_grid_info_t *info, MPI_Comm *active_procs);
void create_file_helper(file_helper_t *h, const dist_grid_info_t *info);
void destroy_file_helper(file_helper_t *h);
int read_file(ptr_t grid, const char *file_name, const file_helper_t* h, MPI_Comm active_procs);
int write_file(ptr_t grid, const char *file_name, const file_helper_t* h, MPI_Comm active_procs);
ptr_t memory_alloc(const dist_grid_info_t *info);

typedef struct {
    data_t norm_1, norm_2, norm_inf;
} check_result_t;

extern check_result_t check_answer(cptr_t ans0, cptr_t ans1, const dist_grid_info_t *info, MPI_Comm active_procs);

#define CHECK(err, err_code) if(err) { return err_code; }
#define CHECK_ERROR(ret, err_code) CHECK(ret != MPI_SUCCESS, err_code)
#define ABORT_IF_ERROR(ret) CHECK_ERROR(ret, MPI_Abort(active_procs, 2))
#define ABORT_IF_NULL(ret) CHECK(ret == NULL, MPI_Abort(active_procs, 1))

int main(int argc, char **argv) {
    int nt, type, thread_scheme, status, p_id;
    dist_grid_info_t info;
    file_helper_t helper;
    double start, end, pre_time;
    ptr_t a0, a1, ans0, ans1;
    MPI_Comm active_procs;

    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &thread_scheme);
    MPI_Comm_rank(MPI_COMM_WORLD, &info.p_id);
    MPI_Comm_size(MPI_COMM_WORLD, &info.p_num);
    ABORT_IF_ERROR(parse_args(&info, &nt, &type, argc, argv))

    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    create_dist_grid(&info, type);
    MPI_Barrier(MPI_COMM_WORLD); 
    pre_time = MPI_Wtime() - start;

    status = find_active_procs(&info, &active_procs);
    a0 = memory_alloc(&info);
    a1 = memory_alloc(&info);
    ABORT_IF_NULL(a0)
    ABORT_IF_NULL(a1)
    if(status == 0) {
        MPI_Comm_rank(active_procs, &p_id);
        create_file_helper(&helper, &info);

        ABORT_IF_ERROR(read_file(a0, argv[6], &helper, active_procs))
    }

    /* warm up */
    ans0 = stencil_27(a0, a1, &info, 1);

    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    ans0 = stencil_27(a0, a1, &info, nt);
    MPI_Barrier(MPI_COMM_WORLD); 
    end = MPI_Wtime();

    if(status == 0) {
        if(argc == 8) {
            check_result_t result;
            ans1 = (ans0 == a0) ? a1 : a0;
            ABORT_IF_ERROR(read_file(ans1, argv[7], &helper, active_procs))
            result = check_answer(ans0, ans1, &info, active_procs);
            printf("errors:\n    1-norm = %.16lf\n    2-norm = %.16lf\n  inf-norm = %.16lf\n", \
                    result.norm_1, result.norm_2, result.norm_inf);
            if(result.norm_inf > 1e-9) {
                printf("Significant numeric error.\n");
            }
        } else if(p_id == 0) { 
            printf("Result unchecked.\n");
        }

        if(p_id == 0) { 
            double t = end - start; 
            double gflops = 1e-9 * info.global_size_x * info.global_size_y * info.global_size_z * nt * (type * 2 - 1) / t;

            printf("%d-point stencil - %s:\nSize (%d x %d x %d), Timestep %d\n",type, version_name, info.global_size_x, info.global_size_y, info.global_size_z, nt);
            printf("Preprocessing time %lfs\n", pre_time);
            printf("Computation time %lfs, Performance %lfGflop/s\n", t, gflops);
        }
        destroy_file_helper(&helper);
    }
    MPI_Free_mem(a1);
    MPI_Free_mem(a0);
    MPI_Comm_free(&active_procs);
    destroy_dist_grid(&info);
    MPI_Finalize();
    return 0;
}

void print_help(const char *argv0, int p_id) {
    if(p_id == 0) {
        printf("USAGE: %s <stencil-type> <nx> <ny> <nz> <nt> <input-file> [<answer-file>]\n  where <stencil-type> is 7 or 27\n", argv0);
    }
}

int parse_args(dist_grid_info_t *info, int *nsteps, int* stencil_type, int argc, char **argv) {
    int nx, ny, nz, nt, type;
    if(argc < 7) {
        print_help(argv[0], info->p_id);
        return MPI_ERR_ARG;
    } 
    type = atoi(argv[1]);
    nx = atoi(argv[2]);
    ny = atoi(argv[3]);
    nz = atoi(argv[4]);
    nt = atoi(argv[5]);
    if(nx <= 0 || ny <= 0 || nz <= 0 || nt <= 0 || (type != 7 && type != 27)) {
        print_help(argv[0], info->p_id);
        return MPI_ERR_ARG;
    }
    info->global_size_x = nx;
    info->global_size_y = ny;
    info->global_size_z = nz;
    *nsteps = nt;
    *stencil_type = type;
    return MPI_SUCCESS;
}

int find_active_procs(const dist_grid_info_t *info, MPI_Comm *active_procs) {
    int status;
    if(info->local_size_x > 0 && info->local_size_y > 0 && info->local_size_z > 0) {
        status = 0;
    } else {
        status = 1;
    }
    MPI_Comm_split(MPI_COMM_WORLD, status, info->p_id, active_procs);
    return status;
}

void create_file_helper(file_helper_t *h, const dist_grid_info_t *info) {
    int size[3], subsize[3], start[3];
    subsize[0] = info->local_size_x; subsize[1] = info->local_size_y; subsize[2] = info->local_size_z;

    size[0] = info->global_size_x;   size[1] = info->global_size_y;   size[2] = info->global_size_z;
    start[0] = info->offset_x;       start[1] = info->offset_y;       start[2] = info->offset_z;
    MPI_Type_create_subarray(3, size, subsize, start, MPI_ORDER_FORTRAN, DATA_TYPE, &h->file_view);

    size[0] = info->local_size_x + 2 * info->halo_size_x;
    size[1] = info->local_size_y + 2 * info->halo_size_y;
    size[2] = info->local_size_z + 2 * info->halo_size_z;
    start[0] = info->halo_size_x;    start[1] = info->halo_size_y;    start[2] = info->halo_size_z;
    MPI_Type_create_subarray(3, size, subsize, start, MPI_ORDER_FORTRAN, DATA_TYPE, &h->array_center);
    MPI_Type_commit(&h->file_view);
    MPI_Type_commit(&h->array_center);
}

void destroy_file_helper(file_helper_t *h) {
    MPI_Type_free(&h->file_view);
    MPI_Type_free(&h->array_center);
}

int read_file(ptr_t grid, const char *file_name, const file_helper_t* h, MPI_Comm active_procs) {
    MPI_File input;
    MPI_Status status;
    int count;
    int ret = MPI_File_open(active_procs, file_name, MPI_MODE_RDONLY, MPI_INFO_NULL, &input);
    CHECK_ERROR(ret, ret)
    ret = MPI_File_set_view(input, 0, DATA_TYPE, h->file_view, "native", MPI_INFO_NULL);
    CHECK_ERROR(ret, ret)
    ret = MPI_File_read_all(input, grid, 1, h->array_center, &status);
    CHECK_ERROR(ret, ret)
    ret = MPI_File_close(&input);
    CHECK_ERROR(ret, ret)
    MPI_Get_count(&status, h->array_center, &count);
    return (count == 1) ? MPI_SUCCESS : MPI_ERR_IO;
}

int write_file(ptr_t grid, const char *file_name, const file_helper_t* h, MPI_Comm active_procs) {
    MPI_File input;
    MPI_Status status;
    int count;
    int ret = MPI_File_open(active_procs, file_name, 5, MPI_INFO_NULL, &input);
    CHECK_ERROR(ret, ret)
    ret = MPI_File_set_view(input, 0, DATA_TYPE, h->file_view, "native", MPI_INFO_NULL);
    CHECK_ERROR(ret, ret)
    ret = MPI_File_write_all(input, grid, 1, h->array_center, &status);
    //ret = MPI_File_read_all(input, grid, 1, h->array_center, &status);
    CHECK_ERROR(ret, ret)
    ret = MPI_File_close(&input);
    CHECK_ERROR(ret, ret)
    MPI_Get_count(&status, h->array_center, &count);
    return (count == 1) ? MPI_SUCCESS : MPI_ERR_IO;
}

void set_zero(ptr_t buff, int size) {
    for(int i = 0; i < size; ++i) {
        buff[i] = 0.0;
    }
}

ptr_t memory_alloc(const dist_grid_info_t *info) {
    ptr_t tmp;
    int x_start = info->halo_size_x, x_end = info->local_size_x + info->halo_size_x;
    int y_start = info->halo_size_y, y_end = info->local_size_y + info->halo_size_y;
    int z_start = info->halo_size_z, z_end = info->local_size_z + info->halo_size_z;
    int nx = info->local_size_x + 2 * info->halo_size_x;
    int ny = info->local_size_y + 2 * info->halo_size_y;
    int nz = info->local_size_z + 2 * info->halo_size_z;
    int ret = MPI_Alloc_mem(sizeof(data_t) * nx * ny * nz, MPI_INFO_NULL, &tmp);
    (void)(x_start); // unused variable
    CHECK_ERROR(ret, NULL)
    /* zeroing halos */
    set_zero(tmp+INDEX(0,0,0,nx,ny), nx*ny*info->halo_size_z);
    for(int z = z_start; z < z_end; ++z) {
        set_zero(tmp+INDEX(0,0,z,nx,ny), nx*info->halo_size_y);
        for(int y = y_start; y < y_end; ++y) {
            set_zero(tmp+INDEX(0,y,z,nx,ny), info->halo_size_x);
            set_zero(tmp+INDEX(x_end,y,z,nx,ny), info->halo_size_x);
        }
        set_zero(tmp+INDEX(0,y_end,z,nx,ny), nx*info->halo_size_y);
    }
    set_zero(tmp+INDEX(0,0,z_end,nx,ny), nx*ny*info->halo_size_z);
    return tmp;
}
