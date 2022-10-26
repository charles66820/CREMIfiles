#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

//
//  Pour compiler : gcc -o block_matmult block_matmult.c  -fopenmp -O3
//  BS divise N
//
//////////////////////////////////////

void printMat(int N, float A[N][N]) {
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      printf(" %f ", A[i][j]);
    }
    printf(" \n ");
  }
}
float reset(int N, float *A) {
  float err;
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      A[i * N + j] = 0.0;
    }
  }
  return err;
}

float compareMat(int N, float *A, float *B) {
  float err = 0.0;
  for (int i = 0; i < N * N; ++i) {
    err = fmax(fabsf(A[i] - B[i]), err);
  }
  return err;
}
//////////////////////////////////////
typedef struct block {
  int shape[2];   //the shape of the block
  float *coeff;   // the coefficients
} block;

typedef struct tiled_matrix {
  int shape[2];             //  he shape of the block matrix
  struct block *mat_block;  //  a pointer on the bloc structure
} tiled_matrix;

// get the number of cols of the matrix 
int nb_cols(tiled_matrix *Abloc) {
  int N_col = 0;
  for (int j = 0; j < Abloc->shape[1]; ++j) {
    N_col += Abloc->mat_block[j].shape[1];
  }
  return N_col;
};
// get the number of rows of the matrix 
int nb_rows(tiled_matrix *Abloc) {
  int N_rows = 0;
  for (int j = 0; j < Abloc->shape[0]; ++j) {
    N_rows += Abloc->mat_block[Abloc->shape[0] * j].shape[1];
  }
  return N_rows;
};
//Convert a dense matrix to a tiled matrix
void to_tiled(int N, float *A, tiled_matrix *Abloc, int BS) {
  // printf(" N= %d BS= %d\n", N, BS);
  // printMat(N, A);
  int nb_bloc = N / BS;
  if (N % BS != 0) {
    nb_bloc += 1;
  }
  // printf("nb_bloc: %d\n", nb_bloc);
  if (Abloc == NULL) {
    Abloc = malloc(sizeof(struct tiled_matrix));
  }
  Abloc->shape[0] = nb_bloc;
  Abloc->shape[1] = nb_bloc;
  Abloc->mat_block = malloc(sizeof(block) * nb_bloc * nb_bloc);

  int pos_row = 0;
  int pos_col = 0;
  for (int i = 0; i < nb_bloc; ++i) {
    int size_bloc_i = BS;
    for (int j = 0; j < nb_bloc; ++j) {
      // printf("\n bloc %d  pos (%d, %d) \n", i * Abloc->shape[0] + j, pos_row,
      //        pos_col);
      struct block *c_block = &(Abloc->mat_block[i * Abloc->shape[0] + j]);
      int size_bloc_j = BS;

      c_block->shape[0] = size_bloc_i;
      c_block->shape[1] = size_bloc_j;
      c_block->coeff = malloc(sizeof(float) * size_bloc_i * size_bloc_j);
      // fill the block
      for (int ii = 0; ii < size_bloc_i; ++ii) {
        for (int jj = 0; jj < size_bloc_j; ++jj) {
          c_block->coeff[jj + ii * size_bloc_j] =
              A[(pos_row + ii) * N + pos_col + jj];
        }
      }
      pos_col += BS;
    }
    pos_col = 0;
    pos_row += BS;
  }
}
void print_struct(struct tiled_matrix *Abloc) {
  printf("Block matrix %d x %d  ptr %d \n", Abloc->shape[0], Abloc->shape[1],
         Abloc->mat_block);
  if (Abloc->mat_block == NULL) {
    return;
  }
  // Abloc = malloc(sizeof(struct tiled_matrix));
  int nb_block = 0;
  for (int i = 0; i < Abloc->shape[0]; ++i) {
    for (int j = 0; j < Abloc->shape[1]; ++j) {
      struct block *c_block = &(Abloc->mat_block[i * Abloc->shape[0] + j]);

      printf("bloc %d  pos (%d, %d) size %d x %d ptr %x \n", nb_block, i, j,
             c_block->shape[0], c_block->shape[1], c_block->coeff);
      ++nb_block;
    }
  }
}
void to_dense(struct tiled_matrix *Abloc, int N, int M, float *A) {
  // printf("inside dense_A: %x  size %d x %d\n", A, N,M);

  int nb_bloc = 0;
  int pos_row = 0;
  int size_bloc_i;
  for (int i = 0; i < Abloc->shape[0]; ++i) {
    int pos_col = 0;
    for (int j = 0; j < Abloc->shape[1]; ++j) {
      struct block *c_block = &Abloc->mat_block[i * Abloc->shape[1] + j];
      size_bloc_i = c_block->shape[0];
      int size_bloc_j = c_block->shape[1];
      // fill the bloc
      for (int ii = 0; ii < size_bloc_i; ++ii) {
        for (int jj = 0; jj < size_bloc_j; ++jj) {
          A[(pos_row + ii) * M + pos_col + jj] =
              c_block->coeff[jj + ii * size_bloc_j];
        }
      }
      pos_col += size_bloc_j;
      ++nb_bloc;
    }
    pos_row += size_bloc_i;
  }
}
// reset the coefficients of a tilted matrix
void reset_tilted_matrix(struct tiled_matrix *Abloc) {
  for (int i = 0; i < Abloc->shape[0]; ++i) {
    for (int j = 0; j < Abloc->shape[1]; ++j) {
      struct block *c_block = &(Abloc->mat_block[i * Abloc->shape[0] + j]);
      for (int k = 0; k < c_block->shape[0] * c_block->shape[1]; ++k) {
        c_block->coeff[k] = 0.0;
      }
    }
  }
}
void printMatptr(int N, float *A) {
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      printf(" %f ", A[i * N + j]);
    }
    printf(" \n ");
  }
}

void block_matmul_seq(tiled_matrix *Abloc, tiled_matrix *Bbloc,
                      tiled_matrix *Cbloc) {

  int nb_row_block = Abloc->shape[0];
  int nb_col_block = Abloc->shape[1];

  for (int i = 0; i < nb_row_block; ++i) {
    for (int j = 0; j < nb_col_block; ++j) {
      struct block *c_block = &Cbloc->mat_block[i * Cbloc->shape[1] + j];

      for (int k = 0; k < nb_col_block; ++k) {
        struct block *a_block = &Abloc->mat_block[i * Abloc->shape[1] + k];
        struct block *b_block = &Bbloc->mat_block[k * Bbloc->shape[1] + j];

        for (int ii = 0; ii < c_block->shape[0]; ++ii) {
          for (int jj = 0; jj < c_block->shape[1]; ++jj) {
            float tmp = 0.0;
            //  #pragma omp simd reduction(+:tmp)
            for (int kk = 0; kk < a_block->shape[1]; ++kk) {
              tmp += a_block->coeff[ii * a_block->shape[1] + kk] *
                     b_block->coeff[kk * b_block->shape[1] + jj];
            }
            c_block->coeff[ii * c_block->shape[1] + jj] += tmp;
          }
        }
      }
    }
  }
}
void block_matmul_FJ(tiled_matrix *Abloc, tiled_matrix *Bbloc,
                     tiled_matrix *Cbloc) {

  int nb_row_block = Abloc->shape[0];
  int nb_col_block = Abloc->shape[1];
  printf("Todo block_matmul_FJ\n");

}


void bloc_matmul_depend(tiled_matrix *Abloc, tiled_matrix *Bbloc,
                        tiled_matrix *Cbloc) {
  int nb_row_block = Abloc->shape[0];
  int nb_col_block = Abloc->shape[1];
  printf("Todo bloc_matmul_depend\n");
}



int main() {
  const int N = 1000;
  const int BS = 100;
  /// 
  float *A, *B, *C;
  A = malloc(sizeof(float) * N * N);
  B = malloc(sizeof(float) * N * N);
  C = malloc(sizeof(float) * N * N);

  srand(time(NULL));
  for (int i = 0; i < N * N; ++i) {
    A[i] = rand() / (double)RAND_MAX;
    B[i] = rand() / (double)RAND_MAX;
    C[i] = 0.0;
  }
  tiled_matrix Abloc = {0, 0, NULL};
  tiled_matrix Bbloc = {0, 0, NULL};
  tiled_matrix Cbloc = {0, 0, NULL};
  tiled_matrix Dbloc = {0, 0, NULL};
  to_tiled(N, A, &Abloc, BS);
  to_tiled(N, B, &Bbloc, BS);
  to_tiled(N, C, &Cbloc, BS);
  to_tiled(N, C, &Dbloc, BS);

  //
  clock_t start, stop;
  double elapsed, dstop, dstart;
  //
  // for cache
  block_matmul_seq(&Abloc, &Bbloc, &Cbloc);
  reset_tilted_matrix(&Cbloc);
  // Start computation
  start = clock();
  block_matmul_seq(&Abloc, &Bbloc, &Cbloc);
  stop = clock();
  elapsed = (double)(stop - start) * 1000.0 / CLOCKS_PER_SEC;
  printf("(bloc) Seq Time elapsed in ms: %f\n", elapsed);

  float *dense_C = malloc(sizeof(float) * N * N);
  to_dense(&Cbloc, N, N, dense_C); //

  // printMatptr( N, dense_C);
  // printf("dense \n");

  //  printMatptr( N, C);
  //  for (int i =0; i < N*N; ++i){
  //   D[i] = dense_C[i] - C[i];
  //  }
  //  printMatptr( N, D);

  //
  // reset_tilted_matrix(&Cbloc);

  /////////////////////////////////////
  // Fork join
  start = clock();
  block_matmul_FJ(&Abloc, &Bbloc, &Dbloc);
  stop = clock();
  elapsed = (double)(stop - start) * 1000.0 / CLOCKS_PER_SEC;
  printf("FJ Time elapsed in ms: %f\n", elapsed);
  to_dense(&Dbloc, N, N, C); //

  float err = compareMat(N, dense_C, C);
  printf("Error: %f \n", err);

  reset_tilted_matrix(&Cbloc);
  start = clock();
  bloc_matmul_depend(&Abloc, &Bbloc, &Cbloc); 
  stop = clock();
  elapsed = (double)(stop - start) * 1000.0 / CLOCKS_PER_SEC;
  printf("tasks depend Time elapsed in ms: %f\n", elapsed);
  to_dense(&Dbloc, N, N, C); 
  // 
  err = compareMat(N, dense_C, C);
  printf("Error: %f \n", err);
}
