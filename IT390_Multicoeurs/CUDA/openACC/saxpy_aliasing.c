

int main(int argc, char **argv){
  int N=1000;
  float a = 3.0f;
  float *x = (float*) malloc ( N * sizeof(float));
  float * restrict y = (float*) malloc ( N * sizeof(float));

  for (int i = 0; i < N; ++i) {
    x[i] = 2.0f;
    y[i] = 1.0f;
  }
#pragma acc kernels
  for (int i = 0; i < N; ++i) {
    y[i] = a * x[i] + y[i];
  }
}
