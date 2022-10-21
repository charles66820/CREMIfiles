#include <math.h>
#include <stdio.h>


int main ()
{	 
  double x, pi, sum = 0.0;
  double PI25DT = 3.141592653589793238462643;
  static long num_steps = 1000000;

  
  double step = 1.0/(double) num_steps;
  int start = 1 ;
  int end   = num_steps;
  
  for (int i=start;i<= end; i++){
    x = (i-0.5)*step;
    sum = sum + 4.0/(1.0+x*x);
  }
  
  pi = step * sum;
  printf("pi := %.16e  %.e\n", pi, fabs(pi - PI25DT));
  return 0 ;
}
