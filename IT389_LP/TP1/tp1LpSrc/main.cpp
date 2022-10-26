
#include <chrono>
#include <cmath>
#include <iostream>

#include "direct.hpp"
#include "my_types.hpp"
#include "utils.hpp"

//
//  g++-11 -O3 -o main main.cpp
//  g++-11 -O3 -o main main.cpp -fopenmp
int main() {
  //

  // Height of the quadtree
  constexpr int two = 2;

  constexpr int h = 6;
  const int nb_part_per_leaf = 10;
  bool mutual{false};

  // number of cells in one dimension
  int n_1 = std::pow(two, h);
  std::cout << "n: " << n_1 << std::endl;
  std::cout << "Number of cells " << n_1 * n_1 << std::endl;
  // read or generate the particles
  std::vector<ParticleT> particles;
  // build grid [0,1]^dimension
  std::vector<LeafT> grid(n_1 * n_1);

  // generate the vector of particles inside each box of the grid
  // the number of particles is nb_part_per_leaf if the define
  // RANDOM_NB_PARTICLES is not used otherwise it is random
  generate_random_particles_in_grid(n_1, nb_part_per_leaf, grid, particles);
  // sort them assorting the index if necessary
  std::cout << " Total number of particles: " << particles.size() << std::endl;
  //
  // build the list of neighbors for each box of the grid
  build_interactions(grid, mutual);
  // timer
  std::chrono::time_point<std::chrono::system_clock> start, end;

  ////////////////////////////////////////////////////////////
  ///
  /// compute the near-field interaction in sequential
  /// It is the reference for the potential and force calculation
  start = std::chrono::system_clock::now();
  compute_near_seq(grid);
  end = std::chrono::system_clock::now();
  std::chrono::duration<double, std::milli> elapsed_seconds = end - start;
  std::cout << "time(seq) " << elapsed_seconds.count() << " ms" << std::endl;
  // copy the particles to obtain the reference vector of the particles
  std::vector<ParticleT> particles_ref(particles);

#ifdef _OPENMP
  // to measure time with openmp you can also consider the function
  // omp_get_wtime() which returns elapsed wall clock time in seconds.
  clear_results(particles);

  start = std::chrono::system_clock::now();
  compute_near_for(grid);
  end = std::chrono::system_clock::now();
  elapsed_seconds = end - start;
  std::cout << "time(omp for) " << elapsed_seconds.count() << " ms"
            << std::endl;
  check_results(particles_ref, particles);

  clear_results(particles);
  start = std::chrono::system_clock::now();
  compute_near_taskloop(grid);
  end = std::chrono::system_clock::now();
  elapsed_seconds = end - start;
  std::cout << "time(omp taskloop) " << elapsed_seconds.count() << "ms"
            << std::endl;
  check_results(particles_ref, particles);

  clear_results(particles);
  start = std::chrono::system_clock::now();
  compute_near_task(grid);
  end = std::chrono::system_clock::now();
  elapsed_seconds = end - start;
  std::cout << "time(omp task) " << elapsed_seconds.count() << "ms"
            << std::endl;
  check_results(particles_ref, particles);

  ////////////////////////////////////////////////////////////

  std::cout << "\n\n ======= Mutual case =======\n";
  mutual = true;
  clear_results(particles);

  build_interactions(grid, mutual);

  start = std::chrono::system_clock::now();
  compute_near_mutual_seq(grid);
  end = std::chrono::system_clock::now();
  elapsed_seconds = end - start;
  std::cout << "time(mutual seq) " << elapsed_seconds.count() << "ms"
            << std::endl;
  check_results(particles_ref, particles);
  std::vector<ParticleT> particles_ref_mut(particles);

  clear_results(particles);
  start = std::chrono::system_clock::now();
  compute_near_mutual_omp(grid);
  end = std::chrono::system_clock::now();
  elapsed_seconds = end - start;
  std::cout << "time(mutual omp for) " << elapsed_seconds.count() << "ms"
            << std::endl;
  check_results(particles_ref_mut, particles);

  clear_results(particles);
  start = std::chrono::system_clock::now();
  compute_near_mutual_task(grid);
  end = std::chrono::system_clock::now();
  elapsed_seconds = end - start;
  std::cout << "time(mutual task) " << elapsed_seconds.count() << "ms"
            << std::endl;
  check_results(particles_ref_mut, particles);
#endif
  //
  //  write_to_fma("grid.fma", particles);
}