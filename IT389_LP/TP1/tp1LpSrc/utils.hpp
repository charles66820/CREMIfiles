#pragma once

#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>
#include <string>

#include "my_types.hpp"

void print_leaf(const LeafT &leaf) {
  std::cout << "leaf " << leaf.index << " coord " << leaf.coord[0] << " "
            << leaf.coord[1] << "\n   particles(" << leaf.nb_part << ") "
            << "\n";
  auto p = leaf.particles;
  for (int i = 0; i < leaf.nb_part; ++i) {
    std::cout << "    " << p[i] << "\n";
  }
  std::cout << "   interactions (" << leaf.interactions.size() << "):";
  for (auto e : leaf.interactions) {
    std::cout << " " << e;
  }
  std::cout << "\n";
}

int set_row_index(const int n_1d, const int i, const int j) {
  return j * n_1d + i;
}
void set_row_coord(int k, const int n_1d, LeafT &leaf) {
  leaf.index = k;
  leaf.coord[1] = k / n_1d;
  leaf.coord[0] = k - n_1d * leaf.coord[1];
}
void set_morton_index(int k, const int n_1d, LeafT &leaf) {}
//
// generate nb_part_per_leaf particles inside each leaf of the grid
void generate_random_particles_in_grid(const int n_1d,
                                       const int nb_part_per_leaf,
                                       std::vector<LeafT> &grid,
                                       std::vector<ParticleT> &particles) {
  const auto seed{33};
  std::mt19937_64 gen(seed);
  std::uniform_real_distribution<> dist(0, 1);
  // distribution uniforme pour le nombre de particules par boite
  std::uniform_int_distribution<> dist1(1, 150);

  value_type dx{value_type(2.0) / value_type(n_1d)},
      dy{value_type(2.0) / value_type(n_1d)};
  value_type x_min{}, y_min{};
  // set the right number of particles
  int nb_particles{0};
  for (int k = 0; k < grid.size(); ++k) {
    auto &leaf = grid[k];
#ifdef RANDOM_NB_PARTICLES
    int nb_part = dist1(gen);  // nb_part_per_leaf;
#else
    int nb_part = nb_part_per_leaf;
#endif
    leaf.nb_part = nb_part;
    nb_particles += leaf.nb_part;
  }
  particles.resize(nb_particles);
  // fill the leaf and particles structures
  int count_part{0};
  for (int k = 0; k < grid.size(); ++k) {
    auto &leaf = grid[k];
    set_row_coord(k, n_1d, leaf);
    x_min = leaf.coord[0] * dx;
    y_min = leaf.coord[1] * dy;
    // generate the particles inside the cell
    leaf.particles = &particles[count_part];

    for (int n = 0; n < leaf.nb_part; ++n) {
      particles[count_part].pos[0] = x_min + dist(gen) * dx;
      particles[count_part].pos[1] = y_min + dist(gen) * dx;
      particles[count_part].q = dist(gen);  // q between (0,1)
      ++count_part;
    }  // end particle loop
  }    // end leaf loop
}

void build_leaf_interactions(const int n_1d, LeafT &leaf, const bool mutual) {
  int i = leaf.coord[0];
  int j = leaf.coord[1];
  if (leaf.interactions.size() > 0) {
    leaf.interactions.clear();
  }
  int mm = 1;
  for (int k = i - mm; k <= i + mm; ++k) {
    if (k < 0 || k >= n_1d) {
      continue;
    }
    for (int l = j - mm; l <= j + mm; ++l) {
      if (l < 0 || l >= n_1d) {
        continue;
      }
      if (k == i && l == j) {
        continue;
      }
      auto index = set_row_index(n_1d, k, l);
      if (mutual && index > leaf.index) {
        continue;
      }
      leaf.interactions.push_back(index);
    }
  }
  std::sort(leaf.interactions.begin(), leaf.interactions.end());
}
void build_interactions(std::vector<LeafT> &grid, const bool mutual) {
  const int n_1d = std::sqrt(grid.size());
  for (auto &leaf : grid) {
    build_leaf_interactions(n_1d, leaf, mutual);
  }
}

void write_to_fma(std::string filename, std::vector<ParticleT> &part) {
  std::ofstream file(filename);

  file << "8  6 " << dimension << " 1" << std::endl;
  file << part.size() << " 0.5 0.5 0.5 \n";
  for (int i = 0; i < part.size(); ++i) {
    file << part[i].pos[0] << " " << part[i].pos[1] << " " << part[i].q << " "
         << part[i].sum << " " << part[i].force[0] << " " << part[i].force[1]
         << "\n";
  }
}

void clear_results(std::vector<ParticleT> &particles) {
  for (auto &p : particles) {
    p.sum = value_type(0.0);
    p.force[0] = value_type(0.0);
    p.force[1] = value_type(0.0);
  }
}

bool check_results(std::vector<ParticleT> &particles_ref,
                   std::vector<ParticleT> &particles) {
  value_type errorSum{0.0};
  value_type errorFX{0.0};
  value_type errorFY{0.0};
  for (int i = 0; i < particles_ref.size(); ++i) {
    auto &p = particles[i];
    auto &p_ref = particles_ref[i];
    errorSum += (p.sum - p_ref.sum) * (p.sum - p_ref.sum);
    errorFX += (p.force[0] - p_ref.force[0]) * (p.force[0] - p_ref.force[0]);
    errorFY += (p.force[1] - p_ref.force[1]) * (p.force[1] - p_ref.force[1]);
    //    std::cout << i <<"  " << p_ref << "  " << p << "\n";
  }
  std::cout << " errorSum: " << std::sqrt(errorSum) << std::endl
            << " errorFX:  " << std::sqrt(errorFX) << std::endl
            << " errorFY:  " << std::sqrt(errorFY) << std::endl;
  bool error{false};
  value_type eps = 1.e-6;
  if (errorSum < eps && errorFX < eps && errorFY < eps) {
    error = true;
  }
  return error;
}