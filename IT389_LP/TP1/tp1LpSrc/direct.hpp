#pragma once

#include "my_types.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif
// kernel 1/r
void p2p_inner(const int n, ParticleT *part) {
#pragma omp simd
  for (int i = 0; i < n; ++i) {
    auto &x1 = part[i].pos;
    auto &F1 = part[i].force;
    auto &pot1 = part[i].sum;
    auto q1 = part[i].q;

    for (int j = 0; j < i; ++j) {
      auto diff = part[j].pos;
      auto q2 = part[j].q;
      auto &F2 = part[j].force;
      auto &pot2 = part[j].sum;

      diff[0] -= x1[0];
      //-diff[0];
      diff[1] -= x1[1];
      // - diff[1];
      value_type inv_norm2 = diff[0] * diff[0] + diff[1] * diff[1];
      value_type inv_norm = std::sqrt(inv_norm2);
      pot1 += q2 / inv_norm;
      pot2 += q1 / inv_norm;
      F1[0] += q2 * diff[0] / (inv_norm * inv_norm2);
      F1[1] += q2 * diff[1] / (inv_norm * inv_norm2);
      F2[0] -= q1 * diff[0] / (inv_norm * inv_norm2);
      F2[1] -= q1 * diff[1] / (inv_norm * inv_norm2);
      //
    }
  };
}
void p2p_outer(const int n, ParticleT *part, const int n_s, ParticleT *part_s) {
#pragma omp simd
  for (int i = 0; i < n; ++i) {
    auto &x1 = part[i].pos;
    auto &F1 = part[i].force;
    auto &pot1 = part[i].sum;
    auto q1 = part[i].q;

    for (int j = 0; j < n_s; ++j) {
      auto diff = part_s[j].pos;
      auto q2 = part_s[j].q;

      diff[0] -= x1[0];
      //-diff[0];
      diff[1] -= x1[1];
      // - diff[1];
      value_type norm2 = diff[0] * diff[0] + diff[1] * diff[1];
      value_type norm = std::sqrt(norm2);
      pot1 += q2 / norm;
      F1[0] += q2 * diff[0] / (norm * norm2);
      F1[1] += q2 * diff[1] / (norm * norm2);
      //
    }
  }
}

void p2p_mutual(const int n, ParticleT *part, const int n_s,
                ParticleT *part_s) {
  for (int i = 0; i < n; ++i) {
    auto &x1 = part[i].pos;
    auto q1 = part[i].q;
    auto &pot1 = part[i].sum;
    auto &F1 = part[i].force;
    value_type accX{0.0}, accY{0.0};
    for (int j = 0; j < n_s; ++j) {
      auto diff = part_s[j].pos;
      auto q2 = part_s[j].q;
      auto &pot2 = part_s[j].sum;
      auto &F2 = part_s[j].force;
      // diff = x2 - x1
      diff[0] -= x1[0];
      diff[1] -= x1[1];
      //
      value_type norm2 = diff[0] * diff[0] + diff[1] * diff[1];
      value_type norm = std::sqrt(norm2);
      pot1 += q2 / norm;
      pot2 += q1 / norm;
      value_type FX = diff[0] / (norm * norm2);
      value_type FY = diff[1] / (norm * norm2);
      F2[0] = F2[0] - q1 * FX;
      F2[1] = F2[1] - q1 * FY;
      accX = accX + q2 * FX;
      accY = accY + q2 * FY;
      //
    }
    F1[0] += accX;
    F1[1] += accY;
  }
}

void compute_near_seq(std::vector<LeafT> &grid) {
  for (int k = 0; k < grid.size(); ++k) {
    auto my_part = grid[k].particles;
    auto n1 = grid[k].nb_part;

    for (auto &p2p_inter : grid[k].interactions) {
      if (grid[p2p_inter].nb_part > 0) {
        p2p_outer(n1, my_part, grid[p2p_inter].nb_part,
                  grid[p2p_inter].particles);
      }
    }
    p2p_inner(n1, my_part);
  }
}
void compute_near_mutual_seq(std::vector<LeafT> &grid) {
  for (int k = 0; k < grid.size(); ++k) {
    auto my_part = grid[k].particles;
    auto n1 = grid[k].nb_part;

    for (auto &p2p_inter : grid[k].interactions) {
      if (grid[p2p_inter].nb_part > 0) {
        p2p_mutual(n1, my_part, grid[p2p_inter].nb_part,
                   grid[p2p_inter].particles);
      }
    }
    p2p_inner(n1, my_part);
  }
}

void compute_near_for(std::vector<LeafT> &grid) {
#pragma omp parallel for schedule(static)
  for (int k = 0; k < grid.size(); ++k) {
    auto my_part = grid[k].particles;
    auto n1 = grid[k].nb_part;

    for (auto &p2p_inter : grid[k].interactions) {
      if (grid[p2p_inter].nb_part > 0) {
        p2p_outer(n1, my_part, grid[p2p_inter].nb_part,
                  grid[p2p_inter].particles);
      }
    }
    p2p_inner(n1, my_part);
  }
}
void compute_near_taskloop(std::vector<LeafT> &grid) {
#pragma omp parallel
#pragma omp single
#pragma omp taskloop
  for (int k = 0; k < grid.size(); ++k) {
    auto my_part = grid[k].particles;
    auto n1 = grid[k].nb_part;

    for (auto &p2p_inter : grid[k].interactions) {
      if (grid[p2p_inter].nb_part > 0) {
        p2p_outer(n1, my_part, grid[p2p_inter].nb_part,
                  grid[p2p_inter].particles);
      }
    }
    p2p_inner(n1, my_part);
  }
}

void compute_near_task(std::vector<LeafT> &grid) {
#pragma omp parallel
#pragma omp single
  for (int k = 0; k < grid.size(); ++k) {
#pragma omp task firstprivate(k) untied
    {
      auto my_part = grid[k].particles;
      auto n1 = grid[k].nb_part;

      for (auto &p2p_inter : grid[k].interactions) {
        if (grid[p2p_inter].nb_part > 0) {
          p2p_outer(n1, my_part, grid[p2p_inter].nb_part,
                    grid[p2p_inter].particles);
        }
      }
      p2p_inner(n1, my_part);
    }
  }
}

void compute_near_mutual_omp(std::vector<LeafT> &grid) {
#pragma omp parallel for schedule(static)
  for (int k = 0; k < grid.size(); ++k) {
    auto my_part = grid[k].particles;
    auto n1 = grid[k].nb_part;

    for (auto &p2p_inter : grid[k].interactions) {
      if (grid[p2p_inter].nb_part > 0) {
        p2p_mutual(n1, my_part, grid[p2p_inter].nb_part,
                   grid[p2p_inter].particles);
      }
    }
    p2p_inner(n1, my_part);
  }
}

void compute_near_mutual_task(std::vector<LeafT> &grid) {
#pragma omp parallel
#pragma omp single
  for (int k = 0; k < grid.size(); ++k) {
#pragma omp task firstprivate(k) untied
    {
      auto my_part = grid[k].particles;
      auto n1 = grid[k].nb_part;

      for (auto &p2p_inter : grid[k].interactions) {
        if (grid[p2p_inter].nb_part > 0) {
          p2p_mutual(n1, my_part, grid[p2p_inter].nb_part,
                     grid[p2p_inter].particles);
        }
      }
      p2p_inner(n1, my_part);
    }
  }
}
