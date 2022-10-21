#pragma once

#include <array>
#include <iostream>
#include <vector>

using value_type = double;

constexpr int dimension = 2;

struct ParticleT {
  std::array<value_type, dimension> pos{};  ///< the position of the particles
  value_type q{};    ///< the physical value associated to the particles
  value_type sum{};  ///< the sum (potential)
  std::array<value_type, dimension> force{};  ///< the force (potential)

  inline friend auto operator<<(std::ostream &os, const ParticleT &part)
      -> std::ostream & {
    os << "[ " << part.pos[0] << ", " << part.pos[1] << "] "
       << " [" << part.q << "] [" << part.sum << ", " << part.force[0] << ", "
       << part.force[1] << "] ";

    return os;
  }
};

struct LeafT {
  int index;                      //< index of the leaf
  std::array<int, 2> coord;       //< coordinate of the leaf in the grid
  int nb_part;                    //< number of particles
  ParticleT *particles;           //< pointer on the particles
  std::vector<int> interactions;  //< vector containing the interaction list
                                  //
};