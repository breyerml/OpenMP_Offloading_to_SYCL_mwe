#include "simulation.hpp"

int main() {

  simulation sim{ "../data/init_data.out", "../data/compute_data_1.out" };

  sim.initialize_arrays();
  sim.compute_monodomain(39);

  return 0;
}
