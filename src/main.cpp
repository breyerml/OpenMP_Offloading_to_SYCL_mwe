#include "simulation.hpp"

int main() {

  simulation sim{ "../data/init_data.bin", "../data/compute_data_1.bin" };

  sim.initialize_arrays();
  sim.compute_monodomain(39);

  return 0;
}
