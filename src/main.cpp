#include "simulation.hpp"

int main() {

  simulation sim{ "../data/init_data_2x2.bin", "../data/compute_data_2x2.bin" };

  sim.initialize_arrays();
  sim.compute_monodomain(10);

  return 0;
}
