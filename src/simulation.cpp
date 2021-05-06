#include "simulation.hpp"
#include "hodgkin_huxley.hpp"

#include <fstream>
#include <iostream>
#ifdef ENABLE_TIMING
#include <chrono>
#endif

#ifdef DEBUG_INFO
#define READ_FROM_FILE(in, val) \
read_from_file(in, val);        \
print_debug_info(#val, val)
#else
#define READ_FROM_FILE(in, val) read_from_file(in, val)
#endif

template <typename T, typename U = int>
void read_from_file(std::ifstream& in, std::vector<T>& vec) {
  U size;
  in.read(reinterpret_cast<char*>(&size), sizeof(U));
  vec.resize(size);
  in.read(reinterpret_cast<char*>(vec.data()), sizeof(T) * size);
}
template <typename T>
void read_from_file(std::ifstream& in, T& val) {
  in.read(reinterpret_cast<char*>(&val), sizeof(T));
}

template <typename T>
void print_debug_info(const std::string_view name, const std::vector<T>& vec) {
  std::cout << name << ": " << vec.size() << '\n';
  for (const T val : vec) {
    if constexpr (std::is_same_v<T, char>) {
      std::cout << static_cast<int>(val) << ' ';
    } else {
      std::cout << val << ' ';
    }
  }
  std::cout << "\n\n";
}
template <typename T>
void print_debug_info(const std::string_view name, const T& val) {
  std::cout << name << ":\n";
  std::cout << val << "\n\n";
}

simulation::simulation(const std::string_view init_params, const std::string_view compute_params) {
  {
    // read parameter for array initialization
    std::ifstream in{ init_params.data(), std::ios::in | std::ios::binary };
    READ_FROM_FILE(in, statesOneInstanceParameter);
    READ_FROM_FILE(in, algebraicsForTransferIndicesParameter);
    READ_FROM_FILE(in, statesForTransferIndicesParameter);
    READ_FROM_FILE(in, firingEventsParameter);
    READ_FROM_FILE(in, setSpecificStatesFrequencyJitterParameter);
    READ_FROM_FILE(in, motorUnitNoParameter);
    READ_FROM_FILE(in, fiberStimulationPointIndexParameter);
    READ_FROM_FILE(in, lastStimulationCheckTimeParameter);
    READ_FROM_FILE(in, setSpecificStatesCallFrequencyParameter);
    READ_FROM_FILE(in, setSpecificStatesRepeatAfterFirstCallParameter);
    READ_FROM_FILE(in, setSpecificStatesCallEnableBeginParameter);
  }
  {
    // read parameter for monodomain computation
    std::ifstream in{ compute_params.data(), std::ios::in | std::ios::binary };
    READ_FROM_FILE(in, parameters);
    READ_FROM_FILE(in, algebraicsForTransfer);
    READ_FROM_FILE(in, statesForTransfer);
    READ_FROM_FILE(in, elementLengths);
    READ_FROM_FILE(in, startTime);
    READ_FROM_FILE(in, timeStepWidthSplitting);
    READ_FROM_FILE(in, nTimeStepsSplitting);
    READ_FROM_FILE(in, dt0D);
    READ_FROM_FILE(in, nTimeSteps0D);
    READ_FROM_FILE(in, dt1D);
    READ_FROM_FILE(in, nTimeSteps1D);
    READ_FROM_FILE(in, prefactor);
    READ_FROM_FILE(in, valueForStimulatedPoint);
  }
}

#undef READ_FROM_FILE

void simulation::initialize_arrays() {
#ifdef ENABLE_TIMING
  auto begin = std::chrono::system_clock::now();
#endif
  std::cout << "Initializing arrays ...\n";

  initializeArrays(statesOneInstanceParameter.data(),
                   algebraicsForTransferIndicesParameter.data(),
                   statesForTransferIndicesParameter.data(),
                   firingEventsParameter.data(),
                   setSpecificStatesFrequencyJitterParameter.data(),
                   motorUnitNoParameter.data(),
                   fiberStimulationPointIndexParameter.data(),
                   lastStimulationCheckTimeParameter.data(),
                   setSpecificStatesCallFrequencyParameter.data(),
                   setSpecificStatesRepeatAfterFirstCallParameter.data(),
                   setSpecificStatesCallEnableBeginParameter.data());

  std::cout << "Done.\n";
#ifdef ENABLE_TIMING
  auto end = std::chrono::system_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
  std::cout << "Elapsed time " << elapsed.count() << " ms\n";
#endif
}

void simulation::compute_monodomain(const std::size_t repeats, const bool print) {
#ifdef ENABLE_TIMING
  auto begin = std::chrono::system_clock::now();
#endif
  std::cout << "Computing monodomain ...\n";

  for (std::size_t i = 1; i <= repeats; ++i) {
    if (print) {
      std::cout << "timestep " << i << '/' << repeats << '\n';
    }
    computeMonodomain(parameters.data(),
                      algebraicsForTransfer.data(),
                      statesForTransfer.data(),
                      elementLengths.data(),
                      startTime,
                      timeStepWidthSplitting,
                      nTimeStepsSplitting,
                      dt0D,
                      nTimeSteps0D,
                      dt1D,
                      nTimeSteps1D,
                      prefactor,
                      valueForStimulatedPoint);
  }
  std::cout << "Done.\n";
#ifdef ENABLE_TIMING
  auto end = std::chrono::system_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
  std::cout << "Elapsed time " << elapsed.count() << " ms\n";
#endif
}