#include <fstream>
#include <iostream>

template <typename T, typename U>
void write_to_file(std::ostream& out, const T* ptr, U size) {
  out.write(reinterpret_cast<char*>(&size), sizeof(U));
  out.write(reinterpret_cast<const char*>(ptr), size * sizeof(T));
}

template <typename T>
void write_to_file(std::ostream& out, T val) {
  out.write(reinterpret_cast<char*>(&val), sizeof(T));
}


#define EXPORT_INIT_PARAMETER_DATA(file_name)                                                \
  std::cout << "Started exporting parameters to '" << file_name << "'\n";                    \
  std::ofstream file(file_name, std::ios::out | std::ios::trunc | std::ios::binary);         \
                                                                                             \
  write_to_file(file, statesOneInstanceParameter, nStates);                                  \
  write_to_file(file, algebraicsForTransferIndicesParameter, nAlgebraicsForTransferIndices); \
  write_to_file(file, statesForTransferIndicesParameter, nStatesForTransferIndices);         \
  write_to_file(file, firingEventsParameter, nFiringEvents);                                 \
  write_to_file(file, setSpecificStatesFrequencyJitterParameter, nFrequencyJitter);          \
  write_to_file(file, motorUnitNoParameter, nFibersToCompute);                               \
  write_to_file(file, fiberStimulationPointIndexParameter, nFibersToCompute);                \
  write_to_file(file, lastStimulationCheckTimeParameter, nFibersToCompute);                  \
  write_to_file(file, setSpecificStatesCallFrequencyParameter, nFibersToCompute);            \
  write_to_file(file, setSpecificStatesRepeatAfterFirstCallParameter, nFibersToCompute);     \
  write_to_file(file, setSpecificStatesCallEnableBeginParameter, nFibersToCompute);          \
                                                                                             \
  file.close();                                                                              \
  std::cout << "Finished exporting parameters!" << std::endl;


#define EXPORT_COMPUTE_PARAMETER_DATA(file_name)                                     \
  std::cout << "Started exporting parameters to '" << file_name << "'\n";            \
  std::ofstream file(file_name, std::ios::out | std::ios::trunc | std::ios::binary); \
                                                                                     \
  write_to_file(file, parameters, nParametersTotal);                                 \
  write_to_file(file, algebraicsForTransfer, nAlgebraicsForTransfer);                \
  write_to_file(file, statesForTransfer, nStatesForTransfer);                        \
  write_to_file(file, elementLengths, nElementLengths);                              \
  write_to_file(file, startTime);                                                    \
  write_to_file(file, timeStepWidthSplitting);                                       \
  write_to_file(file, nTimeStepsSplitting);                                          \
  write_to_file(file, dt0D);                                                         \
  write_to_file(file, nTimeSteps0D);                                                 \
  write_to_file(file, dt1D);                                                         \
  write_to_file(file, nTimeSteps1D);                                                 \
  write_to_file(file, prefactor);                                                    \
  write_to_file(file, valueForStimulatedPoint);                                      \
                                                                                    \
  file.close();                                                                     \
  std::cout << "Finished exporting parameters!" << std::endl;
