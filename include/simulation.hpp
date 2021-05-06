#ifndef MWE_INCLUDE_SIMULATION_HPP_
#define MWE_INCLUDE_SIMULATION_HPP_

#include <string_view>
#include <vector>

class simulation {
 public:
  simulation(std::string_view init_params, std::string_view compute_params);

  void initialize_arrays();
  void compute_monodomain();

 private:
  // parameter for array initialization
  std::vector<double> statesOneInstanceParameter{};
  std::vector<int> algebraicsForTransferIndicesParameter{};
  std::vector<int> statesForTransferIndicesParameter{};
  std::vector<char> firingEventsParameter{};
  std::vector<double> setSpecificStatesFrequencyJitterParameter{};
  std::vector<int> motorUnitNoParameter{};
  std::vector<int> fiberStimulationPointIndexParameter{};
  std::vector<double> lastStimulationCheckTimeParameter{};
  std::vector<double> setSpecificStatesCallFrequencyParameter{};
  std::vector<double> setSpecificStatesRepeatAfterFirstCallParameter{};
  std::vector<double> setSpecificStatesCallEnableBeginParameter{};

  // parameter for monodomain computation
  std::vector<float> parameters{};
  std::vector<double> algebraicsForTransfer{};
  std::vector<double> statesForTransfer{};
  std::vector<float> elementLengths{};
  double startTime{};
  double timeStepWidthSplitting{};
  int nTimeStepsSplitting{};
  double dt0D{};
  int nTimeSteps0D{};
  double dt1D{};
  int nTimeSteps1D{};
  double prefactor{};
  double valueForStimulatedPoint{};
};

#endif // MWE_INCLUDE_SIMULATION_HPP_
