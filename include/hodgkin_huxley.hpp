#ifndef MWE_INCLUDE_HODGKIN_HUXLEY_HPP_
#define MWE_INCLUDE_HODGKIN_HUXLEY_HPP_

void initializeArrays(const double* statesOneInstanceParameter,
                      [[maybe_unused]] const int* algebraicsForTransferIndicesParameter,
                      const int* statesForTransferIndicesParameter,
                      const char* firingEventsParameter,
                      const double* setSpecificStatesFrequencyJitterParameter,
                      const int* motorUnitNoParameter,
                      const int* fiberStimulationPointIndexParameter,
                      const double* lastStimulationCheckTimeParameter,
                      const double* setSpecificStatesCallFrequencyParameter,
                      const double* setSpecificStatesRepeatAfterFirstCallParameter,
                      const double* setSpecificStatesCallEnableBeginParameter);

void computeMonodomain(const float* parameters,
                       double* algebraicsForTransfer,
                       double* statesForTransfer,
                       const float* elementLengths,
                       double startTime,
                       double timeStepWidthSplitting,
                       int nTimeStepsSplitting,
                       double dt0D,
                       int nTimeSteps0D,
                       double dt1D,
                       int nTimeSteps1D,
                       double prefactor,
                       double valueForStimulatedPoint);

#endif // MWE_INCLUDE_HODGKIN_HUXLEY_HPP_
