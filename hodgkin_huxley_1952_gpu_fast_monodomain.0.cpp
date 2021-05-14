#include "hodgkin_huxley.hpp" // TODO: remove for actual simulation

typedef double real;
#include <cmath>
//#include <omp.h>
#include <iostream>
#include <vector>

#pragma omp declare target
/*
   There are a total of 9 entries in the algebraic variable array.
   There are a total of 4 entries in each of the rate and state variable arrays.
   There are a total of 9 entries in the constant variable array.
 */
/*
 * VOI is time in component environment (millisecond).
 * STATES[0] is V in component membrane (millivolt).
 * CONSTANTS[0] is E_R in component membrane (millivolt).
 * CONSTANTS[1] is Cm in component membrane (microF_per_cm2).
 * ALGEBRAIC[0] is i_Na in component sodium_channel (microA_per_cm2).
 * ALGEBRAIC[4] is i_K in component potassium_channel (microA_per_cm2).
 * ALGEBRAIC[8] is i_L in component leakage_current (microA_per_cm2).
 * CONSTANTS[2] is i_Stim in component membrane (microA_per_cm2).
 * CONSTANTS[3] is g_Na in component sodium_channel (milliS_per_cm2).
 * CONSTANTS[6] is E_Na in component sodium_channel (millivolt).
 * STATES[1] is m in component sodium_channel_m_gate (dimensionless).
 * STATES[2] is h in component sodium_channel_h_gate (dimensionless).
 * ALGEBRAIC[1] is alpha_m in component sodium_channel_m_gate (per_millisecond).
 * ALGEBRAIC[5] is beta_m in component sodium_channel_m_gate (per_millisecond).
 * ALGEBRAIC[2] is alpha_h in component sodium_channel_h_gate (per_millisecond).
 * ALGEBRAIC[6] is beta_h in component sodium_channel_h_gate (per_millisecond).
 * CONSTANTS[4] is g_K in component potassium_channel (milliS_per_cm2).
 * CONSTANTS[7] is E_K in component potassium_channel (millivolt).
 * STATES[3] is n in component potassium_channel_n_gate (dimensionless).
 * ALGEBRAIC[3] is alpha_n in component potassium_channel_n_gate (per_millisecond).
 * ALGEBRAIC[7] is beta_n in component potassium_channel_n_gate (per_millisecond).
 * CONSTANTS[5] is g_L in component leakage_current (milliS_per_cm2).
 * CONSTANTS[8] is E_L in component leakage_current (millivolt).
 * RATES[0] is d/dt V in component membrane (millivolt).
 * RATES[1] is d/dt m in component sodium_channel_m_gate (dimensionless).
 * RATES[2] is d/dt h in component sodium_channel_h_gate (dimensionless).
 * RATES[3] is d/dt n in component potassium_channel_n_gate (dimensionless).
 */


real log(real x)
{
  // Taylor expansion of the log function around x=1
  // Note: std::log does not work on GPU,
  // however, if this code runs on CPU, it is fine
#pragma omp \
#ifndef GPU
  return std::log(x);
#pragma omp \
#endif

  // Taylor approximation around 1, 3 or 9
  if (x < 2)
  {
    real t1 = x-1;
    real t1_2 = t1*t1;
    real t1_4 = t1_2*t1_2;
    real t1_6 = t1_4*t1_2;
    real result1 = t1 - 0.5*t1_2 + 1./3*t1_2*t1 - 0.25*t1_4 + 0.2*t1_4*t1 - 1./6*t1_6;
    return result1;
  }
  else if (x < 6)
  {
    real t3 = x-3;
    real t3_2 = t3*t3;
    real t3_4 = t3_2*t3_2;
    real result3 = 1.0986122886681098 + 1./3*t3 - 0.05555555555555555*t3_2 + 0.012345679012345678*t3_2*t3 - 0.0030864197530864196*t3_4;
    return result3;
  }
  else
  {
    real t9 = x-9;
    real t9_2 = t9*t9;
    real t9_4 = t9_2*t9_2;
    real result9 = 2.1972245773362196 + 1./9*t9 - 0.006172839506172839*t9_2 + 0.0004572473708276177*t9_2*t9 - 3.8103947568968146e-05*t9_4;
    return result9;
  }

  // The relative error of this implementation is below 0.04614465854334056 for x in [0.2,19].
  return 0.0;

}

/* This file was created by opendihu at 2021/5/6 16:01:08.
 * It is designed for the FastMonodomainSolver and contains code for offloading to GPU.
  */

// helper functions
real exponential(real x);
real pow2(real x);
real pow3(real x);
real pow4(real x);

real exponential(real x)
{
  //return Vc::exp(x);
  // it was determined the x is always in the range [-12,+12]

  // exp(x) = lim n→∞ (1 + x/n)^n, we set n=1024
  x = 1.0 + x / 1024.;
  for (int i = 0; i < 10; i++)
  {
    x *= x;
  }
  return x;

  // relative error of this implementation:
  // x    rel error
  // 0    0
  // 1    0.00048784455634225593
  // 3    0.0043763626896140342
  // 5    0.012093715791500804
  // 9    0.038557535762274039
  // 12   0.067389808619653505
}

real pow2(real x)
{
  return x*x;
}
real pow3(real x)
{
  return x*(pow2(x));
}

real pow4(real x)
{
  return pow2(pow2(x));
}


#pragma omp end declare target

// global size constants
const int nInstancesPerFiber = 1481;
const int nElementsOnFiber = 1480;
const int nFibersToCompute = 4;
const long long nInstancesToCompute = 5924;  // = nInstancesPerFiber*nFibersToCompute;
const int nStates = 4;
const int firingEventsNRows = 2;
const int firingEventsNColumns = 100;
const int frequencyJitterNColumns = 100;
const int nStatesTotal = 23696;  // = nInstancesToCompute*nStates;
const int nParametersTotal = 5924;  // = nInstancesToCompute*1;
const int nElementLengths = 5920;  // = nElementsOnFiber*nFibersToCompute;
const int nFiringEvents = 200;  // = firingEventsNRows*firingEventsNColumns;
const int nFrequencyJitter = 400;  // = nFibersToCompute*frequencyJitterNColumns;
const int nAlgebraicsForTransferIndices = 0;
const int nAlgebraicsForTransfer = 0;  // = nInstancesToCompute*nAlgebraicsForTransferIndices;
const int nStatesForTransferIndices = 1;
const int nStatesForTransfer = 5924;  // = nInstancesToCompute*nStatesForTransferIndices;


// the following code is generated by FastMonodomainSolver for offloading to GPU
// global variables to be stored on the target device
#pragma omp declare target
real states[nStatesTotal]                                     __attribute__ ((aligned (64)));             // including state 0 which is stored in vmValues
real statesOneInstance[nStates]                               __attribute__ ((aligned (64)));
int statesForTransferIndices[nStatesForTransferIndices]       __attribute__ ((aligned (64)));
char firingEvents[nFiringEvents]                              __attribute__ ((aligned (64)));
real setSpecificStatesFrequencyJitter[nFrequencyJitter]       __attribute__ ((aligned (64)));
char fiberIsCurrentlyStimulated[nFibersToCompute]             __attribute__ ((aligned (64)));
int motorUnitNo[nFibersToCompute]                             __attribute__ ((aligned (64)));
int fiberStimulationPointIndex[nFibersToCompute]              __attribute__ ((aligned (64)));
real lastStimulationCheckTime[nFibersToCompute]               __attribute__ ((aligned (64)));
real setSpecificStatesCallFrequency[nFibersToCompute]         __attribute__ ((aligned (64)));
real setSpecificStatesRepeatAfterFirstCall[nFibersToCompute]  __attribute__ ((aligned (64)));
real setSpecificStatesCallEnableBegin[nFibersToCompute]       __attribute__ ((aligned (64)));
real currentJitter[nFibersToCompute]                          __attribute__ ((aligned (64)));
int jitterIndex[nFibersToCompute]                             __attribute__ ((aligned (64)));

real vmValues[nInstancesToCompute]                            __attribute__ ((aligned (64)));

#pragma omp end declare target


// TODO: re-enable for actual simulation
//#ifdef __cplusplus
//extern "C"
//#endif
void initializeArrays(const double *statesOneInstanceParameter, const int *algebraicsForTransferIndicesParameter, const int *statesForTransferIndicesParameter,
                      const char *firingEventsParameter, const double *setSpecificStatesFrequencyJitterParameter, const int *motorUnitNoParameter,
                      const int *fiberStimulationPointIndexParameter, const double *lastStimulationCheckTimeParameter,
                      const double *setSpecificStatesCallFrequencyParameter, const double *setSpecificStatesRepeatAfterFirstCallParameter,
                      const double *setSpecificStatesCallEnableBeginParameter)
{
  for (int i = 0; i < nStates; i++)
    statesOneInstance[i] = statesOneInstanceParameter[i];



  for (int i = 0; i < nStatesForTransferIndices; i++)
    statesForTransferIndices[i] = statesForTransferIndicesParameter[i];

  for (int i = 0; i < nFiringEvents; i++)
    firingEvents[i] = firingEventsParameter[i];

  for (int i = 0; i < nFrequencyJitter; i++)
    setSpecificStatesFrequencyJitter[i] = setSpecificStatesFrequencyJitterParameter[i];

  for (int fiberNo = 0; fiberNo < nFibersToCompute; fiberNo++)
  {
    motorUnitNo[fiberNo] = motorUnitNoParameter[fiberNo];
    fiberStimulationPointIndex[fiberNo] = fiberStimulationPointIndexParameter[fiberNo];
    lastStimulationCheckTime[fiberNo] = lastStimulationCheckTimeParameter[fiberNo];
    setSpecificStatesCallFrequency[fiberNo] = setSpecificStatesCallFrequencyParameter[fiberNo];
    setSpecificStatesRepeatAfterFirstCall[fiberNo] = setSpecificStatesRepeatAfterFirstCallParameter[fiberNo];
    setSpecificStatesCallEnableBegin[fiberNo] = setSpecificStatesCallEnableBeginParameter[fiberNo];
  }

  // set variables to zero
  for (int fiberNo = 0; fiberNo < nFibersToCompute; fiberNo++)
  {
    fiberIsCurrentlyStimulated[fiberNo] = 0;
    currentJitter[fiberNo] = 0;
    jitterIndex[fiberNo] = 0;
  }

  // initialize vmValues
  const double state0 = statesOneInstance[0];
  for (int instanceToComputeNo = 0; instanceToComputeNo < nInstancesToCompute; instanceToComputeNo++)
  {
    vmValues[instanceToComputeNo] = state0;
  }


  // map values to target
  #pragma omp target update to(states[:nStatesTotal], statesOneInstance[:nStates], \
    statesForTransferIndices[:nStatesForTransferIndices], \
    firingEvents[:nFiringEvents], setSpecificStatesFrequencyJitter[:nFrequencyJitter], \
    motorUnitNo[:nFibersToCompute], fiberStimulationPointIndex[:nFibersToCompute], \
    lastStimulationCheckTime[:nFibersToCompute], setSpecificStatesCallFrequency[:nFibersToCompute], \
    setSpecificStatesRepeatAfterFirstCall[:nFibersToCompute], setSpecificStatesCallEnableBegin[:nFibersToCompute], \
    currentJitter[:nFibersToCompute], jitterIndex[:nFibersToCompute], vmValues[:nInstancesToCompute])

  // initialize states
  #pragma omp target
  {
    // copy given values to variables on target
    for (int fiberNo = 0; fiberNo < nFibersToCompute; fiberNo++)
    {
      for (int instanceNo = 0; instanceNo < nInstancesPerFiber; instanceNo++)
      {
        int instanceToComputeNo = fiberNo*nInstancesPerFiber + instanceNo;

        // The entries in states[0] to states[1*nInstancesToCompute - 1] are not used.
        // State zero is stored in vmValues instead.
        for (int stateNo = 1; stateNo < nStates; stateNo++)
        {
          states[stateNo*nInstancesToCompute + instanceToComputeNo] = statesOneInstance[stateNo];
        }
      }
    }
  }
}

// compute the total monodomain equation
// TODO: re-enable for actual simulation
//#ifdef __cplusplus
//extern "C"
//#endif
void computeMonodomain(const float *parameters,
                       double *algebraicsForTransfer, double *statesForTransfer, const float *elementLengths,
                       double startTime, double timeStepWidthSplitting, int nTimeStepsSplitting, double dt0D, int nTimeSteps0D, double dt1D, int nTimeSteps1D,
                       double prefactor, double valueForStimulatedPoint)
{


  // map data to and from GPU
  #pragma omp target data map(to: parameters[:nParametersTotal], elementLengths[:nElementLengths]) \
       map(from: statesForTransfer[:nStatesForTransfer])
  {

  // loop over splitting time steps
  #pragma omp target teams


  for (int timeStepNo = 0; timeStepNo < nTimeStepsSplitting; timeStepNo++)
  {
    // perform Strang splitting
    real currentTimeSplitting = startTime + timeStepNo * timeStepWidthSplitting;

    // compute midTimeSplitting once per step to reuse it. [currentTime, midTimeSplitting=currentTime+0.5*timeStepWidth, currentTime+timeStepWidth]
    real midTimeSplitting = currentTimeSplitting + 0.5 * timeStepWidthSplitting;
    bool storeAlgebraicsForTransferSplitting = false;   // do not store the computed algebraics values in the algebraicsForTransfer vector for communication, because this is the first 0D computation and they will be changed in the second 0D computation

    // perform Strang splitting:
    // 0D: [currentTimeSplitting, currentTimeSplitting + dt0D*nTimeSteps0D]
    // 1D: [currentTimeSplitting, currentTimeSplitting + dt1D*nTimeSteps1D]
    // 0D: [midTimeSplitting,     midTimeSplitting + dt0D*nTimeSteps0D]

    // advance 0D in [currentTimeSplitting, currentTimeSplitting + dt0D*nTimeSteps0D]
    // ------------------------------------------------------------

    // loop over fibers that will be computed on this rank
    #pragma omp distribute parallel for simd collapse(2)
    for (int fiberNo = 0; fiberNo < nFibersToCompute; fiberNo++)
    {
      // loop over instances to compute here
      for (int instanceNo = 0; instanceNo < nInstancesPerFiber; instanceNo++)
      {
        int instanceToComputeNo = fiberNo*nInstancesPerFiber + instanceNo;    // index of instance over all fibers

        // determine if current point is at center of fiber
        int fiberCenterIndex = fiberStimulationPointIndex[fiberNo];
        bool currentPointIsInCenter = (unsigned long)(fiberCenterIndex+1 - instanceNo) < 3;

        // loop over 0D timesteps
        for (int timeStepNo = 0; timeStepNo < nTimeSteps0D; timeStepNo++)
        {
          real currentTime = currentTimeSplitting + timeStepNo * dt0D;

          // determine if fiber gets stimulated
          // check if current point will be stimulated
          bool stimulateCurrentPoint = false;
          if (currentPointIsInCenter)
          {
            // check if time has come to call setSpecificStates
            bool checkStimulation = false;

            if (currentTime >= lastStimulationCheckTime[fiberNo] + 1./(setSpecificStatesCallFrequency[fiberNo]+currentJitter[fiberNo])
                && currentTime >= setSpecificStatesCallEnableBegin[fiberNo]-1e-13)
            {
              checkStimulation = true;

              // if current stimulation is over
              if (setSpecificStatesRepeatAfterFirstCall[fiberNo] != 0
                  && currentTime - (lastStimulationCheckTime[fiberNo] + 1./(setSpecificStatesCallFrequency[fiberNo] + currentJitter[fiberNo])) > setSpecificStatesRepeatAfterFirstCall[fiberNo])
              {
                // advance time of last call to specificStates
                lastStimulationCheckTime[fiberNo] += 1./(setSpecificStatesCallFrequency[fiberNo] + currentJitter[fiberNo]);

                // compute new jitter value
                real jitterFactor = 0.0;
                if (frequencyJitterNColumns > 0)
                  jitterFactor = setSpecificStatesFrequencyJitter[fiberNo*frequencyJitterNColumns + jitterIndex[fiberNo] % frequencyJitterNColumns];
                currentJitter[fiberNo] = jitterFactor * setSpecificStatesCallFrequency[fiberNo];

                jitterIndex[fiberNo]++;

                checkStimulation = false;
              }
            }

            // instead of calling setSpecificStates, directly determine whether to stimulate from the firingEvents file
            int firingEventsTimeStepNo = int(currentTime * setSpecificStatesCallFrequency[fiberNo] + 0.5);
            int firingEventsIndex = (firingEventsTimeStepNo % firingEventsNRows)*firingEventsNColumns + (motorUnitNo[fiberNo] % firingEventsNColumns);
            // firingEvents_[timeStepNo*nMotorUnits + motorUnitNo[fiberNo]]

            stimulateCurrentPoint = checkStimulation && firingEvents[firingEventsIndex];
            fiberIsCurrentlyStimulated[fiberNo] = stimulateCurrentPoint? 1: 0;

            // output to console
            if (stimulateCurrentPoint && fiberCenterIndex == instanceNo)
            {
//              if (omp_is_initial_device())
//                printf("t: %f, stimulate fiber %d (local no.), MU %d (computation on CPU)\n", currentTime, fiberNo, motorUnitNo[fiberNo]);
//              else
//                printf("t: %f, stimulate fiber %d (local no.), MU %d (computation on GPU)\n", currentTime, fiberNo, motorUnitNo[fiberNo]);
            }
          }
          const bool storeAlgebraicsForTransfer = false;

          //if (stimulateCurrentPoint)
          //  printf("stimulate fiberNo: %d, indexInFiber: %d (center: %d) \n", fiberNo, instanceNo, fiberCenterIndex);


          // CellML define constants
          const real constant0 = -75;
          const real constant1 = 1;
          const real constant2 = 0;
          const real constant3 = 120;
          const real constant4 = 36;
          const real constant5 = 0.3;
          const real constant6 = constant0+115.000;
          const real constant7 = constant0 - 12.0000;
          const real constant8 = constant0+10.6130;

          // compute new rates, rhs(y_n)
          const real algebraic1 = ( - 0.100000*(vmValues[instanceToComputeNo]+50.0000))/(exponential(- (vmValues[instanceToComputeNo]+50.0000)/10.0000) - 1.00000);
          const real algebraic5 =  4.00000*exponential(- (vmValues[instanceToComputeNo]+75.0000)/18.0000);
          const real rate1 =  algebraic1*(1.00000 - states[5924+instanceToComputeNo]) -  algebraic5*states[5924+instanceToComputeNo];
          const real algebraic2 =  0.0700000*exponential(- (vmValues[instanceToComputeNo]+75.0000)/20.0000);
          const real algebraic6 = 1.00000/(exponential(- (vmValues[instanceToComputeNo]+45.0000)/10.0000)+1.00000);
          const real rate2 =  algebraic2*(1.00000 - states[11848+instanceToComputeNo]) -  algebraic6*states[11848+instanceToComputeNo];
          const real algebraic3 = ( - 0.0100000*(vmValues[instanceToComputeNo]+65.0000))/(exponential(- (vmValues[instanceToComputeNo]+65.0000)/10.0000) - 1.00000);
          const real algebraic7 =  0.125000*exponential((vmValues[instanceToComputeNo]+75.0000)/80.0000);
          const real rate3 =  algebraic3*(1.00000 - states[17772+instanceToComputeNo]) -  algebraic7*states[17772+instanceToComputeNo];
          const real algebraic0 =  constant3*pow3(states[5924+instanceToComputeNo])*states[11848+instanceToComputeNo]*(vmValues[instanceToComputeNo] - constant6);
          const real algebraic4 =  constant4*pow4(states[17772+instanceToComputeNo])*(vmValues[instanceToComputeNo] - constant7);
          const real algebraic8 =  constant5*(vmValues[instanceToComputeNo] - constant8);
          const real rate0 = - (- parameters[0+instanceToComputeNo]+algebraic0+algebraic4+algebraic8)/constant1;

          // algebraic step
          // compute y* = y_n + dt*rhs(y_n), y_n = state, rhs(y_n) = rate, y* = intermediateState
          states[0+instanceToComputeNo] = vmValues[instanceToComputeNo] + dt0D*rate0;
          states[5924+instanceToComputeNo] = states[5924+instanceToComputeNo] + dt0D*rate1;
          states[11848+instanceToComputeNo] = states[11848+instanceToComputeNo] + dt0D*rate2;
          states[17772+instanceToComputeNo] = states[17772+instanceToComputeNo] + dt0D*rate3;


          // if stimulation, set value of Vm (state0)
          if (stimulateCurrentPoint)
          {
            states[0+instanceToComputeNo] = valueForStimulatedPoint;
          }
          // compute new rates, rhs(y*)
          const real intermediateAlgebraic1 = ( - 0.100000*(states[0+instanceToComputeNo]+50.0000))/(exponential(- (states[0+instanceToComputeNo]+50.0000)/10.0000) - 1.00000);
          const real intermediateAlgebraic5 =  4.00000*exponential(- (states[0+instanceToComputeNo]+75.0000)/18.0000);
          const real intermediateRate1 =  intermediateAlgebraic1*(1.00000 - states[5924+instanceToComputeNo]) -  intermediateAlgebraic5*states[5924+instanceToComputeNo];
          const real intermediateAlgebraic2 =  0.0700000*exponential(- (states[0+instanceToComputeNo]+75.0000)/20.0000);
          const real intermediateAlgebraic6 = 1.00000/(exponential(- (states[0+instanceToComputeNo]+45.0000)/10.0000)+1.00000);
          const real intermediateRate2 =  intermediateAlgebraic2*(1.00000 - states[11848+instanceToComputeNo]) -  intermediateAlgebraic6*states[11848+instanceToComputeNo];
          const real intermediateAlgebraic3 = ( - 0.0100000*(states[0+instanceToComputeNo]+65.0000))/(exponential(- (states[0+instanceToComputeNo]+65.0000)/10.0000) - 1.00000);
          const real intermediateAlgebraic7 =  0.125000*exponential((states[0+instanceToComputeNo]+75.0000)/80.0000);
          const real intermediateRate3 =  intermediateAlgebraic3*(1.00000 - states[17772+instanceToComputeNo]) -  intermediateAlgebraic7*states[17772+instanceToComputeNo];
          const real intermediateAlgebraic0 =  constant3*pow3(states[5924+instanceToComputeNo])*states[11848+instanceToComputeNo]*(states[0+instanceToComputeNo] - constant6);
          const real intermediateAlgebraic4 =  constant4*pow4(states[17772+instanceToComputeNo])*(states[0+instanceToComputeNo] - constant7);
          const real intermediateAlgebraic8 =  constant5*(states[0+instanceToComputeNo] - constant8);
          const real intermediateRate0 = - (- parameters[0+instanceToComputeNo]+intermediateAlgebraic0+intermediateAlgebraic4+intermediateAlgebraic8)/constant1;

          // final step
          // y_n+1 = y_n + 0.5*[rhs(y_n) + rhs(y*)]
          vmValues[instanceToComputeNo] += 0.5*dt0D*(rate0 + intermediateRate0);
          states[5924+instanceToComputeNo] += 0.5*dt0D*(rate1 + intermediateRate1);
          states[11848+instanceToComputeNo] += 0.5*dt0D*(rate2 + intermediateRate2);
          states[17772+instanceToComputeNo] += 0.5*dt0D*(rate3 + intermediateRate3);

          if (stimulateCurrentPoint)
          {
            vmValues[instanceToComputeNo] = valueForStimulatedPoint;
          }

          // store algebraics for transfer
          if (storeAlgebraicsForTransfer)
          {

            for (int i = 0; i < nStatesForTransferIndices; i++)
            {
              const int stateIndex = statesForTransferIndices[i];

              switch (stateIndex)
              {
                case 0:
                  statesForTransfer[i*nInstancesToCompute + instanceToComputeNo] = vmValues[instanceToComputeNo];
                  break;
                case 1:
                  statesForTransfer[i*nInstancesToCompute + instanceToComputeNo] = states[5924+instanceToComputeNo];
                  break;
                case 2:
                  statesForTransfer[i*nInstancesToCompute + instanceToComputeNo] = states[11848+instanceToComputeNo];
                  break;
                case 3:
                  statesForTransfer[i*nInstancesToCompute + instanceToComputeNo] = states[17772+instanceToComputeNo];
                  break;

              }
            }
          }
        }  // loop over 0D timesteps
      }  // loop over instances
    }  // loop over fibers

    // advance 1D in [currentTimeSplitting, currentTimeSplitting + dt1D*nTimeSteps1D]
    // ------------------------------------------------------------

    // Implicit Euler step:
    // (K - 1/dt*M) u^{n+1} = -1/dt*M u^{n})
    // Crank-Nicolson step:
    // (1/2*K - 1/dt*M) u^{n+1} = (-1/2*K -1/dt*M) u^{n})

    // stencil K: 1/h*[_-1_  1  ]*prefactor
    // stencil M:   h*[_1/3_ 1/6]
    const real dt = dt1D;

    // loop over fibers
    #pragma omp distribute parallel for
    for (int fiberNo = 0; fiberNo < nFibersToCompute; fiberNo++)
    {

      const int nValues = nInstancesPerFiber;

      // [ b c     ] [x]   [d]
      // [ a b c   ] [x] = [d]
      // [   a b c ] [x]   [d]
      // [     a b ] [x]   [d]

      // Thomas algorithm
      // forward substitution
      // c'_0 = c_0 / b_0
      // c'_i = c_i / (b_i - c'_{i-1}*a_i)

      // d'_0 = d_0 / b_0
      // d'_i = (d_i - d'_{i-1}*a_i) / (b_i - c'_{i-1}*a_i)

      // backward substitution
      // x_n = d'_n
      // x_i = d'_i - c'_i * x_{i+1}

      // helper buffers c', d' for Thomas algorithm
      real cIntermediate[nInstancesPerFiber-1];
      real dIntermediate[nInstancesPerFiber];

      // perform forward substitution
      // loop over entries / rows of matrices, this is equal to the instances of the current fiber
      for (int valueNo = 0; valueNo < nValues; valueNo++)
      {
        int instanceToComputeNo = fiberNo*nInstancesPerFiber + valueNo;

        real a = 0;
        real b = 0;
        real c = 0;
        real d = 0;

        real u_previous = 0;
        real u_center = 0;
        real u_next = 0;

        u_center = vmValues[instanceToComputeNo];  // state 0 of the current instance

        // contribution from left element
        if (valueNo > 0)
        {
          u_previous = vmValues[instanceToComputeNo - 1];  // state 0 of the left instance

          // stencil K: 1/h*[1   _-1_ ]*prefactor
          // stencil M:   h*[1/6 _1/3_]

          real h_left = elementLengths[fiberNo*nElementsOnFiber + valueNo-1];
          real k_left = 1./h_left*(1) * prefactor;
          real m_left = h_left*1./6;

          a = (k_left - 1/dt*m_left);   // formula for implicit Euler

          real k_right = 1./h_left*(-1) * prefactor;
          real m_right = h_left*1./3;

          b += (k_right - 1/dt*m_right);     // formula for implicit Euler
          d += (-1/dt*m_left) * u_previous + (-1/dt*m_right) * u_center;

        }

        // contribution from right element
        if (valueNo < nValues-1)
        {
          u_next = vmValues[instanceToComputeNo + 1];  // state 0 of the right instance

          // stencil K: 1/h*[_-1_  1  ]*prefactor
          // stencil M:   h*[_1/3_ 1/6]

          real h_right = elementLengths[fiberNo*nElementsOnFiber + valueNo];
          real k_right = 1./h_right*(1) * prefactor;
          real m_right = h_right*1./6;

          c = (k_right - 1/dt*m_right);     // formula for implicit Euler


          real k_left = 1./h_right*(-1) * prefactor;
          real m_left = h_right*1./3;

          b += (k_left - 1/dt*m_left);
          d += (-1/dt*m_left) * u_center + (-1/dt*m_right) * u_next;     // formula for implicit Euler

        }

        if (valueNo == 0)
        {
          // c'_0 = c_0 / b_0
          cIntermediate[valueNo] = c / b;

          // d'_0 = d_0 / b_0
          dIntermediate[valueNo] = d / b;
        }
        else
        {
          if (valueNo != nValues-1)
          {
            // c'_i = c_i / (b_i - c'_{i-1}*a_i)
            cIntermediate[valueNo] = c / (b - cIntermediate[valueNo-1]*a);
          }

          // d'_i = (d_i - d'_{i-1}*a_i) / (b_i - c'_{i-1}*a_i)
          dIntermediate[valueNo] = (d - dIntermediate[valueNo-1]*a) / (b - cIntermediate[valueNo-1]*a);
        }
      }

      // perform backward substitution
      // x_n = d'_n
      vmValues[nValues-1] = dIntermediate[nValues-1];  // state 0 of the point (nValues-1)

      real previousValue = dIntermediate[nValues-1];

      // loop over entries / rows of matrices
      for (int valueNo = nValues-2; valueNo >= 0; valueNo--)
      {
        int instanceToComputeNo = fiberNo*nInstancesPerFiber + valueNo;

        // x_i = d'_i - c'_i * x_{i+1}
        real resultValue = dIntermediate[valueNo] - cIntermediate[valueNo] * previousValue;
        vmValues[instanceToComputeNo] = resultValue;

        previousValue = resultValue;
      }
    }

    // advance 0D in [midTimeSplitting,     midTimeSplitting + dt0D*nTimeSteps0D]
    // ------------------------------------------------------------
    // in the last timestep, store the computed algebraics values in the algebraicsForTransfer vector for communication
    storeAlgebraicsForTransferSplitting = timeStepNo == nTimeStepsSplitting-1;

    // loop over fibers that will be computed on this rank

    #pragma omp distribute parallel for simd collapse(2)
    for (int fiberNo = 0; fiberNo < nFibersToCompute; fiberNo++)
    {
      // loop over instances to compute here
      for (int instanceNo = 0; instanceNo < nInstancesPerFiber; instanceNo++)
      {
        int instanceToComputeNo = fiberNo*nInstancesPerFiber + instanceNo;    // index of instance over all fibers

        // determine if current point is at center of fiber
        int fiberCenterIndex = fiberStimulationPointIndex[fiberNo];
        bool currentPointIsInCenter = (unsigned long)(fiberCenterIndex+1 - instanceNo) < 3;

        // loop over 0D timesteps
        for (int timeStepNo = 0; timeStepNo < nTimeSteps0D; timeStepNo++)
        {
          real currentTime = midTimeSplitting + timeStepNo * dt0D;

          // determine if fiber gets stimulated
          // check if current point will be stimulated
          bool stimulateCurrentPoint = false;
          if (currentPointIsInCenter)
          {
            // check if time has come to call setSpecificStates
            bool checkStimulation = false;

            if (currentTime >= lastStimulationCheckTime[fiberNo] + 1./(setSpecificStatesCallFrequency[fiberNo]+currentJitter[fiberNo])
                && currentTime >= setSpecificStatesCallEnableBegin[fiberNo]-1e-13)
            {
              checkStimulation = true;

              // if current stimulation is over
              if (setSpecificStatesRepeatAfterFirstCall[fiberNo] != 0
                  && currentTime - (lastStimulationCheckTime[fiberNo] + 1./(setSpecificStatesCallFrequency[fiberNo] + currentJitter[fiberNo])) > setSpecificStatesRepeatAfterFirstCall[fiberNo])
              {
                // advance time of last call to specificStates
                lastStimulationCheckTime[fiberNo] += 1./(setSpecificStatesCallFrequency[fiberNo] + currentJitter[fiberNo]);

                // compute new jitter value
                real jitterFactor = 0.0;
                if (frequencyJitterNColumns > 0)
                  jitterFactor = setSpecificStatesFrequencyJitter[fiberNo*frequencyJitterNColumns + jitterIndex[fiberNo] % frequencyJitterNColumns];
                currentJitter[fiberNo] = jitterFactor * setSpecificStatesCallFrequency[fiberNo];

                jitterIndex[fiberNo]++;

                checkStimulation = false;
              }
            }

            // instead of calling setSpecificStates, directly determine whether to stimulate from the firingEvents file
            int firingEventsTimeStepNo = int(currentTime * setSpecificStatesCallFrequency[fiberNo] + 0.5);
            int firingEventsIndex = (firingEventsTimeStepNo % firingEventsNRows)*firingEventsNColumns + (motorUnitNo[fiberNo] % firingEventsNColumns);
            // firingEvents_[timeStepNo*nMotorUnits + motorUnitNo[fiberNo]]

            stimulateCurrentPoint = checkStimulation && firingEvents[firingEventsIndex];
            fiberIsCurrentlyStimulated[fiberNo] = stimulateCurrentPoint? 1: 0;

            // output to console
            if (stimulateCurrentPoint && fiberCenterIndex == instanceNo)
            {
//              if (omp_is_initial_device())
//                printf("t: %f, stimulate fiber %d (local no.), MU %d (computation on CPU)\n", currentTime, fiberNo, motorUnitNo[fiberNo]);
//              else
//                printf("t: %f, stimulate fiber %d (local no.), MU %d (computation on GPU)\n", currentTime, fiberNo, motorUnitNo[fiberNo]);
            }
          }
          const bool storeAlgebraicsForTransfer = storeAlgebraicsForTransferSplitting && timeStepNo == nTimeSteps0D-1;

          //if (stimulateCurrentPoint)
          //  printf("stimulate fiberNo: %d, indexInFiber: %d (center: %d) \n", fiberNo, instanceNo, fiberCenterIndex);


          // CellML define constants
          const real constant0 = -75;
          const real constant1 = 1;
          const real constant2 = 0;
          const real constant3 = 120;
          const real constant4 = 36;
          const real constant5 = 0.3;
          const real constant6 = constant0+115.000;
          const real constant7 = constant0 - 12.0000;
          const real constant8 = constant0+10.6130;

          // compute new rates, rhs(y_n)
          const real algebraic1 = ( - 0.100000*(vmValues[instanceToComputeNo]+50.0000))/(exponential(- (vmValues[instanceToComputeNo]+50.0000)/10.0000) - 1.00000);
          const real algebraic5 =  4.00000*exponential(- (vmValues[instanceToComputeNo]+75.0000)/18.0000);
          const real rate1 =  algebraic1*(1.00000 - states[5924+instanceToComputeNo]) -  algebraic5*states[5924+instanceToComputeNo];
          const real algebraic2 =  0.0700000*exponential(- (vmValues[instanceToComputeNo]+75.0000)/20.0000);
          const real algebraic6 = 1.00000/(exponential(- (vmValues[instanceToComputeNo]+45.0000)/10.0000)+1.00000);
          const real rate2 =  algebraic2*(1.00000 - states[11848+instanceToComputeNo]) -  algebraic6*states[11848+instanceToComputeNo];
          const real algebraic3 = ( - 0.0100000*(vmValues[instanceToComputeNo]+65.0000))/(exponential(- (vmValues[instanceToComputeNo]+65.0000)/10.0000) - 1.00000);
          const real algebraic7 =  0.125000*exponential((vmValues[instanceToComputeNo]+75.0000)/80.0000);
          const real rate3 =  algebraic3*(1.00000 - states[17772+instanceToComputeNo]) -  algebraic7*states[17772+instanceToComputeNo];
          const real algebraic0 =  constant3*pow3(states[5924+instanceToComputeNo])*states[11848+instanceToComputeNo]*(vmValues[instanceToComputeNo] - constant6);
          const real algebraic4 =  constant4*pow4(states[17772+instanceToComputeNo])*(vmValues[instanceToComputeNo] - constant7);
          const real algebraic8 =  constant5*(vmValues[instanceToComputeNo] - constant8);
          const real rate0 = - (- parameters[0+instanceToComputeNo]+algebraic0+algebraic4+algebraic8)/constant1;

          // algebraic step
          // compute y* = y_n + dt*rhs(y_n), y_n = state, rhs(y_n) = rate, y* = intermediateState
          states[0+instanceToComputeNo] = vmValues[instanceToComputeNo] + dt0D*rate0;
          states[5924+instanceToComputeNo] = states[5924+instanceToComputeNo] + dt0D*rate1;
          states[11848+instanceToComputeNo] = states[11848+instanceToComputeNo] + dt0D*rate2;
          states[17772+instanceToComputeNo] = states[17772+instanceToComputeNo] + dt0D*rate3;


          // if stimulation, set value of Vm (state0)
          if (stimulateCurrentPoint)
          {
            states[0+instanceToComputeNo] = valueForStimulatedPoint;
          }
          // compute new rates, rhs(y*)
          const real intermediateAlgebraic1 = ( - 0.100000*(states[0+instanceToComputeNo]+50.0000))/(exponential(- (states[0+instanceToComputeNo]+50.0000)/10.0000) - 1.00000);
          const real intermediateAlgebraic5 =  4.00000*exponential(- (states[0+instanceToComputeNo]+75.0000)/18.0000);
          const real intermediateRate1 =  intermediateAlgebraic1*(1.00000 - states[5924+instanceToComputeNo]) -  intermediateAlgebraic5*states[5924+instanceToComputeNo];
          const real intermediateAlgebraic2 =  0.0700000*exponential(- (states[0+instanceToComputeNo]+75.0000)/20.0000);
          const real intermediateAlgebraic6 = 1.00000/(exponential(- (states[0+instanceToComputeNo]+45.0000)/10.0000)+1.00000);
          const real intermediateRate2 =  intermediateAlgebraic2*(1.00000 - states[11848+instanceToComputeNo]) -  intermediateAlgebraic6*states[11848+instanceToComputeNo];
          const real intermediateAlgebraic3 = ( - 0.0100000*(states[0+instanceToComputeNo]+65.0000))/(exponential(- (states[0+instanceToComputeNo]+65.0000)/10.0000) - 1.00000);
          const real intermediateAlgebraic7 =  0.125000*exponential((states[0+instanceToComputeNo]+75.0000)/80.0000);
          const real intermediateRate3 =  intermediateAlgebraic3*(1.00000 - states[17772+instanceToComputeNo]) -  intermediateAlgebraic7*states[17772+instanceToComputeNo];
          const real intermediateAlgebraic0 =  constant3*pow3(states[5924+instanceToComputeNo])*states[11848+instanceToComputeNo]*(states[0+instanceToComputeNo] - constant6);
          const real intermediateAlgebraic4 =  constant4*pow4(states[17772+instanceToComputeNo])*(states[0+instanceToComputeNo] - constant7);
          const real intermediateAlgebraic8 =  constant5*(states[0+instanceToComputeNo] - constant8);
          const real intermediateRate0 = - (- parameters[0+instanceToComputeNo]+intermediateAlgebraic0+intermediateAlgebraic4+intermediateAlgebraic8)/constant1;

          // final step
          // y_n+1 = y_n + 0.5*[rhs(y_n) + rhs(y*)]
          vmValues[instanceToComputeNo] += 0.5*dt0D*(rate0 + intermediateRate0);
          states[5924+instanceToComputeNo] += 0.5*dt0D*(rate1 + intermediateRate1);
          states[11848+instanceToComputeNo] += 0.5*dt0D*(rate2 + intermediateRate2);
          states[17772+instanceToComputeNo] += 0.5*dt0D*(rate3 + intermediateRate3);

          if (stimulateCurrentPoint)
          {
            vmValues[instanceToComputeNo] = valueForStimulatedPoint;
          }

          // store algebraics for transfer
          if (storeAlgebraicsForTransfer)
          {

            for (int i = 0; i < nStatesForTransferIndices; i++)
            {
              const int stateIndex = statesForTransferIndices[i];

              switch (stateIndex)
              {
                case 0:
                  statesForTransfer[i*nInstancesToCompute + instanceToComputeNo] = vmValues[instanceToComputeNo];
                  break;
                case 1:
                  statesForTransfer[i*nInstancesToCompute + instanceToComputeNo] = states[5924+instanceToComputeNo];
                  break;
                case 2:
                  statesForTransfer[i*nInstancesToCompute + instanceToComputeNo] = states[11848+instanceToComputeNo];
                  break;
                case 3:
                  statesForTransfer[i*nInstancesToCompute + instanceToComputeNo] = states[17772+instanceToComputeNo];
                  break;

              }
            }
          }


        }  // loop over 0D timesteps
      }  // loop over instances
    }  // loop over fibers
  } // loop over splitting timesteps

  } // end pragma omp target

  // map back from GPU to host
  //#pragma omp target update from(statesForTransfer[:nStatesForTransfer])

}
