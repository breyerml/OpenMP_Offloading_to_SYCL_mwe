/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*
  Porting a Simulation of the Activation of Muscle Fibers from OpenMP Offloading to SYCL

  Author:   Alexander Strack
  Date:     30. September 2021
  Description:
  This Code was ported and optimised from a auto-generated OpenMP Offloading File.
  It computes the activation of muscle fibers solving the Monodomain equation using
  the Hodkin-Huxley model. The equation is split via Strang-Splitting in two 0D
  and one 1D part. The 0D parts are computed via the Method of Heun, while the 1D
  is using an implicit Euler Scheme which is solved by the Thomas Alogrithm
*/
/*
  Running the full simulation in OpenDiHu with DPC++ installed on GPU

  1. Clone the [OpenDiHu github repo](https://github.com/maierbn/opendihu)
  2. Build the library (`cd opendihu && make`)
  3. Export the OpenDiHu root directory: `export OPENDIHU_HOME=/path/to/opendihu`
  4. Create the directory `$OPENDIHU_HOME/examples/electrophysiology/input`
  5. Copy the files
    * `hodgkin_huxley_1952.c`
    * `left_biceps_brachii_2x2fibers.bin` or 37x37 or 109x109 respectively
    * `MU_fibre_distribution_10MUs.txt`
    * `MU_firing_times_always.txt`

    from the `simulation_files/` folder of this repo to the previously created folder
    6. Go to `$OPENDIHU_HOME/examples/electrophysiology/fibers/fibers_emg/build_release` and
    run `./fast_fibers_emg ../settings_fibers_emg.py gpu.py`
    7. Copy your `hodgkin_huxley_1952_gpu_fast_monodomain.0.cpp` file
    to `$OPENDIHU_HOME/examples/electrophysiology/fibers/fibers_emg/build_release/src`
    8. Uncomment the option `libraryFilename` and change its value to `my_lib.so` in
    the `$OPENDIHU_HOME/examples/electrophysiology/fibers/fibers_emg/settings_fibers_emg.py` file
    9. Go to the `$OPENDIHU_HOME/examples/electrophysiology/fibers/fibers_emg/build_release/src` directory and
    run `clang++ hodgkin_huxley_1952_gpu_fast_monodomain.0.cpp -O3 -fPIC -shared -std=c++17 -sycl-std=2020 -fsycl -fsycl-targets=nvptx64-nvidia-cuda-sycldevice -o ../my_lib.so`.
    Make sure that the resulting `my_lib.so` is placed in the same directory as the simulation
    executable `fast_fibers_emg`.
    10. Rerun `./fast_fibers_emg ../settings_fibers_emg.py gpu.py` to use SYCL for the GPU support!
*/

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//#include "hodgkin_huxley.hpp" // TODO: remove for simulation in OpenDiHu

#include <CL/sycl.hpp>
#include <iostream>
#include <chrono>
namespace sycl = cl::sycl;
typedef double real;
//typedef float real;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// PARAMETERS AND STORAGE

// 0 -- use double sycl::exp                        potential: -75.1719661926
// 1 -- use float sycl::exp                         potential: -75.1719662059
// 2 -- use Benjamin Maiers approximation double    potential: -75.1700559855
// 3 -- use Benjamin Maiers approximation float     potential: -75.1700517257
constexpr char choose_exp = 0;

// after the last timestep, store the computed state values in the states_for_transfer vector for communication
constexpr bool store_states_for_transfer = true;

// Create queue to use the GPU device explicitly
sycl::queue dev_Q{ sycl::gpu_selector{} };
// Create queue to use the CPU device explicitly
//sycl::queue dev_Q{ sycl::cpu_selector{} };

// CellML define constants
constexpr real constant_0 = -75;
constexpr real constant_1 = 1;
//real constant_2 = 0; // unused
constexpr real constant_3 = 120;
constexpr real constant_4 = 36;
constexpr real constant_5 = 0.3;
constexpr real constant_6 = constant_0 + 115.0;
constexpr real constant_7 = constant_0 - 12.0;
constexpr real constant_8 = constant_0 + 10.613;
constexpr real one_over_constant_1 = 1. / constant_1;

// global size constants
constexpr int n_instances_per_fiber = 1481;
constexpr int n_elements_on_fiber = 1480;
constexpr int n_fibers_to_compute = 1369;
constexpr long long n_instances_to_compute = n_instances_per_fiber * n_fibers_to_compute;
constexpr int n_states = 4;
constexpr int firing_events_n_rows = 2;
constexpr int firing_events_n_columns = 100;
constexpr int frequency_jitter_n_columns = 100;
constexpr int n_states_total = n_instances_to_compute * n_states;
constexpr int n_parameters_total = n_instances_to_compute * 1;
constexpr int n_element_lengths = n_elements_on_fiber * n_fibers_to_compute;
constexpr int n_firing_events = firing_events_n_rows * firing_events_n_columns;
constexpr int n_frequency_jitter = n_fibers_to_compute * frequency_jitter_n_columns;
constexpr int n_states_for_transfer_indices = 1;
constexpr int n_states_for_transfer = n_instances_to_compute * n_states_for_transfer_indices;

constexpr int n_work_items_per_group = 128;
constexpr int n_groups = n_instances_to_compute / n_work_items_per_group + 1;

//Define workload size
sycl::range global{n_groups * n_work_items_per_group};
sycl::range local{n_work_items_per_group};

// Declare device allocated memory
real *states = sycl::malloc_device<real>(n_states_total, dev_Q);  // including state 0 which ware the vm values
real *states_one_instance = sycl::malloc_device<real>(n_states, dev_Q);
int *states_for_transfer_indices = sycl::malloc_device<int>(n_states_for_transfer_indices, dev_Q);

char *firing_events = sycl::malloc_device<char>(n_firing_events, dev_Q);
real *set_specific_states_frequency_jitter = sycl::malloc_device<real>(n_frequency_jitter, dev_Q);
char *fiber_is_currently_stimulated = sycl::malloc_device<char>(n_fibers_to_compute, dev_Q);
int  *motor_unit_no = sycl::malloc_device<int>(n_fibers_to_compute, dev_Q);
int  *fiber_stimulation_point_index = sycl::malloc_device<int>(n_fibers_to_compute, dev_Q);
real *last_stimulation_check_time = sycl::malloc_device<real>(n_fibers_to_compute, dev_Q);
real *set_specific_states_call_frequency = sycl::malloc_device<real>(n_fibers_to_compute , dev_Q);
real *set_specific_states_repeat_after_first_call = sycl::malloc_device<real>(n_fibers_to_compute, dev_Q);
real *set_specific_states_call_enable_begin = sycl::malloc_device<real>(n_fibers_to_compute, dev_Q);
real *current_jitter = sycl::malloc_device<real>(n_fibers_to_compute, dev_Q);
int  *jitter_index = sycl::malloc_device<int>(n_fibers_to_compute, dev_Q);

// Declare device allocated memory for constant values
float *element_lengths_device = sycl::malloc_device<float>(n_element_lengths, dev_Q);
float *parameters_device = sycl::malloc_device<float>(n_parameters_total, dev_Q);

// Declare device allocated memory for Tridiagonal Matrix
real *a = sycl::malloc_device<real>(n_instances_to_compute, dev_Q);
real *b = sycl::malloc_device<real>(n_instances_to_compute, dev_Q);
real *c = sycl::malloc_device<real>(n_instances_to_compute, dev_Q);
real *d = sycl::malloc_device<real>(n_instances_to_compute, dev_Q);

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// CLASSES
class Compute_0D
{
  // advance 0D in [time_splitting, time_splitting + dt_0D*n_time_steps_0D]
  // ------------------------------------------------------------
public:
  Compute_0D(
    real dt_0D_value,
    int n_time_steps_0D_value,
    real value_for_stimulated_point_value,
    real time_splitting_value,

    real *states_ref,
    const float *parameters_ref,
    char *firing_events_ref,
    real *set_specific_states_frequency_jitter_ref,
    char *fiber_is_currently_stimulated_ref,
    int  *motor_unit_no_ref,
    int  *fiber_stimulation_point_index_ref,
    real *last_stimulation_check_time_ref,
    real *set_specific_states_call_frequency_ref,
    real *set_specific_states_repeat_after_first_call_ref,
    real *set_specific_states_call_enable_begin_ref,
    real *current_jitter_ref,
    int  *jitter_index_ref):

    dt_0D{dt_0D_value},
    n_time_steps_0D{n_time_steps_0D_value},
    value_for_stimulated_point{value_for_stimulated_point_value},
    time_splitting{time_splitting_value},

    states{states_ref},
    parameters{parameters_ref},
    firing_events{firing_events_ref},
    set_specific_states_frequency_jitter{set_specific_states_frequency_jitter_ref},
    fiber_is_currently_stimulated{fiber_is_currently_stimulated_ref},
    motor_unit_no{motor_unit_no_ref},
    fiber_stimulation_point_index{fiber_stimulation_point_index_ref},
    last_stimulation_check_time{last_stimulation_check_time_ref},
    set_specific_states_call_frequency{set_specific_states_call_frequency_ref},
    set_specific_states_repeat_after_first_call{set_specific_states_repeat_after_first_call_ref},
    set_specific_states_call_enable_begin{set_specific_states_call_enable_begin_ref},
    current_jitter{current_jitter_ref},
    jitter_index{jitter_index_ref}
  {}

  void operator()(sycl::nd_item<1> it) const
  {
    //compute current instance and check if its a valid instance
    int instance_to_compute_no = it.get_global_id(0);

    if (instance_to_compute_no < n_instances_to_compute)
    {
      // compute properties of instance
      int instance_no = instance_to_compute_no % n_instances_per_fiber;
      int fiber_no = instance_to_compute_no / n_instances_per_fiber;

      // write global data to private memory
      real state_0 = states[instance_to_compute_no];
      real state_1 = states[1 * n_instances_to_compute + instance_to_compute_no];
      real state_2 = states[2 * n_instances_to_compute + instance_to_compute_no];
      real state_3 = states[3 * n_instances_to_compute + instance_to_compute_no];

      real parameter = parameters[instance_to_compute_no];

      // local variables used for Heun
      real algebraic_0;
      real algebraic_1;
      real algebraic_2;
      real algebraic_3;
      real algebraic_4;
      real algebraic_5;
      real algebraic_6;
      real algebraic_7;
      real algebraic_8;

      real rate_0;
      real rate_1;
      real rate_2;
      real rate_3;

      real intermediate_algebraic_0;
      real intermediate_algebraic_1;
      real intermediate_algebraic_2;
      real intermediate_algebraic_3;
      real intermediate_algebraic_4;
      real intermediate_algebraic_5;
      real intermediate_algebraic_6;
      real intermediate_algebraic_7;
      real intermediate_algebraic_8;

      real intermediate_rate_0;
      real intermediate_rate_1;
      real intermediate_rate_2;
      real intermediate_rate_3;

      real intermediate_state_0;
      real intermediate_state_1;
      real intermediate_state_2;
      real intermediate_state_3;

      // determine if current point is at center of fiber
      int fiber_center_index = fiber_stimulation_point_index[fiber_no];
      bool current_point_is_in_center = (unsigned long)(fiber_center_index + 1 - instance_no) < 3;

      // loop over 0D timesteps
      for (int time_step_no = 0; time_step_no < n_time_steps_0D; time_step_no++)
      {
        real current_time = time_splitting + time_step_no * dt_0D;

        // determine if fiber gets stimulated
        // check if current point will be stimulated
        bool stimulate_current_point = false;

        if (current_point_is_in_center)
        {
          // check if time has come to call set_specific_states
          bool check_stimulation = false;

          if (current_time >= last_stimulation_check_time[fiber_no] + 1. / (set_specific_states_call_frequency[fiber_no] + current_jitter[fiber_no])
              && current_time >= set_specific_states_call_enable_begin[fiber_no] - 1e-13)
          {
            check_stimulation = true;

            // if current stimulation is over
            if (set_specific_states_repeat_after_first_call[fiber_no] != 0
                && current_time - (last_stimulation_check_time[fiber_no] + 1. / (set_specific_states_call_frequency[fiber_no] + current_jitter[fiber_no])) > set_specific_states_repeat_after_first_call[fiber_no])
            {
              // advance time of last call to specific_states
              last_stimulation_check_time[fiber_no] += 1. / (set_specific_states_call_frequency[fiber_no] + current_jitter[fiber_no]);

              // compute new jitter value
              real jitter_factor = 0.0;
              if (frequency_jitter_n_columns > 0)
              {
                jitter_factor = set_specific_states_frequency_jitter[fiber_no * frequency_jitter_n_columns + jitter_index[fiber_no] % frequency_jitter_n_columns];
              }
              current_jitter[fiber_no] = jitter_factor * set_specific_states_call_frequency[fiber_no];

              jitter_index[fiber_no]++;

              check_stimulation = false;
            }
          }

          // instead of calling set_specific_states, directly determine whether to stimulate from the firing_events file
          int firing_events_time_step_no = int(current_time * set_specific_states_call_frequency[fiber_no] + 0.5);
          int firing_events_index = (firing_events_time_step_no % firing_events_n_rows) * firing_events_n_columns + (motor_unit_no[fiber_no] % firing_events_n_columns);

          stimulate_current_point = check_stimulation && firing_events[firing_events_index];
          fiber_is_currently_stimulated[fiber_no] = stimulate_current_point? 1 : 0;
        }

        // START OF 0D COMPUTATION VIA METHOD OF HEUN
        // compute new rates, rhs(y_n)
        algebraic_1 = ( - 0.1 * (state_0 + 50.0)) / (exponential(- (state_0 + 50.0) * 0.1) - 1.0);
        algebraic_5 = 4.0 * exponential(- (state_0 + 75.0) / 18.0);
        rate_1 = algebraic_1 * (1.0 - state_1) - algebraic_5 * state_1;

        algebraic_2 = 0.07 * exponential(- (state_0 + 75.0) * 0.05);
        algebraic_6 = 1.0 / (exponential(- (state_0 + 45.0) * 0.1) + 1.0);
        rate_2 = algebraic_2 * (1.0 - state_2) - algebraic_6 * state_2;

        algebraic_3 = ( -0.01 * (state_0 + 65.0)) / (exponential(- (state_0 + 65.0) * 0.1) - 1.0);
        algebraic_7 = 0.125 * exponential((state_0 + 75.0) * 0.0125);
        rate_3 = algebraic_3 * (1.0 - state_3) - algebraic_7 * state_3;

        algebraic_0 =  constant_3 * state_1 * state_1 * state_1 * state_2 * (state_0 - constant_6);
        algebraic_4 =  constant_4 * state_3 * state_3 * state_3 * state_3 * (state_0 - constant_7);
        algebraic_8 =  constant_5 * (state_0 - constant_8);
        rate_0 = - (- parameter + algebraic_0 + algebraic_4 + algebraic_8) * one_over_constant_1;

        // algebraic step
        // compute y* = y_n + dt*rhs(y_n), y_n = state, rhs(y_n) = rate, y* = intermediateState
        intermediate_state_0 = ((int) stimulate_current_point) * value_for_stimulated_point
                             + (1 - (int) stimulate_current_point) * (state_0 + dt_0D * rate_0);
        intermediate_state_1 = state_1 + dt_0D * rate_1;
        intermediate_state_2 = state_2 + dt_0D * rate_2;
        intermediate_state_3 = state_3 + dt_0D * rate_3;

        // compute new rates, rhs(y*)
        intermediate_algebraic_1 = ( - 0.1 * (intermediate_state_0 + 50.0))/(exponential(- (intermediate_state_0 + 50.0) * 0.1) - 1.0);
        intermediate_algebraic_5 = 4.0 * exponential(- (intermediate_state_0 + 75.0) / 18.0);
        intermediate_rate_1 = intermediate_algebraic_1 * (1.0 - intermediate_state_1) - intermediate_algebraic_5 * intermediate_state_1;

        intermediate_algebraic_2 = 0.07 * exponential(- (intermediate_state_0 + 75.0) * 0.05);
        intermediate_algebraic_6 = 1.0 / (exponential(- (intermediate_state_0 + 45.0) * 0.1) + 1.0);
        intermediate_rate_2 = intermediate_algebraic_2 * (1.0 - intermediate_state_2) - intermediate_algebraic_6 * intermediate_state_2;

        intermediate_algebraic_3 = ( - 0.01 * (intermediate_state_0 + 65.0)) / (exponential(- (intermediate_state_0 + 65.0) * 0.1) - 1.0);
        intermediate_algebraic_7 = 0.125 * exponential((intermediate_state_0 + 75.0) * 0.0125);
        intermediate_rate_3 = intermediate_algebraic_3 * (1.0 - intermediate_state_3) - intermediate_algebraic_7 * intermediate_state_3;

        intermediate_algebraic_0 = constant_3 * intermediate_state_1 * intermediate_state_1 * intermediate_state_1 * intermediate_state_2 * (intermediate_state_0 - constant_6);
        intermediate_algebraic_4 = constant_4 * intermediate_state_3 * intermediate_state_3 * intermediate_state_3 * intermediate_state_3 * (intermediate_state_0 - constant_7);
        intermediate_algebraic_8 = constant_5 * (intermediate_state_0 - constant_8);
        intermediate_rate_0 = - (- parameter + intermediate_algebraic_0 + intermediate_algebraic_4 + intermediate_algebraic_8) * one_over_constant_1;

        // final step
        // y_n+1 = y_n + 0.5*[rhs(y_n) + rhs(y*)]
        state_0 = ((int) stimulate_current_point)* value_for_stimulated_point
                + (1 - (int) stimulate_current_point) * (state_0 + 0.5 * dt_0D * (rate_0 + intermediate_rate_0));
        state_1 += 0.5 * dt_0D * (rate_1 + intermediate_rate_1);
        state_2 += 0.5 * dt_0D * (rate_2 + intermediate_rate_2);
        state_3 += 0.5 * dt_0D * (rate_3 + intermediate_rate_3);
      }  // loop over 0D timesteps

      // write local data to global memory
      states[instance_to_compute_no] = state_0;
      states[1 * n_instances_to_compute + instance_to_compute_no] = state_1;
      states[2 * n_instances_to_compute + instance_to_compute_no] = state_2;
      states[3 * n_instances_to_compute + instance_to_compute_no] = state_3;
    } //endif for valid instances
  } // sycl parallel_for over fibers and instances per fiber

private:
  real dt_0D;
  int n_time_steps_0D;
  real value_for_stimulated_point;
  real time_splitting;

  real *states;
  const float *parameters;
  char *firing_events;
  real *set_specific_states_frequency_jitter;
  char *fiber_is_currently_stimulated;
  int  *motor_unit_no;
  int  *fiber_stimulation_point_index;
  real *last_stimulation_check_time;
  real *set_specific_states_call_frequency;
  real *set_specific_states_repeat_after_first_call;
  real *set_specific_states_call_enable_begin;
  real *current_jitter;
  int  *jitter_index;

  // helper functions
  real exponential(real x) const
  {
    if constexpr(choose_exp == 0)
    {
      return sycl::exp(x);
    }
    else if constexpr(choose_exp == 1)
    {
      return sycl::exp(static_cast<float>(x));
    }
    else if constexpr(choose_exp == 2)
    {
      // it was determined the x is always in the range [-12,+12]
      // exp(x) = lim n→∞ (1 + x/n)^n, we set n=1024
      x = 1.0 + x / 1024.;
      //x = 1.0 + x * 0.0009765625;
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
    else
    {
      // exp(x) = lim n→∞ (1 + x/n)^n, we set n=1024
      float y = 1.0f + static_cast<float>(x) / 1024.f;
      //x = 1.0 + x * 0.0009765625;
      for (int i = 0; i < 10; i++)
      {
        y *= y;
      }
      return y;
    }
  }
};

class Compute_1D_matrix
{
  // advance 1D in [currentTimeSplitting, currentTimeSplitting + dt1D*nTimeSteps1D]
  // ------------------------------------------------------------

  // Implicit Euler step:
  // (K - 1/dt*M) u^{n+1} = -1/dt*M u^{n})

  // stencil K: 1/h*[_-1_  1  ]*prefactor
  // stencil M:   h*[_1/3_ 1/6]
public:
  Compute_1D_matrix(
    real dt_1D_value,
    real prefactor_value,

    real *states_ref,
    const float *element_lengths_ref,

    real *a_ref,
    real *b_ref,
    real *c_ref,
    real *d_ref):

    dt{dt_1D_value},
    prefactor{prefactor_value},

    states{states_ref},
    element_lengths{element_lengths_ref},

    a{a_ref},
    b{b_ref},
    c{c_ref},
    d{d_ref}
  {}

  void operator()(sycl::nd_item<1> it) const
  {
    // compute 'matrix' for thomas algorithm
    // [ b c     ] [x]   [d]
    // [ a b c   ] [x] = [d]
    // [   a b c ] [x]   [d]
    // [     a b ] [x]   [d]

    //compute current instance and check if its a valid instance
    int instance_to_compute_no = it.get_global_id(0);

    if (instance_to_compute_no < n_instances_to_compute)
    {
      int value_no = instance_to_compute_no % n_instances_per_fiber;
      int fiber_no = instance_to_compute_no / n_instances_per_fiber;

      real a_local = 0;
      real b_local = 0;
      real c_local = 0;
      real d_local = 0;

      real u_previous = 0;
      real u_center = 0;
      real u_next = 0;

      u_center = states[instance_to_compute_no];  // state_0 of the current instance

      real one_over_dt = 1. / dt;

      // contribution from left element
      if (value_no > 0)
      {
        u_previous = states[instance_to_compute_no - 1];  // state_0 of the left instance

        // stencil K: 1/h*[1   _-1_ ]*prefactor
        // stencil M:   h*[1/6 _1/3_]

        real h_left = element_lengths[fiber_no * n_elements_on_fiber + value_no - 1];
        real k_left = 1./h_left * (1) * prefactor;
        real m_left = h_left * 1./6;

        a_local = (k_left - one_over_dt * m_left);   // formula for implicit Euler

        real k_right = 1./h_left * (-1) * prefactor;
        real m_right = h_left * 1./3;

        b_local += (k_right - one_over_dt * m_right);     // formula for implicit Euler
        d_local += (-one_over_dt * m_left) * u_previous + (-one_over_dt * m_right) * u_center;
      }

      // contribution from right element
      if (value_no < n_values - 1)
      {
        u_next = states[instance_to_compute_no + 1];  // state 0 of the right instance

        // stencil K: 1/h*[_-1_  1  ]*prefactor
        // stencil M:   h*[_1/3_ 1/6]

        real h_right = element_lengths[fiber_no * n_elements_on_fiber + value_no];
        real k_right = 1./h_right * (1) * prefactor;
        real m_right = h_right * 1./6;

        c_local = (k_right - one_over_dt * m_right);     // formula for implicit Euler

        real k_left = 1./h_right * (-1) * prefactor;
        real m_left = h_right * 1./3;

        b_local += (k_left - one_over_dt * m_left);
        d_local += (-one_over_dt * m_left) * u_center + (-one_over_dt * m_right) * u_next;     // formula for implicit Euler
      }

      // write local data to global memory
      a[instance_to_compute_no] = a_local;
      b[instance_to_compute_no] = b_local;
      c[instance_to_compute_no] = c_local;
      d[instance_to_compute_no] = d_local;
    } // endif for valid instances
  } // sycl parallel_for over fibers and instances per fiber

private:
  real dt;
  real prefactor;

  real *states;
  const float *element_lengths;

  real *a;
  real *b;
  real *c;
  real *d;

  //renaming variable
  const int n_values = n_instances_per_fiber;
};

class Compute_1D_thomas
{
  // advance 1D in [currentTimeSplitting, currentTimeSplitting + dt1D*nTimeSteps1D]
  // ------------------------------------------------------------

  // Implicit Euler step:
  // (K - 1/dt*M) u^{n+1} = -1/dt*M u^{n})
public:
  Compute_1D_thomas(
    real *states_ref,

    real *a_ref,
    real *b_ref,
    real *c_ref,
    real *d_ref):

    states{states_ref},
    a{a_ref},
    b{b_ref},
    c{c_ref},
    d{d_ref}
  {}
  void operator()(sycl::nd_item<1> fibers) const
  {
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

    //get current fiber and check if its a valid fiber
    int fiber_no = fibers.get_global_id(0);;

    if (fiber_no < n_fibers_to_compute)
    {
      // helper buffers c', d' for Thomas algorithm
      real c_intermediate[n_instances_per_fiber - 1];
      real d_intermediate[n_instances_per_fiber];

      // Thomas Algorithm:
      // perform forward substitution
      // loop over entries / rows of matrices, this is equal to the instances of the current fiber
      for (int value_no = 0; value_no < n_values; value_no++)
      {
        int instance_to_compute_no = fiber_no * n_instances_per_fiber + value_no;

        real a_local = a[instance_to_compute_no];
        real b_local = b[instance_to_compute_no];
        real c_local = c[instance_to_compute_no];
        real d_local = d[instance_to_compute_no];

        if (value_no == 0)
        {
          // c'_0 = c_0 / b_0
          c_intermediate[value_no] = c_local / b_local;

          // d'_0 = d_0 / b_0
          d_intermediate[value_no] = d_local / b_local;
        }
        else
        {
          if (value_no != n_values - 1)
          {
            // c'_i = c_i / (b_i - c'_{i-1}*a_i)
            c_intermediate[value_no] = c_local / (b_local - c_intermediate[value_no - 1] * a_local);
          }

          // d'_i = (d_i - d'_{i-1}*a_i) / (b_i - c'_{i-1}*a_i)
          d_intermediate[value_no] = (d_local - d_intermediate[value_no - 1] * a_local) / (b_local - c_intermediate[value_no - 1] * a_local);
        }
      }

      // perform backward substitution
      // x_n = d'_n
      states[fiber_no * n_instances_per_fiber + n_values - 1] = d_intermediate[n_values - 1];  // state_0 of the point (n_values - 1)

      real previous_value = d_intermediate[n_values - 1];

      // loop over entries / rows of matrices
      for (int value_no = n_values - 2; value_no >= 0; value_no--)
      {
        int instance_to_compute_no = fiber_no * n_instances_per_fiber + value_no;

        // x_i = d'_i - c'_i * x_{i+1}
        real result_value = d_intermediate[value_no] - c_intermediate[value_no] * previous_value;
        states[instance_to_compute_no] = result_value;

        previous_value = result_value;
      }
    } // endif for valid instances
  } // sycl parallel_for over fibers

private:
  real *states;

  real *a;
  real *b;
  real *c;
  real *d;

  //renaming variable
  const int n_values = n_instances_per_fiber;
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// FUNCTIONS
void which_device()
{
  // print used device
  std::cout << "Selected device: " <<
  dev_Q.get_device().get_info<sycl::info::device::name>() << "\n";
}

void free_memory()
{
  // SYCL free the allocated USM
  free(states, dev_Q);
  free(states_one_instance, dev_Q);
  free(states_for_transfer_indices, dev_Q);

  free(firing_events, dev_Q);
  free(set_specific_states_frequency_jitter, dev_Q);
  free(fiber_is_currently_stimulated, dev_Q);
  free(motor_unit_no, dev_Q);
  free(fiber_stimulation_point_index, dev_Q);
  free(last_stimulation_check_time, dev_Q);
  free(set_specific_states_call_frequency, dev_Q);
  free(set_specific_states_repeat_after_first_call, dev_Q);
  free(set_specific_states_call_enable_begin, dev_Q);
  free(current_jitter, dev_Q);
  free(jitter_index, dev_Q);

  free(parameters_device, dev_Q);
  free(element_lengths_device, dev_Q);

  free(a, dev_Q);
  free(b, dev_Q);
  free(c, dev_Q);
  free(d, dev_Q);
}

#ifdef __cplusplus
extern "C"
#endif
void initializeArrays(const double *states_one_instance_parameter,
                      const int* algebraics_for_transfer_indices_parameter_unused,  //unused -> remove for final version
                      const int* states_for_transfer_indices_parameter,
                      const char *firing_events_parameter,
                      const double *set_specific_states_frequency_jitter_parameter,
                      const int *motor_unit_no_parameter,
                      const int *fiber_stimulation_point_index_parameter,
                      const double *last_stimulation_check_time_parameter,
                      const double *set_specific_states_call_frequency_parameter,
                      const double *set_specific_states_repeat_after_first_call_parameter,
                      const double *set_specific_states_call_enable_begin_parameter)
{
  // Identify the device e.g. which CPU or GPU - OPTIONAL
  which_device();

  // Initialize device memory
  dev_Q.submit([&](sycl::handler &h)
  {
      // copy parameter data to device memory
      h.memcpy(states_one_instance, &states_one_instance_parameter[0], n_states * sizeof(real));
  });

  dev_Q.submit([&](sycl::handler &h)
  {
      // copy parameter data to device memory
      h.memcpy(states_for_transfer_indices, &states_for_transfer_indices_parameter[0], n_states_for_transfer_indices * sizeof(int));
  });

  dev_Q.submit([&](sycl::handler &h)
  {
      // copy parameter data to device memory
      h.memcpy(firing_events, &firing_events_parameter[0], n_firing_events * sizeof(char));
  });

  dev_Q.submit([&](sycl::handler &h)
  {
      // copy parameter data to device memory
      h.memcpy(set_specific_states_frequency_jitter, &set_specific_states_frequency_jitter_parameter[0], n_frequency_jitter * sizeof(real));
  });

  dev_Q.submit([&](sycl::handler &h)
  {
      // copy parameter data to device memory
      h.memcpy(motor_unit_no, &motor_unit_no_parameter[0], n_fibers_to_compute * sizeof(int));
  });

  dev_Q.submit([&](sycl::handler &h)
  {
      // copy parameter data to device memory
      h.memcpy(fiber_stimulation_point_index, &fiber_stimulation_point_index_parameter[0], n_fibers_to_compute * sizeof(int));
  });

  dev_Q.submit([&](sycl::handler &h)
  {
      // copy parameter data to device memory
      h.memcpy(last_stimulation_check_time, &last_stimulation_check_time_parameter[0], n_fibers_to_compute * sizeof(real));
  });

  dev_Q.submit([&](sycl::handler &h)
  {
      // copy parameter data to device memory
      h.memcpy(set_specific_states_call_frequency, &set_specific_states_call_frequency_parameter[0], n_fibers_to_compute * sizeof(real));
  });

  dev_Q.submit([&](sycl::handler &h)
  {
      // copy parameter data to device memory
      h.memcpy(set_specific_states_repeat_after_first_call, &set_specific_states_repeat_after_first_call_parameter[0], n_fibers_to_compute * sizeof(real));
  });

  dev_Q.submit([&](sycl::handler &h)
  {
      // copy parameter data to device memory
      h.memcpy(set_specific_states_call_enable_begin, &set_specific_states_call_enable_begin_parameter[0], n_fibers_to_compute * sizeof(real));
  });

  // set variables to zero
  dev_Q.submit([&](sycl::handler &h)
  {
      // set device memory to 0
      h.memset(fiber_is_currently_stimulated, 0, n_fibers_to_compute * sizeof(char));
  });

  dev_Q.submit([&](sycl::handler &h)
  {
      // set device memory to 0
      h.memset(current_jitter, 0, n_fibers_to_compute * sizeof(int));
  });

  dev_Q.submit([&](sycl::handler &h)
  {
      // set device memory to 0
      h.memset(jitter_index, 0, n_fibers_to_compute * sizeof(int));
  });

  // initialize states
  dev_Q.submit([&](sycl::handler &h)
  {
      real *states_ref = states;
      real *states_one_instance_ref = states_one_instance;

      h.parallel_for<class Initialize>(sycl::nd_range{global, local}, [=](sycl::nd_item<1> it)
      {
        //compute current instance and check if its a valid instance
        int instance_to_compute_no = it.get_global_id(0);

        if (instance_to_compute_no < n_instances_to_compute)
        {
          for (int state_no = 0; state_no < n_states; state_no++)
          {
            states_ref[state_no * n_instances_to_compute + instance_to_compute_no] = states_one_instance_ref[state_no];
          }
        }
      });
  });

  // wait initialize to complete
  dev_Q.wait();
}

#ifdef __cplusplus
extern "C"
#endif
void computeMonodomain(const float *parameters,
                       double* algebraics_for_transfer_unused, //unused but required to match constructor
                       double* states_for_transfer,
                       const float *element_lengths,
                       double start_time,
                       double time_step_width_splitting,
                       int n_time_steps_splitting,
                       double dt_0D,
                       int n_time_steps_0D,
                       double dt_1D,
                       int n_time_steps_1D_unused, //unused but required to match constructor
                       double prefactor,
                       double value_for_stimulated_point)
{
  // Start time - OPTIONAL
  auto start = std::chrono::system_clock::now();

  // Copy constant memory to device
  auto copy_parameters = dev_Q.submit([&](sycl::handler &h)
  {
      // copy host data to device memory
      h.memcpy(parameters_device, &parameters[0], n_parameters_total * sizeof(float));
  });

  auto copy_element_length = dev_Q.submit([&](sycl::handler &h)
  {
      // copy host data to device memory
      h.memcpy(element_lengths_device, &element_lengths[0], n_element_lengths * sizeof(float));
  });

  // Computation
  // loop over splitting time steps
  for (int time_step_no = 0; time_step_no < n_time_steps_splitting; time_step_no++)
  {
    // perform Strang splitting
    real current_time_splitting = start_time + time_step_no * time_step_width_splitting;
    //printf("Current Time: %f \n", current_time_splitting);

    // compute midTimeSplitting once per step to reuse it. [currentTime, midTimeSplitting=currentTime+0.5*timeStepWidth, currentTime+timeStepWidth]
    real mid_time_splitting = current_time_splitting + 0.5 * time_step_width_splitting;

    // perform Strang splitting:
    //////////////////////////////////////////////////////////////////////
    // 0D: [current_time_splitting, current_time_splitting + dt_0D*n_time_steps_0D]
    {
      auto cg_first_compute_0D = [&](sycl::handler& h)
      {
        //event dependancy
        if (time_step_no == 0)
        {
          h.depends_on(copy_parameters);
        }

        h.parallel_for(sycl::nd_range{global, local}, Compute_0D(dt_0D, n_time_steps_0D, value_for_stimulated_point, current_time_splitting, states, parameters_device,
          firing_events, set_specific_states_frequency_jitter, fiber_is_currently_stimulated, motor_unit_no, fiber_stimulation_point_index,
          last_stimulation_check_time, set_specific_states_call_frequency, set_specific_states_repeat_after_first_call,
          set_specific_states_call_enable_begin, current_jitter, jitter_index));
      };
      auto event_for_first_compute_0D = dev_Q.submit(cg_first_compute_0D);

    //////////////////////////////////////////////////////////////////////
    // 1D: [current_time_splitting, current_time_splitting + dt_1D*n_time_steps_1D]
      //STEP 1: compute the entries of the Tridiagonal matrix
      auto cg_compute_1D_matrix = [&](sycl::handler& h)
      {
        //event dependancy
        if (time_step_no == 0)
        {
          h.depends_on(copy_element_length);
        }
        h.depends_on(event_for_first_compute_0D);

        h.parallel_for(sycl::nd_range{global, local}, Compute_1D_matrix(dt_1D, prefactor, states, element_lengths_device, a, b, c, d));
      };
      auto event_for_compute_1D_matrix = dev_Q.submit(cg_compute_1D_matrix);

      //STEP 2: solve Tridiagonal matrix with thomas Algorithm
      auto cg_for_compute_1D_thomas = [&](sycl::handler& h)
      {
        //event dependancy
        h.depends_on(event_for_compute_1D_matrix);

        //Define workload size for thomas - runs over fiber not over intstances to compute
        constexpr int n_groups_thomas = n_fibers_to_compute / n_work_items_per_group + 1;
        sycl::range global_thomas{n_groups_thomas * n_work_items_per_group};

        //exactly one action call -- access shared memory on device
        h.parallel_for(sycl::nd_range{global_thomas, local}, Compute_1D_thomas(states, a, b, c, d));
      };
      auto event_for_compute_1D_thomas = dev_Q.submit(cg_for_compute_1D_thomas);

    //////////////////////////////////////////////////////////////////////
    // 0D: [mid_time_splitting,     mid_time_splitting + dt_0D*n_time_steps_0D]
      auto cg_second_compute_0D = [&](sycl::handler& h)
      {
        //event dependancy
        h.depends_on(event_for_compute_1D_thomas);

        h.parallel_for(sycl::nd_range{global, local}, Compute_0D(dt_0D, n_time_steps_0D, value_for_stimulated_point, mid_time_splitting, states, parameters_device,
          firing_events, set_specific_states_frequency_jitter, fiber_is_currently_stimulated, motor_unit_no, fiber_stimulation_point_index,
          last_stimulation_check_time, set_specific_states_call_frequency, set_specific_states_repeat_after_first_call,
          set_specific_states_call_enable_begin, current_jitter, jitter_index));
      };
      dev_Q.submit(cg_second_compute_0D);
      dev_Q.wait();
    }
  }

  // store states for transfer
  //dev_Q.wait();
  if (store_states_for_transfer)
  {
    sycl::buffer<double, 1> states_buffer(&states_for_transfer[0], n_states_for_transfer);

    dev_Q.submit([&](sycl::handler &h)
    {
      //references to USM data
      real *states_ref = states;
      int *states_for_transfer_indices_ref = states_for_transfer_indices;

      //accessor for buffers
      auto states_buffer_accessor = states_buffer.get_access<sycl::access::mode::write>(h);

      h.parallel_for<class Transfer>(sycl::nd_range{global, local}, [=](sycl::nd_item<1> it)
      {
        //compute current instance and check if its a valid instance
        int instance_to_compute_no = it.get_global_id(0);

        if (instance_to_compute_no < n_instances_to_compute)
        {
          for (int i = 0; i < n_states_for_transfer_indices; i++)
          {
            const int state_index = states_for_transfer_indices_ref[i];
            states_buffer_accessor[i * n_instances_to_compute + instance_to_compute_no] = states_ref[state_index * n_instances_to_compute + instance_to_compute_no];
          }
        }
      });
    });
  }

  // check if fibers went NaN - OPTIONAL
  for (size_t i = 0; i < n_fibers_to_compute; i++)
  {
    if (std::isnan(states_for_transfer[i * n_instances_per_fiber + 781]))
    {
      std::cout << "Fiber " << i << " is NaN \n";
    }
  }

  // print run time - OPTIONAL
  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout.precision(2);
  std::cout << dev_Q.get_device().get_info<sycl::info::device::name>() << " computation time: " << elapsed.count() << "s\n";
  //Postprocessing
  //free_memory(); // TODO: remove for simulation in OpenDiHu
}
