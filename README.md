# Porting a Simulation of the Activation of Muscle Fibers from OpenMP Offloading to SYCL

Minimum working example for easier development. Requires GNU GCC 10 or newer (with CUDA offloading support).

```bash
$ mkdir build && cd build
$ cmake ..
$ make
$ ./prog
```

Optional CMake options:

```bash
-DDEBUG_INFO=ON/OFF         print the read parameter values
-DENABLE_TIMING=ON/OFF      print timing information 
```

## Running the full simulation

1. Clone the [OpenDiHu github repo](https://github.com/maierbn/opendihu)
2. Build the library (`cd opendihu && make`)
3. Export the OpenDiHu root directory: `export OPENDIHU_HOME=/path/to/opendihu`
4. Create the directory `$OPENDIHU_HOME/examples/electrophysiology/input`
5. Copy the files
    * `hodgkin_huxley_1952.c`
    * `left_biceps_brachii_2x2fibers.bin`
    * `MU_fibre_distribution_10MUs.txt`
    * `MU_firing_times_always.txt`

   from the `simulation_files/` folder of this repo to the previously created folder
6. Go to `$OPENDIHU_HOME/examples/electrophysiology/fibers/fibers_emg/build_release` and
   run `./fast_fibers_emg ../settings_fibers_emg.py gpu.py`
7. Copy your `hodgkin_huxley_1952_gpu_fast_monodomain.0.cpp` file
   to `$OPENDIUH_HOME/examples/electrophysiology/fibers/fibers_emg/build_release/src`
8. Uncomment the option `libraryFilename` and change its value to `my_lib.so` in
   the `$OPENDIUH_HOME/examples/electrophysiology/fibers/fibers_emg/settings_fibers_emg.py` file
9. Go to the `$OPENDIUH_HOME/examples/electrophysiology/fibers/fibers_emg/build_release/src` directory and
   run `dpcpp hodgkin_huxley_1952_gpu_fast_monodomain.0.cpp -O3 -fPIC -shared -std=c++17 -sycl-std=2020 -o my_lib.so`
10. Rerun `./fast_fibers_emg ../settings_fibers_emg.py gpu.py` to use SYCL for the GPU support!
