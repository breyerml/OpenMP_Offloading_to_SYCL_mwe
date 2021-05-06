# Porting a Simulation of the Activation of Muscle Fibers from OpenMP Offloading to SYCL

Minimum working example for easier development. 
Requires a `C++17` compliant compiler.

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