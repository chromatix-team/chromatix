# The Convergent Born Series Iteration in JAX

## Overview of this Python sub-module

### __init__.py
Configuration of the log object. You can safely replace log.debug, log.info, ... statements by `print`.

### example_solve2d.py
Defines a 2D light scattering problem (just a glass slab), and solves it as an example. 

The waves are polarized and scatter of an isotropic material with refractive index 1.5.

### electro_solver.py
The implementation of the preconditioning and the solver. Note that the preconditioning uses a 
guess for the permittivity_bias. Ideally this should be automated as e.g. in MacroMax. The solver already implements 
anisotropic materials, though this code has not been tested and may be inefficient. Magnetic and cross terms as used 
for chiral materials are not implemented as these would add significantly more complexity. The same goes for more 
complex boundary procedures.

### benchmark_solve2d.py
Script to quantify the execution speed of the solver by running the example_solve2d multiple times.

### example_inverse_design2d.py
This script optimizes the refractive index to deposit as much light as possible into a target area.