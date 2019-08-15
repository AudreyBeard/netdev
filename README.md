# netdev
A repository for fast, easy, and readable neural network development.
Inspired by [netharn](github.com/erotemic/netharn)

# Functionality:
---
- Class prototype that implements typical NN model functionality:
  - hyperparameters:
    - input shape
    - output len
    - nice name
    [ ] initializer (TODO)
    - Work directory
    - reset
    - verbosity
 - Provides cache saving and loading built in through the
   [ubelt](https://github.com/erotemic/ubelt) caching module

# Modules:
---
## netskel.py

### NetworkSkeleton
Backbone of the entire package, designed to serve as a simple and intuitive
superclass for network implementations. Includes the following functionality
(subject to change):
- `parameters(self)`
- `to(self, device)`
- `on_epoch(self, epoch, error)`
- `cache(self)`
- `load_cache(self)`

## utils/

### clf_utils.py
- `to_onehot`
- `compute_error`

### general_utils.py
- `deprecated`
- `isiterable`
- `check_constraints`

#### ParameterRegister
Container class for registering and managing constraints, defaults, and current
parameter values.
- Checks values against defined constraints
- sets uninitialized parameters to default values
- produces hashable string for caching
- supports unsetting parameters 

### net_utils.py
- `shape_for_shape`
- `output_shape_for`



