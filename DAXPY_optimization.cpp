// DAXPY optimization with C++ Standard Library (lambdas) using GPU Acceleration. 

// par and par_unseq
/*
The difference between par and par_unseq is that par_unseq allows the implementation to
interleave the execution of multiple function calls in the same thread, while par does not.
This means that par_unseq can use vectorization or instruction-level parallelism to optimize the performance
of the algorithm, but it also requires stronger guarantees from the programmer that the function calls are independent 
and do not introduce data races or side effects.
For example, if you want to apply a function to each element of a vector, you can use par_unseq if 
the function is pure and does not modify any shared state, but you should use par if the function 
acquires a lock or performs I/O operations.
*/


// From Loop:

/// DAXPY: AX + Y: raw loop sequential version
void daxpy(double a, std::vector<double> const &x, std::vector<double> &y) {
  assert(x.size() == y.size());
  for (std::size_t i = 0; i < y.size(); ++i) {
    y[i] += a * x[i];
  }
}

// To Lambda:
// Using [&] because "a" is a stack-variable that in the CPU memory
/// DAXPY: AX + Y: sequential algorithm version
void daxpy(double a, std::vector<double> const &x, std::vector<double> &y) {
    assert(x.size() == y.size());
    // DONE: Implement using SEQUENTIAL transform algorithm    
    std::transform(x.begin(), x.end(), y.begin(),y.begin(),
                  [&](double x, double y) {return x * a + y;});
}

// ------------------------------------------------------------------ //

// From Loop:

/// Intialize vectors `x` and `y`: raw loop sequential version
void initialize(std::vector<double> &x, std::vector<double> &y) {
  assert(x.size() == y.size());
  for (std::size_t i = 0; i < x.size(); ++i) {
    x[i] = (double)i;
    y[i] = 2.;
  }
}

// To Lambda:

/// Intialize vectors `x` and `y`: raw loop sequential version
void initialize(std::vector<double> &x, std::vector<double> &y) {
  assert(x.size() == y.size());
  // DONE: Initialize `x` using SEQUENTIAL std::for_each_n algorithm with std::views::iota
    auto ints = std::views::iota(0);
    std::for_each_n(ints.begin(), x.size(), [&x] (int i) {x[i]=(double)i;});
  // DONE: Initialize `y` using SEQUENTIAL std::fill_n algorithm
    std::fill_n(y.begin(), y.size(), 2.);
}

// ------------------------------------------------------------------ //

// From lambda using CPU:

/// Intialize vectors `x` and `y`: raw loop sequential version
void initialize(std::vector<double> &x, std::vector<double> &y) {
  assert(x.size() == y.size());
  // DONE: Initialize `x` using SEQUENTIAL std::for_each_n algorithm with std::views::iota
  auto ints = std::views::iota(0);
  std::for_each_n(ints.begin(), x.size(), [&x](int i) { x[i] = (double)i; });
  // DONE: Initialize `y` using SEQUENTIAL std::fill_n algorithm
  std::fill_n(y.begin(), y.size(), 2.);
}

// To lambda using GPU:
// The changes:
//      1. [&x] -> [x = x.data()] : call x by reference.
//      2. std::execution::par_unseq :for parallelism
/// Intialize vectors `x` and `y`: parallel algorithm version
void initialize(std::vector<double> &x, std::vector<double> &y) {
  assert(x.size() == y.size());
  // DONE: Parallelize initialization of `x`
  auto ints = std::views::iota(0);
  std::for_each_n(std::execution::par_unseq, ints.begin(), x.size(), [x = x.data()](int i) {x[i] = (double)i;});
  // DONE: Parallelize initialization of `y`
  std::fill_n(std::execution::par_unseq, y.begin(), y.size(), 2.);
}

// also

// From lambda using CPU:

/// DAXPY: AX + Y: sequential algorithm version
void daxpy(double a, std::vector<double> const &x, std::vector<double> &y) {
  assert(x.size() == y.size());
  // DONE: Implement using SEQUENTIAL transform algorithm
  std::transform(x.begin(), x.end(), y.begin(), y.begin(),
                 [&](double x, double y) { return a * x + y; });
}

// To lambda using GPU:
// The changes:
//      1. [&] -> [a] : We can't use CPU memory on the GPU. So we use a by value.
//      2. std::execution::par_unseq :for parallelism

/// DAXPY: AX + Y: sequential algorithm version
void daxpy(double a, std::vector<double> const &x, std::vector<double> &y) {
  assert(x.size() == y.size());
  /// DONE: Parallelize DAXPY computation. Notice that `a` is captured by value.
  std::transform(std::execution::par_unseq, x.begin(), x.end(), y.begin(), y.begin(),
                 [a](double x, double y) { return a * x + y; });
}

