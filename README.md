Description
---

`Hilbert` is a linear algebra library with focus on mathematical elegance and performance at high dimensions. The intended use is quantum mechanics, i.e. the implementation of vectors is mainly designed with quantum states ("Schrödinger wavefunctions"), linear operators mainly with quantum observables in mind. But most of it can as well be applied to other problems in which Hilbert spaces are relevant.

The rather common basis-dependent way of looking at vectors as arrays, linear operators as matrices is strongly avoided here. Obviously, those _are_ used as the underlying digital representations, but on the interface you are only dealing with vectors and operators as completely basis-independent objects. It is thus fine to define a state as some superposition of vectors defined in one basis, apply an operator on it that was defined in another basis, and finally calculate the expectation of some observable defined in a third one, without having to think about which transformations are needed at each step: this is in the `hilbertSpace` class's responsibility.



Design
---

`Hilbert` is a C++ template library. It uses C++11 features, in particular the functional extensions (lambdas etc.) to allow painless definition of the linear-algebra objects without having to resort to the aforementioned plain-array representations. Not only is this safer and mathematically more appealing, it also allows the implementation great freedom right away for optimising the storage location and thus possibility of using fast hardware for carrying out the actual expensive calculations. 

These calculations can be done using routines from various BLAS / LAPACK libraries, which are highly optimised but very difficult, unsafe and mathematically un-canonical to use directly. To safely deal with the caveats of those libraries without sacrificing performance again or being overly wasteful on memory, RAII is used consequently, fully utilising its C++11 augmentations through move semantics and smart pointers.



Dependencies / low-level libraries
---

Currently, the only supported low-level back end is [CUBLAS](http://developer.download.nvidia.com/compute/DevZone/docs/html/CUDALibraries/doc/CUBLAS_Library.pdf) and [CULA](http://www.culatools.com/), plus some custom operations implemented directly in CUDA. This combination allows the critical operations to be carried out seamlessly on a GPU, which is for many systems and problems the fastest possibility there is. But it is definitely not an acceptable default on the long run, for obvious reasons. The default should be [GSL](http://www.gnu.org/software/gsl/). 

Furthermore, two small auxiliary libraries are used: [Lambdalike](https://github.com/leftaroundabout/cxx-lambdalike) and [Cqtx](https://github.com/leftaroundabout/cqtx). The latter is not really in a sane state and only required for giving e.g. observable operators an actual physical dimension, in particular Hamiltonians. This should perhaps be changed to use something more standard, such as [boost::units](http://www.boost.org/doc/libs/1_37_0/doc/html/boost_units.html), which is also much more efficient.
