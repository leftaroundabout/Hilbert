  // Copyright 2012 Justus Sagemüller.
  // This file is part of the Hilbert library.
   //This library is free software: you can redistribute it and/or modify
  // it under the terms of the GNU General Public License as published by
 //  the Free Software Foundation, either version 3 of the License, or
//   (at your option) any later version.
   //This library is distributed in the hope that it will be useful,
  // but WITHOUT ANY WARRANTY; without even the implied warranty of
 //  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//   GNU General Public License for more details.
  // You should have received a copy of the GNU General Public License
 //  along with this library.  If not, see <http://www.gnu.org/licenses/>.

#ifndef ACCELERATED_VECTOR_OBJECTS
#define ACCELERATED_VECTOR_OBJECTS

      //C++ interface to low-level linear algebra routines, as can
     // be calculated at high-performance through e.g. CUDA/CULA.
    //  This interface itself does not aim at much safety/abstraction,
   //   dimension checks are generally not performed. Actual
  //    applications should use higher-level abstraction layers
 //     such as the hilbertSpace classes.


#include <vector>
#include <complex>
#include <set>
#include <cassert>

#ifdef CUDA_ACCELERATION
#   define USE_CUDA_TO_ACCELERATE_DOUBLE_VECTORS
#   define USE_CUDA_TO_ACCELERATE_COMPLEXDOUBLE_VECTORS
#endif

#if defined(USE_CUDA_TO_ACCELERATE_DOUBLE_VECTORS)\
 || defined(USE_CUDA_TO_ACCELERATE_COMPLEXDOUBLE_VECTORS)
#  include "cublas_v2.h"
#  include "cusparse.h"
#  include "cula.h"
   namespace acceleratedVectors {
     template<typename NumT>class accelVector;
     template<typename NumT>class accelSparseMatrix;
     template<typename NumT>class accelGeneralMatrix;
     template<typename NumT>class accelEigenbasisTransform;
     class accelHandle {
       static unsigned instancescount;    //debatable in terms of thread safety
       cublasHandle_t cublashandle;
       cusparseHandle_t cusparsehandle;
       cublasStatus_t cublasstatus;
       cusparseStatus_t cusparsestatus;
       culaStatus culastatus;
      public:
       accelHandle() {
         cublasstatus = cublasCreate(&cublashandle);
         assert(cublasstatus==CUBLAS_STATUS_SUCCESS);
         cusparsestatus = cusparseCreate(&cusparsehandle);
         assert(cusparsestatus==CUSPARSE_STATUS_SUCCESS);
         if(instancescount++==0)
           culastatus = culaInitialize();
       }
       accelHandle(const accelHandle& cp) =delete;
       accelHandle(accelHandle&& mov) noexcept
         : cublashandle(mov.cublashandle)
         , cusparsehandle(mov.cusparsehandle)
         , cublasstatus(mov.cublasstatus)
         , cusparsestatus(mov.cusparsestatus)
         , culastatus(mov.culastatus)
       {}
       accelHandle&operator=(accelHandle cp) {
         cublasDestroy(cublashandle);
         cusparseDestroy(cusparsehandle);
         --instancescount;
         cublashandle = cp.cublashandle;
         cusparsehandle = cp.cusparsehandle;
         cublasstatus = cp.cublasstatus;
         cusparsestatus = cp.cusparsestatus;
         culastatus = cp.culastatus;
         return *this;
       }
       ~accelHandle() {
         cublasDestroy(cublashandle);
         cusparseDestroy(cusparsehandle);
         if(--instancescount==0)
           culaShutdown();
       }
#      ifdef USE_CUDA_TO_ACCELERATE_DOUBLE_VECTORS
       friend class accelVector<double>;
#      endif
#      ifdef USE_CUDA_TO_ACCELERATE_COMPLEXDOUBLE_VECTORS
       friend class accelVector<std::complex<double>>;
       friend class accelSparseMatrix<std::complex<double>>;
       friend class accelGeneralMatrix<std::complex<double>>;
       friend class accelEigenbasisTransform<std::complex<double>>;
#      endif
     };
     unsigned accelHandle::instancescount = 0;
   }
#  define CUDAerrcheckcdH                 \
      if(!cudahandle){                    \
        printout_error(cudahandle);       \
        assert(!"CUDA operation");        \
      }
#  define CUDAerrcheckcdA(handle)            \
      if(!(handle)){              \
        printout_error(handle); \
        assert(!"CUDA operation");        \
      }
#  define CUDAerrcheckcdO(obj)            \
      if(!(obj).cudahandle){              \
        printout_error((obj).cudahandle); \
        assert(!"CUDA operation");        \
      }

#else/*if !(defined(USE_CUDA_TO_ACCELERATE_DOUBLE_VECTORS) \
 || defined(USE_CUDA_TO_ACCELERATE_COMPLEXDOUBLE_VECTORS))  */
   namespace acceleratedVectors {
     struct accelHandle{
       accelHandle(const accelHandle& cp) =delete;
       accelHandle&operator=(const accelHandle& cp) =delete;
       accelHandle() {};
     };
   }
#endif



namespace acceleratedVectors {



template<typename NumT>
class accelVector {
 public:
  typedef std::vector<NumT> HostArray;
 private:
  HostArray v;
  
 public:
  class iterator {
    unsigned i; accelVector* domain;
    iterator(unsigned i, accelVector* domain): i(i),domain(domain) {}
   public:
    auto operator*()const -> NumT& {return (*domain)[i];}
    iterator& operator++(){++i; return *this;}
    auto operator!=(const iterator& o)->bool{return i!=o.i;}
    friend class accelVector<NumT>;
  };
  auto begin()->iterator {return iterator(0,this);}
  auto end()->iterator {return iterator(v.size(),this);}
  
  auto allocated_dimension()const -> unsigned {return v.size();}

  auto operator[](unsigned i)      ->       NumT& { return v[i]; }  //avoid these
  auto operator[](unsigned i)const -> const NumT& { return v[i]; } // if possible
 /*(may require the whole vector to be copied into host memory and back to the accelerated device)*/
  
        //The BLAS axpy operation:  *this += α⋅x
  auto axpyze(const NumT& alpha, const accelVector& x) -> accelVector&;
  auto axpyzed(const NumT& alpha, const accelVector& x)const -> accelVector {
    return accelVector(*this).axpyze(alpha,x);                              }
  
  auto operator+=(const accelVector& y) -> accelVector& { return axpyze(1, y); }
  auto operator+(const accelVector& y)const -> accelVector {
    return accelVector(*this)+=y;                          }

  auto operator*(const accelVector& y)const -> NumT;
  
  void clear() {v.clear();}
  void accelerate() {}
  
  accelVector() {}   //Any operation other than = on such an uninitialized vector is undefined.
  accelVector(const accelHandle& h, HostArray v): v(v) {}
  accelVector(const accelHandle& h, unsigned n, const NumT& init=NumT())
    : v(HostArray(n, init)) {}
  
  auto released()const -> const HostArray& { return v; }
};

template<typename NumT>auto
accelVector<NumT>::axpyze(const NumT& alpha, const accelVector& x)
                                                   -> accelVector& {
  for(unsigned i=v.size(); i-->0;) v[i] += alpha * x[i];
  return *this;
}

template<typename NumT>auto
accelVector<NumT>::operator*(const accelVector& y)const -> NumT {
  NumT accum=0;
  for(unsigned i=v.size(); i-->0;)
    accum += v[i] * y[i];
  return accum;
}



template<typename NumT>        //No host implementation yet, so only
class accelGeneralMatrix {  // works in device specializations
 public:
  void gemv( const NumT& alpha, const accelVector<NumT>& x, const NumT& beta, accelVector<NumT>& y )const {
    std::cerr << "Working with instance of templatized accelGeneralMatrix, which currently has no implementation.\n";
    abort();
  }

  auto invmv( const NumT& alpha, const accelVector<NumT>& x )const -> accelVector<NumT> {
    std::cerr << "Working with instance of templatized accelGeneralMatrix, which currently has no implementation.\n";
    abort();
  }
  
  auto scaled_inverse(const NumT& alpha)const -> accelGeneralMatrix {
    std::cerr << "Working with instance of templatized accelGeneralMatrix, which currently has no implementation.\n";
    abort();
  }

  auto operator+=(const accelGeneralMatrix& o) -> accelGeneralMatrix& {
    std::cerr << "Working with instance of templatized accelGeneralMatrix, which currently has no implementation.\n";
    abort();
  }
  auto operator+(const accelGeneralMatrix& o)const -> accelGeneralMatrix {
    return accelGeneralMatrix(*this)+=o;                                }

  auto axpyze(const NumT& alpha, const accelGeneralMatrix& o) -> accelGeneralMatrix& {
    std::cerr << "Working with instance of templatized accelGeneralMatrix, which currently has no implementation.\n";
    abort();
  }

  auto operator*(const accelGeneralMatrix& rmtl)const -> accelGeneralMatrix {
    std::cerr << "Working with instance of templatized accelGeneralMatrix, which currently has no implementation.\n";
    abort();
  }

  void clear() {
    std::cerr << "Working with instance of templatized accelGeneralMatrix, which currently has no implementation.\n";
    abort();
  }

  accelGeneralMatrix() {
    std::cerr << "Working with instance of templatized accelGeneralMatrix, which currently has no implementation.\n";
    abort();
  }

  accelGeneralMatrix( const accelHandle& h
                    , unsigned covardim, unsigned cntvardim ) {
    std::cerr << "Working with instance of templatized accelGeneralMatrix, which currently has no implementation.\n";
    abort();
  }

  accelGeneralMatrix(const accelGeneralMatrix& cpy) {
    std::cerr << "Working with instance of templatized accelGeneralMatrix, which currently has no implementation.\n";
    abort();
  }
  accelGeneralMatrix(accelGeneralMatrix&& mov) noexcept {
    std::cerr << "Working with instance of templatized accelGeneralMatrix, which currently has no implementation.\n";
    abort();
  }

  accelGeneralMatrix& operator=(accelGeneralMatrix cpy) {
    std::cerr << "Working with instance of templatized accelGeneralMatrix, which currently has no implementation.\n";
    abort();
  }

  friend class accelSparseMatrix<NumT>;
  friend class accelEigenbasisTransform<NumT>;
};

template<typename NumT> class hostSparseMatrix;

template<typename NumT>
class accelSparseMatrix {
  typedef hostSparseMatrix<NumT> HostSpM;
  HostSpM m;
 public:
  typedef typename HostSpM::HostSparseMatrix HostSparseMatrix;

/*  auto operator()(const accelVector<NumT>& v)const -> accelVector<NumT>{
    return m(v);                                                       }  */
  void gemv( const NumT& alpha                   //BLAS gemv operation
           , const accelVector<NumT>& x         // (actually csrmv)
           , const NumT& beta                  //  y <- α(*this)x + β y
           , accelVector<NumT>& y        )const {
    m.gemv(alpha,x, beta,y);                    }
  auto operator+=(const accelSparseMatrix& o) -> accelSparseMatrix& {
    m+=o.m; return *this;                                           }
  auto operator+(const accelSparseMatrix& o)const -> accelSparseMatrix {
    return accelSparseMatrix(*this)+=o;                                }

  auto axpyze(const NumT& alpha, const accelSparseMatrix& o) -> accelSparseMatrix& {
    m.axpyze(alpha, o.m);
    return *this;
  }
  accelSparseMatrix& operator+=(const NumT& alpha) {  //o = id
    m+=alpha;
    return *this;
  }
  
  auto operator*(const accelSparseMatrix& rmtl)const -> accelSparseMatrix {
    accelSparseMatrix result;
    result.m = m*rmtl.m;
    return result;
  }

  auto invmv( const NumT& alpha                                          //inverse matrix multiplication
            , const accelVector<NumT>& x )const -> accelVector<NumT> {  // invmv(α,x) = (*this)⁻¹α⋅x
    std::cerr << "Inverse application of non-device accelSparseMatrix not implemented yet.\n";
    abort();
  }
  auto scaled_inverse(const NumT& alpha)const -> accelGeneralMatrix<NumT> {
    std::cerr << "Inverse of non-device accelSparseMatrix not implemented yet.\n";
    abort();
  }

  void clear() {m.clear();}

  void affirm_hermitian() { m.affirm_hermitian(); }
  void affirm_unitary() { m.affirm_unitary(); }

  accelSparseMatrix() {}
  accelSparseMatrix( const accelHandle& h
                   , unsigned covardim, unsigned cntvardim
                   , HostSparseMatrix m)
    : m(covardim, cntvardim, std::move(m))  {}
//  accelSparseMatrix(const accelHandle& h)
//    : m(std::move(m))   {}
  accelSparseMatrix(const accelHandle& h, unsigned n, const NumT& init=NumT())   //diagonal matrix
    : m(n,init)          {}
  
};



template<typename NumT>
struct cooSparseMatrixEntry {
  unsigned cntvar_idx, covar_idx;
  mutable NumT entry;
};
struct sparseMatrix_cntvarMajorOrdering{
  template<typename NumT>auto
  operator()( const cooSparseMatrixEntry<NumT>& l
            , const cooSparseMatrixEntry<NumT>& r) -> bool {
    if ( l.cntvar_idx != r.cntvar_idx ) return l.cntvar_idx < r.cntvar_idx;
     return l.covar_idx < r.covar_idx;
  }
};

template<typename NumT>
class hostSparseMatrix {
 public:
  typedef std::set<cooSparseMatrixEntry<NumT>, sparseMatrix_cntvarMajorOrdering>
           HostSparseMatrix;
 private:
  unsigned covardim, cntvardim;
  HostSparseMatrix m;
  
  enum class matAttribute { general
                          , hermitian
                          , unitary   };
  matAttribute attribute;
 public:

  auto covariant_dimension()const -> unsigned {return covardim;}
  auto contravariant_dimension()const -> unsigned {return cntvardim;}
  auto nnz()const -> unsigned {return m.size();}
  auto contents()const -> const HostSparseMatrix& {return m;}

//  auto operator()(const accelVector<NumT>&)const -> accelVector<NumT>;
  void gemv( const NumT& alpha, const accelVector<NumT>& x
           , const NumT& beta, accelVector<NumT>& y        )const;
  hostSparseMatrix& operator+=(const hostSparseMatrix&);
  auto operator+(const hostSparseMatrix& o)const -> hostSparseMatrix {
    return hostSparseMatrix(*this)+=o;                               }

  hostSparseMatrix& axpyze(const NumT& alpha, const hostSparseMatrix& o);
  hostSparseMatrix& operator+=(const NumT& alpha);
  
  auto operator*(const hostSparseMatrix&)const -> hostSparseMatrix;

  void clear() {m.clear();}

  auto is_hermitian()const -> bool { return attribute == matAttribute::hermitian; }
  auto is_unitary()const -> bool { return attribute == matAttribute::unitary; }
  void affirm_hermitian() { attribute = matAttribute::hermitian; }
  void affirm_unitary() { attribute = matAttribute::unitary; }

  hostSparseMatrix() {}
  hostSparseMatrix(unsigned covardim, unsigned cntvardim)
    : covardim(covardim)
    , cntvardim(cntvardim)  
    , attribute(matAttribute::general)
  {}
  hostSparseMatrix(unsigned covardim, unsigned cntvardim, HostSparseMatrix cstrm)
    : covardim(covardim)
    , cntvardim(cntvardim)
    , m(std::move(cstrm))
    , attribute(matAttribute::general)                                  {
    for(auto& e: m){
      if(e.covar_idx >= covardim) {
        std::cerr<<e.covar_idx<<" >= "<<covardim<<std::endl;
        assert(e.covar_idx < covardim);
      }
      if(e.covar_idx >= covardim) {
        std::cerr<<e.cntvar_idx<<" >= "<<cntvardim<<std::endl;
        assert(e.cntvar_idx < cntvardim);
      }
    }
   // std::cout << "checked given sparse matrix with nnz "<<m.size()<<"; covar-dim: "<<covardim<<", cntvardim: "<<cntvardim<<std::endl;
  }
  hostSparseMatrix(unsigned n, const NumT& init=NumT())   //Diagonal matrix
    : covardim(n)
    , cntvardim(n)
    , attribute(matAttribute::general)                          {
    for(unsigned j=0; j<n; ++j)
      m.insert(cooSparseMatrixEntry<NumT>{j, j, init});
  }
};
/*
template<typename NumT>auto
hostSparseMatrix<NumT>::
operator()(const accelVector<NumT>& v)const -> accelVector<NumT> {
  accelVector<NumT> result(cntvardim, 0);
  for (auto& e: m)
    result[e.cntvar_idx] += e.entry * v[e.covar_idx];
  return result;
}
*/
template<typename NumT>void
hostSparseMatrix<NumT>::
gemv( const NumT& alpha, const accelVector<NumT>& x
           , const NumT& beta, accelVector<NumT>& y        )const {
  assert(x.allocated_dimension() == covardim
    || (std::cerr<<"Wrong dimension "<<x.allocated_dimension()<<", should be "<<covardim<<std::endl &&0) );
  assert(y.allocated_dimension() == cntvardim
    || (std::cerr<<"Wrong dimension "<<y.allocated_dimension()<<", should be "<<covardim<<std::endl &&0) );
  for (unsigned j=0; j<cntvardim; ++j)
    y[j] *= beta;
  for (auto& e: m)
    y[e.cntvar_idx] += alpha * e.entry * x[e.covar_idx];
}

template<typename NumT>auto
hostSparseMatrix<NumT>::
operator+=(const hostSparseMatrix& other) -> hostSparseMatrix<NumT>& {
  if(is_hermitian() && !other.is_hermitian()) attribute=matAttribute::general;
  for (auto& e: other.m) {
    auto ex = m.insert(e);
    if (!ex.second)
      ex.first->entry += e.entry;
    if (abs(ex.first->entry) == 0) m.erase(ex.first);
  }
  return *this;
}

template<typename NumT>auto
hostSparseMatrix<NumT>::
axpyze(const NumT& alpha, const hostSparseMatrix& o) -> hostSparseMatrix<NumT>& {
  if(is_hermitian() && !o.is_hermitian()) attribute=matAttribute::general;
  for (auto e: o.m) {
    e.entry*=alpha;
    auto ex = m.insert(e);
    if (!ex.second)
      ex.first->entry += e.entry;
    if (abs(ex.first->entry) == 0) m.erase(ex.first);
  }
  return *this;
}

template<typename NumT>auto
hostSparseMatrix<NumT>::
operator+=(const NumT& alpha) -> hostSparseMatrix<NumT>& {
  for (unsigned j=0; j<cntvardim; ++j) {
    cooSparseMatrixEntry<NumT> e{j, j, alpha};
    auto ex = m.insert(e);
    if (!ex.second)
      ex.first->entry += e.entry;
    if (abs(ex.first->entry) == 0) m.erase(ex.first);
  }
  return *this;
}

template<typename NumT>auto
hostSparseMatrix<NumT>::
operator*(const hostSparseMatrix& o)const -> hostSparseMatrix<NumT> {
  hostSparseMatrix result;
  result.covardim = o.covardim;
  result.cntvardim = cntvardim;
  for (auto& c: o.m) {
    for (auto& d: m) {
      if(d.covar_idx==c.cntvar_idx) {  //inefficient, should iterate over just this range
        cooSparseMatrixEntry<NumT> e{ d.cntvar_idx, c.covar_idx
                                    , d.entry * c.entry         };
        auto ex = result.m.insert(e);
        if (!ex.second)
          ex.first->entry += e.entry;
        if (abs(ex.first->entry) == 0) result.m.erase(ex.first);
      }
    }
  }

  bool hermitian=true;          //Exhaustive check, again rather inefficient.
  for(auto d: result.m) {
    std::swap(d.covar_idx,d.cntvar_idx);
    auto ex = result.m.find(d);
    if( ex==result.m.end()
       || abs(ex->entry - conj(d.entry)) > std::max( abs(ex->entry)
                                                   , abs(d.entry)   ) * 1e-8 ) {
      hermitian=false; break;
    }
  }
  if(hermitian) result.affirm_hermitian();
  return result;
}



template<typename NumT>              //No host implementation yet, so only
class accelEigenbasisTransform {};  // works in device specializations




}//namespace acceleratedVectors;


#ifdef USE_CUDA_TO_ACCELERATE_DOUBLE_VECTORS

#include "cudaDaccel.h"

namespace acceleratedVectors{

auto operator!(const cudaDoubleHilbertspcVectHandle& h) -> bool {
  return cudaDoubleHilbertspcVect_bad(&h)!=0;                   }
void printout_error(const cudaDoubleHilbertspcVectHandle& h) {
  cudaDoubleHilbertspcVect_printerrmsg(&h);                  }
// void statusreset(const cudaDoubleHilbertspcVectHandle& h) {
//   cudaDoubleHilbertspcVect_resetStatus(&h);               }


template<>
class accelVector<double> {

  typedef double NumT;
 public:
  typedef std::vector<NumT> HostArray;

  class iterator {
    unsigned i; accelVector* domain;
    iterator(unsigned i, accelVector* domain): i(i),domain(domain) {}
   public:
    auto operator*()const -> NumT& {return (*domain)[i];}
    iterator& operator++(){++i; return *this;}
    auto operator!=(const iterator& o)->bool{return i!=o.i;}
    friend class accelVector<NumT>;
  };
  auto begin()->iterator {enter_host_state();return iterator(0,this);}
  auto end()->iterator {enter_host_state();return iterator(hostvct.size(),this);}

 private:

  enum class datastate
    { device
    , host
    , undefined };
  mutable datastate statenow; //Changing the memory location between host and device
                             // is not considered a modification of the vector object.
  mutable HostArray hostvct;
  mutable cudaDoubleHilbertspcVectHandle cudahandle;

  void cpy_devicevect_to_host(HostArray& tgt)const {
    assert(statenow==datastate::device);
    tgt.resize(cudahandle.vect_dimension);
    get_cudaDoubleHilbertspcVect(
                        tgt.data()
                      , &cudahandle);                        CUDAerrcheckcdH
  }
  void enter_host_state()const {
    if (statenow == datastate::device){
      cpy_devicevect_to_host(hostvct);
      statenow = datastate::host;
      delete_cudaDoubleHilbertspcVect( &cudahandle );   CUDAerrcheckcdH
    }
  }
  void cpy_hostvect_to_device(const HostArray& src)const {  //NOT actually const, but this allows it to be
    if(statenow==datastate::device){                       // used for (pseudo-const) storage-realignment
      delete_cudaDoubleHilbertspcVect( &cudahandle );   CUDAerrcheckcdH
    }
    cudahandle = new_cudaDoubleHilbertspcVect(
                        src.data()
                      , src.size()
                      , cudahandle.cublashandle );           CUDAerrcheckcdH
    statenow=datastate::device;
  }
  void enter_device_state()const {
    if (statenow == datastate::host){
      cpy_hostvect_to_device(hostvct);
      hostvct.clear();
    }
  }

 public:
  auto dimension()const -> unsigned { switch (statenow) {
    case(datastate::device): return cudahandle.vect_dimension;                 default/*
    case(datastate::host)*/: return hostvct.size();
  }}

  auto operator[](unsigned i) -> NumT&{
    enter_host_state();
    return hostvct[i];
  }
  auto operator[](unsigned i)const -> const NumT&{
    enter_host_state();
    return hostvct[i];
  }
  
  auto axpyze(const NumT& alpha, const accelVector& x) -> accelVector&{
    enter_device_state();  x.enter_device_state();
    axpy_cudaDoubleHilbertspcVect(
               alpha
             , &x.cudahandle
             , &cudahandle            );                     CUDAerrcheckcdH
    return *this;
  }
  auto axpyzed(const NumT& alpha, const accelVector& x)const -> accelVector {
    return accelVector(*this).axpyze(alpha,x);                              }
  
  auto operator+=(const accelVector& y) -> accelVector& {
    return axpyze( 1, y );                                }
  auto operator+(const accelVector& y)const -> accelVector {
    return accelVector(*this)+=y;                           }

  auto operator*(const accelVector& y)const -> NumT {
    enter_device_state(); y.enter_device_state();
    double result
         = dot_cudaDoubleHilbertspcVect(
                     &cudahandle
                   , &y.cudahandle    );                     CUDAerrcheckcdH
    return result;
  }

  void clear() {
    if(statenow==datastate::device) {
      delete_cudaDoubleHilbertspcVect( &cudahandle );   CUDAerrcheckcdH
     }else{
      hostvct.clear();
    }
    statenow = datastate::undefined;
  }
  void accelerate() { enter_device_state(); }
  
  accelVector(): statenow(datastate::undefined) {}
  accelVector(const accelHandle& h, HostArray v)
    : statenow(datastate::host)
    , hostvct(std::move(v))
    , cudahandle(unassigned_cudaDoubleHilbertspcVectHandle(&h.cublashandle)) {}
  accelVector(const accelHandle& h, unsigned n)
    : statenow(datastate::device)
    , cudahandle(undefined_cudaDoubleHilbertspcVect(n, &h.cublashandle)) {
    CUDAerrcheckcdH
  }
  accelVector(const accelHandle& h, unsigned n, const NumT& init)
    : statenow(datastate::host)
    , hostvct(HostArray(n, init))
    , cudahandle(unassigned_cudaDoubleHilbertspcVectHandle(&h.cublashandle)) {}
  accelVector(const accelVector& cpy)      //the data will always be
    : statenow(cpy.statenow)              // copied to the device.
    , cudahandle(cpy.cudahandle)     {
    if(statenow==datastate::device){
      cudahandle = copy_cudaDoubleHilbertspcVect(&cudahandle); CUDAerrcheckcdH
     }else if(statenow==datastate::host){
      cpy_hostvect_to_device(cpy.hostvct);
      statenow=datastate::device;
    }
  }
  accelVector(accelVector&& mov)
    : statenow(mov.statenow)
    , hostvct(std::move(mov.hostvct))
    , cudahandle(mov.cudahandle)      {
    mov.statenow = datastate::host;    //prevents mov's destructor from deleting
  }                                   // the device vector handled by cudahandle.

  accelVector& operator=(accelVector cpy) {
    if(statenow==datastate::device) {
      delete_cudaDoubleHilbertspcVect( &cudahandle );   CUDAerrcheckcdH
    }
    statenow = cpy.statenow;
    hostvct = std::move(cpy.hostvct);
    cudahandle = cpy.cudahandle;
    cpy.statenow = datastate::host;
    return *this;
  }
  
  auto released()const -> HostArray {
    if(statenow==datastate::host) return hostvct;
    HostArray result;
    cpy_devicevect_to_host(result);
    return result;
  }
  
  ~accelVector() {
    if(statenow==datastate::device) {
      delete_cudaDoubleHilbertspcVect( &cudahandle );   CUDAerrcheckcdH
    }
  }
#ifdef USE_CUDA_TO_ACCELERATE_COMPLEXDOUBLE_VECTORS
  friend class accelVector<std::complex<double>>;
  friend class accelEigenbasisTransform<std::complex<double>>;
#endif  
};

}//namespace acceleratedVectors;
#endif//def USE_CUDA_TO_ACCELERATE_DOUBLE_VECTORS



#ifdef USE_CUDA_TO_ACCELERATE_COMPLEXDOUBLE_VECTORS

#include "cudaZaccel.h"
#include "cuComplex.h"


namespace acceleratedVectors{

using std::complex;

#define DEFINE_CUDACOMPLEXHILBERTACCEL_ERRORHANDLERS(CudahandlePrefx) \
auto operator!(const CudahandlePrefx##Handle& h) -> bool {            \
  return CudahandlePrefx##_bad(&h)!=0;                   }            \
void printout_error(const CudahandlePrefx##Handle& h) {               \
  CudahandlePrefx##_printerrmsg(&h);                  }               \
void statusreset(CudahandlePrefx##Handle& h) {                        \
  CudahandlePrefx##_resetStatus(&h);         }
DEFINE_CUDACOMPLEXHILBERTACCEL_ERRORHANDLERS(cudaCmplxDoubleHilbertspcVect)
DEFINE_CUDACOMPLEXHILBERTACCEL_ERRORHANDLERS(cudaCmplxDoubleHilbertspcGenM)
DEFINE_CUDACOMPLEXHILBERTACCEL_ERRORHANDLERS(cudaCmplxDoubleHilbertspcSparseM)
DEFINE_CUDACOMPLEXHILBERTACCEL_ERRORHANDLERS(cudaCmplxDoubleHilbertspcEigenbasisTransform)

#if 0
auto operator!(const cudaCmplxDoubleHilbertspcVectHandle& h) -> bool {
  return cudaCmplxDoubleHilbertspcVect_bad(&h)!=0;                   }
void printout_error(const cudaCmplxDoubleHilbertspcVectHandle& h) {
  cudaCmplxDoubleHilbertspcVect_printerrmsg(&h);                  }
void statusreset(cudaCmplxDoubleHilbertspcVectHandle& h) {
  cudaCmplxDoubleHilbertspcVect_resetStatus(&h);         }
auto operator!(const cudaCmplxDoubleHilbertspcGenMHandle& h) -> bool {
  return cudaCmplxDoubleHilbertspcGenM_bad(&h)!=0;                   }
void printout_error(const cudaCmplxDoubleHilbertspcGenMHandle& h) {
  cudaCmplxDoubleHilbertspcGenM_printerrmsg(&h);                  }
void statusreset(cudaCmplxDoubleHilbertspcGenMHandle& h) {
  cudaCmplxDoubleHilbertspcGenM_resetStatus(&h);         }
auto operator!(const cudaCmplxDoubleHilbertspcSparseMHandle& h) -> bool {
  return cudaCmplxDoubleHilbertspcSparseM_bad(&h)!=0;                   }
void printout_error(const cudaCmplxDoubleHilbertspcSparseMHandle& h) {
  cudaCmplxDoubleHilbertspcSparseM_printerrmsg(&h);                  }
void statusreset(cudaCmplxDoubleHilbertspcSparseMHandle& h) {
  cudaCmplxDoubleHilbertspcSparseM_resetStatus(&h);         }
#endif


namespace check_stdComplex_to_cuComplex_binary_compatibility{
  const complex<double> testarr[] = { complex<double>(0.,.5)
                                    , complex<double>(1.,1.5) };
  const cuDoubleComplex* cucomplexd
     = reinterpret_cast<const cuDoubleComplex*>(testarr);
  auto tester() -> bool {
    assert( cuCreal(cucomplexd[0])==0. && cuCimag(cucomplexd[0])==.5
                && cuCreal(cucomplexd[1])==1. && cuCimag(cucomplexd[1])==1.5 );
    return true;
  }
  const bool ok = tester();
  bool good(){return ok;}
};



template<>
class accelVector<complex<double>> {

  typedef complex<double> NumT;
 public:
  typedef std::vector<NumT> HostArray;

  class iterator {
    unsigned i; accelVector* domain;
    iterator(unsigned i, accelVector* domain): i(i),domain(domain) {}
   public:
    auto operator*()const -> NumT& {return (*domain)[i];}
    iterator& operator++(){++i; return *this;}
    auto operator!=(const iterator& o)->bool{return i!=o.i;}
    friend class accelVector<NumT>;
  };
  auto begin()->iterator {enter_host_state();return iterator(0,this);}
  auto end()->iterator {enter_host_state();return iterator(hostvct.size(),this);}

 private:

  enum class datastate
    { device
    , host
    , undefined };
  mutable datastate statenow; //Changing the memory location between host and device
                             // is not considered a modification of the vector object.
  mutable HostArray hostvct;
  mutable cudaCmplxDoubleHilbertspcVectHandle cudahandle;
  
  void cpy_devicevect_to_host(HostArray& tgt)const {
    assert(statenow==datastate::device);
    tgt.resize(cudahandle.vect_dimension);
    get_cudaCmplxDoubleHilbertspcVect(
                        reinterpret_cast<cuDoubleComplex*>(tgt.data())
                      , &cudahandle);                        CUDAerrcheckcdH
  }
  void enter_host_state()const {
    if (statenow == datastate::device){
      cpy_devicevect_to_host(hostvct);
      statenow = datastate::host;
      delete_cudaCmplxDoubleHilbertspcVect( &cudahandle );   CUDAerrcheckcdH
    }
  }
  void cpy_hostvect_to_device(const HostArray& src)const {  //NOT actually const, but this allows it to be
    if(statenow==datastate::device){                       // used for (pseudo-const) storage-realignment
      delete_cudaCmplxDoubleHilbertspcVect( &cudahandle );   CUDAerrcheckcdH
    }
    cudahandle = new_cudaCmplxDoubleHilbertspcVect(
                        reinterpret_cast<const cuDoubleComplex*>(src.data())
                      , src.size()
                      , cudahandle.cublashandle );           CUDAerrcheckcdH
    statenow=datastate::device;
  }
  void enter_device_state()const {
    if (statenow == datastate::host){
      cpy_hostvect_to_device(hostvct);
      hostvct.clear();
    }
  }

 public:
  auto operator[](unsigned i) -> NumT&{
    enter_host_state();
    return hostvct[i];
  }
  auto operator[](unsigned i)const -> const NumT&{
    enter_host_state();
    return hostvct[i];
  }
  
  auto axpyze(const NumT& alpha, const accelVector& x) -> accelVector&{
    enter_device_state();  x.enter_device_state();
    axpy_cudaCmplxDoubleHilbertspcVect(
               *reinterpret_cast<const cuDoubleComplex*>(&alpha)
             , &x.cudahandle
             , &cudahandle            );                     CUDAerrcheckcdH
    return *this;
  }
  auto axpyzed(const NumT& alpha, const accelVector& x)const -> accelVector {
    return accelVector(*this).axpyze(alpha,x);                              }
  
  auto operator+=(const accelVector& y) -> accelVector& {
    return axpyze( NumT(1,0), y );                      }
  auto operator+(const accelVector& y)const -> accelVector {
    return accelVector(*this)+=y;                          }

  auto operator*(const accelVector& y)const -> NumT {
    enter_device_state(); y.enter_device_state();
    cuDoubleComplex result
         = dotc_cudaCmplxDoubleHilbertspcVect(
                     &cudahandle
                   , &y.cudahandle    );                     CUDAerrcheckcdH
    return NumT(cuCreal(result), cuCimag(result));
  }
  
  auto phaserotations(double alpha, const accelVector<double>& thetas)const
                                                 -> accelVector {
    enter_device_state();
#ifdef USE_CUDA_TO_ACCELERATE_DOUBLE_VECTORS
    thetas.enter_device_state();
    accelVector result(cudahandle);
    result.cudahandle
      = phaserotation_cudaCmplxDoubleHilbertspcVect( alpha
                                                   , thetas.cudahandle.vector
                                                   , &cudahandle ); CUDAerrcheckcdH
    return result;
#else
    static_assert(false,
      "accelVector<double> and accelVector<complex<double>> need to be implemented on the same device to allow accelerated phase rotations.");
#endif
  }

  void clear() {
    if(statenow==datastate::device) {
      delete_cudaCmplxDoubleHilbertspcVect( &cudahandle );   CUDAerrcheckcdH
     }else{
      hostvct.clear();
    }
    statenow = datastate::undefined;
  }
  void accelerate() { enter_device_state(); }

  auto dimension()const -> unsigned { switch (statenow) {
    case(datastate::device): return cudahandle.vect_dimension;                 default/*
    case(datastate::host)*/: return hostvct.size();
  }}
  
 private:
  accelVector(const cudaCmplxDoubleHilbertspcVectHandle& h)               //these
    : statenow(datastate::device)                                        // constructors
    , cudahandle(h) {}                                                  //  do NOT copy
  accelVector(const cudaCmplxDoubleHilbertspcVectHandle& h, unsigned n)//   the handled
    : statenow(datastate::device)                                     //    object.
    , cudahandle(undefined_cudaCmplxDoubleHilbertspcVect(n, h.cublashandle)) {}
 public:
  accelVector(): statenow(datastate::undefined) {}
  accelVector(const accelHandle& h, HostArray v)
    : statenow(datastate::host)
    , hostvct(std::move(v))
    , cudahandle(unassigned_cudaCmplxDoubleHilbertspcVectHandle(&h.cublashandle)) {}
  accelVector(const accelHandle& h, unsigned n)
    : statenow(datastate::device)
    , cudahandle(undefined_cudaCmplxDoubleHilbertspcVect(n, &h.cublashandle)) {
    CUDAerrcheckcdH
  }
  accelVector(const accelHandle& h, unsigned n, const NumT& init)
    : statenow(datastate::host)
    , hostvct(HostArray(n, init))
    , cudahandle(unassigned_cudaCmplxDoubleHilbertspcVectHandle(&h.cublashandle)) {}
  accelVector(const cublasHandle_t* h, unsigned n, const NumT& init)
    : statenow(datastate::host)
    , hostvct(HostArray(n, init))
    , cudahandle(unassigned_cudaCmplxDoubleHilbertspcVectHandle(h)) {}
  accelVector(const accelVector& cpy)
    : statenow(cpy.statenow)             
    , cudahandle(cpy.cudahandle)     {
    if(statenow==datastate::device){
      cudahandle = copy_cudaCmplxDoubleHilbertspcVect(&cudahandle); CUDAerrcheckcdH
     }else if(statenow==datastate::host){
      cpy_hostvect_to_device(cpy.hostvct);
      statenow=datastate::device;
    }
  }
  accelVector(accelVector&& mov)
    : statenow(mov.statenow)
    , hostvct(std::move(mov.hostvct))
    , cudahandle(mov.cudahandle)      {
    mov.statenow = datastate::host;    //prevents mov's destructor from deleting
  }                                   // the device vector handled by cudahandle.

  accelVector& operator=(accelVector cpy) {
    if(statenow==datastate::device) {
      delete_cudaCmplxDoubleHilbertspcVect( &cudahandle );   CUDAerrcheckcdH
    }
    statenow = cpy.statenow;
    hostvct = std::move(cpy.hostvct);
    cudahandle = cpy.cudahandle;
    cpy.statenow = datastate::host;
    return *this;
  }
  
  auto released()const -> HostArray {
    if(statenow==datastate::host) return hostvct;
    HostArray result;
    cpy_devicevect_to_host(result);
    return result;
  }
  
  ~accelVector() {
    if(statenow==datastate::device) {
      delete_cudaCmplxDoubleHilbertspcVect( &cudahandle );   CUDAerrcheckcdH
    }
  }
  
  friend class accelSparseMatrix<complex<double>>;
  friend class accelGeneralMatrix<complex<double>>;
  friend class accelEigenbasisTransform<complex<double>>;
};


template<>
class accelGeneralMatrix<complex<double>> {
  typedef complex<double> NumT;

  enum class datastate
    { device
//    , host
    , undefined };
  mutable datastate statenow;

  mutable cudaCmplxDoubleHilbertspcGenMHandle cudahandle;//Dense matrices are, as of now, always
                                                        // stored on device. Still using mutable
                                                       //  since the low-level functions don't
                                                      //   properly const-qualify passed pointers.
 public:

/*  auto operator()(const accelVector<NumT>& v)const -> accelVector<NumT>{
    return m(v);                                                       }  */
  void gemv( const NumT& alpha                   //BLAS gemv operation
           , const accelVector<NumT>& x         // y <- α(*this)x + β y
           , const NumT& beta
           , accelVector<NumT>& y        )const {
    gemv_cudaCmplxDoubleHilbertspcGenMToVect(
                  *reinterpret_cast<const cuDoubleComplex*>(&alpha)
                , *reinterpret_cast<const cuDoubleComplex*>(&beta)
                , &cudahandle
                , &x.cudahandle
                , &y.cudahandle                                     ); CUDAerrcheckcdH
  }

  auto invmv( const NumT& alpha                                          //inverse matrix multiplication
            , const accelVector<NumT>& x )const -> accelVector<NumT> {  // invmv(α,x) = (*this)⁻¹α⋅x, very
                                                                       //  expensive for non-triangular matrices.
    auto mutablecpy = copy_cudaCmplxDoubleHilbertspcGenM              //   (At the moment, even for those)
                                 (&cudahandle);           CUDAerrcheckcdH
      
    auto result = x;
    invapply_cudaCmplxDoubleHilbertspcGenMToVect(&mutablecpy, &result.cudahandle);
    if(!mutablecpy) {printout_error(mutablecpy); assert(!!mutablecpy);}
    if(alpha!=1.) result.axpyze(alpha-1.,result);

    delete_cudaCmplxDoubleHilbertspcGenM(&mutablecpy);

    return result;
  }
  
  auto scaled_inverse(const NumT& alpha)const -> accelGeneralMatrix {
    accelGeneralMatrix result;
    result.cudahandle
       = inverted_cudaCmplxDoubleHilbertspcGenM( *reinterpret_cast<const cuDoubleComplex*>(&alpha)
                                               , &cudahandle                                       );
    return result;
  }

  auto operator+=(const accelGeneralMatrix& o) -> accelGeneralMatrix& {
    axpy_cudaCmplxDoubleHilbertspcGenM( make_cuDoubleComplex(1.,0.)
                                      , &o.cudahandle
                                      , &cudahandle                 );
    return *this;
  }
  auto operator+(const accelGeneralMatrix& o)const -> accelGeneralMatrix {
    return accelGeneralMatrix(*this)+=o;                                }

  auto axpyze(const NumT& alpha, const accelGeneralMatrix& o) -> accelGeneralMatrix& {
    axpy_cudaCmplxDoubleHilbertspcGenM( *reinterpret_cast<const cuDoubleComplex*>(&alpha)
                                      , &o.cudahandle
                                      , &cudahandle                                        );
    return *this;
  }

  auto operator*(const accelGeneralMatrix& rmtl)const -> accelGeneralMatrix {
    accelGeneralMatrix result;
    std::cerr << "Multiplication of dense matrices not implemented yet\n";
    abort();
    return result;
  }

  void clear() {
    delete_cudaCmplxDoubleHilbertspcGenM( &cudahandle );   CUDAerrcheckcdH
    statenow = datastate::undefined;
  }

  accelGeneralMatrix(): statenow(datastate::undefined) {}
/*  accelGeneralMatrix(const accelHandle& h, unsigned covardim, unsigned cntvardim)
    : statenow(datastate::device)
    , cudahandle(unassigned_cudaCmplxDoubleHilbertspcGenMHandle(&h.cublashandle))
  {}*/
  accelGeneralMatrix( const accelHandle& h
                    , unsigned covardim, unsigned cntvardim )
    : statenow(datastate::device)
    , cudahandle(undefined_cudaCmplxDoubleHilbertspcGenM( cuCmplxDouble_GENERIC_MATRIX
                                                        , cuCmplxDouble_MATPACKING_NONE
                                                        , covardim
                                                        , cntvardim
                                                        , &h.cublashandle) )
  {}
/*  accelGeneralMatrix(const accelHandle& h, unsigned n, const NumT& init=NumT())
    : acchandles(&h)
    , statenow(datastate::host)
    , host_spam(n,init)
    , cudahandle(unassigned_cudaCmplxDoubleHilbertspcSparseMHandle(&h.cusparsehandle))
  {}*/

  accelGeneralMatrix(const accelGeneralMatrix& cpy)  
    : statenow(cpy.statenow)                       
    , cudahandle(cpy.cudahandle)     {
    if(statenow==datastate::device){
      cudahandle = copy_cudaCmplxDoubleHilbertspcGenM(&cudahandle); CUDAerrcheckcdH
    }
  }
  accelGeneralMatrix(accelGeneralMatrix&& mov) noexcept
    : statenow(mov.statenow)
    , cudahandle(mov.cudahandle)      {
    mov.statenow = datastate::undefined; //prevents mov's destructor from deleting
  }                                     // the device vector handled by cudahandle.

  accelGeneralMatrix& operator=(accelGeneralMatrix cpy) {
    if(statenow==datastate::device) {
      delete_cudaCmplxDoubleHilbertspcGenM( &cudahandle );   CUDAerrcheckcdH
    }
    statenow = cpy.statenow;
    cudahandle = cpy.cudahandle;
    cpy.statenow = datastate::undefined;
    return *this;
  }

  ~accelGeneralMatrix() {
    if(statenow==datastate::device) {
      delete_cudaCmplxDoubleHilbertspcGenM( &cudahandle );   CUDAerrcheckcdH
    }
  }

  friend class accelSparseMatrix<complex<double>>;
  friend class accelEigenbasisTransform<complex<double>>;
};



template<>
class accelSparseMatrix<complex<double>> {
  typedef complex<double> NumT;
  typedef hostSparseMatrix<NumT> HostSpM;

  const accelHandle* acchandles;

  enum class datastate
    { device
    , host
    , undefined };
  mutable datastate statenow; //Changing the memory location between host and device
                             // is not considered a modification of the object.
  mutable HostSpM host_spam;
  mutable cudaCmplxDoubleHilbertspcSparseMHandle cudahandle;

  void cpy_devicespam_to_host(HostSpM& tgt)const {
    assert(statenow==datastate::device);

    cuCmplxDoubleSparseMatEntries carrayptrs = cudahandle.matrix;

    std::vector<int> coo_covarids(carrayptrs.nnz);
    std::vector<int> coo_cntvarids(carrayptrs.nnz);
    std::vector<NumT> entries(carrayptrs.nnz);

    carrayptrs.row = coo_cntvarids.data();
    carrayptrs.col = coo_covarids.data();
    carrayptrs.entries = reinterpret_cast<cuDoubleComplex*>(entries.data());

    get_cudaCmplxDoubleHilbertspcSparseM(
                        carrayptrs
                      , &cudahandle);                        CUDAerrcheckcdH
    
    typename HostSpM::HostSparseMatrix mcstr;
    
    for(unsigned i=0; i<carrayptrs.nnz; ++i) {
      mcstr.insert( cooSparseMatrixEntry<NumT> { coo_cntvarids[i]
                                               , coo_covarids[i]
                                               , entries[i]       } );
      if(cusparseGetMatType(cudahandle.matrxdescript)==CUSPARSE_MATRIX_TYPE_HERMITIAN
          && coo_cntvarids[i] > coo_covarids[i] )
        mcstr.insert( cooSparseMatrixEntry<NumT> { coo_covarids[i]
                                                 , coo_cntvarids[i]
                                                 , conj(entries[i]) } );
    }

    tgt = HostSpM(carrayptrs.covardim,carrayptrs.cntvardim,mcstr);

    if(cusparseGetMatType(cudahandle.matrxdescript)==CUSPARSE_MATRIX_TYPE_HERMITIAN)
      tgt.affirm_hermitian();
  }
  void enter_host_state()const {
    if (statenow == datastate::device){
      cpy_devicespam_to_host(host_spam);
      statenow = datastate::host;
      delete_cudaCmplxDoubleHilbertspcSparseM( &cudahandle );   CUDAerrcheckcdH
    }
  }

  void cpy_hostspam_to_device(const HostSpM& src)const {   //NOT actually const, but this allows it to be
    if(statenow==datastate::device){                      // used for (pseudo-const) storage-realignment
      delete_cudaCmplxDoubleHilbertspcSparseM( &cudahandle );   CUDAerrcheckcdH
    }
    
    std::vector<int> coo_covarids(src.nnz());
    std::vector<int> coo_cntvarids(src.nnz());
    std::vector<NumT> entries(src.nnz());
    
    unsigned j=0;
    for(auto& e: src.contents()) {
      if(!src.is_hermitian() || e.covar_idx<=e.cntvar_idx) {  //only use lower part
        coo_covarids[j] = e.covar_idx;                       // of Hermitian matrix
        coo_cntvarids[j] = e.cntvar_idx;
        entries[j++] = e.entry;
      }
    }

    const_cuCmplxDoubleSparseMatEntries carrayptrs {
        coo_cntvarids.data()
      , src.covariant_dimension()
      , coo_covarids.data()
      , src.contravariant_dimension()
      , reinterpret_cast<cuDoubleComplex*>(entries.data())
      , j
    };

    cudahandle = new_cudaCmplxDoubleHilbertspcSparseM(
                        carrayptrs
                      , &acchandles->cusparsehandle );           CUDAerrcheckcdH

    if(src.is_hermitian()){
//      std::cout << "Created hermitian matrix with nnz " << src.nnz() << ", whereof " << j << " in lower triangle.\n";
      cusparseSetMatType( cudahandle.matrxdescript, CUSPARSE_MATRIX_TYPE_HERMITIAN );
      cusparseSetMatFillMode( cudahandle.matrxdescript, CUSPARSE_FILL_MODE_LOWER );
    }
    statenow=datastate::device;
  }
  void enter_device_state()const {
    if (statenow == datastate::host){
      cpy_hostspam_to_device(host_spam);
      host_spam.clear();
    }
  }

 public:
  typedef typename HostSpM::HostSparseMatrix HostSparseMatrix;

/*  auto operator()(const accelVector<NumT>& v)const -> accelVector<NumT>{
    return m(v);                                                       }  */
  void gemv( const NumT& alpha                   //BLAS gemv operation
           , const accelVector<NumT>& x         // (actually csrmv)
           , const NumT& beta                  //  y <- α(*this)x + β y
           , accelVector<NumT>& y        )const {
    enter_device_state();
    x.enter_device_state();
    y.enter_device_state();
    gemv_cudaCmplxDoubleHilbertspcSparseMToVect(
                  *reinterpret_cast<const cuDoubleComplex*>(&alpha)
                , *reinterpret_cast<const cuDoubleComplex*>(&beta)
                , &cudahandle
                , &x.cudahandle
                , &y.cudahandle                                     ); CUDAerrcheckcdH
  }

  auto invmv( const NumT& alpha                                          //inverse matrix multiplication
            , const accelVector<NumT>& x )const -> accelVector<NumT> {  // invmv(α,x) = (*this)⁻¹α⋅x,
    enter_device_state();                                              //  very expensive for matrices
    x.enter_device_state();                                           //   not triangular or unitary.
    accelVector<NumT> result(x.cudahandle, x.dimension());

    gemv_cudaCmplxDoubleHilbertspcSparseMToVect(
                  *reinterpret_cast<const cuDoubleComplex*>(&alpha)
                , *reinterpret_cast<const cuDoubleComplex*>(&alpha)
                , &cudahandle
                , &x.cudahandle
                , &result.cudahandle                                ); CUDAerrcheckcdH

    geninvapply_cudaCmplxDoubleHilbertspcTriangSparseMToVect(
                  *reinterpret_cast<const cuDoubleComplex*>(&alpha)
                , &cudahandle
                , &x.cudahandle
                , &result.cudahandle                                 );

    if(!cudahandle) {     //triangular direct solve failed, fallback to dense inverse
      statusreset(cudahandle);
      
      auto densified = copy_cudaCmplxDoubleHilbertspcSparseMToGenM
                                  (&cudahandle, &acchandles->cublashandle);  CUDAerrcheckcdH
      
      static bool triang_fail_warned = false;
      if(!triang_fail_warned && densified.matrix.matrixtype != cuCmplxDouble_UNITARY_MATRIX) {
        std::cerr << "Warning: perform inefficient single-vector non-triangular matrix inverse application.\n";
        triang_fail_warned=true;
      }
      
/*Check if dense matrix is ok:      gemv_cudaCmplxDoubleHilbertspcGenMToVect( make_cuDoubleComplex(.41,.75), make_cuDoubleComplex(.93,-.51)
                                              , &densified
                                              , &x.cudahandle
                                              , &result.cudahandle );
      if(!densified) {printout_error(densified); assert(!!densified);}
      if(!result.cudahandle) {printout_error(result.cudahandle); assert(!!result.cudahandle);}*/

      result = x;
      invapply_cudaCmplxDoubleHilbertspcGenMToVect(&densified, &result.cudahandle);
      if(!densified) {printout_error(densified); assert(!!densified);}
      if(alpha!=1.) result.axpyze(alpha-1.,result);
      delete_cudaCmplxDoubleHilbertspcGenM(&densified);
    }

    return result;
  }

  auto scaled_inverse(const NumT& alpha)const -> accelGeneralMatrix<NumT> {
    accelGeneralMatrix<NumT> result;
    result.cudahandle
      = inverted_cudaCmplxDoubleHilbertspcSparseM_asGenM( *reinterpret_cast<const cuDoubleComplex*>(&alpha)
                                                        , &cudahandle
                                                        , &acchandles->cublashandle );
    return result;
  }

  auto operator+=(const accelSparseMatrix& o) -> accelSparseMatrix& {
    enter_host_state(); o.enter_host_state();
    host_spam += o.host_spam;
    return *this;
  }
  auto operator+(const accelSparseMatrix& o)const -> accelSparseMatrix {
    return accelSparseMatrix(*this)+=o;                                }

  auto axpyze(const NumT& alpha, const accelSparseMatrix& o) -> accelSparseMatrix& {
    enter_host_state(); o.enter_host_state();
    host_spam.axpyze(alpha, o.host_spam);
    return *this;
  }

  auto operator+=(const NumT& alpha) -> accelSparseMatrix& {
    enter_host_state();
    host_spam += alpha;
    return *this;
  }

  auto operator*(const accelSparseMatrix& rmtl)const -> accelSparseMatrix {
    accelSparseMatrix result;
    result.acchandles = acchandles;
    enter_host_state();
    if(rmtl.statenow==datastate::host) {
      result.host_spam = host_spam*rmtl.host_spam;
     }else{
      HostSpM tmp;
      rmtl.cpy_devicespam_to_host(tmp);
      result.host_spam = host_spam*tmp;
    }
    result.statenow = datastate::host;
    return result;
  }

  void clear() {
    if(statenow==datastate::device) {
      delete_cudaCmplxDoubleHilbertspcSparseM( &cudahandle );   CUDAerrcheckcdH
     }else{
      host_spam.clear();
    }
    statenow = datastate::undefined;
  }

  void affirm_hermitian() {
    if(statenow==datastate::host)
      host_spam.affirm_hermitian();
     else
      cusparseSetMatType(cudahandle.matrxdescript, CUSPARSE_MATRIX_TYPE_HERMITIAN);
  }
  void affirm_unitary() {
    if(statenow==datastate::host)
      host_spam.affirm_hermitian();
  }

  accelSparseMatrix(): statenow(datastate::undefined) {}
  accelSparseMatrix(const accelHandle& h, unsigned covardim, unsigned cntvardim)
    : acchandles(&h)
    , statenow(datastate::host)
    , host_spam(covardim,cntvardim)
    , cudahandle(unassigned_cudaCmplxDoubleHilbertspcSparseMHandle(&h.cusparsehandle))
  {}
  accelSparseMatrix( const accelHandle& h
                   , unsigned covardim, unsigned cntvardim
                   , HostSparseMatrix m)
    : acchandles(&h)
    , statenow(datastate::host)
    , host_spam(covardim,cntvardim,std::move(m))
    , cudahandle(unassigned_cudaCmplxDoubleHilbertspcSparseMHandle(&h.cusparsehandle))
  {}
  accelSparseMatrix(const accelHandle& h, unsigned n, const NumT& init=NumT())
    : acchandles(&h)
    , statenow(datastate::host)
    , host_spam(n,init)
    , cudahandle(unassigned_cudaCmplxDoubleHilbertspcSparseMHandle(&h.cusparsehandle))
  {}

  accelSparseMatrix(const accelSparseMatrix& cpy)  
    : acchandles(cpy.acchandles)
    , statenow(cpy.statenow)                       
    , cudahandle(cpy.cudahandle)     {
    if(statenow==datastate::device){
      cudahandle = copy_cudaCmplxDoubleHilbertspcSparseM(&cudahandle); CUDAerrcheckcdH
     }else if(statenow==datastate::host){
      cpy_hostspam_to_device(cpy.host_spam);
      statenow=datastate::device;
    }
  }
  accelSparseMatrix(accelSparseMatrix&& mov) noexcept
    : acchandles(mov.acchandles)
    , statenow(mov.statenow)
    , host_spam(std::move(mov.host_spam))
    , cudahandle(mov.cudahandle)      {
    mov.statenow = datastate::host;    //prevents mov's destructor from deleting
  }                                   // the device vector handled by cudahandle.

  accelSparseMatrix& operator=(accelSparseMatrix cpy) {
    acchandles = cpy.acchandles;
    if(statenow==datastate::device) {
      delete_cudaCmplxDoubleHilbertspcSparseM( &cudahandle );   CUDAerrcheckcdH
    }
    statenow = cpy.statenow;
    host_spam = std::move(cpy.host_spam);
    cudahandle = cpy.cudahandle;
    cpy.statenow = datastate::host;
    return *this;
  }

  ~accelSparseMatrix() {
    if(statenow==datastate::device) {
      delete_cudaCmplxDoubleHilbertspcSparseM( &cudahandle );   CUDAerrcheckcdH
    }
  }

  friend class accelEigenbasisTransform<complex<double>>;
};



template<>
class accelEigenbasisTransform<complex<double>> {
  typedef complex<double> NumT;

  enum class datastate
    { device
//    , host
    , undefined };
  mutable datastate statenow;

  mutable cudaCmplxDoubleHilbertspcEigenbasisTransformHandle cudahandle;

 public:

/*  auto operator()(const accelVector<NumT>& v)const -> accelVector<NumT>{
    return m(v);                                                       }  */

  auto to_eigenbasis(const accelVector<NumT>& v)const -> accelVector<NumT>{
    accelVector<NumT> result(v.cudahandle);
    result.cudahandle = cudaCmplxDoubleHilbertspcVect_transformToEigenbasis
                              ( &cudahandle
                              , &v.cudahandle );    CUDAerrcheckcdO(result)
    return result;
  }
  auto from_eigenbasis(const accelVector<NumT>& v)const -> accelVector<NumT>{
    accelVector<NumT> result(v.cudahandle);
    result.cudahandle = cudaCmplxDoubleHilbertspcVect_transformFromEigenbasis
                              ( &cudahandle
                              , &v.cudahandle );    CUDAerrcheckcdO(result)
    return result;
  }
  
  auto real_eigenvalues()const -> accelVector<double> {
    accelVector<double> result;

    if(cudahandle.n_realeigenvals > 0) {
      result.cudahandle.cublashandle = cudahandle.cublashandle;
      result.cudahandle.cudastat = cudahandle.cudastat;
      result.cudahandle.cublasstat = cudahandle.cublasstat;
      result.statenow = accelVector<double>::datastate::device;
      result.cudahandle.vect_dimension = cudahandle.n_realeigenvals;
      result.cudahandle.vector = cudahandle.real_eigenvals;
      
      result.cudahandle = copy_cudaDoubleHilbertspcVect(&result.cudahandle);
     }else{
      result.statenow = accelVector<double>::datastate::host;
      result.cudahandle = unassigned_cudaDoubleHilbertspcVectHandle(cudahandle.cublashandle);
    }

    return result;
  }
  auto complex_eigenvalues()const -> accelVector<complex<double>> {

    accelVector<complex<double>> result;

    if(cudahandle.n_cplxeigenvals > 0) {
      result.cudahandle.cublashandle = cudahandle.cublashandle;
      result.cudahandle.cudastat = cudahandle.cudastat;
      result.cudahandle.cublasstat = cudahandle.cublasstat;
      result.statenow = accelVector<complex<double>>::datastate::device;
      result.cudahandle.vect_dimension = cudahandle.n_cplxeigenvals;
      result.cudahandle.vector = cudahandle.complex_eigenvals;
    
      result.cudahandle = copy_cudaCmplxDoubleHilbertspcVect(&result.cudahandle);
     }else{
      result.statenow = accelVector<complex<double>>::datastate::host;
      result.cudahandle = unassigned_cudaCmplxDoubleHilbertspcVectHandle(cudahandle.cublashandle);
    }
    return result;
  }

  void clear() {
    if(statenow==datastate::device) {
      delete_cudaCmplxDoubleHilbertspcEigenbasisTransform( &cudahandle );
    }
    statenow = datastate::undefined;
  }

  accelEigenbasisTransform(): statenow(datastate::undefined) {}
  
  explicit accelEigenbasisTransform(const accelGeneralMatrix<complex<double>>& opert)
    : statenow(datastate::device)                                             {
    cudahandle = eigenbasistransform_of_cudaCmplxDoubleHilbertspcGenM(
                                   &opert.cudahandle                 );  CUDAerrcheckcdH
  }
  explicit accelEigenbasisTransform(const accelSparseMatrix<complex<double>>& opert)
    : statenow(datastate::device)                                             {
    opert.enter_device_state();
    auto opdensfd = copy_cudaCmplxDoubleHilbertspcSparseMToGenM
                      (&opert.cudahandle, &opert.acchandles->cublashandle);  CUDAerrcheckcdA(opdensfd)

    cudahandle = eigenbasistransform_of_cudaCmplxDoubleHilbertspcGenM(
                                           &opdensfd                 );  CUDAerrcheckcdH

    delete_cudaCmplxDoubleHilbertspcGenM(&opdensfd);
  }
/*  accelGeneralMatrix(const accelHandle& h, unsigned covardim, unsigned cntvardim)
    : statenow(datastate::device)
    , cudahandle(unassigned_cudaCmplxDoubleHilbertspcGenMHandle(&h.cublashandle))
  {}*/
/*  accelGeneralMatrix( const accelHandle& h
                    , unsigned covardim, unsigned cntvardim )
    : statenow(datastate::device)
    , cudahandle(undefined_cudaCmplxDoubleHilbertspcGenM( cuCmplxDouble_GENERIC_MATRIX
                                                        , cuCmplxDouble_MATPACKING_NONE
                                                        , covardim
                                                        , cntvardim
                                                        , &h.cublashandle) )
  {}*/
/*  accelGeneralMatrix(const accelHandle& h, unsigned n, const NumT& init=NumT())
    : acchandles(&h)
    , statenow(datastate::host)
    , host_spam(n,init)
    , cudahandle(unassigned_cudaCmplxDoubleHilbertspcSparseMHandle(&h.cusparsehandle))
  {}*/

  accelEigenbasisTransform(const accelEigenbasisTransform& cpy)  =delete;
/*    : statenow(cpy.statenow)                       
    , cudahandle(cpy.cudahandle)     {
    if(statenow==datastate::device){
      cudahandle = copy_cudaCmplxDoubleHilbertspcGenM(&cudahandle); CUDAerrcheckcdH
    }
  }*/
  accelEigenbasisTransform(accelEigenbasisTransform&& mov) noexcept
    : statenow(mov.statenow)
    , cudahandle(mov.cudahandle)      {
    mov.statenow = datastate::undefined;
  }

  accelEigenbasisTransform& operator=(accelEigenbasisTransform cpy) {
    if(statenow==datastate::device) {
      delete_cudaCmplxDoubleHilbertspcEigenbasisTransform( &cudahandle );   CUDAerrcheckcdH
    }
    statenow = cpy.statenow;
    cudahandle = cpy.cudahandle;
    cpy.statenow = datastate::undefined;
    return *this;
  }

  ~accelEigenbasisTransform() {
    if(statenow==datastate::device) {
      delete_cudaCmplxDoubleHilbertspcEigenbasisTransform( &cudahandle );   CUDAerrcheckcdH
    }
  }

  friend class accelGeneralMatrix<complex<double>>;
};




}//namespace acceleratedVectors;

#endif//def USE_CUDA_TO_ACCELERATE_COMPLEXDOUBLE_VECTORS

namespace acceleratedVectors {

using std::complex;

template<typename UlNum>
class accelVector<complex<UlNum>> {

  typedef complex<UlNum> NumT;
 public:
  typedef std::vector<NumT> HostArray;
 private:
  HostArray v;
  
 public:
  auto allocated_dimension()const -> unsigned {return v.size();}

  auto operator[](unsigned i)      ->       NumT& { return v[i]; }  //avoid these
  auto operator[](unsigned i)const -> const NumT& { return v[i]; } // if possible
 /*(may require the whole vector to be copied into host memory and back to the accelerated device)*/
  
  auto axpyze(const NumT& alpha, const accelVector& x) -> accelVector&{
    for(unsigned i=v.size(); i-->0;) v[i] += alpha * x[i];
    return *this;
  }
  auto axpyzed(const NumT& alpha, const accelVector& x)const -> accelVector {
    return accelVector(*this).axpyze(alpha,x);                              }
  
  auto operator+=(const accelVector& y) -> accelVector& { return axpyze(1, y); }
  auto operator+(const accelVector& y)const -> accelVector {
    return accelVector(*this)+=y;                           }

  auto operator*(const accelVector& y)const -> NumT{
    NumT accum=0;
    for(unsigned i=v.size(); i-->0;) accum += std::conj(v[i]) * y[i];
    return accum;
  }

  void clear() { v.clear(); }
  void accelerate() {}

  auto phaserotations(UlNum alpha, const accelVector<UlNum>& thetas)const
                                                 -> accelVector {
    accelVector result(*this);
    for(unsigned i=0; i<v.size(); ++i)
      result[i] *= std::polar(1., alpha * thetas[i]);
    return result;
  }
  
  accelVector() {}
  accelVector(const accelHandle& h, HostArray v): v(v) {}
  accelVector(const accelHandle& h, unsigned n, const NumT& init=NumT())
    : v(HostArray(n, init)) {}
  
  auto released()const -> const HostArray& { return v; }
};



}//namespace acceleratedVectors;





#endif