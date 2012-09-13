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


#ifndef HILBERTSPACES_FOR_GENERAL_QUANTUM_MECHANICS
#define HILBERTSPACES_FOR_GENERAL_QUANTUM_MECHANICS

#include<memory>
#include<utility>
#include<typeinfo>
#include<vector>
#include<array>
#include<list>
#include<unordered_map>
#include<unordered_set>
#include<complex>
#include<cmath>
#include<cassert>
#include<functional>
#include<iostream>

#include "acceldvector.hpp"
//#include "basistrans.hpp"

#include "lambdalike/maybe.hpp"
#include "lambdalike/polyliteral.hpp"

namespace hilbert {

using std::complex;
using std::abs; using std::norm;

using namespace acceleratedVectors;


template<class FieldT>
class generalHilbertBasis;
template<class FieldT>
class generalHilbertBasisTransform;
template<class FieldT>                       //defined in basistrans.hpp
class sharedBasisTransform;                 //


template<class DomainVector, class CoDomainVector>class linearMap;     //defined in
template<class DomainVector, class CoDomainVector>class linearLambda; // linearmap.hpp
template<class DomainVect, class CoDomainVect>class linearZeroLambda;//
template<class DomainVect, class CoDomainVect>class linearMapSum;   //

template<class DomainVector, class CoDomainVector>auto
linear_lambda(const std::function< DomainVector(CoDomainVector) >& f)
   -> linearMap<DomainVector,CoDomainVector>;


  //Abstract CRTP interface class for objects implemented in terms of Vectors. They
 // should actually behave as such, i.e. when supporting addition and scalar operations
//  then these should commute with linear operations on the hilbertSpace::vector members.
template<class Hilbertspace, class VctGeneralization>
struct generalizedVector{
  typedef typename Hilbertspace::FieldT FieldT;
#if VIRTUAL_TEMPLATE_FUNCTIONS_were_allowed
  template<class CoCanonBOb> virtual auto
  linear_mapped(const linearMap< typename Hilbertspace::vector
                               , typename hilbertSpace< CoCanonBOb
                                                      , FieldT
                                                      >::vector
                               >& f)const
       -> VctGeneralization<decltype(f(typename Hilbertspace::vector()))>  =0;
  template<class PreCanonBOb> virtual auto
  linear_applyaccumulate( const linearMap< typename hilbertSpace< PreCanonBOb
                                                                , FieldT
                                                                >::vector
                                         , typename Hilbertspace::vector      >& f
                        , const FieldT& alpha
                        , const "generalizedVector<PreCanonBOb>"::vector& x
                        , const FieldT& beta                               ) =0;
#else
  //can't declare virtual templates here. Must be implemented in specialization!
#endif

  //direct access functions to the underlying vector object. Deprecated, as
 // generalized vectors should not need to hold exacltly one hilbertSpace::vector.
  virtual auto plain_hilbertspace_vector() -> typename Hilbertspace::vector&            =0;
  virtual auto plain_hilbertspace_vector()const -> const typename Hilbertspace::vector& =0;
};





struct hilbertSpaceBaseObject {}; //abstract fundamental base class for objects
                                 // represented by the basis vectors of a Hilbert space.

template<typename FieldT>
struct genericEigenbasisObj : hilbertSpaceBaseObject {
  FieldT eigenvalue;
  genericEigenbasisObj(const FieldT& init)
    : eigenvalue(init)  {}
};



template<class CanonicalBaseObj, class Field>class hilbertSpaceContext;


       //Objects ℋ of the hilbertSpace class are collections of some kind of
      // objects (e.g. trigonometric functions) representing an orthonormal
     //  basis. Together with a field (such as ℝ represented by double or ℂ
    //   by complex<double>) this defines an actual mathematical Hilbert space,
   //    the vectors wherein are linear combinations of the base objects. Since
  //     these are taken to be orthonormal, the scalar product in ℋ is then
 //      simply the euclidean/unitary scalar product in the ℝⁿ / ℂⁿ / ... spanned
//       by the coordinate tuples which describe the linear combinations.
      // The choice of one particular canonical basis, while more securely
     //  wrapped than in plain array-"vector"/matrix representations, is still
    //   mathematically not very appealing. Alternative baseis require extra
   //    hilbertSpace objects; it is aimed to treat these objects essentially
  //     as copies of one another. The benefit of this is that equal spaces
 //      are merely a special case (the intersection) of sub- and superspaces.
    //   The vector/linear-map operations tend to be optimized for high dimensions
   //    (e.g. 100s to 10000s, depending on whether there are non-sparse mappings
  //     involved), so e.g. ℝ³ is not very efficiently modelled by a hilbertSpace.
template< class CanonicalBaseObj
        , class Field = complex<double> >
class hilbertSpace : private generalHilbertBasis<Field> {
 public:
  typedef CanonicalBaseObj CanonBaseObjT;
  typedef std::vector<CanonBaseObjT> BasisObjects;
  typedef Field FieldT;
  friend class hilbertSpaceContext<CanonBaseObjT,Field>;
  typedef hilbertSpaceContext<CanonBaseObjT,Field> Context;
 private:
  BasisObjects canonical_orthonormal_base;
  Context context;
  accelHandle accelhandle;
  
  using generalHilbertBasis<Field>::subspaces;
  using generalHilbertBasis<Field>::superspaces;

 public:
  auto dimension()const -> unsigned           {
    return canonical_orthonormal_base.size(); }
  auto basiscontext()const -> const Context& {return context;}

  class vector; friend class vector;

  template<class CoHilbertBaseObj> class generalLinearMap;            //defined
  template<class CoHilbertBaseObj> friend class generalLinearMap;    // in
  template<class CoHilbertBaseObj> class sparseLinearMap;           //  linearmap.hpp
  template<class CoHilbertBaseObj> friend class sparseLinearMap;

  template<unsigned BlockSize>class staticBlocksdiagAccessor;template<unsigned BkSz>friend class staticBlocksdiagAccessor;
  

  class vector : public generalizedVector<hilbertSpace,vector> {
   public:
    typedef hilbertSpace Hilbertspace;           //<CanonBaseObjT,FieldT>
    typedef Field DomainFieldT;
   private:
    const Hilbertspace* domain;        //note that vectors are only
                                      // valid as long as their
    accelVector<FieldT> components;  //  Hilbert space is in scope
    
   public:

    auto operator+=(const vector&) -> vector&;          //vector addition
    auto operator+(vector v)const -> vector {
      return v+=(*this);                    }
    auto operator-=(const vector&) -> vector&;          //substraction
    auto operator-(const vector& v)const -> vector {
      return vector(*this)-=v;                     }
    auto axpyze(const FieldT&, const vector&) -> vector&;            //the BLAS axpy operation
    auto axpyzed(const FieldT& a, const vector& v)const -> vector { // *this += a*x
      return vector(*this).axpyze(a,v);                           }

    auto operator*=(const FieldT&) -> vector&;           //multiplication
    auto operator*(const FieldT& p)const -> vector {     // with a scalar
      return vector(*this)*=p;                     }

    auto operator*(const vector &)const -> FieldT;    //scalar-multiplication

    typedef decltype(abs(std::declval<FieldT>())) NormType;
              //as in std::complex, v.norm() ≡ |v|² = 〈v|v〉, instead of |v|.
    auto norm()const -> NormType;

    auto normalize() -> vector&;                //makes the norm equal to one
    auto normalized()const -> vector    {
      return vector(*this).normalize(); }

    template<class BaseobjOperator>             //apply an operation to every
    vector& baseobj_operation(BaseobjOperator);// component of the vector, WRT the
                                              //  canonical basis of the Hilbert space
    template<class BaseobjOperator>
    auto baseobj_developed(BaseobjOperator o)const -> vector {
      return vector(*this).baseobj_operation(o);             }

    auto undefined_linear_component()const -> CanonBaseObjT; //only for type deduction purposes

    auto domain_Hilbertspace()const -> const decltype(domain) {return domain;}

    auto canonical_base_decomp()const -> std::vector<std::pair<CanonBaseObjT,FieldT>>;

    template<class oCBO>
    auto to_basis(const hilbertSpace<oCBO,FieldT>&)const
           -> lambdalike::maybe<typename hilbertSpace<oCBO,FieldT>::vector>;       //defined in basistrans.hpp
    template<class oCBO>
    auto using_basis(const hilbertSpace<oCBO,FieldT>& tgb)const
           -> typename hilbertSpace<oCBO,FieldT>::vector {
      for(auto& r: to_basis(tgb)) return r;
      std::cerr << "Trying to use a basis to represent vector in non-isomorphic space." << std::endl;
      abort();
    }

    auto implementation_componentsarray_fmapped(const std::function<
                   accelVector<FieldT>(const accelVector<FieldT>&)  >&)const
                           -> vector;

/*    template<class LinMap>auto                         // x[f] ≡ 〈x|f|x〉
    operator[](const LinMap& f) -> FieldT               {
      auto x = using_basis(*f.domain()); return x*f(x); }*/
     
    vector(){}                            //uninitialized (the resulting vector will be invalid)
    vector(int zero)             //null vector
      : domain(nullptr) {
      assert(zero==0);
    }
    vector(const hilbertSpace* nd, int zero)         //null vector
      : domain(nd) {
      assert(zero==0);
      if(nd)
        components = accelVector<FieldT>( nd->accelhandle
                                        , std::vector<FieldT>(nd->dimension(), 0) );
    }

    template<class CntHilbertBaseObj,class OFlT>template<class CoHilbertBaseObj>
    friend class hilbertSpace<CntHilbertBaseObj,OFlT>::sparseLinearMap;
    friend class generalHilbertBasisTransform<FieldT>;
    template<unsigned BlockSize>
    friend class hilbertSpace<CanonBaseObjT,FieldT>::staticBlocksdiagAccessor;
    friend
 class hilbertSpace<CanonBaseObjT,FieldT>; };

 private:
  mutable std::unique_ptr<vector> zero_vector_memoized;
 public:
  auto zero_vector()const -> vector;    //create null vector

  auto uninitialized_vector()const -> vector;    //create valid, but undefined
                                                // vector in this Hilbert space
  template<class RandomGen>
  auto random_vector(RandomGen&)const -> vector;    //create randomized vector

  template<class RandomGen>
  auto random_endolinmap(RandomGen& diag_gen, RandomGen& nondiag_gen)const
       -> sparseLinearMap<CanonBaseObjT>;
                 // should perhaps yield a general linearMap, not sparse?
  template<class RandomGen>
  auto random_endolinmap_fromnewgen(RandomGen diag_gen, RandomGen nondiag_gen)const
       -> sparseLinearMap<CanonBaseObjT>              {
    return random_endolinmap(diag_gen, nondiag_gen);  }

  template<class RandomGen>
  auto random_orthonormal_basis(RandomGen& r)const -> std::vector<vector> {
    return random_orthonormal_system(r, dimension());                     }
  template< class VectorT=vector, class RandomGen=std::function<FieldT()> >
  auto random_orthonormal_system(RandomGen&, unsigned)const -> std::vector<VectorT>;

  auto canonical_orthonormal_basis()const -> std::vector<vector>;
  template<class GuardPred>
  auto canonical_orthonormal_subbasis(GuardPred)const -> std::vector<vector>;


  template<typename VectorSet, class Sumee>
  auto sum_over(const VectorSet& vs, Sumee f)const
     -> decltype(*vs.begin()*1.);  //Should actually be decltype(f(*vs.begin())), but g++-4.6 has problems with that
  template<typename VectorSet>
  auto sum_over(const VectorSet& s)const -> vector    {
    return sum_over(s, [](const vector& v) -> vector {return v;});  }
    

  template<class BtoCompAssoc>                                     //BtoCompAssoc :: CanonBaseObjT -> FieldT
  auto baseobj_superposition(const BtoCompAssoc&)const -> vector;


  template<class FnT, typename VctGeneralz=vector> auto
  linear_lambda(const FnT& f)const
     -> linearLambda< vector, decltype(f(std::declval<VctGeneralz>)) >  {
    return linearLambda<vector, decltype(f(std::declval<VctGeneralz>))>(f);
  }

  template<typename VctGeneralz=vector>
  auto zero_map()const -> linearZeroLambda<VctGeneralz,VctGeneralz>       {
    return linearZeroLambda<VctGeneralz,VctGeneralz>( [&](const VctGeneralz& _)      {
                                                        return VctGeneralz(this, 0); }
                                                    , 0
                                                    );
  }

  auto baseobj_diagonal(const std::function<FieldT(CanonBaseObjT)>&)const
        -> sparseLinearMap<CanonBaseObjT>;


   //Operator helper class for "nearly diagonal" operators, meaning operators which
  // act non-trivially only on independent subspaces spanned by BlockSize-sized
 //  subsets of the canonical basis given by the CanonicalBaseObjs.
  template<unsigned BlockSize>
  class staticBlocksdiagAccessor {
    const hilbertSpace<CanonBaseObjT,FieldT>* domain;

    typedef std::array<unsigned, BlockSize> BlockFinder;
    typedef std::array<FieldT*, BlockSize> BlockAccessor;
    typedef std::array<const CanonicalBaseObj*, BlockSize> BlockBasis;
    std::vector<BlockFinder> blocks;
#if 0
    auto basis_of_blockfd(const BlockFinder&)const -> BlockBasis;
    auto access_to_blockfd( const BlockFinder&
                          , hilbertSpace<CanonBaseObjT,FieldT>::vector&) const
                          -> BlockAccessor;
#endif
 //these need to be specialized, intended to work on functions with BlockSize as the number of arguments.
    template<class AmplitudeFn>
    static auto amplitudefn_on_accessor(AmplitudeFn f, const BlockAccessor &acc)
                                    -> decltype(f(*acc[0],*acc[1]))             {
      return f(*acc[0],*acc[1]);                                                }
    template<class BaseObjFn>
    static auto baseobjfn_on_blckbasis(BaseObjFn f, const BlockBasis &b)
                                    -> decltype(f(*b[0],*b[1]))          {
      return f(*b[0],*b[1]);                                             }

  //Search for tuples of basis objects satisfying as block-diagonal matrix generators
    template<class BlockOkPredicate>void             //implemented as a simple exhaustive search for tuples.
    construct_from_predicate(BlockOkPredicate);     // fulfilling a predicate. May be very slow for large spaces.
    template<class BlockAssociator,class HashFn>void            //implemented as a direct one-to-one association;
    construct_from_direct_association2(BlockAssociator); // requires a function CanonBaseObjT->maybe<CanonBaseObjT>
                                                        //  and a hash on the CanonBaseObjs.
   public:
    template<class BlockOkPredicate>      //The Hilbertspace dimension should be an integer multiple of BlockSize.
    staticBlocksdiagAccessor(const hilbertSpace& H, const BlockOkPredicate& a)
      : domain(&H) {
      construct_from_predicate(a);
    }
    template<class BlockAssociator,class HashFn>
    staticBlocksdiagAccessor(const hilbertSpace& H, const BlockAssociator& a, HashFn hfexample)
      : domain(&H) {
      construct_from_direct_association2<BlockAssociator,HashFn>(a);
    }
    
    template<class BlockOperator, typename HSpaceVectorT>
    void apply(BlockOperator, generalizedVector<hilbertSpace,HSpaceVectorT>&)const;
    
    template<class BlockMatrix>
    auto blockmatrix_impl(const BlockMatrix&)const
             -> hilbertSpace::sparseLinearMap<CanonBaseObjT>;
    template<class BlockMatrix>
    auto blockmatrix(const BlockMatrix& m)const -> hilbertSpace::sparseLinearMap<CanonBaseObjT>{
      return blockmatrix_impl(m);                                                              }
    auto blockmatrix(std::initializer_list<std::initializer_list<FieldT>> l)const
             -> hilbertSpace::sparseLinearMap<CanonBaseObjT>{
      return blockmatrix_impl(l);                           }
  };
  
  template<unsigned BlockSize, class BlockAssociator>
  auto static_blocksdiag_accessor(BlockAssociator b)
                                ->staticBlocksdiagAccessor<BlockSize> {
    return staticBlocksdiagAccessor<BlockSize>(*this, b);             }

  template<class BlockAssociator,class HashFn = std::hash<CanonicalBaseObj>>
  auto direct_stat_blocksdiag_accessor(BlockAssociator b)->staticBlocksdiagAccessor<2> {
    return staticBlocksdiagAccessor<2>(*this, b, HashFn());
  }

  template<class CBOt, class EigenvalueHandler>auto
  eigenbasis_gen( const linearMap<vector,vector>&
            , EigenvalueHandler               )const    // = [](const accelEigenbasisTransform& trf){return trf.real_eigenvalues();}
        -> hilbertSpace<CBOt, FieldT>; //defined in linearmap.hpp

  auto as_vector(const CanonBaseObjT& obj)const -> vector;
/*  struct defaultEigenvalueHandler {
    auto operator() -> std::vector<FieldT> {}
  };

  template<class GeneralizedLinMap>
  auto eigenbasis(const linearMap<vector,vector>& op)
         -> hilbertSpace<genericEigenbasisObj<declval()>,FieldT>*/



  explicit hilbertSpace(unsigned ndim)        //build up Hilbert space of desired
    : canonical_orthonormal_base(ndim)       // dimension from default constructors,
    , context( canonical_orthonormal_base   //  the resulting objects taken
             , accelhandle                )//   as orthonormal
  {}
  explicit hilbertSpace(int ndim) : canonical_orthonormal_base(ndim)
                                  , context(canonical_orthonormal_base, accelhandle) {}

  template<class BObjCntInitT>                                 //build from a container
  explicit hilbertSpace(const BObjCntInitT& init)             // of objects convertible
    : canonical_orthonormal_base(init.begin(), init.end())   //  to CanonicalBaseObjs
    , context(canonical_orthonormal_base, accelhandle)
  {}
  
  hilbertSpace(const hilbertSpace& cpy) =delete;
  hilbertSpace(hilbertSpace&& mov) noexcept
    : generalHilbertBasis<Field>(std::forward<generalHilbertBasis<Field>>(mov))
    , canonical_orthonormal_base(std::move(mov.canonical_orthonormal_base))
    , context(std::move(mov.context))
    , accelhandle(std::move(mov.accelhandle))
  {}
  
  template<class OCBO, class OFieldT>
  friend class hilbertSpace;
  friend class generalHilbertBasisTransform<Field>;
};



template<class Hilbertspace>
struct generalizedVector<Hilbertspace,typename Hilbertspace::vector>{
  typedef typename Hilbertspace::vector Vector;
  typedef typename Hilbertspace::FieldT FieldT;
  auto plain_hilbertspace_vector() -> Vector& {
    return static_cast<Vector&>(*this);       }
  auto plain_hilbertspace_vector()const -> const Vector& {
    return static_cast<const Vector&>(*this);            }
  template<class CoCanonBOb>auto
  linear_mapped(const linearMap< Vector
                               , typename hilbertSpace< CoCanonBOb
                                                      , FieldT
                                                      >::vector
                               >& f)const
        -> decltype(f(Vector()))                                             {
    return f(static_cast<const Vector&>(*this));                            }
  template<class PreCanonBOb> void
  linear_applyaccumulate( const linearMap< typename hilbertSpace< PreCanonBOb
                                                                , FieldT
                                                                >::vector
                                         , Vector                             >& f
                        , const FieldT& alpha
                        , const typename hilbertSpace<PreCanonBOb,FieldT>::vector& x
                        , const FieldT& beta                                  ) {
    f.accumulate_apply(alpha, x, beta, static_cast<Vector&>(*this));
  }
};


}
#include "basistrans.hpp"
namespace hilbert{



   //Accelerable variable contexts of Hilbert space basis objects (e.g. the
  // energies of a quantum mechanics energy basis as a GPU vector). To give
 //  flexibility in declaring such baseis' objects as derived instances of base
//   classes, a somewhat roundabout template type resolving mechanism is employed.
  // Might be doable in a cleaner fashion with C++11 type traits.

struct non_Context {
  struct nothing{};
  template<class CBObj, class FldT> auto
  basis_context(const std::vector<CBObj>& o, const FldT& v, const accelHandle& h)
     -> nothing {return nothing();}
};


auto eigenenergy_context             //For actual eigenenergy basis objects, overload
  (const hilbertSpaceBaseObject* _) // this function with a version that takes a
   -> non_Context       {          //  more derived base class (null)pointer and
  return non_Context(); }         //   returns a nontrivial basis-context factory.

template<class CanonicalBaseObj, class Field>
class hilbertSpaceContext {
  typedef std::vector<CanonicalBaseObj> BasisObjects;
  typedef const CanonicalBaseObj* CBOcptr;
  
  typedef decltype( eigenenergy_context(std::declval<CBOcptr>())
                       .basis_context( std::declval<BasisObjects>()
                                     , Field()
                                     , std::declval<accelHandle>()  ) ) EnergyCtxt;
  EnergyCtxt energy_ctxt;
 public:
  hilbertSpaceContext(const BasisObjects& objs, const accelHandle& h)
    : energy_ctxt(eigenenergy_context((CBOcptr)nullptr)
           .basis_context(objs, Field(), h)) {}

  auto eigenenergies()const -> const EnergyCtxt& { return energy_ctxt; }
};





template<class CBO,class FldT>
auto abs(const typename hilbertSpace<CBO,FldT>::vector& v) -> decltype(abs(FldT())) {
  return sqrt(v.norm()); }



template<class CBO,class FldT>auto
hilbertSpace < CBO,      FldT    > ::
vector::axpyze(const FldT& alpha, const vector& other) -> vector& {
  if(!domain) return *this = other;
  assert(domain == other.domain);
  components.axpyze(alpha, other.components);
  return *this;
}

using lambdalike::one;

template<class CBO,class FldT>auto
hilbertSpace < CBO,      FldT    > ::
vector::operator+=(const vector& other) -> vector&   {
  if(!domain) return *this = other;   // (!domain) only true for zero vector
  if(!other.domain) return *this;
  assert(domain == other.domain);
  components.axpyze(one, other.components);
  return *this;
}

using lambdalike::minusone;
using lambdalike::minustwo;

template<class CBO,class FldT>auto
hilbertSpace < CBO,      FldT    > ::
vector::operator-=(const vector& other) -> vector&   {
  if(!other.domain) return *this;
  if(!domain) {
    *this = other.axpyzed(minustwo,other);    //inefficient, but affordable as
                                             // this case is expected to be seldom
   }else{
    assert(domain == other.domain);
    components.axpyze(minusone, other.components);
  }
  return *this;
}



template<class CBO,class FldT>auto
hilbertSpace < CBO,      FldT    > ::
vector::operator*=(const FldT& a) -> vector& {
  components.axpyze(a-1., components);    //inefficient; generalizedVectors
  return *this;                          // should use an external-multiplier
}                                       //  memoization approach instead.


#ifdef USE_NO_ACCELLERATED_SCALAR_PRODUCT
  //Specialized multiplication for complex numbers, as these require
 // conjugation in the scalar product.
template<class CmvT> auto
posdefinit_multiply(complex<CmvT>a, complex<CmvT>b) -> complex<CmvT> {
  return conj(a) * b;                                                }
template<class FldT> auto
posdefinit_multiply(FldT a, FldT b) -> FldT  { return a * b; }
#endif//def USE_NO_ACCELLERATED_SCALAR_PRODUCT

template<class CBO,class FldT>auto hilbertSpace
          <    CBO,      FldT    > ::
vector::operator*(const vector& other)const -> FldT   {
  assert(domain && domain == other.domain);
#ifdef USE_NO_ACCELLERATED_SCALAR_PRODUCT
  FldT result = 0;
  for(unsigned i=0; i < domain->dimension(); ++i)
    result += posdefinit_multiply(components[i], other.components[i]);
  return result;
#else
  return components * other.components;
#endif//def USE_NO_ACCELLERATED_SCALAR_PRODUCT
}

template<class CBO,class FldT>auto
hilbertSpace < CBO,      FldT    > ::                  //norm ≡ ||²
vector::norm()const -> NormType               {
#ifdef USE_NO_ACCELLERATED_SCALAR_PRODUCT
  decltype(abs(FldT())) result = 0;
  for(unsigned i=0; i < domain->dimension(); ++i)
    result += norm(components[i]);
  return result;
#else
  return abs(components * components);
#endif//def USE_NO_ACCELLERATED_SCALAR_PRODUCT
}
template<class CBO,class FldT>auto
hilbertSpace < CBO,      FldT    > ::
vector::normalize() -> vector&               {
  return *this *= (1/sqrt(norm()));          }



template<class CBO,class FldT> template<class BaseobjOperator>auto
hilbertSpace < CBO,      FldT    >                                 ::
vector::baseobj_operation(BaseobjOperator operate) -> vector&             {
  for (unsigned i=0; i < domain->dimension(); ++i) {
    operate( domain->canonical_orthonormal_base[i]
           , components[i] );
  }
  return *this;
}







using lambdalike::zero;

template<class CBO,class FldT>auto
hilbertSpace < CBO,      FldT    > ::
zero_vector()const -> vector {
  if (!zero_vector_memoized) {
    zero_vector_memoized.reset(new vector(0));
    zero_vector_memoized->domain = this;
    zero_vector_memoized->components = accelVector<FldT>( accelhandle
                                                        , dimension()
                                                        , zero        );
    zero_vector_memoized->components.accelerate();
  }
  return *zero_vector_memoized;
}

template<class CBO,class FldT>auto
hilbertSpace < CBO,      FldT    > ::
uninitialized_vector()const -> vector {
  vector result(0);
  result.domain = this;
  result.components = accelVector<FldT>( accelhandle
                                       , dimension() );
  return result;
}

template<class CBO,class FldT>auto
hilbertSpace < CBO,      FldT    > ::
as_vector(const CBO& obj)const -> vector {
  vector v = zero_vector();
  for (auto i=dimension(); i-->0;)
    v.components[i]  =  obj==canonical_orthonormal_base[i]
                         ? 1   
                         : 0;
  return v;
}

template<class CBO,class FldT> template<class RandomGen> auto
hilbertSpace < CBO,      FldT    >                            ::
random_vector(RandomGen& rnd_amplitude)const -> vector            {
  vector v = zero_vector();
  for (auto i=dimension(); i-->0;) v.components[i] = rnd_amplitude();
  return v;
}

template<class CBO,class FldT>template<class VectorT, class RandomGen> auto
hilbertSpace < CBO,      FldT    >                            ::
random_orthonormal_system(RandomGen& r, unsigned n)const -> std::vector<VectorT> {
  assert(n <= dimension());
  std::vector<VectorT> new_basis;
  for ( VectorT v; new_basis.size() < n; new_basis.push_back(v) ) {
    decltype(v.norm()) normn;
    do{
      v = random_vector(r);
      for (auto& w : new_basis)
        v -= w * (w*v);
      normn = v.norm();
    }while (abs(normn)<1e-20);   //prevent division-by-zero error
    v *= 1./sqrt(normn);
  }
  return new_basis;
}

template<class CBO,class FldT> template<class GuardPred> auto
hilbertSpace < CBO,      FldT    >                            ::
canonical_orthonormal_subbasis(GuardPred g)const -> std::vector<vector> {
  std::vector<vector> new_basis;
  for ( unsigned i=0; i<dimension(); ++i )
    if (g(canonical_orthonormal_base[i]))
      new_basis.push_back( [&](){ vector v=zero_vector();
                                  v.components[i] = 1;
                                  return v;               }() );
  return new_basis;
}



template<class CBO,class FldT> auto
hilbertSpace < CBO,      FldT    >  ::
canonical_orthonormal_basis()const -> std::vector<vector> {
  std::vector<vector> new_basis(dimension(), zero_vector());
  for ( unsigned i=0; i<dimension(); ++i )
    new_basis[i].components[i] = 1;
  return new_basis;
}


template<class CBO,class FldT> auto
hilbertSpace < CBO,      FldT    >  ::
vector::canonical_base_decomp()const -> std::vector<std::pair<CBO,FldT>> {
  std::vector<std::pair<CBO,FldT>> result;
  auto cmps = components.released();
  for(unsigned i=0; i < domain->dimension(); ++i)
    result.push_back(std::make_pair(
        domain->canonical_orthonormal_base[i]
      , cmps[i]
      ));
  return result;
}


template<class CBO,class FldT> auto
hilbertSpace < CBO,      FldT    >  ::vector::
implementation_componentsarray_fmapped
      (const std::function< accelVector<FldT>(const accelVector<FldT>&) >& f) const
          -> vector                                                {
  assert(domain);
  vector result;
  result.domain = domain;
  result.components = f(components);
  return result;
}



  
template<class CBO,class FldT> template<class StateS, class Sumee> auto
hilbertSpace < CBO,      FldT    >                                    ::
sum_over(const StateS& states, Sumee f)const -> decltype(*states.begin()*1.) {
  decltype(*states.begin()*1.) result(zero_vector());
  for (auto& s: states) result += f(s);
  return result;
}


template<class CBO,class FldT> template<class BtoCompAssoc> auto
hilbertSpace < CBO,      FldT    >                                    ::
baseobj_superposition(const BtoCompAssoc& f)const -> vector {
  auto result = zero_vector();
  for(unsigned i=0; i<dimension(); ++i)
    result.components[i] = f(canonical_orthonormal_base[i]);
  return result;
}

template<class CBO,class FldT> auto
hilbertSpace < CBO,      FldT   >::
baseobj_diagonal(const std::function<FldT(CBO)>&asc)const -> sparseLinearMap<CBO> {
  typename accelSparseMatrix<FldT>::HostSparseMatrix accum;
  bool is_hermitian = true;
  for(unsigned i=0; i<dimension(); ++i) {
    auto elem = asc(canonical_orthonormal_base[i]);
    if(elem != conj(elem)) is_hermitian = false;
    if(abs(elem) > 0)
      accum.insert( cooSparseMatrixEntry<FldT>{ i, i, elem } );
  }
  accelSparseMatrix<FldT> result( accelhandle 
                                , dimension(), dimension()
                                , std::move(accum)         );
  if(is_hermitian) result.affirm_hermitian();
  return sparseLinearMap<CBO>( *this, *this, result );
}



template<class CBO, class FldT>template<class RandomGen>auto
hilbertSpace < CBO ,      FldT>::
random_endolinmap(RandomGen& diag_gen, RandomGen& nondiag_gen)const
       -> sparseLinearMap<CanonBaseObjT> {
  typename accelSparseMatrix<FldT>::HostSparseMatrix accum;
  for(unsigned i=0; i<dimension(); ++i) {
    for(unsigned j=0; j<dimension(); ++j) {
      auto elem = (j==i)
             ? diag_gen()
             : nondiag_gen();
      if(abs(elem) > 0)
        accum.insert( cooSparseMatrixEntry<FldT>{ i, j, elem } );
    }
  }
  accelSparseMatrix<FldT> result( accelhandle 
                                , dimension(), dimension()
                                , std::move(accum)         );
  return sparseLinearMap<CBO>( *this, *this, result );
}






#if 0
template<class CBO,class FldT>template<unsigned BlckSz>auto
hilbertSpace < CBO,      FldT > ::staticBlocksdiagAccessor<BlckSz>::
basis_of_blockfd(const BlockFinder& b)const -> BlockBasis                {
  BlockBasis result;
  for(unsigned j=0; j<BlckSz; ++j)
    result[j] = &(domain->canonical_orthonormal_base[b[j]]);
  return result;
}
template<class CBO,class FldT>template<unsigned BlckSz>auto
hilbertSpace < CBO,      FldT > ::staticBlocksdiagAccessor<BlckSz>::
access_to_blockfd(const BlockFinder& b, hilbertSpace<CBO,FldT>::vector& v) const
                -> BlockAccessor                                           {
  BlockAccessor result;
  for(unsigned j=0; j<BlckSz; ++j)
    result[j] = &(v.components[b[j]]);
  return result;
}

template<class CBO,class FldT> template<> template<class Fn>auto
hilbertSpace < CBO,      FldT > ::staticBlocksdiagAccessor<1>::
static auto amplitudefn_on_accessor(Fn f, const BlockAccessor &ba)
                                    -> decltype(f(*ba[0]))      {
  return f(*ba[0]);
}
template<class CBO,class FldT> template<> template<class Fn>auto
hilbertSpace < CBO,      FldT > ::staticBlocksdiagAccessor<1>::
static auto baseobjfn_on_blckbasis(Fn f, const BlockBasis &ba)
                                    -> decltype(f(*ba[0]))      {
  return f(*ba[0]);
}
#endif





template<class CBO,class FldT>template<unsigned BlckSz>template<class BlckAssoc>void
hilbertSpace < CBO,      FldT > ::staticBlocksdiagAccessor<BlckSz>::
construct_from_predicate( BlckAssoc is_block )                             {
  auto unassigned = [&](){ std::list<unsigned>a;
                           for(auto i=domain->dimension(); i-->0;) a.push_front(i);
                           return a;
                         }();

  typedef std::array<std::list<unsigned>::iterator, BlckSz> LitArr;

  auto blockcandidate_ok = [&](const LitArr& cnd) -> bool {
    for(unsigned i = 0; i<BlckSz; ++i) {
      for(unsigned j = 0; j<i; ++j)
        if(cnd[i]==cnd[j]) return false;
    }
    return true;
  };

  auto candidate_stepfwd = [&](LitArr& cnd) -> bool {
    for(unsigned i=BlckSz; i-->0;)
      if(++cnd[i]!=unassigned.end()) {
        for(unsigned j=i+1; j<BlckSz; ++j)
          cnd[j] = unassigned.begin();
        return true;
      }
    return false;
  };

  auto reset_candidate = [&]() -> LitArr {
    LitArr f;
    for(auto&j: f) j=unassigned.begin();
    return f;
  };
  
  auto basis_from_candidate = [&](const LitArr& cnd){
    BlockBasis b;
    for(auto j=BlckSz; j-->0;)
      b[j] = &domain->canonical_orthonormal_base[*cnd[j]];
    return b;
  };

  auto finder_from_candidate = [&](const LitArr& cnd){
    BlockFinder b;
    for(auto j=BlckSz; j-->0;) b[j] = *cnd[j];
    return b;
  };

  for ( LitArr searcher=reset_candidate()
      ; blocks.size() < domain->dimension()/BlckSz && candidate_stepfwd(searcher); ){
    if ( blockcandidate_ok(searcher)
        && baseobjfn_on_blckbasis(is_block, basis_from_candidate(searcher)) ){
      blocks.push_back(finder_from_candidate(searcher));
      for (auto i: searcher) unassigned.erase(i);
      searcher = reset_candidate();
    }
  }
}



template<class CBO,class FldT>template<unsigned BlckSz>template<class BlckAssoc,class HashFn>void
hilbertSpace < CBO,      FldT > ::staticBlocksdiagAccessor<BlckSz>::
construct_from_direct_association2( BlckAssoc associator )                    {

  static_assert(BlckSz==2, "Creating a block-diagonal operator with block size other than 2x2 using a direct-association function is not yet supported.");

  auto bare_baseelems_ids = [&](){ std::unordered_map<CBO,int,HashFn>m;
                                   for(auto i=domain->dimension(); i-->0;)
                                     m[domain->canonical_orthonormal_base[i]] = i;
                                   return m;                             }();

  for (auto o: bare_baseelems_ids) {
    if (o.second>=0) {
//      std::cout << o.first.i << " <-> ";
      for (auto associated : associator(o.first)) {
//        std::cout << associated.i;
        auto assocd_loc = bare_baseelems_ids.find(associated);
        assert(assocd_loc != bare_baseelems_ids.end());
        if(assocd_loc->second >= 0)
          blocks.push_back( BlockFinder{{ o.second
                                        , assocd_loc->second }} );
//         else std::cout << "   (already associated)";
        assocd_loc->second = -1;
      }
//      std::cout << std::endl;
    }
  }
}





template<class CBO,class FldT>template<unsigned BlckSz>
template<class BlkOp,class HSpcVct>void
hilbertSpace < CBO,      FldT > ::staticBlocksdiagAccessor<BlckSz>::
apply(BlkOp O, generalizedVector<hilbertSpace<CBO,FldT>,HSpcVct>& vect)const {
  hilbertSpace<CBO,FldT>::vector& v = vect.plain_hilbertspace_vector();
  for (auto&b: blocks)
    amplitudefn_on_accessor(O, [&](){ BlockAccessor r;
                                      for(unsigned j=0; j<BlckSz; ++j)
                                        r[j] = &(v.components[b[j]]);
                                      return r;
                                    }()                                );
}

template<class CBO,class FldT>template<unsigned BlckSz>template<class BlckMat>auto
hilbertSpace < CBO,      FldT > ::staticBlocksdiagAccessor<BlckSz>::
blockmatrix_impl(const BlckMat& blcm)const
             -> hilbertSpace<CBO,FldT>::sparseLinearMap<CBO>                       {
  assert(blcm.size()==BlckSz || !"Nearly-diagonal-block-matrix initialized with wrong block size.");
  for(auto row: blcm) assert(row.size() == BlckSz);
  
    //copy using only begin() and end() iterators, to allow call with initializer_lists.
  auto blockmat = [&](){
          std::array<std::array<FldT, BlckSz>, BlckSz> blmes;
          unsigned i=0, j=0;
          for(auto row: blcm) {
            for(auto elem: row) {
              blmes[i][j] = elem; j++;
            } i++; j=0;
          }                                         return blmes;     }();

  typename accelSparseMatrix<FldT>::HostSparseMatrix accum;
  for (auto&b: blocks)
    for(unsigned i=0; i<BlckSz; ++i)
      for(unsigned j=0; j<BlckSz; ++j)
        if(abs(blockmat[i][j]) > 0)
          accum.insert( cooSparseMatrixEntry<FldT>{ b[i], b[j], blockmat[i][j] } );

  accelSparseMatrix<FldT> result( domain->accelhandle 
                                , domain->dimension(), domain->dimension()
                                , std::move(accum)                         );
  
  for(unsigned i=0; i<BlckSz; ++i)
    for(unsigned j=0; j<=i; ++j)
      if(blockmat[i][j] != conj(blockmat[j][i]))
        goto skip_hermitian;
  result.affirm_hermitian();
  
  skip_hermitian:

  return sparseLinearMap<CBO>( *domain, *domain, result );
}


}

#endif