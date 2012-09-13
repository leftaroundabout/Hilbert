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


#ifndef QUANTUM_MECHANICAL_HILBERTSPACE_STATES
#define QUANTUM_MECHANICAL_HILBERTSPACE_STATES

#include <complex>

#include "hilbertsp.hpp"
#include "linearmap.hpp"
#include "cqtx/cqtx.h"

namespace hilbert {

using std::complex;
using std::polar;
using cqtx::physquantity;
using cqtx::abs;

template<typename CalcT>class eigenEnergyState;

    //quantumState is mostly a wrapper around a Hilbertspace vector with an
   // attached physical dimension; in general this is NOT an actual quantum
  //  state (which in the normal sense would be a normalized vector) but the
 //   result of applying a possibly non-unitary operator to such a state, e.g.
//    an observable such as momentum: p|ψ〉 would still be a quantumState object.
template <class HilbertBaseObjT, typename PhysT = physquantity, typename CalcT=double>
class quantumState : public generalizedVector< hilbertSpace< HilbertBaseObjT
                                                           , complex<CalcT> >
                                             , quantumState< HilbertBaseObjT
                                                           , PhysT
                                                           , CalcT> >
                   , lhsMultipliable< quantumState< HilbertBaseObjT
                                                         , PhysT
                                                         , CalcT> >           {
  typedef complex<PhysT> complexPhys;
  typedef complex<CalcT> complexCalc;
 public:
  typedef hilbertSpace<HilbertBaseObjT, complexCalc> Hilbertspace;
  typedef complexPhys DomainFieldT;
  typedef typename Hilbertspace::vector StateVector;
  typedef quantumState<HilbertBaseObjT, PhysT, CalcT> State;
 private:
  StateVector statevector;

  PhysT physical_dimension;

  friend class generalizedVector<Hilbertspace, quantumState>;
  
 public:

  auto operator+=(const State&) -> State&;          //vector addition
  auto operator+(State v)const -> State {
    return v+=State(*this);             }
  auto operator-=(const State&) -> State&;          //substraction
  auto operator-(const State& v)const -> State {
    return State(*this)-=v;                    }

  auto operator*=(const PhysT&) -> State&;           //multiplication
  auto operator*=(const DomainFieldT&) -> State&;    // with a scalar
  auto operator*=(const complexCalc&) -> State&;
  auto operator*=(const CalcT&) -> State&;
  template<typename Multiplicator>
  auto operator*(const Multiplicator& p)const -> State {
    return State(*this)*=p;                            }
  template<typename Multiplicator>
  auto lefthand_multiplied(const Multiplicator& p) -> State& {
    return State(*this)*=p;                                  }

  auto operator*(const State &)const -> complex<PhysT>;    //scalar-multiplication

            //as in std::complex, v.norm() ≡ |v|² = 〈v|v〉, instead of |v|.
  auto norm()const -> decltype(abs(PhysT()));

  auto normalize() -> State&;                //makes the norm equal to one
  auto normalized()const -> State    {
    return State(*this).normalize(); }

  template<class BaseobjOperator>            //apply an operation to every
  State& baseobj_operation(BaseobjOperator);// component of the state vector, WRT the
                                           //  canonical basis of the Hilbert space
  template<class BaseobjOperator>
  auto baseobj_developed(BaseobjOperator o)const -> State {
    return State(*this).baseobj_operation(o);             }

  auto canonical_base_decomp()const -> std::vector<std::pair<HilbertBaseObjT,complexPhys>>;

  auto domain_Hilbertspace()const -> const decltype(statevector.domain_Hilbertspace()) {
    return statevector.domain_Hilbertspace();                                          }

  template<class oCBO>
  auto to_basis(const hilbertSpace<oCBO,complexCalc>& tgb)const
         -> lambdalike::maybe<quantumState<oCBO,PhysT,CalcT>> {
    for(auto& stres : statevector.to_basis(tgb))
      return lambdalike::just(quantumState<oCBO,PhysT,CalcT>( std::move(stres)
                                                            , physical_dimension) );
    return lambdalike::nothing;
  }
  template<class oCBO>
  auto using_basis(const hilbertSpace<oCBO,complexCalc>& tgb)const
         -> quantumState<oCBO,PhysT,CalcT> {
    for(auto& r: to_basis(tgb)) return r;
    std::cerr << "Trying to use a basis to represent vector in non-isomorphic space." << std::endl;
    abort();
  }

  template<class Operator>auto                       //For observables A,
  operator[](const Operator& A)const -> PhysT {     // the expectation value,
    auto btransformed = using_basis(*A.domain());  //  ψ[A] ≡ 〈ψ|A|ψ〉.
    complexPhys sclProd = btransformed * A(btransformed);
    if(abs(std::imag(sclProd)) >= abs(std::real(sclProd)) * 1e-8) {
      std::cerr << "Non-real expectation value ("<<std::real(sclProd)<<" + " << std::imag(sclProd) << "*i)"<<std::endl
                 << "    (i.e. the investigated operator is not hermitian)." << std::endl;
      assert(abs(std::imag(sclProd)) < abs(std::real(sclProd)) * 1e-6);
      std::cerr << "  Taking this as a floating-point uncertainty, return real part only. Resume.\n";
    }
    return std::real(sclProd);
  }

  quantumState(){}                            //uninitialized (the resulting vector will be invalid)
  quantumState(int zero)            //null vector
    : statevector(0)
    , physical_dimension(1) {}
  quantumState(const Hilbertspace* nd, int zero)       //null vector that is actually
    : statevector(nd? nd->uninitialized_vector() : 0) // usable in a specific space
    , physical_dimension(0) {}
  quantumState(const StateVector& vct, const PhysT& physdim=1)
    : statevector(vct)
    , physical_dimension(physdim) {}


  template<class HBO,typename ClcT>
  friend auto time_evolution( const quantumState<HBO,physquantity,ClcT>& st
                            , const physquantity& t                )
                             -> quantumState<HBO,physquantity,ClcT>;
  template<class PureHilbertVector, class PhT, class ClcT>
  friend struct quantumStateWrapper;
};

template<class HilbertBaseObjT, typename PhysT, typename CalcT>
struct generalizedVector< hilbertSpace<HilbertBaseObjT,complex<CalcT>>
                        , quantumState<HilbertBaseObjT,PhysT,CalcT>     > {
 public:
  typedef complex<CalcT> UnderlyingField;
  typedef complex<PhysT> FieldT;
  typedef hilbertSpace<HilbertBaseObjT,UnderlyingField> Hilbertspace;
  typedef typename Hilbertspace::vector HilbertVector;
  typedef quantumState<HilbertBaseObjT,PhysT,CalcT> Vector;

 private:
  auto polymorph() -> Vector&           {
    return static_cast<Vector&>(*this); }
  auto polymorph()const -> const Vector&      {
    return static_cast<const Vector&>(*this); }
/*  
  template<class OtherHBO>
  auto hilbertvect_embed(const typename hilbertSpace<OtherHBO,UnderlyingField>::vector& v)
      -> quantumState<OtherHBO,PhysT,CalcT> {
    return quantumState<OtherHBO,PhysT,CalcT>(v, polymorph().physical_dimension);
  }

  template<class OtherHBO>
  auto hilbertvect_embed(const VctT& v) -> quantumState<VctT,PhysT,CalcT> {
    return quantumState<VctT,PhysT,CalcT>(v, polymorph().physical_dimension);
  }
*/
 public:

  auto plain_hilbertspace_vector()      ->       HilbertVector& {
    return polymorph().statevector;                             }
  auto plain_hilbertspace_vector()const -> const HilbertVector& {
    return polymorph().statevector;                             }

  auto
  linear_mapped(const linearMap<HilbertVector,HilbertVector>& f)const -> Vector {
    return Vector( f(polymorph().statevector), polymorph().physical_dimension );
  }

  template<class CoCanonBOb>auto
  linear_mapped(const linearMap< HilbertVector
                               , typename hilbertSpace< CoCanonBOb
                                                      , UnderlyingField
                                                      >::vector
                               >& f)const
     -> quantumState< CoCanonBOb //decltype( f(HilbertVector()).undefined_linear_component() )
                    , PhysT
                    , CalcT                                                       > {
    return quantumState<  /* ...                                   ... */ decltype(f(HilbertVector()).undefined_linear_component()),PhysT,CalcT>
             ( f(polymorph().statevector)
             , polymorph().physical_dimension );
  }
  template<class PreCanonBOb> void
  linear_applyaccumulate( const linearMap< typename hilbertSpace< PreCanonBOb
                                                                , UnderlyingField
                                                                >::vector
                                         , HilbertVector                     >& f
                        , const FieldT& alpha
                        , const quantumState<PreCanonBOb,PhysT,CalcT>& x
                        , const FieldT& beta                                  ) {
    if(abs(polymorph().physical_dimension) > 0) {
      if(abs(x.physical_dimension) > 0) {
        auto absbeta=abs(beta);
        polymorph().physical_dimension *= absbeta;
        assert( polymorph().physical_dimension
                    .compatible(abs(alpha)*x.physical_dimension) );
        f.accumulate_apply( unphysicalize_cast(alpha * x.physical_dimension
                                                  /polymorph().physical_dimension)
                          , x.statevector
                          , unphysicalize_cast(beta/absbeta)
                          , polymorph().statevector            );
      }
     }else{
      auto absalpha = abs(alpha);
      f.accumulate_apply( unphysicalize_cast(alpha/absalpha), x.statevector
                        , 0.                                , polymorph().statevector );
      polymorph().physical_dimension = absalpha*x.physical_dimension;
    }
  }

};


template <class HBO,typename PhT,typename ClcT>auto quantumState<HBO,PhT,ClcT>::
operator+=(const State& addv) -> State& {
  /*if (physical_dimension == addv.physical_dimension) {
    statevector += addv.statevector;
   }else if(physical_dimension == -addv.physical_dimension){
    statevector -= addv.statevector;
   }else*/
  if(addv.physical_dimension!=0){
    if(physical_dimension!=0) {
      assert(physical_dimension.compatible(addv.physical_dimension));
      statevector.axpyze( complexCalc( (addv.physical_dimension
                                    / physical_dimension).dbl() , 0.)
                        , addv.statevector                            );
     }else{
      *this = addv;
    }
  }
  return *this;
}
/*
template <class HBO,typename ClcT>auto quantumState<HBO,physquantity,ClcT>::
operator+=(const State& addv) -> State& {
  if (physical_dimension == addv.physical_dimension) {
    statevector += addv.statevector;
   }else if(physical_dimension == -addv.physical_dimension){
    statevector -= addv.statevector;
   }else if(addv.physical_dimension==0){
   }else{
    assert(physical_dimension.compatible(addv.physical_dimension));
    statevector += (addv.statevector * complex((addv.physical_dimension
                                               / physical_dimension   ).dbl(),0);
  }
  return *this;
}
*/
template <class HBO,typename PhT,typename ClcT>auto quantumState<HBO,PhT,ClcT>::
operator-=(const State& addv) -> State& {
  /*if (physical_dimension == addv.physical_dimension) {
    statevector -= addv.statevector;
   }else if(physical_dimension == -addv.physical_dimension){
    statevector += addv.statevector;
   }else*/
  if(addv.physical_dimension!=0){
    if(physical_dimension!=0){
      assert(physical_dimension.compatible(addv.physical_dimension));
      statevector.axpyze( complexCalc( -(addv.physical_dimension
                                    / physical_dimension).dbl() , 0.)
                        , addv.statevector                            );
//      statevector.axpyze(addv.statevector, complexPhys(-(addv.physical_dimension
  //                                                     / physical_dimension).dbl(), 0) );
     }else{
      statevector = addv.statevector;
      physical_dimension = -addv.physical_dimension;
    }
  }
  return *this;
}
/*
template <class HBO,typename ClcT>auto quantumState<HBO,physquantity,ClcT>::
operator-=(const State& addv) -> State& {
  if (physical_dimension == addv.physical_dimension) {
    statevector -= addv.statevector;
   }else if(physical_dimension == -addv.physical_dimension){
    statevector += addv.statevector;
   }else if(addv.physical_dimension==0){
   }else{
    assert(physical_dimension.compatible(addv.physical_dimension));
    statevector -= (addv.statevector * complex((addv.physical_dimension
                                               / physical_dimension   ).dbl(),0);
  }
  return *this;
}
*/
template <class HBO,typename PhysT,typename ClcT>auto quantumState<HBO,PhysT,ClcT>::
operator*=(const PhysT& multpl) -> State& {
  physical_dimension *= multpl;
  return *this;
}
template <class HBO,typename PhysT,typename ClcT>auto quantumState<HBO,PhysT,ClcT>::
operator*=(const complexCalc& multpl) -> State& {
  statevector *= multpl;
  return *this;
}
template <class HBO,typename PhysT,typename ClcT>auto quantumState<HBO,PhysT,ClcT>::
operator*=(const ClcT& multpl) -> State& {
  physical_dimension *= multpl;
  return *this;
}
template <class HBO,typename PhysT,typename ClcT>auto quantumState<HBO,PhysT,ClcT>::
operator*=(const complexPhys& multpl) -> State& {
  physical_dimension *= multpl.physdim();
  statevector *= multpl.complexdir();
  return *this;
}
/*template <class HBO,typename PhysT,typename ClcT>auto
operator*(quantumState<HBO,PhysT,ClcT>s, const PhysT& multpl) -> quantumState<HBO,PhysT,ClcT> {
  return s*=multpl;
}*/


template <class HBO,typename PhysT,typename ClcT>auto quantumState<HBO,PhysT,ClcT>::
operator*(const State& mtplSt)const -> complexPhys {
  return complexPhys( physical_dimension*mtplSt.physical_dimension
                    , statevector * mtplSt.statevector             );
}

template <class HBO>auto
operator*( const physquantity& p
         , typename hilbertSpace<HBO,complex<double>>::vector v
         ) -> quantumState<HBO,physquantity,double> {
  return quantumState<HBO,physquantity,double>(v,p);
}


template <class HBO,typename PhysT,typename ClcT>auto quantumState<HBO,PhysT,ClcT>::
norm()const -> decltype(abs(PhysT())) {
  auto result = abs(physical_dimension);
  result *= result;
  result *= statevector.norm();
  return result;
}

template <class HBO,typename PhysT,typename ClcT>auto quantumState<HBO,PhysT,ClcT>::
normalize() -> State& {
  auto oldstnorm = sqrt(statevector.norm());
  ClcT fldim = statevector.domain_Hilbertspace()->dimension();
  if( oldstnorm < fldim*fldim             //should be a safe range for float types,
       && oldstnorm > 1/fldim ) {        // here it's not necessary to (expensively)
    physical_dimension = 1./oldstnorm;  //  normalize the actual state vector.
   }else{                                 
    statevector *= 1./oldstnorm;
    physical_dimension = 1.;
  }
  return *this;
}

template <class HBO,typename PhysT,typename ClcT> template<class BaseobjOperator>
auto quantumState<HBO,PhysT,ClcT>::
baseobj_operation(BaseobjOperator o) -> State&{
  statevector.baseobj_operation(o);
  return *this;
}
                                         
template <class HBO,typename PhysT,typename ClcT>auto quantumState<HBO,PhysT,ClcT>::
canonical_base_decomp()const -> std::vector<std::pair<HBO,complexPhys>>{
  std::vector<std::pair<HBO,complexPhys>> result;
  for(auto oc : statevector.canonical_base_decomp())
    result.push_back(std::make_pair(oc.first, physical_dimension * oc.second));
  return result;
}


template<class PureHilbertVector, class PhysT, class CalcT=double>
struct quantumStateWrapper {
  typedef quantumState< typename PureHilbertVector::Hilbertspace::CanonBaseObjT
                      , PhysT
                      , CalcT
                      > WrappedVector;
  static auto wrap(PureHilbertVector vct) -> WrappedVector {
    return WrappedVector(std::move(vct));                  }
  static auto unwrap(const WrappedVector& vct) -> const PureHilbertVector& {
    return vct.statevector;                                                }
  static auto unwrap(WrappedVector& vct) -> PureHilbertVector& {
    return vct.statevector;                                    }
};




namespace Planck = cqtx::stdPhysUnitsandConsts::Planck;

     //Base class for hilbertSpace::CanonBaseObjT-objects that represent energy
    // eigenstates. States in such a Hilbert space can be trivially time-developed
   //  by rotating the phase of the eigenenergy-components, i.e. multiplying
  //   the amplitude with exp(iℏ⋅E⋅t).
template<typename CalcT=double>
class eigenEnergyState : public hilbertSpaceBaseObject {
  CalcT energy_in_plancks; //Planck units are arbitrary here, required is just
 public:                  // a unit system in which ℏ has the value 1, so we don't
                         //  need an extra multiplication when time-developing states.

  explicit eigenEnergyState(const physquantity& E)
    : energy_in_plancks(E[Planck::energy])
  {}
  virtual ~eigenEnergyState(){}

  virtual auto energy()const -> physquantity   {
    return energy_in_plancks * Planck::energy; }

  virtual auto operator<(const eigenEnergyState& cmp)const -> bool {   //problematic for
    return energy_in_plancks < cmp.energy_in_plancks;              }  // degenerate energy bases

  template <class HBO,typename PhysT,typename ClcT>
  friend quantumState<HBO,PhysT,ClcT>& time_develop(quantumState<HBO,PhysT,ClcT>&,const physquantity&);
  template <class HBO,typename PhysT,typename ClcT>
  friend quantumState<HBO,PhysT,ClcT>& quantum_energized(quantumState<HBO,PhysT,ClcT>&);
  friend class accellEigenEnergyArray;
};


struct accellEigenEnergyArray {
  template<class CBObj> auto
  basis_context( const std::vector<CBObj>& objs
               , const complex<double>& someval
               , const accelHandle& h           )
            -> accelVector<double>              {
    std::vector<double> ens;
    for(auto&e : objs) ens.push_back(e.energy_in_plancks);
    return accelVector<double>(h, std::move(ens));
  }
};

auto eigenenergy_context(const eigenEnergyState<double>* p)
        -> accellEigenEnergyArray {return accellEigenEnergyArray();}


 //Will only work for Hilbert spaces with baseobjs derived from eigenEnergyState.
template <class HBO,typename PhysT,typename ClcT>
quantumState<HBO,PhysT,ClcT>& time_develop( quantumState<HBO,PhysT,ClcT>& st
                                          , const physquantity& t)           {
  double tnat = t[Planck::time];
  return st.baseobj_operation( [&]( const eigenEnergyState<ClcT>& en
                                  , complex<ClcT>& a                 ){
                                 a *= polar(1., -en.energy_in_plancks * tnat);
                               }                                               );
}
template <class HBO,typename ClcT>auto
time_evolution(const quantumState<HBO,physquantity,ClcT>& st, const physquantity& t)
             -> quantumState<HBO,physquantity,ClcT>                                       {
  quantumState<HBO,physquantity,ClcT> result;
  auto&u = st.statevector; auto&v = result.statevector;
  v = st.statevector.implementation_componentsarray_fmapped(
                      [&](const accelVector<complex<ClcT>>& x) {
                        return x.phaserotations( physquantity(t)[Planck::time]
                                               , u.domain_Hilbertspace()
                                                   ->basiscontext()
                                                      .eigenenergies() );
                      }                                                  );
  result.physical_dimension = st.physical_dimension;
  return result;
}
template <class Vector>auto
time_evolution(const Vector& st, const physquantity& t)
             -> Vector                                  {
  typedef typename Vector::Hilbertspace Hilbertspace;
  typedef typename Hilbertspace::FieldT FieldT;
  return st.implementation_componentsarray_fmapped(
                      [&](const accelVector<FieldT>& x) {
                        return x.phaserotations( physquantity(t)[Planck::time]
                                               , st.domain_Hilbertspace()
                                                   ->basiscontext()
                                                      .eigenenergies() );
                      }                                                  );
}


template <class HBO,typename PhysT,typename ClcT>
quantumState<HBO,PhysT,ClcT>& quantum_energized( quantumState<HBO,PhysT,ClcT>& st ){
  return st.baseobj_operation( [&]( const eigenEnergyState<ClcT>& en
                                  , complex<ClcT>& a                 ){
                                 a *= en.energy_in_plancks;
                               } ) *= Planck::energy;
}
template <class HBO,typename PhysT,typename ClcT>auto
direct_hamiltonian(quantumState<HBO,PhysT,ClcT> st)
                 -> quantumState<HBO,PhysT,ClcT>     {
  return quantum_energized(st);
}


template<class HBOT, typename PhysT, typename ClcT>auto
energy_eigenbasis( const linearMap< quantumState<HBOT,PhysT,ClcT>
                                  , quantumState<HBOT,PhysT,ClcT> >& hamiltonian )
         -> hilbertSpace<eigenEnergyState<ClcT>, complex<ClcT>>        {
  if(auto gnm = dynamic_cast<const generalizedLinMap< quantumState<HBOT,PhysT,ClcT>
                                                    , typename hilbertSpace<HBOT,complex<ClcT>>::vector
                                                    , typename hilbertSpace<HBOT,complex<ClcT>>::vector
                                                    , quantumState<HBOT,PhysT,ClcT>
                                                    >*
                            >(hamiltonian.implemented()) ){
    assert(abs(imag(gnm->intn_multiplier())) < abs(real(gnm->intn_multiplier())) * 1e-8);
    return gnm->domain()->template eigenbasis_gen<eigenEnergyState<ClcT>>
                        ( gnm->special_map()
                        , [&](const accelEigenbasisTransform<complex<ClcT>> &transform) {
                            auto eigenensnat = transform.real_eigenvalues();
                            std::vector<eigenEnergyState<ClcT>> eigenenergies;
                            for(unsigned i=0; i<gnm->domain()->dimension(); ++i)
                              eigenenergies.push_back(eigenEnergyState<ClcT>
                                  (eigenensnat[i] * real(gnm->intn_multiplier())));
                            return eigenenergies;
                          }
                        );
   }else{
    std::cerr << "Can't create energy eigenbasis for Hamiltonian not implemented as a Hilbert space linear map.\n";
    std::cerr << "Actual type is " << typeid(*hamiltonian.implemented()).name() << std::endl;
    abort();
  }
/*  hilbertSpace<eigenEnergyState<ClcT>, complex<ClcT>> result;
  return result;*/
}


template<class CBO, class PhysT, class CalcT>
struct eigenbasisFactory<quantumState<CBO,PhysT,CalcT>> {
  typedef quantumState<CBO,PhysT,CalcT> DomVect;
  typedef complex<CalcT> complexCalc;
  typedef complex<PhysT> complexPhys;

  struct eigenvalueHandler {
    complexPhys multiplier;
    auto operator()(const accelEigenbasisTransform<complexCalc>& eigtransf)const
            -> std::vector<genericEigenbasisObj<complexPhys>> {
      std::vector<genericEigenbasisObj<complexPhys>> result;
      auto cpleigvals = eigtransf.complex_eigenvalues();
      std::cout << cpleigvals.dimension() << " complex eigenvalues\n";
      for(auto& eigv: cpleigvals)
        result.push_back(genericEigenbasisObj<complexPhys>(multiplier * eigv));
      auto realeigvals = eigtransf.real_eigenvalues();
      std::cout << realeigvals.dimension() << " real eigenvalues\n";
      if(result.size()==0)
        for(auto& eigv: realeigvals)
          result.push_back(genericEigenbasisObj<complexPhys>(multiplier * complexPhys(eigv)));
      return result;
    }
    eigenvalueHandler(const complexPhys& multip): multiplier(multip) {}
  };

  static auto
  eigenbasis(const linearMap<DomVect, DomVect>& op)
     -> hilbertSpace< genericEigenbasisObj<complexPhys>, complexCalc > {
    if(auto gnm = dynamic_cast<const generalizedLinMap< quantumState<CBO,PhysT,CalcT>
                                                      , typename hilbertSpace<CBO,complex<CalcT>>::vector
                                                      , typename hilbertSpace<CBO,complex<CalcT>>::vector
                                                      , quantumState<CBO,PhysT,CalcT>
                                                      >* >(op.implemented()) ){
      return gnm->domain()->template eigenbasis_gen<genericEigenbasisObj<complexPhys>,eigenvalueHandler>
                          ( gnm->special_map()
                          , eigenvalueHandler(gnm->intn_multiplier()) );
     }else{
      std::cerr << "Can't create eigenbasis for operator not implemented as a Hilbert space linear map.\n";
      std::cerr << "Actual type is " << typeid(*op.implemented()).name() << std::endl;
      abort();
    }
  }
};



}
#endif