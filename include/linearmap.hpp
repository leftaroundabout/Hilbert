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


#ifndef GENERAL_LINEAR_MAPPINGS_ON_HILBERTSPACES_FOR_QM
#define GENERAL_LINEAR_MAPPINGS_ON_HILBERTSPACES_FOR_QM

      //Algebra of linear mappings between Hilbert spaces. The aim is to provide
     // an abstract interface for treating those mappings in all the intuitive
    //  ways (e.g. addition A+B, composition as multiplication A*B, inverse with
   //   division operator A/B...) while internally keeping all computations as
  //    efficient as possible, using arbitrarily combinable representations ranging
 //     from opaque matrix-free descriptions over plain multiplicators to various
//      sparse or dense GPU-accelerated matrices.

#include "lambdalike/maybe.hpp"
using namespace lambdalike;

#include<utility>

#include<hilbertsp.hpp>
//#include <cxxabi.h>



namespace hilbert {


template<class DomVct, class CoDomVct>
class linearMap;
template<class DomVct, class CoDomVct>auto
linear_lambda(const std::function<DomVct(CoDomVct)>& f)-> linearMap<DomVct,CoDomVct>;
template<class DomVct, class CoDomVct>
class linearMapSum;
template<class DomVct, class CoDomVct>
class invertedLinearmap;
template<class DomVect,class SpclzDomVect
        ,class SpclzCoDomVect,class CoDomVct>
class generalizedLinMap;
template< class DomainVector, class InterDomainVector, class CoDomainVector >
class linearMapComposition;


   //base class for linear mappings/operators from one vector space into another
  // one. Conceptually an abstract class, but it can actually be used as a concrete
 //  class, in which case it acts as a wrapper around a derived object. That also
//   means that any derived instance can virtually act as any other derived instance!
template<class DomainVector, class CoDomainVector = DomainVector>
class linearMap {
 public:
  typedef typename DomainVector::Hilbertspace Domain;
  typedef typename CoDomainVector::Hilbertspace CoDomain;
  typedef typename DomainVector::DomainFieldT FieldT;
  typedef typename CoDomainVector::DomainFieldT CoFieldT;

 protected: mutable                                 //polymorphic; must either hold an implementation
  std::unique_ptr<linearMap> implement_override;   // instance or point to NULL. In a base instance,
                                                  //  must point to a derived instance!
                                                    //mutable to allow conservative changes of implementation,
 public:                                           // e.g. from one in terms of scalar product sums to a full
  virtual void matricisize()const{                //  matrix to enable inverse mapping.
    implement_override->matricisize();
  }

  auto implemented() -> linearMap*                                       {      //always returns a
    return implement_override? implement_override->implemented() : this; }     // pointer to the object
  auto implemented()const -> const linearMap*                            {    //  that actually implements
    return implement_override? implement_override->implemented() : this; }   //   the linear mapping.
  
  virtual auto                                                       //Plain application
  operator()(const DomainVector& v)const -> CoDomainVector {        // of the linear mapping
    return (*implement_override)(v);                       }       //  on a vector.

  virtual auto                                                      //Plain application
  inv_appliedto(const CoDomainVector& v)const -> DomainVector {    // of the inverse linear
    return implement_override->inv_appliedto(v);              }   //  mapping on a vector.

  virtual void                                                          //Essentially the BLAS
  accumulate_apply( const CoFieldT& alpha, const DomainVector& v       // gemv operation,
                  , const CoFieldT& beta, CoDomainVector& acc)const { //  acc <- α⋅v + β⋅acc
    implement_override->accumulate_apply(alpha, v, beta, acc);
  }


/*  template<typename VctGenrlz> auto                           //Operators on vectors
  operator()(const generalizedVector<Domain,VctGenrlz>& v)const// should also work on
      -> decltype(v.linear_mapped(linearMap())) {             //  generalizations of such.
    return v.linear_mapped(*this);                           //   At the moment, this
  }*/  // is only enabled in the hilbertSpace::linearMap classes (see comment in
      //  hilbertSpace::generalLinearMap) and only works for endomorphismisms (TODO).

/*  template<class OtherBasisVector>auto    //TODO Likewise, operators should be able to operate
  operator()(const OtherBasisVector& v)    // on vectors defined in another basis of the same space.
     -> CoDomainVector {  // actually: -> decltype(std::declval<linearMap>()(v.using_basis(std::declval<Domain>()))) {
                        // but that can hang the compiler in an infinite recursion consuming all memory
    return (*this)(v.using_basis(*domain()));
  }*/

  template<class AltntvDomVect, class AltntvCoDomVect>void               //Generalization
  accumulate_apply( const CoFieldT& alpha, const AltntvDomVect& v       // of accumulate_apply
                  , const CoFieldT& beta, AltntvCoDomVect& acc)const { //  works already.
    acc.linear_appyaccumulate(*this, alpha, v, beta);
  }

  virtual linearMap& operator+=(const linearMap&);
  virtual linearMap& operator+=(linearMap&& addm);

  auto operator+(const linearMap& addm)const -> linearMap{
    return linearMap(*this)+=addm;                       }
  auto operator+(linearMap&& addm)const -> linearMap{
    return addm += *this;                           }

  template<class SpclzDomVect,class SpclzCoDomVect>
  linearMap& operator+=(const linearMap<SpclzDomVect,SpclzCoDomVect>& addm){
     return *this += generalizedLinMap< DomainVector,  SpclzDomVect
                                      , SpclzCoDomVect,CoDomainVector >(addm);
  }

  linearMap& operator+=(const std::function< CoDomainVector(DomainVector) >& f){
    return *this += linear_lambda(f);
  }

  virtual auto
  inverse()const -> linearMap<CoDomainVector,DomainVector> {
    return implement_override
               ? implement_override->inverse()
               : invertedLinearmap<DomainVector,CoDomainVector>(*this);
  }
  
  virtual linearMap& lcompose(const linearMap<CoDomainVector,CoDomainVector>&);
  virtual linearMap& lcompose(linearMap<CoDomainVector,CoDomainVector>&&);
  virtual linearMap& operator*=(const linearMap<DomainVector,DomainVector>&); //right composition
  virtual linearMap& operator*=(linearMap<DomainVector,DomainVector>&&);
  
  virtual auto
  domain()const -> const Domain* {
    return implement_override
               ? implement_override->domain()
               : nullptr;
  }
  virtual auto
  codomain()const -> const CoDomain* {
    return implement_override
               ? implement_override->codomain()
               : nullptr;
  }
  
//   virtual auto
//   eigenbasis()const -> hilbertSpace<genericEigenbasisObj<FieldT>,FieldT>;

  virtual auto
  clone()const -> linearMap*{return implement_override->clone();}
  virtual auto
  moved() -> linearMap*{return implement_override->moved();}
//  virtual auto moved() -> linearMap*{return new linearMap(std::move(*this));}

  template<class Derived>               //all derived instances should (via CRTP) derive from
  struct implementation : linearMap {  // implementation, rather than directly from linearMap.
    auto operator()(const DomainVector&)const -> CoDomainVector =0;
    auto inv_appliedto(const CoDomainVector& y)const -> DomainVector         {
      if (linearMap::implement_override){
        linearMap::implement_override->matricisize();
        return linearMap::implement_override->inv_appliedto(y);
       }else{
        static_cast<const Derived*>(this)->matricisize();
        return static_cast<const Derived*>(this)->inv_appliedto(y);
      }
    }
    void accumulate_apply( const CoFieldT& alpha, const DomainVector& v
                         , const CoFieldT& beta,CoDomainVector& acc   )const{
      (acc*=beta) += operator()(v) * alpha;                                 }
    void matricisize()const {
      if (linearMap::implement_override)
        linearMap::implement_override->matricisize();
      std::cerr << "Casting " << typeid(Derived).name()
                << "\n         to a plain matrix type not supported.\n";
      abort();
    }
    auto clone()const -> linearMap* {
      return linearMap::implement_override
                ? this->implemented()->clone()
                : new Derived(static_cast<const Derived&>(*this));
    }
    auto moved() -> linearMap*          {
      return linearMap::implement_override
                ? linearMap::implement_override.release()
                : new Derived(std::move(static_cast<Derived&>(*this)));
    }
    linearMap& operator=(linearMap cpy) {
      clear();
      linearMap::implement_override.reset(cpy.moved());
      return *this;
    }
   protected:
    virtual void clear()=0;  //set the state ready for copy assignment

    implementation(const implementation& cpy) {   // does NOT copy a non-overridden 
      if(cpy.implement_override)            //  implementation instance
        linearMap::implement_override.reset(cpy.clone());
    }
    implementation(implementation&& mov) noexcept{
      if(mov.implement_override)
        linearMap::implement_override.reset(mov.moved());
    }
    implementation(){}
  };
    
  linearMap(const linearMap& cpy)
    : implement_override(cpy.implemented()->clone())
  {}
  linearMap(linearMap&& mov) noexcept{
    if (!mov.implement_override) implement_override.reset(mov.moved());
     else                 implement_override = std::move(mov.implement_override);
  }
  virtual linearMap& operator=(linearMap cpy) {
    implement_override.reset(cpy.moved());
    return *this;
  }
  
  virtual auto                                                 //an implementation instance which
  supports_native_addition(const linearMap& f)const -> bool { // can efficiently perform linear
    return implement_override                                //  map-addition (e.g. a simple
         && implement_override->supports_native_addition(f);//   matrix representation) should
  }                                                        //    advertise this by returning true.
  virtual auto
  supports_native_rcomposition(const linearMap<DomainVector,DomainVector>& f)const -> bool {
    return implement_override
         && implement_override->supports_native_rcomposition(f);
  }
  virtual auto
  supports_native_lcomposition(const linearMap<CoDomainVector,CoDomainVector>& f)const -> bool {
    return implement_override
         && implement_override->supports_native_lcomposition(f);
  }
                                                           

 protected:
  linearMap()                         //use only to initialize the base
    : implement_override(nullptr) {} // slice of an implementation instance!

 public:

  virtual ~linearMap(){}
  
  friend class linearMapSum<DomainVector, CoDomainVector>;
};



#if 0
//Specialization for simple hilbertSpace::vectors.
template<class DomainCBO, class CoDomainCBO, class FieldT>
class linearMap< typename hilbertSpace<DomainCBO,FieldT>::vector
               , typename hilbertSpace<CoDomainCBO,FieldT>::vector > {
  typedef hilbertSpace<DomainCBO,FieldT> Domain;
  typedef typename Domain::vector DomainVector;
  typedef hilbertSpace<CoDomainCBO,FieldT> CoDomain;
  typedef typename CoDomain::vector CoDomainVector;

  std::unique_ptr<linearMap> implement_override;
                                         
 public:
  virtual auto
  operator()(const DomainVector& v)const -> CoDomainVector {
    return (*implement_override)(v);                              }
  template<typename HSpaceVectT> auto                              //Operators on hilbertSpace::vectors
  operator()(const generalizedVector<Domain,HSpaceVectT>& v)const // should automatically generalize to
      -> decltype(v.linear_mapped(linearMap())) {                //  Operators on the corresponding
    return v.linear_mapped(*this);                              //   vector generalizations.
  }

 //    virtual auto
//  operator+(const linearMap& addm)const -> linearMap;

  virtual auto
  clone()const -> linearMap*{return new linearMap(*this);}
  virtual auto
  moved() -> linearMap*{return new linearMap(std::move(*this));}

  template<class Derived>               //all derived instances should (via CRTP) derive from
  struct implementation : linearMap {  // implementation, rather than directly from linearMap.
    auto operator()(const DomainVector&)const -> CoDomainVector =0;
    auto clone()const -> linearMap*                           {
      return new Derived(static_cast<const Derived&>(*this)); }
    auto moved() -> linearMap*                                      {
      return new Derived(static_cast<Derived&&>(std::move(*this))); }
    implementation(){}
  };
  
  linearMap(const linearMap& cpy)
    : implement_override( cpy.implement_override ? cpy.implement_override->clone()
                                   : cpy.clone()              )
  {}
  linearMap(linearMap&& mov)
    : implement_override( mov.implement_override ? std::move(mov.implement_override)
                                   : mov.moved()                )
  {}
  virtual linearMap& operator=(linearMap cpy) {
    implement_override = ( cpy.implement_override ? std::move(cpy.implement_override)
                                    : cpy.moved()                );
  }
 protected:
  linearMap()                  //use only to initialize the base
    : implement_override(nullptr) {} // slice of an implementation instance!
 public:

  virtual ~linearMap(){}
};
#endif



    //linearMap wrapper around a generic std::function, which can conveniently
   // be defined with a C++11 lambda function. Of course, that function should
  //  itself be linear: this is not enforced but required for the methods to
 //   behave correctly.
template<class DomainVector, class CoDomainVector = DomainVector>
class linearLambda : public linearMap<DomainVector,CoDomainVector>
                          ::template implementation<linearLambda< DomainVector
                                                                , CoDomainVector> > {
  using linearMap<DomainVector,CoDomainVector>::template implementation<linearLambda>::linearMap::implement_override;
                           
  std::function< CoDomainVector(DomainVector) > f;
  
  void clear(){}
 public:
  auto operator()(const DomainVector& v)const -> CoDomainVector {
    return implement_override? (*implement_override)(v) : f(v);
  }
  
  template<class Lambda>
  linearLambda(const Lambda& init)
    : f(init)
  {}
};

template<class DomainVector, class CoDomainVector = DomainVector>auto
linear_lambda(const std::function< DomainVector(CoDomainVector) >& f)
   -> linearMap<DomainVector,CoDomainVector> {    
  return linearMap<DomainVector,CoDomainVector>(
           linearLambda<DomainVector,CoDomainVector>(f) );
}



   //specialized linearLambda: the initializing function should not only be
  // linear, but constant, ᴀʟᴡᴀʏꜱ returning a zero vector.
template<class DomainVector, class CoDomainVector = DomainVector>
class linearZeroLambda : public linearMap<DomainVector,CoDomainVector>
                          ::template implementation<linearZeroLambda< DomainVector
                                                                    , CoDomainVector> > {
  typedef linearMap<DomainVector,CoDomainVector> LinearMap;
  using LinearMap::template implementation<linearZeroLambda>::linearMap::implement_override;
                           
  std::function< CoDomainVector(DomainVector) > f;

  void clear(){}
 public:
  auto supports_native_addition(const LinearMap& addc)const -> bool {
    return !implement_override || implement_override->supports_native_addition(addc);
  }
  auto supports_native_rcomposition(const linearMap<DomainVector,DomainVector>& f)const -> bool {
    return !implement_override || implement_override->supports_native_addition(f);
  }
  auto supports_native_lcomposition(const linearMap<CoDomainVector,CoDomainVector>& f)const -> bool {
    return !implement_override || implement_override->supports_native_addition(f);
  }
  
  auto operator()(const DomainVector& v)const -> CoDomainVector {
    return implement_override? (*implement_override)(v) : f(v);
  }

  LinearMap& operator+=(const LinearMap& addm){          //adding something to a
    if(implement_override) (*implement_override)+=addm; // zero mapping results
     else implement_override.reset(addm.clone());      //  in the added mapping
    return *this;                                     //   alone.
  }
  LinearMap& operator+=(LinearMap&& addm){
    if(implement_override) (*implement_override)+=std::move(addm);
     else implement_override.reset(addm.moved());
    return *this;
  }
  
  template<class SDV,class SCDV>                                  //Should not be
  LinearMap& operator+=(const linearMap<SDV,SCDV>& addm){        // necessary (because
     return*this+=generalizedLinMap<DomainVector, SDV           //  defined in base
                                   ,SCDV,CoDomainVector>(addm);//   class), but
  }                                                           //    apparently is.
  
  LinearMap& operator*=(const linearMap<DomainVector,DomainVector>& f) { //composition with zero map
    return implement_override? *implement_override*=f : *this;         }// yields a zero map again
  LinearMap& operator*=(linearMap<DomainVector,DomainVector>&& f)         {
    return implement_override? *implement_override*=std::move(f) : *this; }
  LinearMap& lcompose(const linearMap<DomainVector,DomainVector>& f)    {
    return implement_override? implement_override->lcompose(f) : *this; }
  LinearMap& lcompose(linearMap<DomainVector,DomainVector>&& f)                    {
    return implement_override? implement_override->lcompose(std::move(f)) : *this; }
  

  template<class Lambda>
  linearZeroLambda(const Lambda& init, int zero)
    : f(init) {
    assert(zero==0);
  }

  linearZeroLambda(const linearZeroLambda& init) {
    if(init.implement_override) implement_override.reset(new LinearMap(*init.implement_override));
     else f = init.f;
  }
  linearZeroLambda(linearZeroLambda&& init) {
    if(init.implement_override) implement_override=std::move((init.implement_override));
     else f = std::move(init.f);
  }
  linearZeroLambda(const LinearMap& init) {
    implement_override.reset(init.copy());
  }
  linearZeroLambda(LinearMap&& init) {
    implement_override.reset(init.moved());
  }
  
};



template<class DomainVector, class CoDomainVector=DomainVector>
class multiplierMap : public linearMap<DomainVector,DomainVector>
                          ::template implementation<multiplierMap<DomainVector,CoDomainVector> > {
  typedef linearMap<DomainVector,DomainVector> LinearMap;
  using LinearMap::template implementation<multiplierMap>::linearMap::implement_override;

  typedef typename DomainVector::Hilbertspace Domain;
  typedef typename DomainVector::DomainFieldT FieldT;

  FieldT multiplier;

  void clear(){}
 public:
  auto operator()(const DomainVector& v)const -> CoDomainVector {
    return implement_override? (*implement_override)(v) : v*multiplier;
  }
    
  multiplierMap(const FieldT& init)
    : multiplier(init)
  {}

   auto operator*(const linearMap<DomainVector,DomainVector>& f) -> LinearMap {
     return linearMap<DomainVector,DomainVector>(f).lcompose(*this);          }
  auto operator*(linearMap<DomainVector,DomainVector>&& f) -> LinearMap {
    return f.lcompose(*this);                                           }
  
  template<class AltDomainVect, class AltCoDomainVect>
  auto operator*(const linearMap<AltDomainVect,AltCoDomainVect>& f)
         -> generalizedLinMap<DomainVector,AltDomainVect, AltCoDomainVect, CoDomainVector> {
    return generalizedLinMap<DomainVector,AltDomainVect, AltCoDomainVect, CoDomainVector>
                (f, multiplier);
  }
  template<class AltDomainVect, class AltCoDomainVect>
  auto operator*(linearMap<AltDomainVect,AltCoDomainVect>&& f)
         -> generalizedLinMap<DomainVector,AltDomainVect, AltCoDomainVect, CoDomainVector> {
    return generalizedLinMap<DomainVector,AltDomainVect, AltCoDomainVect, CoDomainVector>
                (std::move(f), multiplier);
  }
  
  auto multiplicator()const -> const FieldT& { return multiplier; }
};


template<class DomainVector>auto
identity_map() -> linearMap<DomainVector,DomainVector> {
  return linearMap<DomainVector,DomainVector>(
            linearLambda<DomainVector,DomainVector>([](const DomainVector& v){
              return v;
            })
         );
}




#if 0                            //Not really a linear mapping, in general! – only
                                // use this to efficiently implement zero-mappings.
template<class DomainVector, class CoDomainVector = DomainVector>
class constLinMap : public linearMap<DomainVector,CoDomainVector>
                      ::template implementation<constLinMap<DomainVector,CoDomainVector>> {
  using linearMap<DomainVector,CoDomainVector>::template implementation<constLinMap>::linearMap::implement_override;

  CoDomainVector c;
 public:
  auto operator()(const DomainVector& v)const -> CoDomainVector {
    return implement_override? (*implement_override)(v) : c;
  }
  
  constLinMap(CoDomainVector init)
    : c(std::move(init))
  {}
};
#endif



template< class OrigDomainVect, class OrigCoDomainVct >
class invertedLinearmap : public linearMap<OrigCoDomainVct,OrigDomainVect>
                          ::template implementation< invertedLinearmap
                                                       <OrigDomainVect,OrigCoDomainVct> > {
  typedef linearMap<OrigCoDomainVct,OrigDomainVect> LinearMap;
  using LinearMap::template implementation<invertedLinearmap>::linearMap::implement_override;


  typedef typename OrigCoDomainVct::Hilbertspace Domain;
  typedef typename OrigCoDomainVct::DomainFieldT FieldT;
  typedef typename OrigDomainVect::Hilbertspace CoDomain;
  typedef typename OrigDomainVect::DomainFieldT CoFieldT;
  
  typedef linearMap<OrigDomainVect,OrigCoDomainVct> OrigDirLinMap;
  maybe<OrigDirLinMap> origmap; //MUST hold an object unless implement_override does

  void clear(){origmap=nothing;}
 public:

  auto operator()(const OrigCoDomainVct& v)const -> OrigDomainVect {
    for(auto&iv: origmap) return iv.inv_appliedto(v);
    return (*implement_override)(v);
  }
  auto inv_appliedto(const OrigDomainVect& v)const -> OrigCoDomainVct {
    for(auto&iv: origmap) return iv(v);
    return implement_override->inv_appliedto(v);
  }
  auto inverse()const -> OrigDirLinMap {
    for(auto&iv: origmap) return iv;
    return implement_override->inverse();
  }
  
  auto domain()const -> const Domain* {
    for(auto&iv: origmap) return iv.codomain();
    return implement_override->domain();
  }
  auto codomain()const -> const CoDomain* {
    for(auto&iv: origmap) return iv.domain();
    return implement_override->codomain();
  }

  invertedLinearmap(OrigDirLinMap originit)
    : origmap(nothing) {
    if(auto cs = dynamic_cast<invertedLinearmap*>(originit.implemented()))
      for(auto&iv: cs->origmap) implement_override.reset(iv.moved());
     else
      origmap = std::move(originit);
  }
};



template< class GnrlzDomVect, class DomainVector
        , class CoDomainVector, class GnrlzCoDomVect >
class generalizedLinMap : public linearMap<GnrlzDomVect,GnrlzCoDomVect>
                          ::template implementation< generalizedLinMap
                                                       <GnrlzDomVect,DomainVector
                                                       ,CoDomainVector,GnrlzCoDomVect> > {
  typedef linearMap<GnrlzDomVect,GnrlzCoDomVect> LinearMap;
  using LinearMap::template implementation<generalizedLinMap>::linearMap::implement_override;
  
  typedef typename DomainVector::Hilbertspace Domain;
  typedef typename DomainVector::DomainFieldT InnerFieldT;
  typedef typename GnrlzDomVect::DomainFieldT OuterFieldT;
  typedef typename CoDomainVector::Hilbertspace CoDomain;
  typedef typename CoDomainVector::DomainFieldT InnerCoFieldT;
  typedef typename GnrlzCoDomVect::DomainFieldT OuterCoFieldT;

  typedef linearMap<DomainVector,CoDomainVector> SpecializedMap;
  maybe<SpecializedMap> spclf;   //MUST hold an object unless implement_override is set
  OuterCoFieldT multiplier;

  void clear(){spclf=nothing;}
 public:

  auto operator()(const GnrlzDomVect& v)const -> GnrlzCoDomVect {
    return implement_override ? (*implement_override)(v)
                              : v.linear_mapped(*spclf) * multiplier;
  }
  auto inv_appliedto(const GnrlzCoDomVect& v)const -> GnrlzDomVect {
    return implement_override
               ? implement_override->inv_appliedto(v)
               : v.linear_mapped( linearLambda<CoDomainVector,DomainVector>(
                                      [&](const CoDomainVector& v)        {
                                        return (*spclf).inv_appliedto(v); } )
                                ) * (OuterCoFieldT(1)/multiplier);
  }

  void accumulate_apply( const OuterCoFieldT& alpha, const GnrlzDomVect& v
                       , const OuterCoFieldT& beta,GnrlzCoDomVect& acc   )const{
    if(implement_override){
      implement_override->accumulate_apply(alpha,v,beta,acc);
     }else{
      acc.linear_applyaccumulate(*spclf,alpha*multiplier,v,beta);
    }
  }

  auto supports_native_addition(const LinearMap& addc)const -> bool {
    return implement_override
               ? implement_override->supports_native_addition(addc)
               : !!dynamic_cast<const generalizedLinMap*>(addc.implemented());
  }

  LinearMap& operator+=(const LinearMap& addm){
    if(implement_override){ (*implement_override)+=addm; return *this; }

    if(abs(multiplier) == 0.) {
      implement_override.reset(addm.clone());
      return *this;
    }

    if( auto addmgn = dynamic_cast<const generalizedLinMap*>(addm.implemented()) ) {
      if(multiplier == addmgn->multiplier) {
        *spclf += *addmgn->spclf;
       }else{
        *spclf += multiplierMap<CoDomainVector>(InnerCoFieldT
                               (addmgn->multiplier/multiplier))
                                    * *addmgn->spclf;
      }
     }else{
      implement_override.reset( new
          linearMapSum< GnrlzDomVect
                      , GnrlzCoDomVect >( generalizedLinMap(std::move(*spclf))
                                        , addm                                 ) );
      spclf = nothing;
    }
    return *this;
  }
  LinearMap& operator+=(LinearMap&& addm){
    if(implement_override){ (*implement_override)+=std::move(addm); return *this; }
    
    if(abs(multiplier) == 0.) {
      implement_override.reset(addm.moved());
      return *this;
    }

    if( auto addmgn = dynamic_cast<generalizedLinMap*>(addm.implemented()) ) {
      if(multiplier == addmgn->multiplier)
        *spclf += std::move(*addmgn->spclf);
       else
        *spclf += (multiplierMap<CoDomainVector>(InnerCoFieldT
                                         (addmgn->multiplier/multiplier)))
                      * //std::move
                       (*addmgn->spclf);
     }else{
      implement_override.reset(new
          linearMapSum< GnrlzDomVect
                      , GnrlzCoDomVect >( generalizedLinMap(std::move(*spclf))
                                        , std::move(addm)                     ) );
      spclf = nothing;
    }
    return *this;
  }

  auto supports_native_rcomposition(const linearMap<GnrlzDomVect
                                                   ,GnrlzDomVect>& f)const -> bool {
    return implement_override
               ? implement_override->supports_native_rcomposition(f)
               : dynamic_cast<const generalizedLinMap<GnrlzDomVect,DomainVector
                                                       ,DomainVector,GnrlzDomVect>*>(f.implemented())
                ||dynamic_cast<const multiplierMap<GnrlzDomVect,GnrlzDomVect>*>(f.implemented());
  }
  auto supports_native_lcomposition(const linearMap<GnrlzCoDomVect
                                                   ,GnrlzCoDomVect>& f)const -> bool {
    return implement_override
               ? implement_override->supports_native_rcomposition(f)
               : dynamic_cast<const generalizedLinMap<GnrlzCoDomVect,CoDomainVector
                                                       ,CoDomainVector,GnrlzCoDomVect>*>(f.implemented())
                ||dynamic_cast<const multiplierMap<GnrlzCoDomVect,GnrlzCoDomVect>*>(f.implemented());
  }

  LinearMap& operator*=(const linearMap<GnrlzDomVect,GnrlzDomVect>& rcompf) {
    if(implement_override){ (*implement_override)*=rcompf; return *this; }

    if(auto rcmmgn = dynamic_cast< const generalizedLinMap<GnrlzDomVect,DomainVector
                                                          ,DomainVector,GnrlzDomVect
                                                          >* >(rcompf.implemented())) {
      multiplier *= rcmmgn->multiplier;
      *spclf *= *rcmmgn->spclf;
     }else if(
      auto fmsp = dynamic_cast<const multiplierMap<GnrlzDomVect,GnrlzDomVect>*>(rcompf.implemented())) {
      multiplier *= fmsp->multiplicator();
     }else{
      implement_override.reset(
          new linearMapComposition< GnrlzDomVect, GnrlzDomVect, GnrlzCoDomVect >
                      ( generalizedLinMap(std::move(*spclf),multiplier)
                      , rcompf                                                     ) );
      spclf = nothing;
    }
    
    return *this;
  }
  LinearMap& operator*=(linearMap<GnrlzDomVect,GnrlzDomVect>&& rcompf) {
    if(implement_override){ (*implement_override)*=std::move(rcompf); return *this; }

    if(auto rcmmgn = dynamic_cast< generalizedLinMap<GnrlzDomVect,DomainVector
                                                    ,DomainVector,GnrlzDomVect
                                                    >* >(rcompf.implemented())) {
      multiplier *= rcmmgn->multiplier;
      *spclf *= std::move(*rcmmgn->spclf);
     }else if(
      auto fmsp = dynamic_cast<multiplierMap<GnrlzDomVect,GnrlzDomVect>*>(rcompf.implemented())) {
      multiplier *= fmsp->multiplicator();
     }else{
      implement_override.reset(
          new linearMapComposition< GnrlzDomVect, GnrlzDomVect, GnrlzCoDomVect >
                      ( generalizedLinMap(std::move(*spclf),multiplier)
                      , std::move(rcompf)                                        ) );
      spclf = nothing;
    }
    return *this;
  }

  LinearMap& lcompose(const linearMap<GnrlzCoDomVect,GnrlzCoDomVect>& lcompf) {
    if(implement_override){ implement_override->lcompose(lcompf); return *this; }

    if(auto rcmmgn = dynamic_cast< const generalizedLinMap<GnrlzCoDomVect,CoDomainVector
                                                          ,CoDomainVector,GnrlzCoDomVect
                                                          >* >(lcompf.implemented())) {
      multiplier *= rcmmgn->multiplier;
      (*spclf).lcompose(*rcmmgn->spclf);
     }else if(
      auto fmsp = dynamic_cast<const multiplierMap<GnrlzCoDomVect,GnrlzCoDomVect>*>(lcompf.implemented())) {
      multiplier *= fmsp->multiplicator();
     }else{
      implement_override.reset(
          new linearMapComposition< GnrlzDomVect, GnrlzCoDomVect, GnrlzCoDomVect >
                      ( lcompf
                      , generalizedLinMap(std::move(*spclf),multiplier) ) );
      spclf = nothing;
    }
    
    return *this;
  }
  LinearMap& lcompose(linearMap<GnrlzCoDomVect,GnrlzCoDomVect>&& lcompf) {
    if(implement_override){ implement_override->lcompose(std::move(lcompf)); return *this; }

    if(auto rcmmgn = dynamic_cast< generalizedLinMap<GnrlzCoDomVect,CoDomainVector
                                                    ,CoDomainVector,GnrlzCoDomVect
                                                    >* >(lcompf.implemented())) {
      multiplier *= rcmmgn->multiplier;
      (*spclf).lcompose(std::move(*rcmmgn->spclf));
     }else if(
      auto fmsp = dynamic_cast<multiplierMap<GnrlzCoDomVect,GnrlzCoDomVect>*>(lcompf.implemented())) {
      multiplier *= fmsp->multiplicator();
     }else{
      implement_override.reset(
          new linearMapComposition< GnrlzDomVect, GnrlzCoDomVect, GnrlzCoDomVect >
                      ( std::move(lcompf)
                      , generalizedLinMap(std::move(*spclf),multiplier) ) );
      spclf = nothing;
    }
    
    return *this;
  }

  generalizedLinMap& rcompose(linearMap<DomainVector,DomainVector>&& f) {
    if(implement_override)
      *implement_override *= generalizedLinMap<GnrlzDomVect,DomainVector
                                              ,DomainVector,GnrlzDomVect>
                                                               (std::move(f));
     else
      *spclf *= std::move(f);
    return *this;
  }

  auto special_map()const -> const SpecializedMap& { return *spclf; }
  auto intn_multiplier()const -> const OuterCoFieldT& { return multiplier; }
  
  auto domain()const -> const Domain* {
    return implement_override
               ? implement_override->domain()
               : (*spclf).domain();
  }
  auto codomain()const -> const CoDomain* {
    return implement_override
               ? implement_override->codomain()
               : (*spclf).codomain();
  }

  generalizedLinMap( SpecializedMap init
                   , OuterCoFieldT multiplier = polymorphic_1<OuterCoFieldT>() )
    : spclf(just(std::move(init)))
    , multiplier(multiplier)
  {}

   //copy/move constructors should be created automatically, but as of
  // GCC4.7 move semantics only take place with explicitly defined ones.
  generalizedLinMap(const generalizedLinMap& cpy)
    : linearMap<GnrlzDomVect,GnrlzCoDomVect>::template implementation<generalizedLinMap<GnrlzDomVect,DomainVector,CoDomainVector,GnrlzCoDomVect>>
        (cpy)
    , spclf(cpy.spclf), multiplier(cpy.multiplier) {}
  generalizedLinMap(generalizedLinMap&& mov) noexcept
    : linearMap<GnrlzDomVect,GnrlzCoDomVect>::template implementation<generalizedLinMap<GnrlzDomVect,DomainVector,CoDomainVector,GnrlzCoDomVect>>
        (std::move(mov))
    , spclf(std::move(mov.spclf)), multiplier(mov.multiplier) {}
  
};

template<class GenVect, class SpecVect>auto
generalize(linearMap<SpecVect,SpecVect> spc)
    -> linearMap<GenVect,GenVect>           {
  return linearMap< GenVect
                  , GenVect >( generalizedLinMap< GenVect
                                                , SpecVect
                                                , SpecVect
                                                , GenVect  >(spc) );
}

template< class GDomVect, class DomVect
        , class CoDomVect, class GCoDomVect >auto
operator*( generalizedLinMap<GDomVect,DomVect,CoDomVect,GCoDomVect>&& l
         , linearMap<DomVect,CoDomVect>&& r                             )
   -> generalizedLinMap<GDomVect,DomVect,CoDomVect,GCoDomVect>            {
  return std::move(l.rcompose(std::move(r)));
}

/*
template<class DomVect,class CoDomVect>
template<class SpclzDomVect,class SpclzCoDomVect>auto
linearMap<     DomVect,      CoDomVect>::
operator+=(const linearMap<SpclzDomVect,SpclzCoDomVect>& addm) -> linearMap& {
  return *this
    += generalizedLinMap<DomVect,SpclzDomVect,SpclzCoDomVect,CoDomVct>(addm);
}
*/


template<class DomainVector, class CoDomainVector = DomainVector>
class linearMapSum : public linearMap<DomainVector,CoDomainVector>
                          ::template implementation<linearMapSum< DomainVector
                                                                , CoDomainVector> > {
                           
  typedef linearMap<DomainVector,CoDomainVector> LinearMap;
  using LinearMap::template implementation<linearMapSum>::linearMap::implement_override;
  typedef typename LinearMap::template implementation<linearMapSum>::linearMap::FieldT FieldT;
  typedef typename LinearMap::template implementation<linearMapSum>::linearMap::CoFieldT CoFieldT;
  typedef typename LinearMap::template implementation<linearMapSum>::linearMap::Domain Domain;
  typedef typename LinearMap::template implementation<linearMapSum>::linearMap::CoDomain CoDomain;
  
//  typedef std::unique_ptr<LinearMap> Implementor;
  
  std::vector<LinearMap> fns;

  void clear(){fns.clear();}
 public:
  auto supports_native_addition(const LinearMap& addc)const -> bool {
    return true;                                                    }

  auto operator()(const DomainVector& v)const -> CoDomainVector {
    if(implement_override) return (*implement_override)(v);
    if(fns.size()==0) return CoDomainVector(0);
    CoDomainVector acc = fns[0](v);
    for(unsigned i=1; i<fns.size(); ++i) fns[i].accumulate_apply(FieldT(1.), v, FieldT(1.), acc);
    return acc;
  }

  void accumulate_apply( const CoFieldT& alpha, const DomainVector& v
                       , const CoFieldT& beta,CoDomainVector& acc   )const{
    if(implement_override) {
      implement_override->accumulate_apply(alpha,v,beta,acc);
     }else{
      bool beta_was_applied=false;
      for(auto&f: fns) {
        if(beta_was_applied){
          f.accumulate_apply(alpha, v, FieldT(1.), acc);
         }else{
          f.accumulate_apply(alpha, v, beta, acc);
          beta_was_applied = true;
        }
      }
      if(!beta_was_applied)
        acc *= beta;
    }
  }

  LinearMap& operator+=(const LinearMap& addm){
    if(implement_override){
      if(implement_override->supports_native_addition(addm)) {
        (*implement_override) += addm;
        return *this;
       }else if(addm.supports_native_addition(*implement_override)) {
        return *this = addm + *implement_override;
      }
      fns.clear();
      fns.push_back(std::move(*implement_override.release()));
    }

    if(auto addmsp = dynamic_cast<const linearMapSum*>(&addm)) {
      for(auto& f: addmsp->fns) *this += f;
      return *this;
    }

    for(auto& f: fns) {
      if(f.supports_native_addition(addm)) {
        f += addm;
        return *this;
       }else if(addm.supports_native_addition(f)){
        f = addm + f;
        return *this;
      }
    }
    fns.push_back(addm);
    return *this;
  }
  LinearMap& operator+=(LinearMap&& addm){
    if(implement_override){
      if(implement_override->supports_native_addition(addm)) {
        (*implement_override) += std::move(addm);
        return *this;
       }else if(addm.supports_native_addition(*implement_override)) {
        return *this = std::move(addm += *implement_override);
      }
      fns.clear();
      fns.push_back(std::move(*implement_override.release()));
    }

    if(auto addmsp = dynamic_cast<linearMapSum*>(&addm)) {
      for(auto& f: addmsp->fns)
        *this += std::move(f);
      return *this;
    }

    for(auto& f: fns) {
      if(f.supports_native_addition(addm)) {
        f += std::move(addm);
        return *this;
       }else if(addm.supports_native_addition(f)){
        f = std::move(addm += f);
        return *this;
      }
    }
    fns.push_back(std::move(addm));
    return *this;
  }
  
  auto domain()const -> const Domain* {
    if (implement_override) return implement_override->domain();
    const Domain* result = nullptr;
    for(auto& f: fns)
      if(auto resd = f.domain()) {
        if(!result || resd==result) {
          result = resd;
         }else{
          std::cerr << "Linear maps with different domains in sum container."; abort();
        }
      }
    return result;
  }
  auto codomain()const -> const CoDomain* {
    if (implement_override) return implement_override->codomain();
    const Domain* result = nullptr;
    for(auto& f: fns)
      if(auto resd = f.codomain()) {
        if(!result || resd==result) {
          result = resd;
         }else{
          std::cerr << "Linear maps with different domains in sum container."; abort();
        }
      }
    return result;
  }

  linearMapSum() {}

  linearMapSum(const LinearMap& f) {
    if(auto fspt = dynamic_cast<const linearMapSum*>(f.implemented()))
      for(auto&fn: fspt->fns)
        fns.push_back(fn);
     else
      implement_override.reset(new LinearMap(f));
  }
  linearMapSum(LinearMap&& f) {
    if(auto fspt = dynamic_cast<linearMapSum*>(f.implemented()))
      for(auto&fn: fspt->fns)
        fns.push_back(std::move(fn));
     else
      implement_override.reset(f.moved());
  }

  linearMapSum(LinearMap f1, LinearMap f2) {
    if(f1.supports_native_addition(f2)) {
      *this = std::move(f1) + std::move(f2);
     }else if(auto fi=dynamic_cast<linearMapSum*>(f2.implemented())){
      fns = fi->fns;
      *this += std::move(f1);
     }else{
      fns = {f1, f2};
    }
  }

  template<class BObjCntInitT>
  linearMapSum(BObjCntInitT init) {
    for(auto&f: init)
      fns.push_back(std::move(f));
  }
};


template<class DomVect,class CoDomVect>auto
linearMap<     DomVect,      CoDomVect>::
operator+=(const linearMap& addm) -> linearMap& {
  if(implement_override && implement_override->supports_native_addition(addm)) {
    (*implement_override) += addm;
   }else if(addm.supports_native_addition(*this)){
    *this = addm + std::move(*this);
   }else{
    implement_override.reset(new linearMapSum<DomVect,CoDomVect>(std::move(*this), addm));
  }
  return *this;
}
template<class DomVect,class CoDomVect>auto
linearMap<     DomVect,      CoDomVect>::
operator+=(linearMap&& addm) -> linearMap& {
  if(implement_override && implement_override->supports_native_addition(addm)) {
    (*implement_override) += std::move(addm);
   }else if(addm.supports_native_addition(*this)){
    *this = std::move(addm += *this);
   }else{
    implement_override.reset( new linearMapSum<DomVect,CoDomVect>( std::move(*this)
                                                                 , std::move(addm)  ) );
  }
  return *this;
}


template<class DomVect,class CoDomVect>auto
linearMap<     DomVect,      CoDomVect>::
lcompose(const linearMap<CoDomVect,CoDomVect>& f) -> linearMap& {
  if (implement_override && implement_override->supports_native_lcomposition(f)) {
    implement_override->lcompose(f);
   }else{
    implement_override.reset(new linearMapComposition< DomVect
                                                     , CoDomVect
                                                     , CoDomVect >( f
                                                                  , std::move(*this) ) );
  }
  return *this;
}
template<class DomVect,class CoDomVect>auto
linearMap<     DomVect,      CoDomVect>::
lcompose(linearMap<CoDomVect,CoDomVect>&& f) -> linearMap& {

  if (implement_override && implement_override->supports_native_lcomposition(f)) {
    implement_override->lcompose(std::move(f));
   }else{
    implement_override.reset(new linearMapComposition< DomVect
                                                     , CoDomVect
                                                     , CoDomVect >( std::move(f)
                                                                  , std::move(*this) ) );
  }
  return *this;
}

template<class DomVect,class CoDomVect>auto
linearMap<     DomVect,      CoDomVect>::
operator*=(const linearMap<DomVect,DomVect>& f) -> linearMap& {
  if (implement_override && implement_override->supports_native_rcomposition(f)) {
    (*implement_override) *= f;
   }else{
    implement_override.reset(new linearMapComposition< DomVect
                                                     , DomVect
                                                     , CoDomVect >( std::move(*this)
                                                                  , f                ) );
  }
  return *this;
}
template<class DomVect,class CoDomVect>auto
linearMap<     DomVect,      CoDomVect>::
operator*=(linearMap<DomVect,DomVect>&& f) -> linearMap& {
  if (implement_override && implement_override->supports_native_rcomposition(f)) {
    (*implement_override) *= std::move(f);
   }else{
    implement_override.reset(new linearMapComposition< DomVect
                                                     , DomVect
                                                     , CoDomVect >( std::move(*this)
                                                                  , std::move(f)     ) );
  }
  return *this;
}



template<class CBO, class FldT>
template<class CoHilbertBaseObj>
class hilbertSpace<CBO,FldT>
::generalLinearMap: public linearMap< vector
                                    , typename hilbertSpace< CoHilbertBaseObj
                                                           , FldT             >::vector >
                     :: template implementation< generalLinearMap<CoHilbertBaseObj> > {
 public:
  typedef hilbertSpace Domain;
  typedef hilbertSpace<CoHilbertBaseObj, FieldT> CoDomain;
  typedef vector DomainVector;
  typedef typename CoDomain::vector CoDomainVector;
//  typedef typename Domain::FieldT FieldT;
  typedef typename CoDomain::FieldT CoFieldT;  //always ≡FieldT

  typedef linearMap<DomainVector,CoDomainVector> LinearMap;

 private:
  using linearMap<DomainVector,CoDomainVector>::template implementation<generalLinearMap>::linearMap::implement_override;

  const Domain* domainp;
  const CoDomain* codomainp;
  
  accelGeneralMatrix<FldT> components;
  CoFieldT multiplier;

  void clear(){components.clear();}
  
  generalLinearMap( const Domain& domainp
                  , const CoDomain& codomainp
                  , accelGeneralMatrix<FldT> components )
    : domainp(&domainp)
    , codomainp(&codomainp)
    , components( std::move(components) )
    , multiplier(1)
  {}
  

 public:
  void matricisize()const{
    if(implement_override)
      implement_override->matricisize();
  }

  auto supports_native_addition(const LinearMap& addc)const -> bool {
    return implement_override ? implement_override->supports_native_addition(addc)
                              : !!dynamic_cast<const generalLinearMap*>(addc.implemented());
                  //             ||typeid(*addc.implemented())==typeid(typename hilbertSpace<CBO,FldT>
                    //                                               ::sparseLinearMap                 );
  }
  auto supports_native_rcomposition(const linearMap<vector,vector>& cmf)const -> bool {
    return implement_override ? implement_override->supports_native_rcomposition(cmf)
                              : dynamic_cast<const generalLinearMap<CBO>*>(cmf.implemented())
                               ||dynamic_cast<const multiplierMap<vector,vector>*>(cmf.implemented());
  }
  auto supports_native_lcomposition(const linearMap<CoDomainVector,CoDomainVector>& cmf)const -> bool {
    return implement_override ? implement_override->supports_native_lcomposition(cmf)
                              : dynamic_cast<const typename CoDomain::template generalLinearMap<CoHilbertBaseObj>*>(cmf.implemented())
                               ||dynamic_cast<const multiplierMap<CoDomainVector,CoDomainVector>*>(cmf.implemented());
  }

  auto operator()(const DomainVector& v)const -> CoDomainVector {
    if (implement_override) return (*implement_override)(v);

    assert(v.domain == domainp);

    CoDomainVector result = domainp->uninitialized_vector();

    components.gemv( multiplier, v.components
                   , zero      , result.components );

    return result;
  }
  template<typename VctGenrlz> auto      //Operators on hilbertSpace::vectors
  operator()(const VctGenrlz& v)const   // should automatically generalize to
      -> VctGenrlz  {                  //  Operators on the corresponding vector
    return v.linear_mapped(*this);    //   generalizations. As of now, only
  }                                  //    works for Hilbert space endomorphisms
                                    //  (decltype, at least in gcc-4.6, is not
                                   //   powerful enough to deduce the return type)

  auto inv_appliedto(const CoDomainVector& v)const -> DomainVector {
    if(implement_override) return implement_override->inv_appliedto(v);

    assert(v.domain == codomainp);
    assert(domainp->dimension() == codomainp->dimension());
    CoDomainVector result;
    result.domain = codomainp;
    result.components = components.invmv( 1./multiplier
                                        , v.components  );
    return result;
  }

  void accumulate_apply( const CoFieldT& alpha, const DomainVector& v
                       , const CoFieldT& beta,CoDomainVector& acc   )const{
    if(implement_override){
      implement_override->accumulate_apply(alpha,v,beta,acc);
     }else{
      assert(v.domain == domainp && acc.domain == codomainp);
      components.gemv(alpha*multiplier, v.components, beta, acc.components);
    }
  }


  LinearMap& operator+=(const LinearMap& addm){
    if(implement_override){ (*implement_override)+=addm; return *this; }

    if(auto addmsp = dynamic_cast<const generalLinearMap*>(addm.implemented())) {
      assert(addmsp->domainp==domainp && addmsp->codomainp==codomainp);
      if(abs(multiplier) > 0) {
        if(abs(addmsp->multiplier) > 0){
          components.axpyze(addmsp->multiplier/multiplier, addmsp->components);}
       }else{
        components = addmsp->components;
        multiplier = addmsp->multiplier;
      }
     }else{
      implement_override.reset( new linearMapSum< DomainVector
                                                , CoDomainVector>( std::move(*this)
                                                                 , addm             ) );
    }
    return *this;
  }
/*  LinearMap& operator+=(LinearMap&& addm){
    if(implement_override){ (*implement_override)+=std::move(addm); return *this; }

    if(auto addmsp = dynamic_cast<sparseLinearMap*>(addm.implemented())) {
      assert(addmsp->domainp==domainp && addmsp->codomainp==codomainp);
      components += std::move(addmsp->components);
     }else{
      implement_override.reset(new linearMapSum<DomainVector,CoDomainVector>(*this, std::move(addm)));
      components.clear();
    }
    return *this;
  }*/

  LinearMap& lcompose(const linearMap<CoDomainVector,CoDomainVector>& f){
    if(implement_override){ implement_override->lcompose(f); return *this; }

    if(auto fmsp = dynamic_cast<const typename CoDomain::template generalLinearMap<CoHilbertBaseObj>*>(f.implemented())) {
      assert(fmsp->codomainp==domainp);
      if(abs(multiplier) > 0) {
        multiplier *= fmsp->multiplier;
        components = fmsp->components * components;
      }
     }else if(
      auto fmsp = dynamic_cast<const multiplierMap<CoDomainVector,CoDomainVector>*>(f.implemented())) {
      multiplier *= fmsp->multiplicator();
     }else{
      implement_override.reset( new linearMapComposition< vector, CoDomainVector
                                                        , CoDomainVector>( f
                                                                         , std::move(*this) ) );
    }
    return *this;
  }
  LinearMap& lcompose(linearMap<CoDomainVector,CoDomainVector>&& f){return lcompose(f);}

  LinearMap& operator*=(const linearMap<vector,vector>& f){
    if(implement_override){ (*implement_override)*=f; return *this; }

    if(auto fmsp = dynamic_cast<const generalLinearMap<CBO>*>(f.implemented())) {
      assert(fmsp->codomainp==domainp);
      if(abs(multiplier) > 0) {
        multiplier *= fmsp->multiplier;
        components = components * fmsp->components;
      }
     }else if(
      auto fmsp = dynamic_cast<const multiplierMap<vector,vector>*>(f.implemented())) {
      multiplier *= fmsp->multiplicator();
     }else{
      implement_override.reset( new linearMapComposition< vector, vector
                                                        , CoDomainVector>( std::move(*this)
                                                                         , f                ) );
    }
    return *this;
  }
  LinearMap& operator*=(linearMap<vector,vector>&& f){return *this += f;}
  
  generalLinearMap& operator*=(const CoFieldT& multiplicator) {
    multiplier *= multiplicator; return(*this);              }
  auto operator*(const CoFieldT& multiplicator)const -> LinearMap {
    return generalLinearMap(*this) *= multiplicator;               }

  auto domain()const -> const Domain* {
    return implement_override
                ? implement_override->domain()
                : domainp;
  }
  auto codomain()const -> const CoDomain* {
    return implement_override
                ? implement_override->codomain()
                : codomainp;
  }

  friend class hilbertSpace<CBO,FldT>;
  template<unsigned BlckSz>
  friend class hilbertSpace<CBO,FldT>::sparseLinearMap;
};


template<class CBO, class FldT>
template<class CoHilbertBaseObj>
class hilbertSpace<CBO,FldT>
::sparseLinearMap: public linearMap< vector
                                   , typename hilbertSpace< CoHilbertBaseObj
                                                          , FldT             >::vector >
                     :: template implementation< sparseLinearMap<CoHilbertBaseObj> > {
 public:
  typedef hilbertSpace Domain;
  typedef hilbertSpace<CoHilbertBaseObj, FieldT> CoDomain;
  typedef vector DomainVector;
  typedef typename CoDomain::vector CoDomainVector;
//  typedef typename Domain::FieldT FieldT;
  typedef typename CoDomain::FieldT CoFieldT;  //always ≡FieldT

  typedef linearMap<DomainVector,CoDomainVector> LinearMap;

 private:
  using linearMap<DomainVector,CoDomainVector>::template implementation<sparseLinearMap>::linearMap::implement_override;

  const Domain* domainp;
  const CoDomain* codomainp;
  
  accelSparseMatrix<FldT> components;
  CoFieldT multiplier;
  
  void clear(){components.clear();}

  sparseLinearMap( const Domain& domainp
                 , const CoDomain& codomainp
                 , typename accelSparseMatrix<FldT>::HostSparseMatrix components )
    : domainp(&domainp)
    , codomainp(&codomainp)
    , components( domainp.accelhandle
                , domainp.dimension(), codomainp.dimension()
                , std::move(components) )
    , multiplier(1)
  {}
  sparseLinearMap( const Domain& domainp
                 , const CoDomain& codomainp
                 , accelSparseMatrix<FldT> components )
    : domainp(&domainp)
    , codomainp(&codomainp)
    , components( std::move(components) )
    , multiplier(1)
  {}
  

 public:
  void matricisize()const{
    if(implement_override)
      implement_override->matricisize();
  }

  auto supports_native_addition(const LinearMap& addc)const -> bool {
    return implement_override ? implement_override->supports_native_addition(addc)
                              : dynamic_cast<const sparseLinearMap*>(addc.implemented())
                               || dynamic_cast<const multiplierMap<DomainVector,CoDomainVector>*>(addc.implemented());
  }
  auto supports_native_rcomposition(const linearMap<vector,vector>& cmf)const -> bool {
    return implement_override ? implement_override->supports_native_rcomposition(cmf)
                              : dynamic_cast<const sparseLinearMap<CBO>*>(cmf.implemented())
                               ||dynamic_cast<const multiplierMap<vector,vector>*>(cmf.implemented());
  }
  auto supports_native_lcomposition(const linearMap<CoDomainVector,CoDomainVector>& cmf)const -> bool {
    return implement_override ? implement_override->supports_native_lcomposition(cmf)
                              : dynamic_cast<const typename CoDomain::template sparseLinearMap<CoHilbertBaseObj>*>(cmf.implemented())
                               ||dynamic_cast<const multiplierMap<CoDomainVector,CoDomainVector>*>(cmf.implemented());
  }

  auto operator()(const DomainVector& v)const -> CoDomainVector {
    if (implement_override) return (*implement_override)(v);

    assert(v.domain == domainp);

    CoDomainVector result = domainp->uninitialized_vector();

    components.gemv( multiplier, v.components
                   , zero      , result.components );

    return result;
  }
  template<typename VctGenrlz> auto  
  operator()(const VctGenrlz& v)const
      -> VctGenrlz  {                
    return v.linear_mapped(*this);   
  }                                  

  auto inv_appliedto(const CoDomainVector& v)const -> DomainVector {
    if(implement_override) return implement_override->inv_appliedto(v);

    assert(v.domain == codomainp);
    assert(domainp->dimension() == codomainp->dimension());
    CoDomainVector result;
    result.domain = codomainp;
    result.components = components.invmv( 1./multiplier
                                        , v.components  );
    return result;
  }

  void accumulate_apply( const CoFieldT& alpha, const DomainVector& v
                       , const CoFieldT& beta,CoDomainVector& acc   )const{
    if(implement_override){
      implement_override->accumulate_apply(alpha,v,beta,acc);
     }else{
      assert(v.domain == domainp && acc.domain == codomainp);
      components.gemv(alpha*multiplier, v.components, beta, acc.components);
    }
  }


  LinearMap& operator+=(const LinearMap& addm){
    if(implement_override){ (*implement_override)+=addm; return *this; }

    if(auto addmsp = dynamic_cast<const sparseLinearMap*>(addm.implemented())) {
      assert(addmsp->domainp==domainp && addmsp->codomainp==codomainp);
      if(abs(multiplier) > 0) {
        if(abs(addmsp->multiplier) > 0)
          components.axpyze(addmsp->multiplier/multiplier, addmsp->components);
       }else{
        components = addmsp->components;
        multiplier = addmsp->multiplier;
      }
     }else if(auto addmsp
               = ( abs(multiplier) > 0 )
                 ? dynamic_cast<const multiplierMap< DomainVector
                                                   , CoDomainVector>*>
                                             (addm.implemented())
                 : nullptr                                              ) {
      assert(domainp == codomainp);
      if(abs(addmsp->multiplicator()) > 0)
        components += addmsp->multiplicator()/multiplier;
     }else{
      implement_override.reset( new linearMapSum< DomainVector
                                                , CoDomainVector>( std::move(*this)
                                                                 , addm             ) );
      components.clear();
    }
    return *this;
  }
  LinearMap& operator+=(LinearMap&& addm){return *this += addm;}

  LinearMap& lcompose(const linearMap<CoDomainVector,CoDomainVector>& f)   {
    if(implement_override){ implement_override->lcompose(f); return *this; }

    if(auto fmsp = dynamic_cast<const typename CoDomain::template sparseLinearMap<CoHilbertBaseObj>*>(f.implemented())) {
      assert(fmsp->codomainp==domainp);
      if(abs(multiplier) > 0) {
        multiplier *= fmsp->multiplier;
        components = fmsp->components * components;
      }
     }else if(
      auto fmsp = dynamic_cast<const multiplierMap<CoDomainVector,CoDomainVector>*>(f.implemented())) {
      multiplier *= fmsp->multiplicator();
     }else{
      implement_override.reset( new linearMapComposition< vector, vector
                                                        , CoDomainVector>( std::move(*this)
                                                                         , f                ) );
    }
    return *this;
  }
  LinearMap& lcompose(linearMap<CoDomainVector,CoDomainVector>&& f){return lcompose(f);}
  LinearMap& operator*=(const linearMap<vector,vector>& f){
    if(implement_override){ (*implement_override)*=f; return *this; }

    if(auto fmsp = dynamic_cast<const sparseLinearMap<CBO>*>(f.implemented())) {
      assert(fmsp->codomainp==domainp);
      if(abs(multiplier) > 0) {
        multiplier *= fmsp->multiplier;
        components = components * fmsp->components;
      }
     }else if(
      auto fmsp = dynamic_cast<const multiplierMap<vector,vector>*>(f.implemented())) {
      multiplier *= fmsp->multiplicator();
     }else{
      implement_override.reset( new linearMapComposition< vector, vector
                                                        , CoDomainVector>( std::move(*this)
                                                                         , f                ) );
    }
    return *this;
  }
  LinearMap& operator*=(linearMap<vector,vector>&& f){return *this *= f;}
  
  sparseLinearMap& operator*=(const CoFieldT& multiplicator) {
    multiplier *= multiplicator; return(*this);              }
  auto operator*(const CoFieldT& multiplicator)const -> LinearMap {
    return sparseLinearMap(*this) *= multiplicator;               }

  auto inverse()const -> linearMap<CoDomainVector,DomainVector> {
    assert(domainp->dimension() == codomainp->dimension());
    if(implement_override) return implement_override->inverse();
    
    return hilbertSpace<CBO,FldT>::generalLinearMap<CoHilbertBaseObj>
                                      ( *domainp
                                      , *codomainp
                                      , components.scaled_inverse(multiplier) );
  }

  auto domain()const -> const Domain* {
    return implement_override
                ? implement_override->domain()
                : domainp;
  }
  auto codomain()const -> const CoDomain* {
    return implement_override
                ? implement_override->codomain()
                : codomainp;
  }

  friend class hilbertSpace<CBO,FldT>;
  template<unsigned BlckSz>
  friend class hilbertSpace<CBO,FldT>::staticBlocksdiagAccessor;
};


template<class CBO, class FldT>template<class CBOt, class EVHandler>auto
hilbertSpace < CBO ,      FldT>::
eigenbasis_gen(const linearMap<vector,vector>& f, EVHandler evhandler)const
     -> hilbertSpace<CBOt, FieldT> {
  f.matricisize();

  auto eigenbasistransf = [&]() -> accelEigenbasisTransform<FieldT> {
    if(auto dense = dynamic_cast<const generalLinearMap<CBO>*>(f.implemented()))
      return accelEigenbasisTransform<FieldT>(dense->components);
     else if(auto sparse = dynamic_cast<const sparseLinearMap<CBO>*>(f.implemented()))
      return accelEigenbasisTransform<FieldT>(sparse->components);
     else
      std::cerr << "Tried to obtain the eigenbasis of an operator that would not assume a matrix representation\n";
      std::cerr << "(Type "<<typeid(*f.implemented()).name()<<"\n";
      abort();
  }();
  hilbertSpace<CBOt, FieldT> result(evhandler(eigenbasistransf));
  sharedBasisTransform<FieldT> transf;
  transf.transfm = std::make_shared< generalHilbertBasisTransform<FieldT>
                                   > ( static_cast<const generalHilbertBasis<FldT>*>(this)
                                     , static_cast<const generalHilbertBasis<FldT>*>(&result)
                                     , std::move(eigenbasistransf)
                                     );

  link_isohilbertbaseis(*this,result,transf);

  return result;
}



/*
template<class DomVect, class CoDomVect>auto
linearMap<DomVect, CoDomVect>::
eigenbasis()const -> hilbertSpace<genericEigenbasisObj<FieldT>,FieldT> {
  std::cerr << "Tried to obtain the eigenbasis of a non-endomorphism linear map.\n";
  abort();
}
template<class DomVect>auto
linearMap<DomVect, DomVect>::
eigenbasis()const -> hilbertSpace< genericEigenbasisObj<typename DomVect::DomainFieldT>
                                 , typename DomVect::DomainFieldT                       > {
  typedef typename DomVect::DomainFieldT FldT;
  if(!domain()) {
    std::cerr << "Tried to obtain the eigenbasis of an operator not defined on a hillbertSpace object.\n";
    abort();
  }
  return domain()->template eigenbasis_gen< genericEigenbasisObj<FldT>
                                          , defaultEigenvalueHandler<FldT> >
                     (*this, defaultEigenvalueHandler<FldT>());
}
*/

template<class DomVect>
struct eigenbasisFactory {
  typedef typename DomVect::DomainFieldT FieldT;

  struct eigenvalueHandler {
    auto operator()(const accelEigenbasisTransform<FieldT>& eigtransf)const
            -> std::vector<genericEigenbasisObj<FieldT>> {
      std::vector<genericEigenbasisObj<FieldT>> result;
      for(auto& eigv: eigtransf.real_eigenvalues())
        result.push_back(genericEigenbasisObj<FieldT>(eigv));
      return result;
    }
  };

  static auto
  eigenbasis(const linearMap<DomVect, DomVect>& op)
     -> hilbertSpace< genericEigenbasisObj<FieldT>, FieldT > {
    if(!op.domain()) {
      std::cerr << "Tried to obtain the eigenbasis of an operator not defined on a hillbertSpace object.\n";
      abort();
    }
    return op.domain()->template eigenbasis_gen< genericEigenbasisObj<FieldT>
                                               , eigenvalueHandler            >
                           (op, eigenvalueHandler());
  }
};

template<class DomVect>auto
eigenbasis(const linearMap<DomVect, DomVect>& op)
   -> decltype(eigenbasisFactory<DomVect>::eigenbasis(op)) {
  return eigenbasisFactory<DomVect>::eigenbasis(op);
}




/*template<class CBO, class FldT, class CoHBO> auto
operator*( typename hilbertSpace<CBO,FldT>::template sparseLinearMap<CoHBO> m
         , const FldT& multiplicator                                 )
              -> typename hilbertSpace<CBO,FldT>::template sparseLinearMap<CoHBO> {
  return m *= multiplicator;
}*/





template< class DomainVector
        , class InterDomainVector = DomainVector
        , class CoDomainVector = InterDomainVector >
class linearMapComposition : public linearMap<DomainVector,CoDomainVector>
                                  ::template implementation< linearMapComposition< DomainVector
                                                                                 , InterDomainVector
                                                                                 , CoDomainVector >
                                                                                 > {
                           
  typedef linearMap<DomainVector,CoDomainVector> LinearMap;
  typedef linearMap<DomainVector,InterDomainVector> RLinearMap;
  typedef linearMap<InterDomainVector,CoDomainVector> LLinearMap;
  using LinearMap::template implementation<linearMapComposition>::linearMap::implement_override;
  
  typedef typename LinearMap::template implementation<linearMapComposition>::linearMap::Domain Domain;
  typedef typename LinearMap::template implementation<linearMapComposition>::linearMap::CoDomain CoDomain;
  
  maybe<LLinearMap> f1;
  maybe<RLinearMap> f2;
  
  void clear(){f1=nothing; f2=nothing;}

 public:
  auto operator()(const DomainVector& v)const -> CoDomainVector {
    return implement_override ? (*implement_override)(v)
                       : (*f1)((*f2)(v));
  }

  auto domain()const -> const Domain* {
    return implement_override
                ? implement_override->domain()
                : (*f1).domain();
  }
  auto codomain()const -> const CoDomain* {
    return implement_override
                ? implement_override->codomain()
                : (*f2).codomain();
  }

  linearMapComposition(LLinearMap f1, RLinearMap f2)
    : f1(just(std::move(f1))), f2(just(std::move(f2))) {}
};



template< class DomainVector
        , class InterDomainVector
        , class CoDomainVector    >
struct linearMapCompositor {
  static auto
  multiply( linearMap<InterDomainVector,CoDomainVector>& f1
          , linearMap<DomainVector,InterDomainVector>& f2   )
               -> linearMap<DomainVector,CoDomainVector>      {
    return linearMap<DomainVector,CoDomainVector>(
              linearMapComposition<DomainVector,InterDomainVector,CoDomainVector>
                   (std::move(f1), std::move(f2))      );
  }
};

template< class DomainVector
        , class CoDomainVector    >
struct linearMapCompositor<DomainVector,DomainVector,CoDomainVector> {
  static auto
  multiply( linearMap<DomainVector,CoDomainVector>& f1
          , linearMap<DomainVector,DomainVector>& f2   )
               -> linearMap<DomainVector,CoDomainVector>      {
    return f1 *= std::move(f2);
  }
};

template< class DomainVector
        , class CoDomainVector    >
struct linearMapCompositor<DomainVector,CoDomainVector,CoDomainVector> {
  static auto
  multiply( linearMap<CoDomainVector,CoDomainVector>& f1
          , linearMap<DomainVector,CoDomainVector>& f2   )
               -> linearMap<DomainVector,CoDomainVector>      {
    return f2.lcompose(std::move(f1));
  }
};

template< class DomainVector >
struct linearMapCompositor<DomainVector,DomainVector,DomainVector> {
  static auto
  multiply( linearMap<DomainVector,DomainVector>& f1
          , linearMap<DomainVector,DomainVector>& f2   )
               -> linearMap<DomainVector,DomainVector>      {
    return f1.supports_native_rcomposition(f2)
              ? f1 *= std::move(f2)
              : f2.lcompose(std::move(f1));
  }
};


template< class DomainVector
        , class InterDomainVector
        , class CoDomainVector    >auto
operator*( const linearMap<InterDomainVector,CoDomainVector>& f1
         , const linearMap<DomainVector,InterDomainVector>& f2   )
    -> linearMap<DomainVector,CoDomainVector> {
  return linearMapCompositor<DomainVector,InterDomainVector,CoDomainVector>
             ::multiply( linearMap<InterDomainVector,CoDomainVector>(f1)
                       , linearMap<InterDomainVector,CoDomainVector>(f2) );
}
template< class DomainVector
        , class InterDomainVector
        , class CoDomainVector    >auto
operator*( linearMap<InterDomainVector,CoDomainVector>&& f1
         , const linearMap<DomainVector,InterDomainVector>& f2   )
    -> linearMap<DomainVector,CoDomainVector> {
  return linearMapCompositor<DomainVector,InterDomainVector,CoDomainVector>
             ::multiply( f1
                       , linearMap<InterDomainVector,CoDomainVector>(f2) );
}
template< class DomainVector
        , class InterDomainVector
        , class CoDomainVector    >auto
operator*( const linearMap<InterDomainVector,CoDomainVector>& f1
         , linearMap<DomainVector,InterDomainVector>&& f2   )
    -> linearMap<DomainVector,CoDomainVector> {
  return linearMapCompositor<DomainVector,InterDomainVector,CoDomainVector>
             ::multiply( linearMap<InterDomainVector,CoDomainVector>(f1)
                       , f2                                              );
}
template< class DomainVector
        , class InterDomainVector
        , class CoDomainVector    >auto
operator*( linearMap<InterDomainVector,CoDomainVector>&& f1
         , linearMap<DomainVector,InterDomainVector>&& f2   )
    -> linearMap<DomainVector,CoDomainVector> {
  return linearMapCompositor<DomainVector,InterDomainVector,CoDomainVector>
             ::multiply(f1,f2);
}
/*
template< class DomainVector
        , class CoDomainVector >auto
operator*( linearMap<DomainVector,CoDomainVector> f1
         , linearMap<DomainVector,DomainVector> f2   )
    -> linearMap<DomainVector,CoDomainVector> {
  return f1*=std::move(f2);
}

template< class DomainVector
        , class CoDomainVector >auto
operator*( linearMap<CoDomainVector,CoDomainVector> f1
         , linearMap<DomainVector,CoDomainVector> f2   )
    -> linearMap<DomainVector,CoDomainVector> {
  return f2.lcompose(std::move(f1));
}
*/

template< class DomainVector
        , class InterDomainVector = DomainVector
        , class CoDomainVector = InterDomainVector >auto
operator/( linearMap<InterDomainVector,CoDomainVector> f1
         , linearMap<InterDomainVector,DomainVector> f2   )
    -> linearMapComposition<DomainVector,InterDomainVector,CoDomainVector> {
  return linearMapComposition<DomainVector,InterDomainVector,CoDomainVector>
            (std::move(f1), f2.inverse());
}

template< class DomainVector
        , class InterDomainVector
        , class Multiplier >auto
operator*( const Multiplier& a
         , linearMap<DomainVector,InterDomainVector> f2   )
    -> linearMapComposition< DomainVector
                           , InterDomainVector
                           , decltype(a * std::declval<InterDomainVector>())> {
  return linearMapComposition<DomainVector,InterDomainVector,decltype(a*InterDomainVector())>(
            linearLambda<InterDomainVector,decltype(a*InterDomainVector())>(
              [=](const InterDomainVector& v){return a*v;}                 )
            , std::move(f2)
         );
}

template< class DomainVector
        , class CoDomainVector
        , class ExtFn >auto
operator*=( linearMap<DomainVector,CoDomainVector>& f0
          , const ExtFn& f1                            )
              -> linearMap<DomainVector,CoDomainVector>&   {
  return f0 = linearMap<DomainVector,CoDomainVector>(
            linearMapComposition<DomainVector,DomainVector,CoDomainVector>
                  ( f0, linearLambda<DomainVector,DomainVector>(f1) )       );
}


template<class DomainVector>auto
operator+(int i, linearMap<DomainVector,DomainVector> f0)
            -> linearMap<DomainVector,DomainVector>        {
  return f0 += (double)i;
}
template<class DomainVector>auto
operator-(int i, linearMap<DomainVector,DomainVector> f0)
            -> linearMap<DomainVector,DomainVector>        {
  f0 *= -1.;
  return f0 += (double)i;
}



template< class DomainVector, class Multiplier>auto
multiplier(const Multiplier& mul)
    -> multiplierMap<DomainVector,DomainVector> {
  return multiplierMap<DomainVector,DomainVector>(mul);
}






}


#endif