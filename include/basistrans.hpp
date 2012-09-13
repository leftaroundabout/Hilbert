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


#ifndef BASE_TRANSFORMATIONS_BETWEEN_BASEIS_OF_SUB_AND_SUPERSPACES
#define BASE_TRANSFORMATIONS_BETWEEN_BASEIS_OF_SUB_AND_SUPERSPACES

#include "hilbertsp.hpp"
/*
#include "linearmap.hpp"
*/


#include<memory>
#include<cassert>
#include<unordered_set>


#include "acceldvector.hpp"


namespace hilbert {



/*template<class CBO, class FieldT>
class hilbertSpace;*/

template<class FieldT>             
class generalHilbertBasisTransform;
template<class FieldT>             
class sharedBasisTransform;        

template<class FieldT>
class generalHilbertBasis {
 protected:
  friend class generalHilbertBasisTransform<FieldT>;
  mutable std::unordered_set<sharedBasisTransform<FieldT>> subspaces     //if subspaces change, it
                                                         , superspaces; // is not modifying this space

  void remove_relationreference(const generalHilbertBasis* oldrf)const;
  void migrate_relationreference( const generalHilbertBasis* oldrf
                                , const generalHilbertBasis* newrf )const;
  generalHilbertBasis() {}
  generalHilbertBasis(const generalHilbertBasis&) =delete;
  generalHilbertBasis(generalHilbertBasis&&) noexcept;
  generalHilbertBasis& operator=(generalHilbertBasis);
  ~generalHilbertBasis();

  template<class CBO, class FldT>
  friend class hilbertSpace;
  template<class FldT>
  friend void link_isohilbertbaseis(const generalHilbertBasis<FldT>&,const generalHilbertBasis<FldT>&,const sharedBasisTransform<FldT>&);
};


using namespace acceleratedVectors;

template<class FieldT>
class generalHilbertBasisTransform {
  typedef const generalHilbertBasis<FieldT>* BasisRef;
  BasisRef basis1, basis2;

  accelEigenbasisTransform<FieldT> transform; //note that every basis transform is an
                                             // eigenbasis transform of some operator.
                                            //  basis2 is always seen as the eigenbasis.
 public:
#if 0
  template<class CBO> auto
  is_basis1(const hilbertSpace<CBO,FieldT>& basiscndt)const -> bool {
    return static_cast<generalHilbertBasis*>(&basiscndt) == basis1; }
  template<class CBO> auto
  is_basis2(const hilbertSpace<CBO,FieldT>& basiscndt)const -> bool {
    return static_cast<generalHilbertBasis*>(&basiscndt) == basis2; }

  template<class CBO> void
  migrate_basis1(const hilbertSpace<CBO,FieldT>* basisnewloc) {
    basis1 = static_cast<generalHilbertBasis*>(basisnewloc);  }
  template<class CBO> void
  migrate_basis2(const hilbertSpace<CBO,FieldT>* basisnewloc) {
    basis2 = static_cast<generalHilbertBasis*>(basisnewloc);  }
#endif
  template<class CBO1, class CBO2> auto
  transform_B1toB2(const typename hilbertSpace<CBO1,FieldT>::vector&)const
                      -> typename hilbertSpace<CBO2,FieldT>::vector  ;
  template<class CBO2, class CBO1> auto
  transform_B2toB1(const typename hilbertSpace<CBO2,FieldT>::vector&)const
                      -> typename hilbertSpace<CBO1,FieldT>::vector  ;
  
  auto hash_B1()const -> size_t { return std::hash<BasisRef>()(basis1); }
  auto hash_B2()const -> size_t { return std::hash<BasisRef>()(basis2); }
  
  generalHilbertBasisTransform( BasisRef basis1
                              , BasisRef basis2
                              , accelEigenbasisTransform<FieldT> transform )
    : basis1(basis1)
    , basis2(basis2)
    , transform(std::move(transform))
  {}

  friend class sharedBasisTransform<FieldT>;
};

template<class FldT>template<class CBO1, class CBO2>auto generalHilbertBasisTransform<FldT>
::transform_B1toB2(const typename hilbertSpace<CBO1,FldT>::vector& v)const
                      -> typename hilbertSpace<CBO2,FldT>::vector          {
  typename hilbertSpace<CBO2,FldT>::vector result;
  assert(v.domain == (static_cast<const hilbertSpace<CBO1,FldT>*>(basis1)));
  result.domain = static_cast<const hilbertSpace<CBO2,FldT>*>(basis2);
  result.components = transform.to_eigenbasis(v.components);
  return result;
}
template<class FldT>template<class CBO2, class CBO1>auto generalHilbertBasisTransform<FldT>
::transform_B2toB1(const typename hilbertSpace<CBO2,FldT>::vector& v)const
                      -> typename hilbertSpace<CBO1,FldT>::vector          {
  typename hilbertSpace<CBO1,FldT>::vector result;
  assert(v.domain == (static_cast<const hilbertSpace<CBO2,FldT>*>(basis2)));
  result.domain = static_cast<const hilbertSpace<CBO1,FldT>*>(basis1);
  result.components = transform.from_eigenbasis(v.components);
  return result;
}




template<class FieldT>
class sharedBasisTransform {
  typedef const generalHilbertBasis<FieldT>* BasisRef;

  enum class transformDirection { forth
                                , back  };
  transformDirection dir;
  std::shared_ptr<generalHilbertBasisTransform<FieldT>> transfm;
 
  virtual auto
  hashval()const -> size_t                          {
    return transfm->hash_B2() ^ transfm->hash_B1(); }
  friend class std::hash<sharedBasisTransform>;
  
 public:
  template<class CBO, class tCBO> auto
  transform(const typename hilbertSpace<CBO,FieldT>::vector& v)const
               -> typename hilbertSpace<tCBO,FieldT>::vector      {switch(dir) {
    case transformDirection::forth :
       return transfm->template transform_B1toB2<CBO,tCBO>(v);                 default/*
    case transformDirection::back*/:
       return transfm->template transform_B2toB1<CBO,tCBO>(v);
  }}

  virtual auto
  this_basis()const -> BasisRef { switch(dir) {
    case transformDirection::forth : return transfm->basis1;                   default/*
    case transformDirection::back*/: return transfm->basis2;
  }}
  virtual auto
  this_basis() -> BasisRef& { switch(dir) {
    case transformDirection::forth : return transfm->basis1;                   default/*
    case transformDirection::back*/: return transfm->basis2;
  }}
  virtual auto
  other_basis()const -> BasisRef { switch(dir) {
    case transformDirection::forth : return transfm->basis2;                   default/*
    case transformDirection::back*/: return transfm->basis1;
  }}
  virtual auto
  other_basis() -> BasisRef& { switch(dir) {
    case transformDirection::forth : return transfm->basis2;                   default/*
    case transformDirection::back*/: return transfm->basis1;
  }}

  auto operator==(const sharedBasisTransform& orf)const -> bool {
    return this_basis()==orf.this_basis() && other_basis()==orf.other_basis();
  }
  
  auto inverse()const -> sharedBasisTransform {
    sharedBasisTransform result(*this);
    result.dir = result.dir==transformDirection::back
                     ? transformDirection::forth
                     : transformDirection::back;
    return result;
  }

 protected:
  sharedBasisTransform() : dir(transformDirection::forth) {}

 public:
  friend class generalHilbertBasis<FieldT>;
  template<class CBO, class FldT>
  friend class hilbertSpace;
  /*template<class CBO1, class CBO2>
  friend class hilbertSpace<CBO1, FieldT>::generalLinearMap<CBO2>;
  template<class CBO1, class CBO2>
  friend class hilbertSpace<CBO1, FieldT>::sparseLinearMap<CBO2>;*/
};

}namespace std { template<class FieldT>class
hash<hilbert::sharedBasisTransform<FieldT>>{ public: auto
  operator()(const hilbert::sharedBasisTransform<FieldT>& rf)const -> size_t {
    return rf.hashval();                                                         }
};}namespace hilbert {


template<class FldT>
class lookupdummyShareBasTransf : public sharedBasisTransform<FldT> {
  typedef const generalHilbertBasis<FldT>* BasisRef;

  BasisRef thisbas, otherbas;
  
  auto hashval()const -> size_t                                              {
    return std::hash<BasisRef>()(thisbas) ^ std::hash<BasisRef>()(otherbas); }
 public:  
  auto this_basis()const -> BasisRef { return thisbas; }
  auto other_basis()const -> BasisRef { return otherbas; }
  auto this_basis() -> BasisRef& { return thisbas; }
  auto other_basis() -> BasisRef& { return otherbas; }
  
  lookupdummyShareBasTransf( BasisRef thisbas
                           , BasisRef otherbas )
    : thisbas(thisbas), otherbas(otherbas)      {}
};

/*template<class FldT, class CBO1, class CBO2>
auto shareBasTransf_lookupdummy( const hilbertSpace<CBO1,FldT>& thisbas
                               , const hilbertSpace<CBO1,FldT>& otherbas ) {
  return lookupdummyShareBasTransf<FldT>(&thisbas, &otherbas);
}*/


template<class FldT>void generalHilbertBasis<FldT>::
remove_relationreference(const generalHilbertBasis* oldrf)const {
  lookupdummyShareBasTransf<FldT> lkdummy(this, oldrf);
  subspaces.erase(lkdummy);
  superspaces.erase(lkdummy);
}

template<class FldT>void generalHilbertBasis<FldT>::
migrate_relationreference( const generalHilbertBasis* oldrf
                         , const generalHilbertBasis* newrf )const {
  lookupdummyShareBasTransf<FldT> lkdummy(this, oldrf);
  auto oldp = subspaces.find(lkdummy);
  if(oldp!=subspaces.end()) {
    auto migrated = *oldp; subspaces.erase(oldp);
    migrated.other_basis() = newrf;
    subspaces.insert(migrated);
  }
  oldp = superspaces.find(lkdummy);
  if(oldp!=superspaces.end()) {
    auto migrated = *oldp; superspaces.erase(oldp);
    migrated.other_basis() = newrf;
    superspaces.insert(migrated);
  }
}

template<class FldT>generalHilbertBasis<FldT>::
generalHilbertBasis(generalHilbertBasis&& cpy) noexcept{
  for(auto& sp: cpy.subspaces) {
    sp.other_basis() -> migrate_relationreference(&cpy, this);
    subspaces.insert(sp);
  }
  for(auto& sp: cpy.superspaces) {
    sp.other_basis() -> migrate_relationreference(&cpy, this);
    superspaces.insert(sp);
  }
}

template<class FldT>generalHilbertBasis<FldT>::
~generalHilbertBasis() {
  for(auto& sp: subspaces)
    sp.other_basis() -> remove_relationreference(this);
  for(auto& sp: superspaces)
    sp.other_basis() -> remove_relationreference(this);
}



template<class FldT>
void link_isohilbertbaseis( const generalHilbertBasis<FldT>& b1
                          , const generalHilbertBasis<FldT>& b2
                          , const sharedBasisTransform<FldT>& transfm ) {
  b1.subspaces.insert(transfm);
  b1.superspaces.insert(transfm);
  b2.subspaces.insert(transfm.inverse());
  b2.superspaces.insert(transfm.inverse());
}



template<class CBO, class FldT>template<class oCBO>auto
hilbertSpace<  CBO,       FldT>::vector
::to_basis(const hilbertSpace<oCBO,FldT>& tgb)const
           -> lambdalike::maybe<typename hilbertSpace<oCBO,FldT>::vector> {
  if(reinterpret_cast<const hilbertSpace*>(&tgb) == domain)
    return just(*(reinterpret_cast<const typename hilbertSpace<oCBO,FldT>::vector*>(this)));
  lookupdummyShareBasTransf<FldT> lkdummy(domain, &tgb);
  auto lookup = domain->superspaces.find(lkdummy);
  if(lookup == domain->superspaces.end()) return lambdalike::nothing;
  return lambdalike::just(lookup->template transform<CBO,oCBO>(*this));
}



#if 0
template<class FieldT, class HBO1, class HBO2>
class hilbertBasisTransform : generalHilbertBasisTransform<FieldT> {
  typedef hilbertSpace<HBO1, FieldT> Basis1; Basis1* basis1;
  typedef typename Basis1::vector B1vector;
  typedef hilbertSpace<HBO2, FieldT> Basis2; Basis2* basis2;
  typedef typename Basis2::vector B2vector;
  using generalHilbertBasisTransform<FieldT>::AnonymVector;

  accelEigenbasisTransform<FieldT> transform; //note that every basis transform is an
                                             // eigenbasis transform of some operator.
                                            //  basis2 is always seen as the eigenbasis.
/*  template<class HBO>struct basisObj{};
  template<>struct basisObj<HBO1>{typedef HBO2 other;};
  template<>struct basisObj<HBO2>{typedef HBO1 other;};*/

 public:                                            
  auto transform_B1toB2(const B1vector&) -> B2vector;
  auto transform_B1toB2(AnonymVector v) -> AnonymVector{
    return static_cast<AnonymVector>(new B2vector(transform_B1toB2(
                               *static_cast<const B1vector*>(v)   ) );
  }
  auto transform_B2toB1(const B2vector&) -> B1vector;
  auto transform_B2toB1(AnonymVector v) -> AnonymVector{
    return static_cast<AnonymVector>(new B1vector(transform_B2toB1(
                               *static_cast<const B2vector*>(v)   ) );
  }
  
  hilbertBasisTransform( Basis2& new_eigendomain
                       , const typename Basis1::template sparseLinearMap<HBO1>& op )
    : basis1(op.domain)
    , basis2(new_eigendomain)
    , transform(op.components)
  {}
};

template<class FldT, class HBO1, class HBO2>auto
hilbertBasisTransform::transform_B1toB2(const B1vector& v) -> B2vector {
  B2vector result;
  result.domain = basis2;
  result.components = transform.to_eigenbasis(v.components);
}
template<class FldT, class HBO2, class HBO1>auto
hilbertBasisTransform::transform_B2toB1(const B2vector& v) -> B1vector {
  B1vector result;
  result.domain = basis1;
  result.components = transform.from_eigenbasis(v.components);
}
#endif


/*
class generalHilbertBasis {
 public:
  virtual auto
  operator==()
  template<class AlternativeImplement>
  struct basisAlternative : hilbertSpace<HBO,FldT>::generalizedBasisRef {
  };
};

template<class FieldT, class HBO1, class HBO2>
class sharedBasisTransform {
  enum class transformDirection { forth
                                , back  };
  transformDirection dir;
  hilbertBasisTransform<FieldT,HBO1,HBO2>* forth;
  
};
*/


}

#endif