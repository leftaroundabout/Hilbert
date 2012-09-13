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


#ifndef COMPLEX_RANDOM_NUMBERS
#define COMPLEX_RANDOM_NUMBERS

#include<complex>
#include<cassert>
#include<random>
#include <functional>


template<typename RealType = double>      //Simple creation of random complex numbers
class simpleComplexNormalDistribution {  // as x+iy of normally-distributed variables x,y
  mutable std::normal_distribution<RealType> rd;
  typedef std::complex<RealType> ComplexT;
  ComplexT mean;
 public:
  simpleComplexNormalDistribution( const ComplexT& mean, RealType stddev )
    : rd(0, stddev)
    , mean(mean)     {}
  simpleComplexNormalDistribution( RealType stddev = 1 )
    : rd(0, stddev)
    , mean(ComplexT(0,0))     {}
  
  template<class RandomGen>
  auto operator()(RandomGen& rg) -> ComplexT {
    return mean + ComplexT(rd(rg), rd(rg));
  }
};




#endif