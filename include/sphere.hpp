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


#ifndef SPHERICAL_COORDINATE_OBJECTS
#define SPHERICAL_COORDINATE_OBJECTS

#include<cassert>
#include<random>
#include<array>
#include <functional>



template<unsigned SphereDim, typename CoordType = double>
class sphericalUniformDistribution {
  mutable std::normal_distribution<CoordType> rd;
  typedef std::array<CoordType, SphereDim+1> Cartesian;
  CoordType r;
 public:
  sphericalUniformDistribution( const CoordType& radius=1 )
    : r(radius)   {}
  
  template<class RandomGen>
  auto operator()(RandomGen& rg) -> Cartesian {
    Cartesian result; CoordType rr = 0;
    for(auto& c: result) {
      c = rd(rg);
      rr += c*c;
    }
    rr = r/sqrt(rr);
    for(auto& c: result)
      c *= rr;
    return result;
  }
};

#endif