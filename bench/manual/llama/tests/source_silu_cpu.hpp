#pragma once
// Independent arithmetic replay of source silu_internal, not true Torch SiLU.
// Complex inputs retain the per-g3/f3 real projections and complex P/Q/w path.
#include <array>
#include <complex>
namespace silu_cpu_reference {
using Z=std::complex<long double>;
inline Z odd_project(Z x,const std::array<double,4>&c){Z sum=0.L,p=x;const Z x2=x*x;for(double a:c){sum+=static_cast<long double>(a)*p;p*=x2;}return {sum.real(),0};}
inline Z sign(Z x){constexpr std::array<double,4> g{4589./1024.,-16577./1024.,25614./1024.,-12860./1024.},f{35./16.,-35./16.,21./16.,-5./16.};return odd_project(odd_project(odd_project(odd_project(x,g),g),f),f);}
inline Z apply(Z v,Z w){const Z vr=v/50.L,s0=sign(vr+8.L/50),s1=sign(vr+4.L/50),s2=sign(vr-4.L/50),v2=v*v,v4=v2*v2,v6=v2*v4;const Z P=(static_cast<long double>(-.0055465625580307)*v2+static_cast<long double>(-.0819767021525476)*v+static_cast<long double>(-.3067541139982155))/2.L;const Z Q=(static_cast<long double>(.0002743776353465)*v6+static_cast<long double>(-.011113046708173)*v4+static_cast<long double>(.2281430841728270)*v2+.5L*v+static_cast<long double>(.0085064025895951))/2.L;return w*((s0-s1)*P+(s1-s2)*Q+(s2+1.L)*v/2.L);}
}
