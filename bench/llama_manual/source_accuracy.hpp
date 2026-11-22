#pragma once
// Post-decryption aggregate only. Never used to select or alter encrypted work.
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace cerium_source {
struct ComplexReferenceSummary {
    std::size_t active_entries=0;
    double maximum_real_error=0,maximum_complex_error=0;
    double maximum_imaginary=0,maximum_absolute_output=0,rmse=0;
    bool passed(double threshold=1e-3) const {
        if(!active_entries)throw std::runtime_error("model precision requires active entries");
        if(!std::isfinite(threshold)||threshold<0)throw std::invalid_argument("invalid complex accuracy threshold");
        return maximum_complex_error<=threshold;
    }
};
class ComplexReferenceMetric {
    ComplexReferenceSummary result_;
    // Scaled sum of squares: avoid double d*d overflow even if the final
    // real-component RMSE is representable (e.g. a finite error of1e308).
    long double squared_scale_=0,normalized_squared_=0;
    static double finite_double(long double v) {
        if(!std::isfinite(v)||v>std::numeric_limits<double>::max()||v<0)
            throw std::runtime_error("nonfinite/unrepresentable model accuracy aggregate");
        const double d=static_cast<double>(v);
        if(!std::isfinite(d))throw std::runtime_error("nonfinite model accuracy aggregate narrowing");
        return d;
    }
public:
    void observe(std::complex<double> value,double reference,bool active) {
        observe(value,std::complex<double>(reference,0),active);
    }
    void observe(std::complex<double> value,std::complex<double> reference,bool active) {
        if(!std::isfinite(value.real())||!std::isfinite(value.imag())||!std::isfinite(reference.real())||!std::isfinite(reference.imag()))
            throw std::runtime_error("nonfinite final model value/reference");
        result_.maximum_imaginary=std::max(result_.maximum_imaginary,std::abs(value.imag()));
        result_.maximum_absolute_output=std::max(result_.maximum_absolute_output,std::abs(value.real()));
        if(!active)return;
        const long double dr=static_cast<long double>(value.real())-reference.real();
        const long double di=static_cast<long double>(value.imag())-reference.imag();
        const long double magnitude=std::hypot(dr,di);
        const double real=finite_double(std::abs(dr)),complex=finite_double(magnitude);
        result_.maximum_real_error=std::max(result_.maximum_real_error,real);
        result_.maximum_complex_error=std::max(result_.maximum_complex_error,complex);
        if(result_.active_entries==std::numeric_limits<std::size_t>::max())
            throw std::overflow_error("model metric count overflow");
        ++result_.active_entries;
        const auto a=std::abs(dr);
        if(a>0) {
            if(squared_scale_<a) {
                const auto ratio=squared_scale_/a;
                normalized_squared_=1+normalized_squared_*ratio*ratio;squared_scale_=a;
            }else {
                const auto ratio=a/squared_scale_;normalized_squared_+=ratio*ratio;
            }
        }
        if(!std::isfinite(normalized_squared_))throw std::runtime_error("nonfinite scaled model sum of squares");
    }
    ComplexReferenceSummary finish() const {
        if(!result_.active_entries)throw std::runtime_error("model metric requires active entries");
        auto result=result_;
        result.rmse=finite_double(squared_scale_*std::sqrt(normalized_squared_/result.active_entries));
        return result;
    }
};
} // namespace cerium_source
