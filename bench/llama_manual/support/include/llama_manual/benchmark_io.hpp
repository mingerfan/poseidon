#pragma once
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace llama_manual::benchmark {
constexpr std::size_t tokens=128,hidden=4096,elements=tokens*hidden;

inline void require_little_endian() {
    const std::uint16_t word=1;
    if(*reinterpret_cast<const unsigned char*>(&word)!=1||sizeof(double)!=8)
        throw std::runtime_error("benchmark tensors require little-endian float64 host");
}

// Header identifies layout and number of completed decoder blocks, preventing
// a 2-block reference from silently being used for a 32-block run.
inline std::vector<double> read_tensor(const std::filesystem::path &path,
                                      const std::string &kind,std::size_t layers) {
    require_little_endian();
    std::ifstream in(path,std::ios::binary);std::string magic,line,label,extra;
    if(!std::getline(in,magic)||magic!="LLAMA_CERIUM_TENSOR_V1"||!std::getline(in,line))
        throw std::invalid_argument("invalid benchmark tensor header: "+path.string());
    std::size_t rows=0,width=0,blocks=0;std::istringstream header(line);
    if(!(header>>label>>rows>>width>>blocks)||(header>>extra)||label!=kind||rows!=tokens||width!=hidden||blocks!=layers)
        throw std::invalid_argument("benchmark tensor kind/shape/block count mismatch: "+path.string());
    std::vector<double> values(elements);
    if(!in.read(reinterpret_cast<char*>(values.data()),elements*sizeof(double))||in.peek()!=std::char_traits<char>::eof())
        throw std::invalid_argument("benchmark tensor payload length mismatch: "+path.string());
    for(double v:values)if(!std::isfinite(v))throw std::invalid_argument("nonfinite benchmark input/reference");
    return values;
}

inline void write_tensor(const std::filesystem::path &path,const std::vector<double> &values,
                         const std::string &kind,std::size_t layers) {
    require_little_endian();
    if(values.size()!=elements||std::filesystem::exists(path))throw std::invalid_argument("output shape or existing tensor path");
    for(double v:values)if(!std::isfinite(v))throw std::invalid_argument("nonfinite benchmark output");
    std::ofstream out(path,std::ios::binary);
    out<<"LLAMA_CERIUM_TENSOR_V1\n"<<kind<<' '<<tokens<<' '<<hidden<<' '<<layers<<'\n';
    out.write(reinterpret_cast<const char*>(values.data()),values.size()*sizeof(double));
    if(!out)throw std::runtime_error("failed to write benchmark tensor");
}
} // namespace llama_manual::benchmark
