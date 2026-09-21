#include "hecate/Support/HEVMHeader.h"
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <type_traits>

static_assert(std::is_standard_layout<HEVMHeader>::value);
static_assert(std::is_standard_layout<ConfigBody>::value);
static_assert(std::is_standard_layout<HEVMOperation>::value);
static_assert(sizeof(HEVMHeader) == 24 && alignof(HEVMHeader) == 8);
static_assert(offsetof(HEVMHeader, magic_number) == 0);
static_assert(offsetof(HEVMHeader, hevm_header_size) == 4);
static_assert(offsetof(HEVMHeader, config_header) == 8);
static_assert(sizeof(HEVMHeader::ConfigHeader) == 16);
static_assert(offsetof(HEVMHeader::ConfigHeader, res_length) == 8);
static_assert(sizeof(ConfigBody) == 40 && alignof(ConfigBody) == 8);
static_assert(offsetof(ConfigBody, num_operations) == 8);
static_assert(offsetof(ConfigBody, num_ctxt_buffer) == 16);
static_assert(offsetof(ConfigBody, num_ptxt_buffer) == 24);
static_assert(offsetof(ConfigBody, init_level) == 32);
static_assert(sizeof(HEVMOperation) == 8 && alignof(HEVMOperation) == 2);
static_assert(offsetof(HEVMOperation, dst) == 2);
static_assert(offsetof(HEVMOperation, lhs) == 4);
static_assert(offsetof(HEVMOperation, rhs) == 6);
static_assert(sizeof(void*) == 8 && sizeof(size_t) == 8 && sizeof(int) == 4 && sizeof(int64_t) == 8);
static_assert(sizeof(double) == 8 && std::numeric_limits<double>::is_iec559);

template <typename T> void hex(const T& value) {
  const auto *bytes = reinterpret_cast<const unsigned char*>(&value);
  for (size_t i = 0; i < sizeof(T); ++i)
    std::cout << std::hex << std::setw(2) << std::setfill('0') << unsigned(bytes[i]);
}
int main() {
  const uint16_t endian = 1;
  if (*reinterpret_cast<const unsigned char*>(&endian) != 1) return 2;
  HEVMHeader header{};
  header.hevm_header_size = 24;
  header.config_header = {1, 2};
  const ConfigBody body{104, 3, 4, 5, 13};
  const HEVMOperation operation{1, 2, 3, 65533};
  std::cout << "{\"header_hex\":\""; hex(header);
  std::cout << "\",\"body_hex\":\""; hex(body);
  std::cout << "\",\"operation_hex\":\""; hex(operation);
  std::cout << "\",\"sizes\":[24,40,8],\"alignments\":[8,8,2],"
               "\"pointer_bytes\":8,\"int_bytes\":4,\"double_bytes\":8}" << '\n';
}
