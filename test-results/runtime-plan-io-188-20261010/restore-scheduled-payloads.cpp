// Experiment only: restore constants removed from diagnostic MLIR. IDs, types,
// placement, Release/reuse and operation order are copied verbatim from the IR.
#include "runtime/json_utils.hpp"
#include "runtime/plaintext_bundle.hpp"
#include <chrono>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <sys/resource.h>
#include <unordered_map>

using namespace fhegpu;
using namespace fhegpu::json_utils;
struct Payload {
  std::string content;
  std::vector<double> values;
};

int main(int argc, char **argv) {
  try {
    if (argc != 6)
      throw std::runtime_error(
          "usage: restore PLAN EXPECTED_SHA256 BUNDLE IR OUTPUT_IR");
    const auto start = std::chrono::steady_clock::now();
    std::unordered_map<std::string, Payload> payloads;
    std::ifstream plan(argv[1], std::ios::binary);
    if (!plan)
      throw std::runtime_error("cannot open plan");
    StrictJsonSax sax(
        argv[1],
        {"values", "external_inputs", "initialization", "execution",
         "finalization", "final_outputs"},
        [&](const std::string &field, std::size_t, Json &&record) {
          if (field != "initialization" && field != "execution")
            return;
          if (record.at("kind") != "encode")
            return;
          Payload payload;
          const auto &value = record.at("payload");
          const auto kind = read_string(value.at("kind"), "restore", "$kind");
          if (kind == "bundle")
            payload.content =
                read_sha256(value.at("content"), "restore", "$content");
          else if (kind == "inline")
            payload.values = value.at("values").get<std::vector<double>>();
          else
            throw std::runtime_error("unsupported payload kind");
          const auto id =
              read_string(record.at("output"), "restore", "$output");
          if (!payloads.emplace(id, std::move(payload)).second)
            throw std::runtime_error("duplicate Encode output");
        });
    HashingInputBuffer buffer(plan, argv[1]);
    std::istream stream(&buffer);
    Json::sax_parse(stream, &sax);
    if (buffer.source_sha256() != argv[2])
      throw std::runtime_error("source plan SHA-256 mismatch");
    const auto root = sax.take_result();
    const auto &ref = root.at("plaintext_bundle");
    PlaintextBundleRef reference{ref.at("id").get<std::string>(),
                                 ref.at("version").get<int>(),
                                 ref.at("manifest_sha256").get<std::string>()};
    std::vector<std::string> required;
    for (const auto &[id, payload] : payloads)
      if (!payload.content.empty())
        required.push_back(payload.content);
    auto bundle = PlaintextBundleLoader::open(
        argv[3], reference, required, std::numeric_limits<std::size_t>::max(),
        false);
    const double extractSeconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count();
    std::ifstream input(argv[4], std::ios::binary);
    std::ofstream output(argv[5], std::ios::binary | std::ios::trunc);
    if (!input || !output)
      throw std::runtime_error("cannot open IR input/output");
    std::string line;
    std::size_t restored = 0;
    const auto restoreStart = std::chrono::steady_clock::now();
    while (std::getline(input, line)) {
      if (line.find("\"ckks.encode\"") != std::string::npos) {
        const std::string marker = "runtime.value_id = ";
        const auto position = line.find(marker);
        if (position == std::string::npos)
          throw std::runtime_error("Encode is missing stable ValueId");
        const auto begin = position + marker.size();
        const auto end = line.find(' ', begin);
        auto found = payloads.find(line.substr(begin, end - begin));
        if (found == payloads.end())
          throw std::runtime_error("Encode has no original payload");
        std::string bytes;
        if (!found->second.content.empty()) {
          const auto path = std::filesystem::path(argv[3]) / "data" /
                            (found->second.content.substr(7) + ".bin");
          const auto count = std::filesystem::file_size(path);
          if (count == 0 || count % 8)
            throw std::runtime_error("invalid float64 blob length");
          bytes.resize(count);
          std::ifstream data(path, std::ios::binary);
          data.read(bytes.data(), bytes.size());
          if (!data || data.peek() != std::char_traits<char>::eof() ||
              source_sha256(bytes) != found->second.content)
            throw std::runtime_error("blob SHA-256/length mismatch");
        } else {
          for (double value : found->second.values) {
            std::uint64_t bits;
            std::memcpy(&bits, &value, 8);
            for (int byte = 0; byte < 8; ++byte)
              bytes.push_back(static_cast<char>(bits >> (byte * 8)));
          }
        }
        bool splat = true;
        for (std::size_t offset = 8; offset < bytes.size(); offset += 8)
          if (std::memcmp(bytes.data(), bytes.data() + offset, 8)) {
            splat = false;
            break;
          }
        std::string dense;
        if (splat) {
          std::uint64_t bits = 0;
          for (int byte = 0; byte < 8; ++byte)
            bits |= static_cast<std::uint64_t>(
                        static_cast<unsigned char>(bytes[byte]))
                    << (byte * 8);
          double value;
          std::memcpy(&value, &bits, 8);
          if (!std::isfinite(value))
            throw std::runtime_error("nonfinite splat");
          std::ostringstream literal;
          literal.imbue(std::locale::classic());
          literal << std::scientific << std::setprecision(17) << value;
          dense = "dense<" + literal.str() + ">";
        } else {
          static constexpr char hex[] = "0123456789ABCDEF";
          dense = "dense<\"0x";
          dense.reserve(bytes.size() * 2 + 16);
          for (unsigned char byte : bytes) {
            dense += hex[byte >> 4];
            dense += hex[byte & 15];
          }
          dense += "\">";
        }
        dense += " : tensor<" + std::to_string(bytes.size() / 8) + "xf64>";
        const auto payloadMarker = line.find("payload = ");
        if (payloadMarker == std::string::npos)
          throw std::runtime_error("missing Encode payload");
        const auto payloadStart =
            payloadMarker + std::string("payload = ").size();
        const auto payloadEnd = line.find("}>", payloadStart);
        if (payloadEnd == std::string::npos)
          throw std::runtime_error("unexpected Encode attribute layout");
        line.replace(payloadStart, payloadEnd - payloadStart, dense);
        ++restored;
      }
      output << line << '\n';
      if (!output)
        throw std::runtime_error("IR write failed");
    }
    if (input.bad())
      throw std::runtime_error("IR read failed");
    output.close();
    if (!output || restored != payloads.size())
      throw std::runtime_error("restored Encode count/write mismatch");
    rusage usage{};
    getrusage(RUSAGE_SELF, &usage);
    std::cout << Json{{"plan_sha256", argv[2]},
                      {"restored_encodes", restored},
                      {"extract_seconds", extractSeconds},
                      {"restore_seconds",
                       std::chrono::duration<double>(
                           std::chrono::steady_clock::now() - restoreStart)
                           .count()},
                      {"output_bytes", std::filesystem::file_size(argv[5])},
                      {"peak_rss_bytes", usage.ru_maxrss * 1024L}}
                     .dump()
              << '\n';
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
