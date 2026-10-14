// Compare raw reads into one resident buffer; no blob hashing or decoding.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <vector>

struct Entry {
    std::uint64_t length;
    char digest[64];
};
static_assert(sizeof(Entry) == 72, "index ABI requires 72-byte records");

int main(int argc, char **argv) {
    try {
        if (argc != 4) throw std::runtime_error("usage: raw-io-comparison files|pack BUNDLE INDEX");
        const bool packed = std::string(argv[1]) == "pack";
        if (!packed && std::string(argv[1]) != "files") throw std::runtime_error("unknown mode");
        const auto index_size = std::filesystem::file_size(argv[3]);
        if (index_size % sizeof(Entry)) throw std::runtime_error("invalid index length");
        std::vector<Entry> entries(index_size / sizeof(Entry));
        std::ifstream index(argv[3], std::ios::binary);
        index.read(reinterpret_cast<char *>(entries.data()), index_size);
        if (!index) throw std::runtime_error("cannot read index");
        std::uint64_t total = 0;
        for (const auto &entry : entries) total += entry.length;
        const std::filesystem::path directory(argv[2]);
        const auto start = std::chrono::steady_clock::now();
        std::unique_ptr<char[]> resident(new char[total]);
        std::uint64_t position = 0;
        if (packed) {
            const auto path = directory / "data.bin";
            if (std::filesystem::file_size(path) != total) throw std::runtime_error("pack length mismatch");
            std::ifstream input(path, std::ios::binary);
            while (position < total) {
                const auto count = std::min<std::uint64_t>(8 * 1024 * 1024, total - position);
                input.read(resident.get() + position, count);
                if (!input) throw std::runtime_error("failed pack read");
                position += count;
            }
            if (input.peek() != std::char_traits<char>::eof()) throw std::runtime_error("pack length changed");
        } else {
            for (std::size_t i = 0; i < entries.size(); ++i) {
                const auto &entry = entries[i];
                const auto path = directory / "data" / (std::string(entry.digest, 64) + ".bin");
                if (std::filesystem::file_size(path) != entry.length) throw std::runtime_error("blob length mismatch");
                std::ifstream input(path, std::ios::binary);
                input.read(resident.get() + position, entry.length);
                if (!input || input.peek() != std::char_traits<char>::eof()) throw std::runtime_error("failed blob read");
                position += entry.length;
                if ((i + 1) % 200000 == 0) std::cerr << "read " << i + 1 << '/' << entries.size() << " blobs\n";
            }
        }
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        rusage usage{};
        getrusage(RUSAGE_SELF, &usage);
        std::cout << "{\"mode\":\"" << argv[1] << "\",\"bytes\":" << position
                  << ",\"blob_count\":" << entries.size() << ",\"raw_read_seconds\":" << seconds
                  << ",\"peak_rss_bytes\":" << usage.ru_maxrss * 1024L
                  << ",\"blob_hashes_verified\":0,\"blob_payloads_decoded\":0}\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
