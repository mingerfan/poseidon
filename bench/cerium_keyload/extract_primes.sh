#!/usr/bin/env bash
# Extract the public Cerium 28-bit list without executing Python.
set -euo pipefail
if [[ $# != 2 ]]; then
  echo "usage: $0 CERIUM_PRIMES_PY OUTPUT_HEADER" >&2
  exit 2
fi
src=$1
out=$2
mkdir -p "$(dirname "$out")"
tmp=$(mktemp "${out}.XXXXXX")
trap 'rm -f "$tmp"' EXIT
awk '
BEGIN {
  print "// Generated from Cerium benchmarks/bootstrap_32K/primes.py; public moduli, not keys."
  print "#pragma once\n#include <array>\n#include <cstdint>"
  print "inline constexpr std::array<std::uint64_t, 64> kCeriumPrimes = {{"
}
/^[[:space:]]*Primes[[:space:]]*=/ {
  active = 1
  sub(/^[^[]*\[/, "")
}
active {
  end = index($0, "]")
  if (end) $0 = substr($0, 1, end - 1)
  sub(/#.*/, "")
  gsub(/,/, " ")
  n = split($0, values, /[[:space:]]+/)
  for (i = 1; i <= n; ++i) {
    if (values[i] == "") continue
    if (values[i] !~ /^[0-9]+$/) { bad = 1; exit }
    print "  " values[i] "ULL,"
    ++count
  }
  if (end) active = 0
}
END {
  print "}};"
  if (bad || active || count != 64) {
    print "expected exactly 64 integer public primes" > "/dev/stderr"
    exit 1
  }
}' "$src" > "$tmp"
mv "$tmp" "$out"
