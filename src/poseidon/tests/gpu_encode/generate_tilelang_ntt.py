"""Generate kernels and a CUDA launcher; TileLang is a build-only dependency."""
import argparse
from importlib.metadata import version
from pathlib import Path
import re

from tilelang.tools.compile_only import compile_kernel_source
from tilelang_ntt import tensor_tam_stage, dif_fused2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arch", default="sm_89")
    args = parser.parse_args()
    if version("tilelang") != "0.1.14":
        raise RuntimeError("This experiment requires TileLang 0.1.14; validate other versions explicitly")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sources, calls = [], {}
    for kind, stages, fn in (("tam", (1, 16, 256), tensor_tam_stage),
                             ("dif", tuple(4 ** s for s in range(8)), dif_fused2)):
        for m in stages:
            source = compile_kernel_source(fn(m), f"cuda -arch={args.arch}")
            source = source.replace("#include <tl_templates/cuda/debug.h>\n", "")
            name = f"encode_tile_{kind}_{m}_kernel"
            # TileLang may reorder pointer/scalar arguments. Use its emitted
            # signature rather than assuming Python declaration order.
            match = re.search(rf"void {name}\(([^)]*)\);", source)
            if not match:
                raise RuntimeError(f"Cannot read emitted signature: {name}")
            parameters = [part.strip().split()[-1].lstrip("*") for part in match[1].split(",")]
            if set(parameters) - {"source", "values", "roots", "matrices", "primes", "ratios", "weights", "batch", "limbs"}:
                raise RuntimeError(f"Unexpected kernel signature: {parameters}")
            arguments = []
            for param in parameters:
                if param == "source":
                    arguments.append("source" if m == 1 else "values")
                elif param == "matrices":
                    # Stage-major table, then limb-major matrices.
                    offset = {1: 0, 16: 256, 256: 4352}[m]
                    arguments.append(f"matrices + {offset} * matrix_limbs")
                else:
                    arguments.append(param)
            grid = f"dim3({256 // m}, {m}, limbs * batch)" if kind == "tam" else "dim3(64, limbs * batch)"
            threads, shared = (64, 4096) if kind == "tam" else (256, 0)
            calls[kind, m] = f"    {name}<<<{grid}, {threads}, {shared}, stream>>>({', '.join(arguments)});\n    if (auto error = cudaGetLastError(); error != cudaSuccess) return error;\n"
            sources.append(source)
    output = '#include "tilelang_ntt.h"\n' + "\n".join(sources)
    output += """
cudaError_t launch_encode_tilelang_ntt(
    const std::uint32_t *source, std::uint32_t *values,
    const std::uint32_t *roots, const std::uint32_t *matrices,
    const std::uint32_t *primes, const std::uint64_t *ratios,
    const std::uint32_t *weights, int limbs, int matrix_limbs, int batch,
    bool tensor, cudaStream_t stream)
{
    if (!source || !values || source == values || !roots || !primes || !ratios ||
        !weights || limbs < 1 || limbs > 64 || matrix_limbs < limbs || batch < 1 || batch > 64 ||
        (tensor && !matrices)) return cudaErrorInvalidValue;
    if (tensor) {
"""
    output += "".join(calls["tam", m] for m in (1, 16, 256))
    output += "".join(calls["dif", m] for m in (4096, 16384))
    output += "    } else {\n" + "".join(calls["dif", 4 ** s] for s in range(8))
    output += "    }\n    return cudaSuccess;\n}\n"
    args.out.write_text(output)
    print(f"Generated 11 TileLang NTT kernels: {args.out}")


if __name__ == "__main__":
    main()
