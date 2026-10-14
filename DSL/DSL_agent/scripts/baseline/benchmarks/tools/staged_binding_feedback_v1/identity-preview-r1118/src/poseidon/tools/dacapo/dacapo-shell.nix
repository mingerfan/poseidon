{ nixpkgsSource ? null, toolchainProfile ? "hecate", platform ? "x86_64-linux" }:
let
  deps = import ./dacapo-dependencies.nix { inherit nixpkgsSource toolchainProfile platform; };
  inherit (deps) pkgs toolchain compiler compilerInfo seal;
in
assert toolchain.version == "18.1.2";
assert seal.version == "4.0.0";
assert pkgs.python310.version == deps.platformConfig.python;
assert pkgs.stdenv.cc.version == "13.2.0";
(pkgs.mkShell.override { stdenv = pkgs.overrideCC pkgs.stdenv compiler; }) {
  packages = [
    pkgs.cmake
    pkgs.git
    pkgs.ninja
    pkgs.python310
    pkgs.pkg-config
    toolchain
    seal
  ];

  POSEIDON_PLATFORM = platform;
  LLVM_ROOT = "${toolchain}";
  MLIR_ROOT = "${toolchain}";
  LLVM_DIR = "${toolchain}/lib/cmake/llvm";
  MLIR_DIR = "${toolchain}/lib/cmake/mlir";
  SEAL_ROOT = "${seal}";
  SEAL_DIR = "${seal}/lib/cmake/SEAL-4.0";
  CC = "${compiler}/bin/${compilerInfo.c_binary}";
  CXX = "${compiler}/bin/${compilerInfo.cxx_binary}";
  HECATE_C_COMPILER_NAME = compilerInfo.c_binary;
  HECATE_CXX_COMPILER_NAME = compilerInfo.cxx_binary;
  HECATE_CXX_COMPILER_VERSION = compilerInfo.version;
  CMAKE_BUILD_PARALLEL_LEVEL = "2";
  # Upstream manylinux wheels do not carry Nix RPATHs. Expose the existing,
  # pinned C++ runtime only to Python subprocesses, never globally/host tools.
  HECATE_PYTHON_LIBRARY_PATH = pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc.lib pkgs.zlib ];

  shellHook = ''
    # The stdenv setup hook resets CC/CXX to bare compiler names. Reassert the
    # wrappers after setup so builds use the selected Nix compiler and runtime.
    export CC="${compiler}/bin/${compilerInfo.c_binary}"
    export CXX="${compiler}/bin/${compilerInfo.cxx_binary}"
    export PATH="${compiler}/bin:$PATH"
    if [ -z "''${POSEIDON_ROOT:-}" ]; then
      POSEIDON_ROOT="$(timeout -k 3s 20s git rev-parse --show-toplevel)" || return 1
      export POSEIDON_ROOT
    fi

    export DACAPO_ROOT="$POSEIDON_ROOT/third_party/dacapo"
    export POSEIDON_WORK_ROOT="''${POSEIDON_WORK_ROOT:-$HOME/poseidon-work}"
    export POSEIDON_PLATFORM_WORK_ROOT="$POSEIDON_WORK_ROOT${if deps.platformConfig.work_subdirectory == "" then "" else "/${deps.platformConfig.work_subdirectory}"}"
    export DACAPO_BUILD_DIR="$POSEIDON_PLATFORM_WORK_ROOT/build-dacapo/hecate-18.1.2-nix"

    echo "Dacapo root: $DACAPO_ROOT"
    echo "Build dir:   $DACAPO_BUILD_DIR"
    echo "Pinned LLVM/MLIR 18.1.2 and SEAL 4.0.0. Build with -j2."
    echo "Python tracing packages and out-of-source Hecate imports are a separate gate."
  '';
}
