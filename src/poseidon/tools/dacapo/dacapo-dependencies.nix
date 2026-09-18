# Evaluation defines a plan only. Realizing these packages needs user approval.
# No <nixpkgs>, channel lookup, fetchTarball or import-from-derivation here.
{ nixpkgsSource ? null }:
let
  lock = builtins.fromJSON (builtins.readFile ./dependency-lock.json);
  candidate = if nixpkgsSource == null then lock.nixpkgs.bundled_path else nixpkgsSource;
  source = if builtins.pathExists candidate then builtins.path {
    path = builtins.toPath candidate;
    name = "poseidon-nixpkgs-${lock.nixpkgs.rev}";
    sha256 = lock.nixpkgs.nar_hash;
  } else throw "Pinned nixpkgs source is missing. Use the approved nix-portable environment or provide nixpkgsSource; no automatic download is allowed.";
  pkgs = import source { inherit (lock) system; config = {}; overlays = []; };
  # Bound each fixed-output source download as well as the caller's build timeout.
  sourceCurlOptions = [ "--connect-timeout" "10" "--max-time" "600" "--retry" "0" ];

  # Bootstrap Clang itself with the pinned GCC stdenv; Dacapo uses Clang 18.1.2.
  # One monorepo build guarantees LLVM, MLIR and Clang have the same version.
  toolchain = pkgs.stdenv.mkDerivation {
    pname = "poseidon-llvm-mlir-clang";
    inherit (lock.llvm) version;
    src = pkgs.fetchurl {
      inherit (lock.llvm) url sha256;
      curlOptsList = sourceCurlOptions;
    };
    nativeBuildInputs = [ pkgs.cmake pkgs.ninja pkgs.python310 ];
    buildInputs = [ pkgs.zlib pkgs.libxml2 ];
    cmakeDir = "../llvm";
    cmakeFlags = [
      "-DLLVM_ENABLE_PROJECTS=clang;mlir"
      "-DLLVM_TARGETS_TO_BUILD=X86"
      "-DLLVM_INSTALL_UTILS=ON"
      "-DLLVM_ENABLE_PIC=ON"
      "-DLLVM_PARALLEL_COMPILE_JOBS=2"
      "-DLLVM_PARALLEL_LINK_JOBS=1"
      "-DLLVM_BUILD_EXAMPLES=OFF"
      "-DLLVM_INCLUDE_BENCHMARKS=OFF"
      "-DMLIR_ENABLE_BINDINGS_PYTHON=OFF"
    ];
    enableParallelBuilding = true;
    # Verify the actual installed tools and CMake export locations, not only attrs.
    doInstallCheck = true;
    installCheckPhase = ''
      runHook preInstallCheck
      test "$($out/bin/llvm-config --version)" = '${lock.llvm.version}'
      $out/bin/mlir-opt --version | grep -F '${lock.llvm.version}'
      $out/bin/clang --version | grep -F '${lock.llvm.version}'
      test -f "$out/lib/cmake/llvm/LLVMConfig.cmake"
      test -f "$out/lib/cmake/mlir/MLIRConfig.cmake"
      runHook postInstallCheck
    '';
    passthru.isClang = true;
  };
  clang = pkgs.wrapCCWith {
    cc = toolchain;
    isClang = true;
    libcxx = null;
    gccForLibs = pkgs.stdenv.cc.cc;
  };

  # SEAL 4.0 requests GSL major version 3. The snapshot's GSL 4 is incompatible.
  # Keep GSL enabled, including its tests; reuse the upstream package test patch.
  msgsl = pkgs.stdenv.mkDerivation {
    pname = "poseidon-microsoft-gsl";
    inherit (lock.msgsl) version;
    src = pkgs.fetchFromGitHub {
      owner = "Microsoft";
      repo = "GSL";
      rev = lock.msgsl.commit;
      hash = lock.msgsl.nar_hash;
      curlOptsList = sourceCurlOptions;
    };
    patches = [ (pkgs.fetchurl {
      url = "https://github.com/microsoft/GSL/commit/f5cf01083baf7e8dc8318db3648bc6098dc32d67.patch";
      hash = "sha256-uouv35crtly8kYhKyvMyZkqwTKt1jXC6dZjw4sQ6uv0=";
      curlOptsList = sourceCurlOptions;
    }) ];
    nativeBuildInputs = [ pkgs.cmake pkgs.pkg-config ];
    buildInputs = [ pkgs.gtest ];
    doCheck = true;
    enableParallelBuilding = true;
  };
  # SEAL's CMake requires the static Zstd target; retain its upstream checks.
  zstd = pkgs.zstd.override { enableStatic = true; };
  seal = pkgs.stdenv.mkDerivation {
    pname = "poseidon-seal";
    inherit (lock.seal) version;
    src = pkgs.fetchurl {
      inherit (lock.seal) url sha256;
      curlOptsList = sourceCurlOptions;
    };
    nativeBuildInputs = [ pkgs.cmake pkgs.ninja pkgs.pkg-config ];
    propagatedBuildInputs = [ msgsl pkgs.zlib zstd ];
    cmakeFlags = [
      # SEAL 4.0's seal.pc.in prepends ${prefix}; Nix otherwise supplies an
      # absolute includedir, producing ${prefix}//nix/store/... in seal.pc.
      "-DCMAKE_INSTALL_INCLUDEDIR=include"
      "-DSEAL_BUILD_DEPS=OFF"
      "-DBUILD_SHARED_LIBS=OFF"
      "-DSEAL_USE_MSGSL=ON"
      "-DSEAL_USE_ZLIB=ON"
      "-DSEAL_USE_ZSTD=ON"
      "-DSEAL_THROW_ON_TRANSPARENT_CIPHERTEXT=ON"
    ];
    enableParallelBuilding = true;
    doInstallCheck = true;
    installCheckPhase = ''
      runHook preInstallCheck
      test -f "$out/lib/cmake/SEAL-4.0/SEALConfig.cmake"
      grep -F 'set(PACKAGE_VERSION "${lock.seal.version}")' \
        "$out/lib/cmake/SEAL-4.0/SEALConfigVersion.cmake"
      test "$(PKG_CONFIG_PATH="$out/lib/pkgconfig" pkg-config --variable=includedir seal)" \
        = "$out/include/SEAL-4.0"
      test -f "$out/include/SEAL-4.0/seal/seal.h"
      runHook postInstallCheck
    '';
  };
in {
  inherit pkgs lock toolchain clang msgsl zstd seal;
  metadata = {
    scope = "dependency_plan_only";
    compiler_build_validated = false;
    encrypted_execution_validated = false;
    inherit (lock) system limits;
    nixpkgs = { inherit (lock.nixpkgs) rev nar_hash; };
    versions = {
      llvm = toolchain.version;
      mlir = toolchain.version;
      clang = clang.version;
      seal = seal.version;
      msgsl = msgsl.version;
      zstd = zstd.version;
      zlib = pkgs.zlib.version;
      cmake = pkgs.cmake.version;
      ninja = pkgs.ninja.version;
      python = pkgs.python310.version;
      bootstrap_gcc = pkgs.stdenv.cc.version;
    };
    toolchain_cmake_flags = toolchain.cmakeFlags;
    seal_cmake_flags = seal.cmakeFlags;
    compiler_derivations = [ toolchain.drvPath clang.drvPath seal.drvPath ];
    compiler_outputs = [ toolchain.outPath clang.outPath seal.outPath ];
    # nix-shell otherwise adds an implicit bashInteractive dependency not covered
    # by nix build shell.nix. Reuse the already-realized stdenv Bash explicitly.
    shell_bash = "${pkgs.bash}/bin/bash";
  };
}
