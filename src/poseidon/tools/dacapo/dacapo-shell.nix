{ pkgs ? import <nixpkgs> {} }:

let
  llvmPackages = pkgs.llvmPackages_16 or pkgs.llvmPackages;
  python = pkgs.python310 or pkgs.python3;
  optional = pkg: if pkg == null then [] else [ pkg ];

  clang = llvmPackages.clang or (pkgs.clang_16 or null);
  llvm = llvmPackages.llvm or null;
  mlir = llvmPackages.mlir or null;
  seal = pkgs.seal or null;
in
pkgs.mkShell {
  packages = [
    pkgs.cmake
    pkgs.git
    pkgs.ninja
    python
  ] ++ optional clang
    ++ optional llvm
    ++ optional mlir
    ++ optional seal;

  shellHook = ''
    if [ -z "''${POSEIDON_ROOT:-}" ]; then
      POSEIDON_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
      export POSEIDON_ROOT
    fi

    export DACAPO_ROOT="$POSEIDON_ROOT/third_party/dacapo"
    export DACAPO_BUILD_DIR="$DACAPO_ROOT/build/nix"

    echo "Dacapo root: $DACAPO_ROOT"
    echo "Build dir:   $DACAPO_BUILD_DIR"
    echo "This shell is for dependency isolation only; do not install Dacapo deps system-wide."
  '';
}
