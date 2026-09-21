"""Load native Hecate libraries without importing Hecate, Torch or a VM backend."""
import argparse
import ctypes
import json
from pathlib import Path
from platform_config import identity, require_platform, loaded_libraries
from hevm_abi import check_elf, require_native_abi


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build", type=Path)
    args = parser.parse_args()
    require_platform()
    abi = require_native_abi(args.build)
    libraries = {}
    # Do not call initFullVM, encryption/decryption, or any bootstrap entrypoint.
    for name, symbols in {
        "libHecateFrontend.so": ("init", "createFunc", "save", "finalize"),
        "libSEAL_HEVM.so": ("initFullVM", "load", "encrypt", "decrypt"),
    }.items():
        path = args.build.resolve() / "lib" / name
        check_elf(path)
        library = ctypes.CDLL(str(path))
        for symbol in symbols:
            getattr(library, symbol)
        libraries[name] = {"path": str(path), "loaded": True, "symbols": list(symbols)}
    print(json.dumps({"platform_identity": identity(), "hevm_abi": abi,
                      "scope": "native_library_loading_only", "libraries": libraries,
                      "mapped_libraries": loaded_libraries(),
                      "hecate_python_import_validated": False,
                      "encrypted_execution_validated": False}, indent=2))


if __name__ == "__main__":
    main()
