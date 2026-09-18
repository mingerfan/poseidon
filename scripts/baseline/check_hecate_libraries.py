"""Load native Hecate libraries without importing Hecate, Torch or a VM backend."""
import argparse
import ctypes
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build", type=Path)
    args = parser.parse_args()
    libraries = {}
    # Do not call initFullVM, encryption/decryption, or any bootstrap entrypoint.
    for name, symbols in {
        "libHecateFrontend.so": ("init", "createFunc", "save", "finalize"),
        "libSEAL_HEVM.so": ("initFullVM", "load", "encrypt", "decrypt"),
    }.items():
        path = args.build.resolve() / "lib" / name
        library = ctypes.CDLL(str(path))
        for symbol in symbols:
            getattr(library, symbol)
        libraries[name] = {"path": str(path), "loaded": True, "symbols": list(symbols)}
    print(json.dumps({"scope": "native_library_loading_only", "libraries": libraries,
                      "hecate_python_import_validated": False,
                      "encrypted_execution_validated": False}, indent=2))


if __name__ == "__main__":
    main()
