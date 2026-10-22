#include "llvm/Config/llvm-config.h"
#include "llvm/Support/Process.h"
#include <gnu/libc-version.h>
#include <iostream>

int main() {
    // Force a real LLVMSupport link, including its libc requirements. Merely
    // executing clang --version did not detect the earlier host-libc mismatch.
    auto columns = llvm::sys::Process::StandardOutColumns();
    std::cout << "LLVM=" << LLVM_VERSION_STRING << " glibc=" << gnu_get_libc_version()
              << " stdout_columns=" << columns << '\n';
}
