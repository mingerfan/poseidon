#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>

namespace
{

namespace fs = std::filesystem;

struct ToolOptions
{
    std::string dacapo_root;
    std::string dump_tool = "poseidon_mgpu_dacapo_hevm_dump";
    std::optional<std::string> config_path;
    std::string summary_path = "/tmp/resnet20-mgpu-preflight.json";
    std::string schedule_path = "/tmp/resnet20-mgpu-schedule.mlir";
    bool print_command = true;
};

struct PathStatus
{
    fs::path path;
    bool exists = false;
    bool regular_file = false;
    std::uintmax_t size = 0;
    std::string diagnostic;

    bool ok() const
    {
        return exists && regular_file;
    }
};

void print_usage(std::ostream &stream)
{
    stream
        << "usage: poseidon_mgpu_resnet20_artifact_check "
           "[--dacapo-root <dir>] "
           "[--dump-tool <path>] "
           "[--config <file>] "
           "[--summary-path <file>] "
           "[--schedule-path <file>] "
           "[--no-command]\n\n"
        << "Checks Dacapo's expected ResNet20 HEAAN GPU artifact paths:\n"
        << "  examples/optimized/dacapo/ResNet.40._hecate_ResNet.hevm\n"
        << "  examples/traced/_hecate_ResNet.cst\n\n"
        << "If --dacapo-root is omitted, DACAPO_ROOT is used.\n";
}

const char *get_env(const char *name)
{
    const char *value = std::getenv(name);
    if (value == nullptr || std::string(value).empty())
    {
        return nullptr;
    }
    return value;
}

std::string require_value(int &index, int argc, char **argv, const std::string &name)
{
    if (++index >= argc)
    {
        throw std::invalid_argument("missing value for " + name);
    }
    return argv[index];
}

ToolOptions parse_args(int argc, char **argv)
{
    ToolOptions options;
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h")
        {
            print_usage(std::cout);
            std::exit(EXIT_SUCCESS);
        }
        if (arg == "--dacapo-root")
        {
            options.dacapo_root = require_value(i, argc, argv, arg);
            continue;
        }
        if (arg == "--dump-tool")
        {
            options.dump_tool = require_value(i, argc, argv, arg);
            continue;
        }
        if (arg == "--config")
        {
            options.config_path = require_value(i, argc, argv, arg);
            continue;
        }
        if (arg == "--summary-path")
        {
            options.summary_path = require_value(i, argc, argv, arg);
            continue;
        }
        if (arg == "--schedule-path")
        {
            options.schedule_path = require_value(i, argc, argv, arg);
            continue;
        }
        if (arg == "--no-command")
        {
            options.print_command = false;
            continue;
        }
        throw std::invalid_argument("unknown argument: " + arg);
    }

    if (options.dacapo_root.empty())
    {
        if (const char *root = get_env("DACAPO_ROOT"))
        {
            options.dacapo_root = root;
        }
    }
    if (options.dacapo_root.empty())
    {
        throw std::invalid_argument("missing --dacapo-root or DACAPO_ROOT");
    }
    return options;
}

fs::path absolute_path(const fs::path &path)
{
    std::error_code error;
    fs::path absolute = fs::absolute(path, error);
    if (error)
    {
        return path;
    }
    return absolute.lexically_normal();
}

PathStatus inspect_file(const fs::path &path)
{
    PathStatus status;
    status.path = absolute_path(path);

    std::error_code error;
    status.exists = fs::exists(status.path, error);
    if (error)
    {
        status.diagnostic = "failed to inspect path: " + error.message();
        return status;
    }
    if (!status.exists)
    {
        status.diagnostic = "missing";
        return status;
    }

    status.regular_file = fs::is_regular_file(status.path, error);
    if (error)
    {
        status.diagnostic = "failed to inspect file type: " + error.message();
        return status;
    }
    if (!status.regular_file)
    {
        status.diagnostic = "not a regular file";
        return status;
    }

    status.size = fs::file_size(status.path, error);
    if (error)
    {
        status.diagnostic = "failed to read file size: " + error.message();
        return status;
    }
    return status;
}

std::string shell_quote(const std::string &text)
{
    std::string output = "'";
    for (const char ch : text)
    {
        if (ch == '\'')
        {
            output += "'\\''";
        }
        else
        {
            output.push_back(ch);
        }
    }
    output.push_back('\'');
    return output;
}

void append_common_dump_outputs(std::ostream &stream, const ToolOptions &options)
{
    stream << " \\\n  --summary-json"
           << " \\\n  --write-summary-json "
           << shell_quote(options.summary_path)
           << " \\\n  --write-schedule "
           << shell_quote(options.schedule_path)
           << " \\\n  --no-schedule";
}

std::string make_dump_command(
    const ToolOptions &options, const fs::path &hevm_path,
    const fs::path &constants_path)
{
    std::ostringstream command;
    command << shell_quote(options.dump_tool)
            << " \\\n  --hevm " << shell_quote(hevm_path.string())
            << " \\\n  --constants " << shell_quote(constants_path.string());

    if (options.config_path.has_value())
    {
        command << " \\\n  --config " << shell_quote(*options.config_path);
        append_common_dump_outputs(command, options);
        return command.str();
    }

    command
        << " \\\n  --devices 8"
        << " \\\n  --upload-device 0"
        << " \\\n  --compute-devices 0,1,2,3,4,5,6,7"
        << " \\\n  --download-device 0"
        << " \\\n  --opcode-summary"
        << " \\\n  --communication-plan"
        << " \\\n  --communication-execution-preflight"
        << " \\\n  --execution-cuda-peer-available"
        << " \\\n  --poseidon-gpu-preflight"
        << " \\\n  --preflight-comm-available"
        << " \\\n  --preflight-relin-keys"
        << " \\\n  --preflight-galois-keys"
        << " \\\n  --require-ready";
    append_common_dump_outputs(command, options);
    return command.str();
}

void print_file_status(
    std::ostream &stream, const std::string &label, const PathStatus &status)
{
    stream << "  " << label << ": " << status.path.string() << '\n';
    stream << "    present: " << (status.ok() ? "true" : "false") << '\n';
    if (status.ok())
    {
        stream << "    bytes: " << status.size << '\n';
    }
    else
    {
        stream << "    diagnostic: " << status.diagnostic << '\n';
    }
}

}  // namespace

int main(int argc, char **argv)
{
    try
    {
        const ToolOptions options = parse_args(argc, argv);
        const fs::path dacapo_root = absolute_path(options.dacapo_root);
        const PathStatus hevm = inspect_file(
            dacapo_root / "examples" / "optimized" / "dacapo" /
            "ResNet.40._hecate_ResNet.hevm");
        const PathStatus constants = inspect_file(
            dacapo_root / "examples" / "traced" / "_hecate_ResNet.cst");
        const bool ready = hevm.ok() && constants.ok();

        std::cout << "resnet20_artifact_check:\n";
        std::cout << "  dacapo_root: " << dacapo_root.string() << '\n';
        print_file_status(std::cout, "hevm", hevm);
        print_file_status(std::cout, "constants", constants);
        std::cout << "  status: " << (ready ? "ready" : "missing_artifacts")
                  << '\n';

        if (!ready)
        {
            std::cout << "  diagnostics:\n";
            if (!hevm.ok())
            {
                std::cout << "    - missing hevm: " << hevm.path.string()
                          << '\n';
            }
            if (!constants.ok())
            {
                std::cout << "    - missing constants: "
                          << constants.path.string() << '\n';
            }
            std::cout << "  generation_hint:\n";
            std::cout << "    - run: hc-trace ResNet\n";
            std::cout << "    - run: hbt dacapo 40 ResNet HEAAN GPU\n";
        }

        if (options.print_command)
        {
            std::cout << "  dump_command:\n";
            std::cout << make_dump_command(options, hevm.path, constants.path)
                      << '\n';
        }

        return ready ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    catch (const std::exception &ex)
    {
        std::cerr << "poseidon_mgpu_resnet20_artifact_check: " << ex.what()
                  << '\n';
        return EXIT_FAILURE;
    }
}
