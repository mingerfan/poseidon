#include "poseidon/runtime_api/rotation_key_basis.h"

#include <cstdlib>
#include <iostream>
#include <set>
#include <stdexcept>

using namespace poseidon::runtime_api;

namespace
{
void require(bool condition, const char *message)
{
    if (!condition) throw std::runtime_error(message);
}
}

int main()
{
    try
    {
        const std::set<int> requests{0, 3, -3, 7, 15, -8, 8, 19};
        const std::set<int> direct{-3, -1, 3, 7, 8};
        const std::set<int> binary{-2, -1, 1, 2, 4, 8};

        require(rotation_key_steps(requests, 16) == direct,
                "default must normalize, deduplicate and select direct keys");
        require(rotation_key_steps(requests, 16, RotationKeyPolicy::Binary) == binary,
                "binary must preserve the compact key basis");
        require(rotation_key_steps({}, 16).empty(), "empty requests must stay empty");
        require(rotation_key_steps({0, 16, -16}, 16).empty(),
                "zero rotations must not create conjugation keys");
        require(parse_rotation_key_policy(nullptr) == RotationKeyPolicy::Direct,
                "unset policy must default to direct");
        require(parse_rotation_key_policy("") == RotationKeyPolicy::Direct,
                "empty policy must default to direct");
        require(parse_rotation_key_policy("direct") == RotationKeyPolicy::Direct,
                "direct policy must be accepted");
        require(parse_rotation_key_policy("binary") == RotationKeyPolicy::Binary,
                "binary policy must be accepted");
        bool invalid_rejected = false;
        try { (void)parse_rotation_key_policy("other"); }
        catch (const std::invalid_argument &) { invalid_rejected = true; }
        require(invalid_rejected, "unknown policy must be rejected");
        bool invalid_slots_rejected = false;
        try { (void)rotation_key_steps({1}, 1); }
        catch (const std::invalid_argument &) { invalid_slots_rejected = true; }
        require(invalid_slots_rejected, "invalid slot count must be rejected");

        unsetenv("POSEIDON_RUNTIME_ROTATION_KEY_POLICY");
        require(rotation_key_policy_from_environment() == RotationKeyPolicy::Direct,
                "unset environment must default to direct");
        setenv("POSEIDON_RUNTIME_ROTATION_KEY_POLICY", "binary", 1);
        require(rotation_key_policy_from_environment() == RotationKeyPolicy::Binary,
                "environment must allow binary opt-in");
        setenv("POSEIDON_RUNTIME_ROTATION_KEY_POLICY", "direct", 1);
        require(rotation_key_policy_from_environment() == RotationKeyPolicy::Direct,
                "environment must allow direct selection");
        setenv("POSEIDON_RUNTIME_ROTATION_KEY_POLICY", "invalid", 1);
        bool invalid_environment_rejected = false;
        try { (void)rotation_key_policy_from_environment(); }
        catch (const std::invalid_argument &) { invalid_environment_rejected = true; }
        require(invalid_environment_rejected, "invalid environment policy must be rejected");
        std::cout << "rotation key policy tests passed\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "rotation key policy test failed: " << error.what() << '\n';
        return 1;
    }
}
