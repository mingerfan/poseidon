// Diagnostic preparation only. Uses public context primes, never saved secrets.
#include "poseidon/poseidon_context.h"
#include "poseidon/keygenerator.h"
#include "poseidon/gpu/gpu_bootstrap_profile.h"
#include "poseidon/gpu/gpu_memory.h"
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <type_traits>

using namespace poseidon;
using namespace poseidon::gpu;
using Clock = std::chrono::steady_clock;
static std::size_t payload(const GpuEvaluationKeyData &key) {
    std::size_t n = 0;
    for (const auto &field : key.fields_) n += field.size() * sizeof(GpuWord);
    return n;
}

namespace {
// Canonical JSON metadata contains integers, strings and exact binary64 bits,
// never locale-dependent floating point, object representations or addresses.
std::string json_quote(std::string_view text) {
    std::string out = "\"";
    constexpr char hex[] = "0123456789abcdef";
    for (unsigned char c : text) {
        if (c == '"' || c == '\\') { out += '\\'; out += c; }
        else if (c < 32) { out += "\\u00"; out += hex[c >> 4]; out += hex[c & 15]; }
        else out += c;
    }
    return out + '"';
}
std::string hex64(std::uint64_t value) {
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::hex << std::setw(16) << std::setfill('0') << value;
    return out.str();
}
std::string json(const std::string &value) { return json_quote(value); }
std::string json(const char *value) { return json_quote(value); }
std::string json(bool value) { return value ? "true" : "false"; }
template<class T> requires std::is_integral_v<T>
std::string json(T value) { return std::to_string(value); }
struct Object {
    std::string text = "{";
    template<class T> Object &add(const char *name, const T &value);
    std::string str() const { return text + '}'; }
};
std::string json(const Object &value) { return value.str(); }
template<class Range, class Encode>
std::string array(const Range &values, Encode encode) {
    std::string out = "[";
    bool first = true;
    for (const auto &value : values) {
        if (!first) out += ',';
        first = false;
        out += encode(value);
    }
    return out + ']';
}
template<class T, std::size_t N>
std::string json(const std::array<T, N> &values) {
    return array(values, [](const auto &v) { return json(v); });
}
template<class T>
std::string json(const std::vector<T> &values) {
    return array(values, [](const auto &v) { return json(v); });
}
template<class T> Object &Object::add(const char *name, const T &value) {
    if (text.size() > 1) text += ',';
    text += json_quote(name) + ':' + json(value);
    return *this;
}
Object &bits(Object &out, const char *name, double value) {
    static_assert(sizeof(double) == sizeof(std::uint64_t));
    return out.add(name, hex64(std::bit_cast<std::uint64_t>(value)));
}

// FNV-1a over length-prefixed canonical metadata and EVERY uint32 residue,
// encoded little endian. This is only a noncollisionproof diagnostic digest.
struct Fingerprint {
    std::uint64_t value = 14695981039346656037ULL;
    void byte(unsigned char v) { value = (value ^ v) * 1099511628211ULL; }
    void integer(std::uint64_t v, unsigned width) {
        for (unsigned i = 0; i < width; ++i) { byte(v & 255); v >>= 8; }
    }
    void text(const std::string &v) {
        integer(v.size(), 8);
        for (unsigned char c : v) byte(c);
    }
};
Object shape(const auto &buffer) {
    return Object{}.add("device", buffer.device_id()).add("elements", buffer.size())
        .add("allocated_bytes", buffer.allocated_bytes());
}
Object poly(const GpuRNSPoly &value) {
    std::vector<Object> shards;
    for (const auto &s : value.shards)
        shards.push_back(Object{}.add("field_index", s.field_index).add("field_offset", s.field_offset)
            .add("limb_begin", s.limb_begin).add("limb_count", s.limb_count)
            .add("coeff_begin", s.coeff_begin).add("coeff_count", s.coeff_count));
    return Object{}.add("poly_id", value.poly_id).add("N", value.degree)
        .add("q_count", value.q_count).add("p_count", value.p_count).add("shards", shards);
}
std::vector<Object> fields(const std::vector<GpuFieldData> &values) {
    std::vector<Object> result;
    for (const auto &f : values)
        result.push_back(shape(f.buffer).add("field_device", f.device_id)
            .add("payload_bytes", f.size() * sizeof(GpuWord)));
    return result;
}
Object plaintext_meta(const GpuPlaintextMeta &value) {
    Object out;
    out.add("parms_id", value.parms_id).add("N", value.degree)
        .add("q_count", value.q_count).add("p_count", value.p_count)
        .add("is_ntt_form", value.is_ntt_form);
    return bits(out, "scale_bits", value.scale);
}
std::string indexed(const std::string &path, auto i) {
    return path + '/' + std::to_string(i);
}

struct PublicAudit {
    Fingerprint hash;
    std::vector<Object> inventory;
    std::size_t plaintext_owners = 0, plaintext_fields = 0, plaintext_words = 0;
    std::size_t qp_diagonals = 0, qp_compact_bytes = 0, qp_expanded_bytes = 0;
    std::size_t group_views = 0, plan_bytes = 0;

    void record(const std::string &path, const char *kind, const Object &metadata) {
        auto item = Object{}.add("path", path).add("kind", kind).add("metadata", metadata);
        hash.text(item.str());
        inventory.push_back(std::move(item));
    }
    template<class T> void words(const DeviceVector<T> &buffer) {
        static_assert(std::is_same_v<T, std::uint32_t>);
        hash.integer(buffer.size(), 8);
        // Bound host scratch even for uncompressed constants. No GPU kernels.
        std::vector<T> host(std::min(buffer.size(), std::size_t{1} << 20));
        if (!host.empty()) gpu_check_cuda(cudaSetDevice(buffer.device_id()), "fingerprint device");
        for (std::size_t offset = 0; offset < buffer.size(); offset += host.size()) {
            const auto count = std::min(host.size(), buffer.size() - offset);
            gpu_check_cuda(cudaMemcpy(host.data(), buffer.data() + offset, count * sizeof(T),
                                     cudaMemcpyDeviceToHost), "fingerprint public words");
            for (std::size_t i = 0; i < count; ++i) hash.integer(host[i], 4);
        }
    }
    void plain(const std::string &path, const GpuPlaintextData &value) {
        record(path, value.fields_.empty() ? "empty_plaintext_slot" : "plaintext_owner",
               plaintext_meta(value.meta).add("fields", fields(value.fields_)).add("poly", poly(value.poly_)));
        plaintext_owners += !value.fields_.empty();
        for (const auto &f : value.fields_) {
            ++plaintext_fields;
            plaintext_words += f.size();
            words(f.buffer);
        }
    }
    void plains(const std::string &path, const std::vector<GpuPlaintextData> &values) {
        record(path, "plaintext_vector", Object{}.add("size", values.size()));
        for (std::size_t i = 0; i < values.size(); ++i) plain(indexed(path, i), values[i]);
    }
    void terms(const std::string &path, const std::vector<GpuEvalModPolynomialTerm> &values) {
        record(path, "evalmod_terms", Object{}.add("size", values.size()));
        for (std::size_t i = 0; i < values.size(); ++i) {
            const auto name = indexed(path, i);
            record(name, "evalmod_term", Object{}.add("degree", values[i].degree));
            plain(name + "/coefficient_plaintext", values[i].coefficient_plaintext);
        }
    }
    void plan(const std::string &path, const GpuDoubleHoistMatrixPlan &value,
              const GpuMatrixPlainQP &owner, const std::string &owner_path) {
        std::vector<Object> terms, refs;
        for (const auto &t : value.terms) {
            terms.push_back(Object{}.add("giant_index", t.giant_index)
                .add("baby_index", t.baby_index).add("diagonal_id", t.diagonal_id));
            const int diagonal = value.giant_steps.at(t.giant_index) + value.baby_steps.at(t.baby_index);
            const auto &meta = value.compressed_plaintexts
                ? owner.compressed_plain_vec_qp.at(diagonal).meta : owner.plain_vec_qp.at(diagonal).meta;
            const auto period = value.compressed_plaintexts
                ? owner.compressed_plain_vec_qp.at(diagonal).period : meta.degree;
            refs.push_back(Object{}.add("diagonal_id", t.diagonal_id)
                .add("owner", indexed(owner_path + (value.compressed_plaintexts ? "/compressed" : "/plain"), diagonal))
                .add("q_word_offset", 0).add("p_word_offset", meta.q_count * period));
        }
        if (value.diagonal_q_ptrs.size() != value.terms.size() ||
            value.diagonal_p_ptrs.size() != value.terms.size())
            throw std::runtime_error("QP plan pointer-table shape mismatch");
        Object out;
        out.add("log_slots", value.log_slots).add("n1", value.n1).add("n2", value.n2)
            .add("rescale_count", value.rescale_count).add("compressed_plaintexts", value.compressed_plaintexts)
            .add("baby_steps", value.baby_steps).add("giant_steps", value.giant_steps)
            .add("group_term_offsets", value.group_term_offsets).add("terms", terms)
            .add("semantic_diagonal_refs", refs)
            .add("diagonal_q_ptrs_shape_only", shape(value.diagonal_q_ptrs))
            .add("diagonal_p_ptrs_shape_only", shape(value.diagonal_p_ptrs))
            .add("diagonal_periods", shape(value.diagonal_periods))
            .add("term_baby_indices", shape(value.term_baby_indices))
            .add("group_term_offsets_device", shape(value.group_term_offsets_device));
        record(path, "double_hoist_plan", out);
        words(value.diagonal_periods); words(value.term_baby_indices); words(value.group_term_offsets_device);
        plan_bytes += value.diagonal_q_ptrs.allocated_bytes() + value.diagonal_p_ptrs.allocated_bytes()
            + value.diagonal_periods.allocated_bytes() + value.term_baby_indices.allocated_bytes()
            + value.group_term_offsets_device.allocated_bytes();
        // Pointer table CONTENTS are deliberately neither downloaded nor hashed.
    }
    template<class Matrix> Object matrix_meta(const Matrix &value) {
        Object out;
        out.add("log_slots", value.log_slots).add("n1", value.n1).add("level", value.level)
            .add("rot_index", value.rot_index);
        return bits(out, "scale_bits", value.scale);
    }
    template<class Group> void group(const std::string &path, const Group &value) {
        Object out;
        out.add("rot_index", value.rot_index()).add("step", value.step())
            .add("rescale_counts", value.rescale_counts()).add("matrices", value.data().size());
        bits(out, "rescale_min_scale_bits", value.rescale_min_scale());
        record(path, "linear_matrix_group", out);
        for (std::size_t i = 0; i < value.data().size(); ++i) {
            const auto name = indexed(path, i);
            const auto &m = value.data()[i];
            if constexpr (std::is_same_v<Group, GpuLinearMatrixGroup>) {
                record(name, "classic_matrix", matrix_meta(m).add("plain_count", m.plain_vec.size()));
                for (const auto &[diagonal, p] : m.plain_vec) plain(indexed(name + "/plain", diagonal), p);
            } else {
                if (m.group_view) throw std::runtime_error("QP owner unexpectedly marked group_view");
                record(name, "qp_matrix_owner", matrix_meta(m).add("group_view", m.group_view)
                    .add("plain_count", m.plain_vec_qp.size()).add("compressed_count", m.compressed_plain_vec_qp.size()));
                for (const auto &[diagonal, p] : m.plain_vec_qp) {
                    ++qp_diagonals; qp_compact_bytes += (p.meta.q_count + p.meta.p_count) * p.meta.degree * sizeof(GpuWord);
                    qp_expanded_bytes += (p.meta.q_count + p.meta.p_count) * p.meta.degree * sizeof(GpuWord);
                    plain(indexed(name + "/plain", diagonal), p);
                }
                for (const auto &[diagonal, p] : m.compressed_plain_vec_qp) {
                    if (!p.period || (p.period & (p.period - 1)) || p.period > p.meta.degree ||
                        p.residues.size() != p.compact_word_count())
                        throw std::runtime_error("invalid compact QP owner shape");
                    record(indexed(name + "/compressed", diagonal), "compressed_qp_plaintext_owner",
                           plaintext_meta(p.meta).add("period", p.period).add("residues", shape(p.residues))
                               .add("exact_device_reconstruction", p.exact_device_reconstruction));
                    ++qp_diagonals; ++plaintext_owners; ++plaintext_fields;
                    plaintext_words += p.residues.size();
                    qp_compact_bytes += p.residues.size() * sizeof(GpuWord);
                    qp_expanded_bytes += p.full_word_count() * sizeof(GpuWord);
                    words(p.residues);
                }
                plan(name + "/plan", m.plan, m, name);
            }
        }
    }
    void views(const std::string &path, const std::vector<std::vector<GpuMatrixPlainQP>> &values,
               const std::string &owner_path, const GpuLinearMatrixGroupQP &owners) {
        record(path, "qp_group_view_layers", Object{}.add("layers", values.size()));
        for (std::size_t i = 0; i < values.size(); ++i) {
            record(indexed(path, i), "qp_group_views", Object{}.add("size", values[i].size()));
            for (std::size_t j = 0; j < values[i].size(); ++j) {
                const auto &v = values[i][j];
                if (!v.group_view || !v.plain_vec_qp.empty() || !v.compressed_plain_vec_qp.empty())
                    throw std::runtime_error("QP alias view unexpectedly owns plaintexts");
                const auto name = indexed(indexed(path, i), j), parent = indexed(owner_path, i);
                record(name, "qp_alias_view_no_plaintext_allocation",
                       matrix_meta(v).add("group_view", v.group_view).add("plaintext_owner_matrix", parent));
                ++group_views;
                plan(name + "/plan", v.plan, owners.data().at(i), parent);
            }
        }
    }
    void evalmod(const GpuBootstrapData::EvalModData &value) {
        Object out;
#define M(x) out.add(#x, value.x)
#define D(x) bits(out, #x "_bits", value.x)
        D(input_scale); D(target_scale); D(output_scale); M(output_parms_id); M(output_q_count);
        out.add("polynomial_basis", static_cast<unsigned>(value.polynomial_basis));
        M(polynomial_degree); M(polynomial_log_split); M(polynomial_flat_bsgs);
        M(polynomial_degree_bound_virtual); M(polynomial_root_anchor_q_count);
        M(logical_rescale_count); M(dynamic_rescale); D(dynamic_min_scale);
        M(polynomial_rescale_count); M(polynomial_final_rescale_count); D(polynomial_output_scale);
        M(rescale_polynomial_terms_individually); M(polynomial_result_node);
        M(double_angle_rescale_counts); M(required_relin_q_counts);
        out.add("basis_steps", value.basis_steps.size()).add("blocks", value.polynomial_blocks.size())
            .add("combine_steps", value.polynomial_combine_steps.size());
        record("eval_mod", "evalmod_schedule", out);
        plain("eval_mod/input_offset_plaintext", value.input_offset_plaintext);
        terms("eval_mod/polynomial_terms", value.polynomial_terms);
        plains("eval_mod/chebyshev_one_plaintexts", value.chebyshev_one_plaintexts);
        plains("eval_mod/polynomial_coefficients", value.polynomial_coefficients);
        plains("eval_mod/double_angle_constants", value.double_angle_constants);
        for (std::size_t i = 0; i < value.basis_steps.size(); ++i) {
            const auto &s = value.basis_steps[i];
            const auto name = indexed("eval_mod/basis_steps", i);
            Object meta;
#define S(x) meta.add(#x, s.x)
#define SD(x) bits(meta, #x "_bits", s.x)
            S(output_degree); S(left_degree); S(right_degree); S(correction_degree);
            SD(pre_rescale_scale); SD(output_scale); S(rescale_count); S(align_left_operand);
            SD(operand_alignment_pre_rescale_scale); SD(operand_alignment_output_scale);
            S(operand_alignment_rescale_count); SD(correction_alignment_pre_rescale_scale);
            S(correction_alignment_rescale_count);
            record(name, "evalmod_basis_step", meta);
            plain(name + "/operand_alignment_plaintext", s.operand_alignment_plaintext);
            plain(name + "/correction_alignment_plaintext", s.correction_alignment_plaintext);
            plain(name + "/correction_plaintext", s.correction_plaintext);
        }
        for (std::size_t i = 0; i < value.polynomial_blocks.size(); ++i) {
            const auto &s = value.polynomial_blocks[i];
            const auto name = indexed("eval_mod/polynomial_blocks", i);
            Object meta;
            S(rescale_count); SD(output_scale); S(output_q_count);
            record(name, "evalmod_block", meta);
            terms(name + "/terms", s.terms);
        }
        for (std::size_t i = 0; i < value.polynomial_combine_steps.size(); ++i) {
            const auto &s = value.polynomial_combine_steps[i];
            const auto name = indexed("eval_mod/polynomial_combine_steps", i);
            Object meta;
            S(output_node); S(quotient_node); S(remainder_node); S(basis_degree); SD(output_scale);
            S(output_q_count); S(quotient_rescale_count); SD(quotient_output_scale);
            S(remainder_rescale_count); SD(product_scale); S(product_q_count);
            SD(product_aligned_scale); SD(remainder_aligned_scale);
            record(name, "evalmod_combine_step", meta);
            plain(name + "/product_scale_plaintext", s.product_scale_plaintext);
            plain(name + "/remainder_scale_plaintext", s.remainder_scale_plaintext);
        }
#undef S
#undef SD
#undef M
#undef D
    }
};

// Keys are freshly random: inventory only, NEVER call PublicAudit::words here.
Object key_inventory(const GpuEvaluationKeyData *value,
                     std::vector<const GpuEvaluationKeyData *> &owners) {
    if (!value) return Object{}.add("present", false);
    auto found = std::find(owners.begin(), owners.end(), value);
    if (found != owners.end()) return Object{}.add("owner_ref", std::size_t(found - owners.begin()));
    const auto owner_id = owners.size();
    owners.push_back(value);
    const auto &m = value->meta;
    std::vector<Object> polys, mappings, overrides;
    for (const auto &p : value->polys_) polys.push_back(poly(p));
    for (const auto &p : value->poly_metadata_)
        mappings.push_back(Object{}.add("poly_id", p.poly_id).add("key_index", p.key_index)
            .add("decomposition_index", p.decomposition_index).add("component_index", p.component_index));
    if (value->level_keys)
        for (const auto &[q_count, key] : *value->level_keys)
            overrides.push_back(Object{}.add("q_count", q_count).add("key", key_inventory(key.get(), owners)));
    return Object{}.add("owner_id", owner_id).add("key_payload_hashed", false)
        .add("key_parms_id", m.key_parms_id).add("N", m.degree).add("q_count", m.q_count).add("p_count", m.p_count)
        .add("key_count", m.key_count).add("decomposition_count", m.decomposition_count)
        .add("component_count", m.component_count).add("galois_format", static_cast<unsigned>(m.galois_format))
        .add("payload_bytes", payload(*value)).add("fields", fields(value->fields_)).add("polys", polys)
        .add("poly_metadata", mappings).add("galois_elts_by_key_index", value->galois_elts_by_key_index)
        .add("galois_key_q0_ptrs_shape_only", shape(value->galois_key_q0_ptrs))
        .add("galois_key_p0_ptrs_shape_only", shape(value->galois_key_p0_ptrs))
        .add("galois_key_q1_ptrs_shape_only", shape(value->galois_key_q1_ptrs))
        .add("galois_key_p1_ptrs_shape_only", shape(value->galois_key_p1_ptrs)).add("level_keys", overrides);
}
Object fingerprint_profile(const GpuBootstrapProfile &profile, const std::vector<Modulus> &q,
                           const std::vector<Modulus> &p) {
    PublicAudit audit;
    std::vector<std::uint64_t> q_values, p_values;
    for (const auto &v : q) q_values.push_back(v.value());
    for (const auto &v : p) p_values.push_back(v.value());
    Object meta;
    meta.add("profile_id", profile.profile_id).add("cuda_device_id", profile.cuda_device_id)
        .add("input_level_min", profile.input_level_min).add("input_level_max", profile.input_level_max)
        .add("input_components", profile.input_components).add("has_input_scale_log2", bool(profile.input_scale_log2))
        .add("input_scale_log2", profile.input_scale_log2.value_or(0)).add("output_level", profile.output_level)
        .add("output_scale_log2", profile.output_scale_log2).add("output_components", profile.output_components)
        .add("N", std::size_t{1} << 16).add("q_primes", q_values).add("p_primes", p_values)
        .add("level_parameters", profile.level_parameters.size());
    audit.record("profile", "public_profile", meta);
    for (std::size_t i = 0; i < profile.level_parameters.size(); ++i) {
        const auto &level = profile.level_parameters[i];
        std::vector<Object> shards;
        for (const auto &s : level.shards)
            shards.push_back(Object{}.add("device", s.device_id).add("limb_begin", s.limb_begin).add("limb_count", s.limb_count));
        audit.record(indexed("level_parameters", i), "parameter_level_metadata",
                     Object{}.add("parms_id", level.parms_id).add("N", level.degree)
                         .add("q_count", level.q_count).add("p_count", level.p_count).add("shards", shards));
    }
    const auto &value = profile.bootstrap_data;
    Object out;
#define M(x) out.add(#x, value.x)
#define D(x) bits(out, #x "_bits", value.x)
    out.add("linear_transform_mode", static_cast<unsigned>(value.linear_transform_mode));
    M(allow_environment_linear_transform_override); M(slot_to_coeff_first); M(input_q_count);
    D(input_scale); D(output_scale_multiplier); M(q0_parms_id); D(q0_over_message_ratio);
    D(raised_scale_override); M(post_raise_integer_multiplier); D(post_raise_scale_multiplier);
    D(slot_to_coeff_input_scale); D(eval_mod_target_scale); M(project_real); M(output_ratio);
    D(slot_to_coeff_output_scale); M(slot_to_coeff_input_q_count); M(slot_to_coeff_input_parms_id);
    out.add("has_sparse_low_evaluator", value.sparse_low_evaluator != nullptr)
        .add("has_sparse_low_owner", bool(value.sparse_low_owner));
    audit.record("bootstrap_data", "bootstrap_schedule", out);
#undef M
#undef D
    audit.plain("post_raise_plaintext", value.post_raise_plaintext);
    audit.evalmod(value.eval_mod);
    audit.plain("eval_mod_input_offset_plaintext", value.eval_mod_input_offset_plaintext);
    audit.plains("eval_mod_polynomial_plaintexts", value.eval_mod_polynomial_plaintexts);
    audit.plains("double_angle_plaintexts", value.double_angle_plaintexts);
    audit.plain("minus_i_plaintext", value.minus_i_plaintext);
    audit.plain("plus_i_plaintext", value.plus_i_plaintext);
    audit.group("coeff_to_slot_matrix", value.coeff_to_slot_matrix);
    audit.group("slot_to_coeff_matrix", value.slot_to_coeff_matrix);
    audit.group("coeff_to_slot_matrix_qp", value.coeff_to_slot_matrix_qp);
    const auto c2s_diagonals = audit.qp_diagonals;
    audit.group("slot_to_coeff_matrix_qp", value.slot_to_coeff_matrix_qp);
    const auto s2c_diagonals = audit.qp_diagonals - c2s_diagonals;
    audit.views("coeff_to_slot_groups_qp", value.coeff_to_slot_groups_qp,
                "coeff_to_slot_matrix_qp", value.coeff_to_slot_matrix_qp);
    audit.views("slot_to_coeff_groups_qp", value.slot_to_coeff_groups_qp,
                "slot_to_coeff_matrix_qp", value.slot_to_coeff_matrix_qp);
    if (c2s_diagonals != 158 || s2c_diagonals != 158)
        throw std::runtime_error("manual profile expected all 158 C2S + 158 S2C QP diagonals");
    std::vector<const GpuEvaluationKeyData *> key_owners;
    Object keys;
    keys.add("relin", key_inventory(profile.relin_keys.get(), key_owners));
    keys.add("galois", key_inventory(profile.galois_keys.get(), key_owners));
    keys.add("sparse_switch_to_ephemeral", key_inventory(value.sparse_switch_to_ephemeral.get(), key_owners));
    keys.add("sparse_switch_to_dense", key_inventory(value.sparse_switch_to_dense.get(), key_owners));
    keys.add("unique_owners", key_owners.size());
    return Object{}.add("schema", "native-public-profile-fingerprint-v1")
        .add("algorithm", "FNV-1a-64: u64-LE-length-prefixed canonical JSON records, then u64-LE word counts and all u32-LE words")
        .add("noncollisionproof", true).add("cryptographic_or_bytewise_equality_proof", false)
        .add("public_fingerprint", hex64(audit.hash.value)).add("all_public_plaintext_words_hashed", true)
        .add("pointer_table_contents_hashed", false)
        .add("plan_pointer_references", "semantic owner/offset identities derived from host plans; device pointer-table contents not inspected")
        .add("key_payloads_hashed", false).add("key_inventory_in_public_fingerprint", false)
        .add("plaintext_owners", audit.plaintext_owners).add("plaintext_fields", audit.plaintext_fields)
        .add("public_plaintext_words", audit.plaintext_words)
        .add("public_plaintext_payload_bytes", audit.plaintext_words * sizeof(GpuWord))
        .add("qp_diagonals", audit.qp_diagonals).add("c2s_diagonals", c2s_diagonals).add("s2c_diagonals", s2c_diagonals)
        .add("qp_compact_payload_bytes", audit.qp_compact_bytes)
        .add("qp_expanded_payload_bytes", audit.qp_expanded_bytes).add("qp_group_alias_views", audit.group_views)
        .add("plan_allocation_bytes_including_pointer_tables", audit.plan_bytes)
        .add("public_inventory", audit.inventory).add("key_inventory", keys);
}
} // namespace

int main(int argc, char **argv) {
    if (argc != 3 && argc != 4) { std::cerr << "native_profile PUBLIC_PRIMES.txt RESULT.json [HASH_RESULT.json]\n"; return 2; }
    try {
        std::ifstream input(argv[1]);
        std::size_t nq = 0, np = 0;
        input >> nq >> np;
        if (nq != 50 || np != 25) throw std::runtime_error("expected public Q50/P25 primes");
        std::vector<Modulus> q, p;
        for (std::size_t i = 0; i < nq + np; ++i) {
            std::uint64_t value = 0;
            if (!(input >> value)) throw std::runtime_error("truncated public primes");
            (i < nq ? q : p).emplace_back(value);
        }
        gpu_check_cuda(cudaSetDevice(0), "set device");
        gpu_check_cuda(cudaFree(nullptr), "CUDA initialization");
        auto start = Clock::now();
        ParametersLiteral params(CKKS, 16, 15, 40, 192, 1, 0, q, p, sec_level_type::none);
        PoseidonContext context(params);
        const double context_s = std::chrono::duration<double>(Clock::now()-start).count();
        std::cerr << "context_s=" << context_s << '\n' << std::flush;
        start = Clock::now();
        KeyGenerator keygen(context);
        const double sk_s = std::chrono::duration<double>(Clock::now()-start).count();
        start = Clock::now();
        auto profile = GpuBootstrapProfileBuilder::build_resnet20_manual(context, keygen, 0);
        gpu_check_cuda(cudaDeviceSynchronize(), "finish profile upload");
        const double profile_s = std::chrono::duration<double>(Clock::now()-start).count();
        // Optional downloads/inventory/hash happen strictly AFTER the recorded
        // native preparation boundary, and do not change the legacy result.
        if (argc == 4) {
            const auto validation_start = Clock::now();
            auto report = fingerprint_profile(profile, q, p);
            report.add("validation_excluded_from_native_profile_s", true)
                .add("bootstrap_executed", false);
            const double validation_s = std::chrono::duration<double>(Clock::now() - validation_start).count();
            std::ofstream hash_output(argv[3]);
            hash_output.imbue(std::locale::classic());
            hash_output << std::setprecision(17) << "{\"validation_s\":" << validation_s << ',' << report.str().substr(1) << '\n';
            if (!hash_output) throw std::runtime_error("cannot write hash result");
        }
        std::size_t actual_rotation_keys = 0;
        for (auto elt : profile.galois_keys->galois_elts_by_key_index) actual_rotation_keys += elt != 0;
        std::ofstream output(argv[2]);
        output << "{\n  \"context_s\":" << context_s << ",\n  \"secret_key_s\":" << sk_s
               << ",\n  \"native_profile_s\":" << profile_s
               << ",\n  \"relin_payload_bytes\":" << payload(*profile.relin_keys)
               << ",\n  \"galois_payload_bytes\":" << payload(*profile.galois_keys)
               << ",\n  \"actual_rotation_keys\":" << actual_rotation_keys
               << ",\n  \"output_level\":" << profile.output_level
               << ",\n  \"bootstrap_executed\":false,\n  \"security_level\":\"none (diagnostic only)\"\n}\n";
        if (!output) throw std::runtime_error("cannot write result");
        std::cout << "native profile prepared; no bootstrap execution/numerical claim\n";
        return 0;
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
