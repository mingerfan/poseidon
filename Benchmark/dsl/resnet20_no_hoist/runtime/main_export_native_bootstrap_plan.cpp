// CPU-only plan export from the same native setup code; no keys or GPU calls.
#define POSEIDON_BOOTSTRAP_RELU_LIBRARY
#include "native_snapshot/Doc/analysis/resnet20_joint_chain_20260911/bootstrap_relu_precision.cpp"
#include <bit>
#include <fstream>

namespace {
std::ostream *records=nullptr;
std::uint64_t constantId=0;
std::uint64_t bits(double value){return std::bit_cast<std::uint64_t>(value);}
void plain(const char *role,const GpuPlaintextData &p){
    *records<<",\""<<role<<"\":{\"constant\":"<<p.poly_.poly_id
        <<",\"q\":"<<p.meta.q_count<<",\"scale_bits\":"<<bits(p.meta.scale)<<'}';
}
void terms(const char *kind,std::size_t block,const auto &v){
    std::size_t index=0;
    for(const auto &term:v){
        *records<<"{\"kind\":\""<<kind<<"\",\"block\":"<<block
            <<",\"index\":"<<index++<<",\"degree\":"<<term.degree;
        plain("coefficient",term.coefficient_plaintext);*records<<"}\n";
    }
}
void exportEvalMod(const BaselineMetadata &m){
    FoldSpec fold(m);const auto p=fold.upload(m);
    *records<<"{\"kind\":\"evalmod\",\"input_q\":46,\"input_scale_bits\":"<<bits(p.input_scale)
        <<",\"target_scale_bits\":"<<bits(p.target_scale)<<",\"output_q\":"<<p.output_q_count
        <<",\"output_scale_bits\":"<<bits(p.output_scale)<<",\"fold_alpha_bits\":"<<bits(fold.alpha)
        <<",\"dynamic\":"<<int(p.dynamic_rescale)<<",\"result_node\":"<<p.polynomial_result_node
        <<",\"polynomial_output_scale_bits\":"<<bits(p.polynomial_output_scale)
        <<",\"polynomial_rescale_count\":"<<p.polynomial_rescale_count
        <<",\"individual_term_rescale\":"<<int(p.rescale_polynomial_terms_individually);
    plain("input_offset",p.input_offset_plaintext);*records<<"}\n";
    for(const auto &s:p.basis_steps){
        *records<<"{\"kind\":\"basis\",\"degree\":"<<s.output_degree<<",\"left\":"<<s.left_degree
            <<",\"right\":"<<s.right_degree<<",\"correction\":"<<s.correction_degree
            <<",\"pre_bits\":"<<bits(s.pre_rescale_scale)<<",\"output_bits\":"<<bits(s.output_scale)
            <<",\"rescale_count\":"<<s.rescale_count<<",\"align_left\":"<<int(s.align_left_operand)
            <<",\"alignment_pre_bits\":"<<bits(s.operand_alignment_pre_rescale_scale)
            <<",\"alignment_output_bits\":"<<bits(s.operand_alignment_output_scale)
            <<",\"alignment_rescale_count\":"<<s.operand_alignment_rescale_count
            <<",\"correction_pre_bits\":"<<bits(s.correction_alignment_pre_rescale_scale)
            <<",\"correction_rescale_count\":"<<s.correction_alignment_rescale_count;
        plain("alignment",s.operand_alignment_plaintext);plain("correction_alignment",s.correction_alignment_plaintext);
        plain("correction_plain",s.correction_plaintext);*records<<"}\n";
    }
    terms("root_term",0,p.polynomial_terms);
    std::size_t index=0;
    for(const auto &b:p.polynomial_blocks){
        *records<<"{\"kind\":\"block\",\"index\":"<<index<<",\"q\":"<<b.output_q_count
            <<",\"scale_bits\":"<<bits(b.output_scale)<<",\"rescale_count\":"<<b.rescale_count<<"}\n";
        terms("block_term",index++,b.terms);
    }
    for(const auto &s:p.polynomial_combine_steps){
        *records<<"{\"kind\":\"combine\",\"node\":"<<s.output_node<<",\"quotient\":"<<s.quotient_node
            <<",\"remainder\":"<<s.remainder_node<<",\"degree\":"<<s.basis_degree
            <<",\"q\":"<<s.output_q_count<<",\"output_bits\":"<<bits(s.output_scale)
            <<",\"quotient_rescale_count\":"<<s.quotient_rescale_count
            <<",\"quotient_output_bits\":"<<bits(s.quotient_output_scale)
            <<",\"remainder_rescale_count\":"<<s.remainder_rescale_count
            <<",\"product_q\":"<<s.product_q_count<<",\"product_bits\":"<<bits(s.product_scale)
            <<",\"product_aligned_bits\":"<<bits(s.product_aligned_scale)
            <<",\"remainder_aligned_bits\":"<<bits(s.remainder_aligned_scale);
        plain("product_alignment",s.product_scale_plaintext);plain("remainder_alignment",s.remainder_scale_plaintext);
        *records<<"}\n";
    }
    for(std::size_t i=0;i<p.double_angle_constants.size();++i){
        *records<<"{\"kind\":\"double_angle\",\"index\":"<<i
            <<",\"rescale_count\":"<<p.double_angle_rescale_counts.at(i);
        plain("constant",p.double_angle_constants[i]);*records<<"}\n";
    }
}
}

std::uint64_t poseidon_bootstrap_export_scalar(std::complex<double> value,std::size_t q,double scale){
    if(!records)throw std::runtime_error("export observer not initialized");
    const auto id=++constantId;
    *records<<"{\"kind\":\"scalar\",\"id\":"<<id<<",\"q\":"<<q
        <<",\"scale_bits\":"<<bits(scale)<<",\"real_bits\":"<<bits(value.real())
        <<",\"imag_bits\":"<<bits(value.imag())<<"}\n";
    return id;
}

int main(int argc,char **argv){
    if(argc!=3)throw std::runtime_error("usage: native_bootstrap_plan_export fixture output.jsonl");
    std::ofstream output(argv[2]);if(!output)throw std::runtime_error("cannot create export");records=&output;
    output<<"{\"kind\":\"scope\",\"gpu_executed\":false,\"keys_generated\":false,"
          "\"native_template_complete\":false,\"note\":\"Native folded EvalMod setup only; DFT and primitive lowering still required\"}\n";
    std::string mode="--metadata-only",profile="q50-fixed45",dataset="grid";
    char *args[]={argv[0],mode.data(),profile.data(),dataset.data(),argv[1]};
    const int rc=run_bootstrap_baseline(5,args,[&](const BaselineMetadata &m){exportEvalMod(m);});
    output.flush();if(!output)throw std::runtime_error("export write failed");
    if(rc==0)std::cout<<"NATIVE_FOLDED_EVALMOD_PLAN_EXPORT PASS gpu_executed=false keys_generated=false scalars="<<constantId<<'\n';
    return rc;
}
