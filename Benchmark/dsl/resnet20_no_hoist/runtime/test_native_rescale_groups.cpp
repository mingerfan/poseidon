#include "hevm_native_rescale_groups.h"
#include <cassert>
#include <fstream>
#include <iostream>
#include <iterator>
std::string read(const char *path){std::ifstream in(path,std::ios::binary);assert(in);return {std::istreambuf_iterator<char>(in),{}};}
struct MockCT {struct {std::uint64_t q_count=50,parms_id=49;double scale=std::exp2(40);} meta;};
struct MockEval {std::size_t single=0,many=0;
    void rescale(const MockCT &src,MockCT &dst){++single;dst=src;--dst.meta.q_count;--dst.meta.parms_id;}
    void rescale_many(const MockCT &src,MockCT &dst,std::uint64_t count){++many;dst=src;dst.meta.q_count-=count;dst.meta.parms_id-=count;}
};
int main(int argc,char **argv){assert(argc==3);auto binary=read(argv[1]),text=read(argv[2]);
    auto groups=poseidon::mgpu::parse_hevm_native_rescale_groups(binary,text);
    std::size_t seq=0,many=0,x2=0;
    for(auto g:groups){seq+=g.mode==0;many+=g.mode==1;x2+=g.mode==1 && g.count==2;}
    assert(seq==74 && many==760 && x2==247);
    MockEval eval;for(auto g:groups){MockCT source,destination;
        poseidon::mgpu::execute_hevm_native_rescale_group(eval,source,destination,g);}
    assert(eval.single==74 && eval.many==760);
    MockCT left,right;right.meta.scale*=1+1e-12;
    poseidon::mgpu::reconcile_hevm_native_scale(left,right);assert(right.meta.scale==left.meta.scale);
    right.meta.scale*=1.001;bool alignment_rejected=false;
    try{poseidon::mgpu::reconcile_hevm_native_scale(left,right);}catch(const std::invalid_argument &){alignment_rejected=true;}
    assert(alignment_rejected);
    auto rejects=[&](const std::string &b,const std::string &t){bool rejected=false;
        try{poseidon::mgpu::parse_hevm_native_rescale_groups(b,t);}catch(const std::invalid_argument &){rejected=true;}assert(rejected);};
    auto corrupt=binary;corrupt.back()^=1;rejects(corrupt,text);
    auto first_end=text.find('\n'),row_end=text.find('\n',first_end+1);
    auto row=text.substr(first_end+1,row_end-first_end-1);rejects(binary,text+row+"\n");
    auto bad=text;bad.replace(first_end+1,row_end-first_end-1,"0 2 0 1 0 1 1");rejects(binary,bad);
    auto wrong=text;wrong.replace(first_end+1,row_end-first_end-1,"0 4 0 1 2 1 1");rejects(binary,wrong);
    std::cout<<"NATIVE_RESCALE_GROUPS PASS groups="<<groups.size()<<" sequential="<<seq<<" many="<<many<<" x2="<<x2<<" negative_tests=4 gpu_executed=false\n";
}
