#pragma once

// The task boundaries are prepared GPU API calls, not new numerical kernels.
// All rescale, polynomial, hoisting and Boot choices stay inside those calls.
#include <cstdint>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace poseidon::benchmark::s2c_first {

struct PreparedApiTask {
    std::string callee;
    std::vector<int> inputs;
    int q_count;
    int latency_us;
    int index=0;
    int h=32,w=32,c=16,k=1;
};

struct PreparedApiGraph {
    std::vector<PreparedApiTask> tasks;
    std::string bootstrap_implementation;

    int add(std::string callee,std::vector<int> inputs,int q,int latency,
        int index=0,int h=32,int w=32,int c=16,int k=1) {
        tasks.push_back({std::move(callee),std::move(inputs),q,latency,index,h,w,c,k});
        return int(tasks.size()-1);
    }

    // This is the outer API program of the manual network. In particular the
    // Q31 -> Q9 ReLU and Q6 -> Q31 Boot are explicit, immutable contracts.
    static PreparedApiGraph resnet20(const std::vector<int> &strides,
        std::string boot) {
        if(strides.size()!=9)throw std::runtime_error("ResNet20 needs nine blocks");
        PreparedApiGraph g;g.bootstrap_implementation=std::move(boot);
        int active=g.add("stem",{},31,10000);
        active=g.add("relu",{active},9,56000);
        int h=32,w=32,c=16,k=1,conv=0;
        for(int block=0;block<9;++block) {
            const int stride=strides.at(block);
            int shortcut=g.add("clone",{active},9,30);
            int first=g.add("conv",{active},6,38000,conv++,h,w,c,k);
            int boot1=g.add("bootstrap."+g.bootstrap_implementation,{first},31,200000);
            int relu1=g.add("relu",{boot1},9,56000);
            const int nh=h/stride,nw=w/stride,nk=k*stride,nc=c*stride;
            int second=g.add("conv",{relu1},6,38000,conv++,nh,nw,nc,nk);
            if(stride==2)
                shortcut=g.add("shortcut",{shortcut},8,14000,block,h,w,c,k);
            int sum=g.add("residual_add",{second,shortcut},6,30,block,nh,nw,nc,nk);
            int boot2=g.add("bootstrap."+g.bootstrap_implementation,{sum},31,200000);
            active=g.add("relu",{boot2},9,56000);
            h=nh;w=nw;c=nc;k=nk;
        }
        g.add("head",{active},4,12000,0,h,w,c,k);
        return g;
    }

    std::vector<int> source_order() const {
        std::vector<int> order(tasks.size());
        std::iota(order.begin(),order.end(),0);return order;
    }

    void write_mlir(const std::string &path,
        const std::vector<std::uint64_t> &moduli) const {
        std::ofstream out(path);
        if(!out)throw std::runtime_error("cannot write MLIR: "+path);
        out<<"module attributes {poseidon.preplanned = true, poseidon.task_granularity = \"prepared_api\", poseidon.bootstrap_implementation = \""
           <<bootstrap_implementation<<"\", ckks.rns_moduli = array<i64: ";
        for(std::size_t i=0;i<moduli.size();++i)out<<(i?", ":"")<<moduli[i];
        out<<">} {\n  func.func @resnet20_manual() -> tensor<1x!ckks.poly<2 * 40 * 3>> {\n";
        const auto type=[&](int id) {
            return "tensor<1x!ckks.poly<2 * 40 * "+std::to_string(tasks.at(id).q_count-1)+">>";
        };
        for(std::size_t id=0;id<tasks.size();++id) {
            const auto &task=tasks[id];
            out<<"    %v"<<id<<" = \"ckks.apicall\"(";
            for(std::size_t i=0;i<task.inputs.size();++i)
                out<<(i?", ":"")<<"%v"<<task.inputs[i];
            out<<") {callee = \""<<task.callee<<"\", call_id = "<<id
               <<" : i64, latency_us = "<<task.latency_us
               <<" : i64, poseidon.q_count = "<<task.q_count<<" : i64}";
            out<<" : (";
            for(std::size_t i=0;i<task.inputs.size();++i)
                out<<(i?", ":"")<<type(task.inputs[i]);
            out<<") -> "<<type(int(id))<<"\n";
        }
        out<<"    return %v"<<tasks.size()-1<<" : "<<type(int(tasks.size()-1))<<"\n  }\n}\n";
    }

};

} // namespace poseidon::benchmark::s2c_first
