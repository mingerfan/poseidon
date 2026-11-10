// Standalone CPU-only test: g++ -std=c++17 test_prepared_api_graph.cpp -o /tmp/test_prepared_api_graph
#include "prepared_api_graph.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <iterator>
#include <map>
#include <sstream>

namespace {
using poseidon::benchmark::s2c_first::PreparedApiGraph;
using poseidon::benchmark::s2c_first::PreparedApiTask;

const std::vector<int> strides{1,1,1,2,1,1,2,1,1};
const std::string native_boot="library_degree22_da3_k12_h32_split2_full1";

void require(bool condition,const std::string &message) {
    if(!condition)throw std::runtime_error(message);
}

template<class Exception,class Function>
void must_reject(Function function,const std::string &message) {
    try {function();}
    catch(const Exception &) {return;}
    throw std::runtime_error(message);
}

void require_inputs(const PreparedApiTask &task,std::vector<int> expected) {
    require(task.inputs==expected,"unexpected dependencies for "+task.callee);
}

void require_layout(const PreparedApiTask &task,int h,int w,int c,int k) {
    require(task.h==h && task.w==w && task.c==c && task.k==k,
        "incorrect layout for "+task.callee);
}

void check_graph(const PreparedApiGraph &graph,const std::string &boot) {
    require(graph.bootstrap_implementation==boot,"Boot configuration was not preserved");
    require(graph.tasks.size()==77,"expected 77 prepared calls");
    const auto order=graph.source_order();
    require(order.size()==graph.tasks.size(),"source order size mismatch");
    std::map<std::string,int> counts;
    std::vector<int> uses(graph.tasks.size());
    for(std::size_t id=0;id<graph.tasks.size();++id) {
        const auto &task=graph.tasks[id];
        require(order[id]==int(id),"source order is not topological");
        require(task.q_count>0 && task.q_count<=50,"invalid output Q count");
        require(task.latency_us>0,"prepared latency must be positive");
        ++counts[task.callee];
        for(int input:task.inputs) {
            require(input>=0 && input<int(id),"dependency is not an earlier task");
            ++uses.at(std::size_t(input));
        }
    }
    require(counts==std::map<std::string,int>{
        {"stem",1},{"conv",18},{"relu",19},{"bootstrap."+boot,18},
        {"clone",9},{"residual_add",9},{"shortcut",2},{"head",1}},
        "prepared call inventory mismatch");
    for(std::size_t id=0;id+1<uses.size();++id)
        require(uses[id]>0,"unreachable prepared task");
    require(uses.back()==0,"final logits must be the sink");

    const auto task_at=[&](int id,const std::string &callee,int q)->const PreparedApiTask & {
        const auto &task=graph.tasks.at(std::size_t(id));
        require(task.callee==callee,"incorrect callee at task "+std::to_string(id));
        require(task.q_count==q,"incorrect output Q at task "+std::to_string(id));
        return task;
    };
    const auto &stem=task_at(0,"stem",31);
    require_inputs(stem,{});
    require_layout(stem,32,32,16,1);
    require_inputs(task_at(1,"relu",9),{0});
    int active=1,cursor=2,conv=0,h=32,w=32,c=16,k=1;
    for(int block=0;block<9;++block) {
        const int stride=strides.at(std::size_t(block));
        int shortcut=cursor++;
        require_inputs(task_at(shortcut,"clone",9),{active});
        const int first=cursor++;
        const auto &conv1=task_at(first,"conv",6);
        require_inputs(conv1,{active});
        require_layout(conv1,h,w,c,k);
        require(conv1.index==conv++,"first Conv index mismatch");
        const int boot1=cursor++;
        require_inputs(task_at(boot1,"bootstrap."+boot,31),{first});
        const int relu1=cursor++;
        require_inputs(task_at(relu1,"relu",9),{boot1});
        const int nh=h/stride,nw=w/stride,nc=c*stride,nk=k*stride;
        const int second=cursor++;
        const auto &conv2=task_at(second,"conv",6);
        require_inputs(conv2,{relu1});
        require_layout(conv2,nh,nw,nc,nk);
        require(conv2.index==conv++,"second Conv index mismatch");
        if(stride==2) {
            const int downsample=cursor++;
            const auto &task=task_at(downsample,"shortcut",8);
            require_inputs(task,{shortcut});
            require_layout(task,h,w,c,k);
            require(task.index==block,"Option-A block index mismatch");
            shortcut=downsample;
        }
        const int sum=cursor++;
        const auto &add=task_at(sum,"residual_add",6);
        require_inputs(add,{second,shortcut});
        require_layout(add,nh,nw,nc,nk);
        require(add.index==block,"residual block index mismatch");
        const int boot2=cursor++;
        require_inputs(task_at(boot2,"bootstrap."+boot,31),{sum});
        active=cursor++;
        require_inputs(task_at(active,"relu",9),{boot2});
        h=nh;w=nw;c=nc;k=nk;
    }
    const auto &head=task_at(cursor++,"head",4);
    require_inputs(head,{active});
    require_layout(head,8,8,64,4);
    require(cursor==int(graph.tasks.size()) && conv==18,"incomplete graph traversal");
}

std::size_t occurrences(const std::string &text,const std::string &needle) {
    std::size_t count=0,offset=0;
    while((offset=text.find(needle,offset))!=std::string::npos) {
        ++count;offset+=needle.size();
    }
    return count;
}

struct TemporaryDirectory {
    std::filesystem::path path;
    TemporaryDirectory() {
        const auto stamp=std::chrono::steady_clock::now().time_since_epoch().count();
        path=std::filesystem::temp_directory_path()/
            ("poseidon_prepared_graph_test_"+std::to_string(stamp));
        require(std::filesystem::create_directory(path),"cannot create test directory");
    }
    ~TemporaryDirectory() {
        std::error_code error;
        std::filesystem::remove_all(path,error);
    }
};

void check_mlir(const PreparedApiGraph &graph,const std::filesystem::path &path) {
    // Exact Q50 prefix from relu_precision_fixture.txt; no GPU/context is needed.
    const std::vector<std::uint64_t> moduli{
        4255252481ULL,4255645697ULL,4258922497ULL,4260364289ULL,4261675009ULL,
        4264427521ULL,4265476097ULL,4266393601ULL,4267442177ULL,1040842753ULL,
        1042415617ULL,1043464193ULL,1045430273ULL,1048707073ULL,1049100289ULL,
        1051721729ULL,1052508161ULL,1053818881ULL,1054212097ULL,1055260673ULL,
        1056178177ULL,1056440321ULL,1060765697ULL,1062469633ULL,1062862849ULL,
        1064697857ULL,1065484289ULL,1068236801ULL,1070727169ULL,1071513601ULL,
        1073479681ULL,2135162881ULL,4272291841ULL,4272685057ULL,1040056321ULL,
        2135818241ULL,4273340417ULL,2142502913ULL,4274126849ULL,4276092929ULL,
        2144468993ULL,2146041857ULL,2146959361ULL,4277403649ULL,4279369729ULL,
        2147352577ULL,4280025089ULL,4280156161ULL,4281204737ULL,1038745601ULL};
    graph.write_mlir(path.string(),moduli);
    std::ifstream input(path);
    require(bool(input),"cannot read generated MLIR");
    const std::string mlir{std::istreambuf_iterator<char>(input),std::istreambuf_iterator<char>()};
    require(occurrences(mlir,"\"ckks.apicall\"")==77,"MLIR lost prepared calls");
    require(occurrences(mlir,"callee = \"bootstrap."+graph.bootstrap_implementation+"\"")==18,
        "MLIR lost native Boot configuration");
    require(mlir.find("poseidon.preplanned = true")!=std::string::npos,
        "MLIR lost preplanned marker");
    require(mlir.find("poseidon.task_granularity = \"prepared_api\"")!=std::string::npos,
        "MLIR lost task granularity");
    require(mlir.find("poseidon.bootstrap_implementation = \""+
        graph.bootstrap_implementation+"\"")!=std::string::npos,"MLIR lost Boot metadata");
    require(mlir.find("ckks.rns_moduli = array<i64: 4255252481, 4255645697, 4258922497, ")
        !=std::string::npos,"MLIR lost exact modulus values");
    require(mlir.find("decrypt_reencrypt")==std::string::npos,
        "native prepared Boot was replaced with re-encryption");

    std::istringstream lines(mlir);
    std::string line;
    std::size_t calls=0;
    while(std::getline(lines,line)) {
        if(line.find("\"ckks.apicall\"")==std::string::npos)continue;
        const auto &task=graph.tasks.at(calls);
        std::ostringstream expected;
        expected<<"    %v"<<calls<<" = \"ckks.apicall\"(";
        for(std::size_t i=0;i<task.inputs.size();++i)
            expected<<(i?", ":"")<<"%v"<<task.inputs[i];
        expected<<") {callee = \""<<task.callee<<"\", call_id = "<<calls
            <<" : i64, latency_us = "<<task.latency_us
            <<" : i64, poseidon.q_count = "<<task.q_count<<" : i64} : (";
        for(std::size_t i=0;i<task.inputs.size();++i) {
            const auto &source=graph.tasks.at(std::size_t(task.inputs[i]));
            expected<<(i?", ":"")<<"tensor<1x!ckks.poly<2 * 40 * "<<source.q_count-1<<">>";
        }
        expected<<") -> tensor<1x!ckks.poly<2 * 40 * "<<task.q_count-1<<">>";
        require(line==expected.str(),"incorrect physical MLIR contract for task "+std::to_string(calls));
        ++calls;
    }
    require(calls==77,"incomplete MLIR traversal");
    require(mlir.find("return %v76 : tensor<1x!ckks.poly<2 * 40 * 3>>")
        !=std::string::npos,"Q4 logits must be exported at physical level 3");
}
} // namespace

int main() {
    try {
        const auto graph=PreparedApiGraph::resnet20(strides,native_boot);
        check_graph(graph,native_boot);
        TemporaryDirectory temporary;
        check_mlir(graph,temporary.path/"native.mlir");
        const auto historical=PreparedApiGraph::resnet20(strides,"application_degree59");
        check_graph(historical,"application_degree59");
        check_mlir(historical,temporary.path/"historical.mlir");
        must_reject<std::runtime_error>([] {
            PreparedApiGraph::resnet20(std::vector<int>(8,1),native_boot);
        },"eight-block topology was accepted");
        must_reject<std::runtime_error>([] {
            PreparedApiGraph::resnet20(std::vector<int>(10,1),native_boot);
        },"ten-block topology was accepted");
        must_reject<std::runtime_error>([&] {
            graph.write_mlir((temporary.path/"missing"/"graph.mlir").string(),{4255252481ULL});
        },"unwritable MLIR destination was accepted");
        must_reject<std::out_of_range>([&] {
            auto invalid=graph;
            invalid.tasks[1].inputs={int(invalid.tasks.size())};
            invalid.write_mlir((temporary.path/"invalid.mlir").string(),{4255252481ULL});
        },"out-of-range task dependency was accepted");
        std::cout<<"Prepared API graph tests PASS: 77 calls, 18 Boots, 19 ReLUs, Q4 logits\n";
        return 0;
    } catch(const std::exception &error) {
        std::cerr<<error.what()<<'\n';
        return 1;
    }
}
