// Independent trusted host observer. Never launch, stop, or modify the target.
#include "/home/lhohy/Code/Poseidon/src/poseidon/util/json.h"
#include "/home/lhohy/Code/Poseidon-resnet20/third_party/ckks-runtime/runtime/utils/sha256.cpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cerrno>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <thread>
#include <unistd.h>
#include <vector>
using J=nlohmann::json;
const std::array<const char*,6> NS={"pid","user","mnt","net","ipc","uts"};
constexpr std::size_t PROC_LIMIT=1024*1024;
void need(bool ok,const char*why){if(!ok)throw std::runtime_error(why);}
std::uint64_t mono(){return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();}
std::uint64_t real(){return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();}
std::string bytes(const std::string&path,std::size_t limit=PROC_LIMIT){
    int fd=::open(path.c_str(),O_RDONLY|O_CLOEXEC|O_NONBLOCK);need(fd>=0,"bounded_input_open");
    std::string result;std::array<char,16384>buffer{};
    try{for(;;){ssize_t n=::read(fd,buffer.data(),buffer.size());if(n<0&&errno==EINTR)continue;need(n>=0,"bounded_input_read");if(!n)break;need(result.size()+n<=limit,"bounded_input_size");result.append(buffer.data(),n);}}catch(...){::close(fd);throw;}
    ::close(fd);return result;
}
std::string link(const std::string&path){std::array<char,8192>buffer{};auto n=::readlink(path.c_str(),buffer.data(),buffer.size());need(n>0&&std::size_t(n)<buffer.size(),"bounded_kernel_link");return std::string(buffer.data(),n);}
std::string proc(int pid,const char*name){return "/proc/"+std::to_string(pid)+"/"+name;}
J identity(const struct stat&s){return J{{"device",std::uint64_t(s.st_dev)},{"inode",std::uint64_t(s.st_ino)},{"kind",std::uint64_t(s.st_mode&S_IFMT)}};}
J signature(const struct stat&s){auto value=identity(s);need(S_ISREG(s.st_mode)&&s.st_size>=0&&std::uint64_t(s.st_size)<=512*1024*1024ULL,"public_executable_shape");value["bytes"]=std::uint64_t(s.st_size);value["mtime_ns"]=std::uint64_t(s.st_mtim.tv_sec)*1000000000ULL+s.st_mtim.tv_nsec;value["ctime_ns"]=std::uint64_t(s.st_ctim.tv_sec)*1000000000ULL+s.st_ctim.tv_nsec;return value;}
J path_identity(const std::string&path){struct stat s{};need(::stat(path.c_str(),&s)==0&&(S_ISREG(s.st_mode)||S_ISDIR(s.st_mode)),"actual_mount_destination_identity");return identity(s);}
J namespaces(int pid){J value=J::object();for(auto name:NS)value[name]=link(proc(pid,(std::string("ns/")+name).c_str()));return value;}
std::pair<J,int>process(int pid,const std::string&boot){
    auto raw=bytes(proc(pid,"stat"));auto end=raw.rfind(") ");need(end!=std::string::npos,"process_stat_suffix");
    std::istringstream input(raw.substr(end+2));std::vector<std::string>fields;std::string field;while(input>>field)fields.push_back(field);
    need(fields.size()>19&&fields[0]!="Z","live_process_stat");
    return {J{{"pid",pid},{"start_ticks",std::stoull(fields[19])},{"boot_id",boot}},std::stoi(fields[1])};
}
J ancestry(int pid,const J&owner,const std::string&boot){
    J rows=J::array();int cursor=pid;
    for(int i=0;i<64;++i){auto p=process(cursor,boot);rows.push_back(J{{"process",p.first},{"parent_pid",p.second}});if(p.first==owner)return rows;need(p.second>1&&p.second!=cursor,"original_owner_ancestry");cursor=p.second;}
    throw std::runtime_error("bounded_owner_ancestry");
}
J nul_strings(const std::string&raw){J rows=J::array();std::size_t at=0;while(at<raw.size()){auto end=raw.find('\0',at);need(end!=std::string::npos,"complete_nul_fields");rows.push_back(raw.substr(at,end-at));at=end+1;}return rows;}
J public_environment(int pid,const J&expected){
    auto rows=nul_strings(bytes(proc(pid,"environ")));J value=J::object();
    for(const auto&item:rows){auto text=item.get<std::string>();auto sep=text.find('=');need(sep!=std::string::npos,"environment_field");auto key=text.substr(0,sep);need((expected.contains(key)||key=="PWD")&&!value.contains(key),"unexpected_environment_name_redacted");value[key]=text.substr(sep+1);}
    return value;
}
void write_new(const std::string&path,const J&value){
    auto raw=value.dump();need(raw.size()<=4*PROC_LIMIT,"bounded_observer_receipt");
    int fd=::open(path.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW|O_CLOEXEC,0400);need(fd>=0,"fresh_observer_output");
    try{std::size_t at=0;while(at<raw.size()){auto n=::write(fd,raw.data()+at,raw.size()-at);if(n<0&&errno==EINTR)continue;need(n>0,"observer_output_write");at+=n;}need(::fsync(fd)==0,"observer_output_sync");}catch(...){::close(fd);throw;}::close(fd);
}
void original_limits(){
    struct Limit{int name;rlim_t soft,hard;};
    const Limit values[]={{RLIMIT_CPU,150,155},{RLIMIT_AS,4ULL*1024*1024*1024,4ULL*1024*1024*1024},
        {RLIMIT_FSIZE,16ULL*1024*1024,16ULL*1024*1024},{RLIMIT_NOFILE,128,128},{RLIMIT_CORE,0,0}};
    for(const auto&item:values){rlimit value{};need(::getrlimit(item.name,&value)==0&&value.rlim_cur==item.soft&&value.rlim_max==item.hard,"unchanged_observer_native_limits");}
}
void capture(const J&config,int pid,const std::string&boot,unsigned attempt){
    const auto&base=config.at("base_contract");const auto owner_pid=base.at("owner").at("pid").get<int>();
    const auto owner=process(owner_pid,boot).first;const auto owner_ns=namespaces(owner_pid);
    need(owner==base.at("owner")&&owner_ns==base.at("owner_namespaces"),"original_live_owner");
    auto begin=mono(),observed=real();auto target=process(pid,boot).first;
    int fd=::open(proc(pid,"exe").c_str(),O_RDONLY|O_NONBLOCK|O_CLOEXEC);need(fd>=0,"live_target_executable_open");
    try{
        struct stat opened{},ended{};need(::fstat(fd,&opened)==0,"opened_executable_stat");auto held=signature(opened);
        for(const auto*name:{"device","inode","kind","bytes"})need(held.at(name)==base.at("executable").at(name),"exact_live_public_executable_identity");
        // Read mappings first, before parsing the command, environment or mounts.
        auto maps=bytes(proc(pid,"maps"));auto argv=nul_strings(bytes(proc(pid,"cmdline")));need(argv==base.at("argv"),"original_exact_target_argv");
        auto chain=ancestry(pid,owner,boot);auto environment=public_environment(pid,base.at("environment"));
        auto target_ns=namespaces(pid);auto cwd=link(proc(pid,"cwd"));auto mounts=bytes(proc(pid,"mountinfo"));auto limits=bytes(proc(pid,"limits"));
        J bindings=J::array();for(const auto&row:base.at("bindings")){auto destination=row.at("destination").get<std::string>();need(!destination.empty()&&destination[0]=='/',"absolute_frozen_mount_destination");bindings.push_back(J{{"destination",destination},{"identity",path_identity(proc(pid,"root")+destination)}});}
        auto end_target=process(pid,boot).first;need(::stat(proc(pid,"exe").c_str(),&ended)==0,"live_end_executable_stat");
        auto end_owner=process(owner_pid,boot).first;auto end_owner_ns=namespaces(owner_pid);
        need(target==end_target&&held==signature(ended)&&owner==end_owner&&owner_ns==end_owner_ns,"stable_live_raw_capture");
        J names=J::array();for(auto it=environment.begin();it!=environment.end();++it)names.push_back(it.key());
        J fields={{"format","poseidon-approved-cpu-compiler-process-observation-v1"},{"contract_binding",base.at("binding")},{"nonce",base.at("nonce")},{"role",base.at("role")},
            {"owner",owner},{"process",target},{"ancestry",chain},{"observed_at_ns",observed},{"argv",argv},{"cwd",cwd},{"environment_names",names},{"public_environment",environment},
            {"namespaces",target_ns},{"owner_namespaces",owner_ns},{"actual_mountinfo",mounts},{"actual_maps",maps},{"actual_limits",limits},{"bindings",bindings},
            {"private_file_reads",0},{"private_key_hashes",0},{"process_memory_reads",0},{"processes_started",0},{"compilation_completed_or_source_semantics_or_native_or_first_delivery_certified",false}};
        J raw={{"format","poseidon-native-fast-compiler-raw-v1"},{"config_binding",config.at("binding")},{"snapshot_fields",fields},{"opened_executable",held},{"end_executable",signature(ended)},
            {"end_target",end_target},{"end_owner",end_owner},{"end_owner_namespaces",end_owner_ns},{"raw_begin_monotonic_ns",begin},{"raw_end_monotonic_ns",mono()},{"attempt",attempt},{"test_only",false}};
        const auto directory=config.at("evidence_directory").get<std::string>();auto raw_path=directory+"/NATIVE_RAW_SNAPSHOT.json";write_new(raw_path,raw);
        const auto hash_begin=mono();need(::lseek(fd,0,SEEK_SET)==0,"held_executable_seek");fhegpu::Sha256 hash;std::array<char,65536>buffer{};std::uint64_t size=0;
        for(;;){auto n=::read(fd,buffer.data(),buffer.size());if(n<0&&errno==EINTR)continue;need(n>=0,"held_executable_read");if(!n)break;hash.update(std::string_view(buffer.data(),n));size+=n;need(size<=held.at("bytes").get<std::uint64_t>(),"held_executable_size_bound");}
        struct stat after{};need(::fstat(fd,&after)==0&&signature(after)==held&&size==held.at("bytes"),"stable_held_executable_after_hash");
        auto executable=identity(after);executable["bytes"]=size;executable["sha256"]=hash.hex_digest();
        write_new(directory+"/NATIVE_EXECUTABLE_HASH.json",J{{"format","poseidon-native-held-executable-hash-v1"},{"config_binding",config.at("binding")},
            {"raw_sha256",fhegpu::sha256_hex(bytes(raw_path,4*PROC_LIMIT))},{"opened_executable",held},{"executable",executable},
            {"hash_begin_monotonic_ns",hash_begin},{"hash_end_monotonic_ns",mono()},{"raw_saved_before_hash",true},{"test_only",false}});
    }catch(...){::close(fd);throw;}::close(fd);
}
int main(int argc,char**argv){try{
    need(argc==3,"observer_config_arguments");original_limits();auto raw=bytes(argv[1]);need(fhegpu::sha256_hex(raw)==argv[2],"externally_pinned_observer_config");auto config=J::parse(raw);
    need(config.at("format")=="poseidon-native-fast-compiler-config-v1"&&config.at("test_only")==false&&config.at("maximum_attempts")==4&&config.at("wall_seconds")==90,"fixed_observer_scope");
    auto boot=bytes("/proc/sys/kernel/random/boot_id");while(!boot.empty()&&(boot.back()=='\n'||boot.back()=='\r'))boot.pop_back();
    const auto&base=config.at("base_contract");need(base.at("owner").at("boot_id")==boot,"same_original_host_boot");
    auto owner_pid=base.at("owner").at("pid").get<int>();need(process(owner_pid,boot).first==base.at("owner"),"prelaunch_owner_live");
    auto directory=config.at("evidence_directory").get<std::string>();write_new(directory+"/NATIVE_OBSERVER_READY.json",J{{"config_binding",config.at("binding")},{"observer",process(::getpid(),boot).first},{"namespaces",namespaces(::getpid())},{"original_limits",true}});
    auto deadline=mono()+90000000000ULL;unsigned attempts=0;int selected=-1;bool saved=false;
    while(mono()<deadline&&attempts<4&&!saved){
        need(process(owner_pid,boot).first==base.at("owner"),"original_owner_still_live");
        if(selected<0){DIR*dir=::opendir("/proc");need(dir!=nullptr,"proc_directory");
            while(auto*item=::readdir(dir)){std::string name=item->d_name;if(name.empty()||!std::all_of(name.begin(),name.end(),[](char c){return c>='0'&&c<='9';}))continue;
                int pid=std::stoi(name);struct stat s{},directory_stat{};
                if(::stat(proc(pid,"exe").c_str(),&s)!=0||::stat(("/proc/"+name).c_str(),&directory_stat)!=0||directory_stat.st_uid!=::getuid())continue;
                if(std::uint64_t(s.st_dev)!=base.at("executable").at("device")||std::uint64_t(s.st_ino)!=base.at("executable").at("inode"))continue;
                try{ancestry(pid,base.at("owner"),boot);selected=pid;break;}catch(const std::exception&){continue;}
            }::closedir(dir);
        }
        if(selected>0){++attempts;try{capture(config,selected,boot,attempts);saved=true;}
            catch(const std::exception&error){write_new(directory+"/NATIVE_REFUSAL_"+std::to_string(attempts)+".json",J{{"attempt",attempts},{"target_pid",selected},{"reason",error.what()},{"config_binding",config.at("binding")}});try{process(selected,boot);}catch(...){break;}if(::access((directory+"/NATIVE_RAW_SNAPSHOT.json").c_str(),F_OK)==0)break;}}
        if(!saved)std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    write_new(directory+"/NATIVE_OBSERVER_TERMINAL.json",J{{"config_binding",config.at("binding")},{"attempts",attempts},{"target_pid",selected},{"snapshot_and_hash_saved",saved},{"target_signals_or_modifications",0}});
    return saved?0:2;
}catch(const std::exception&error){std::cerr<<"native_observer_refusal: "<<error.what()<<'\n';return 1;}}
