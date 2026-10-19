#include "api/mock_api.hpp"
#include "runtime/runtime.hpp"
#include "runtime/plan_reader.hpp"
#include "runtime/operator_spec_reader.hpp"
#include "runtime/binary_plan_codec.hpp"
#include "runtime/utils/sha256.hpp"
#include <iostream>
#include <atomic>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
using namespace fhegpu;
void require(bool ok,const char *message) {if(!ok)throw std::runtime_error(message);}
template <class F> void expect_error(F &&run, const char *message) {
    try { run(); }
    catch (const std::exception &e) {
        if (std::string(e.what()).find(message) != std::string::npos) return;
        throw std::runtime_error(std::string("unexpected error, expected '") + message + "': " + e.what());
    }
    throw std::runtime_error(std::string("expected error containing '") + message + "'");
}
class PrepareApi : public MockVecApi {
public:
    explicit PrepareApi(std::shared_ptr<MockCluster> cluster,std::size_t devices=1)
        :MockVecApi(0,std::move(cluster)),devices(devices) {}
    const std::size_t devices;
    std::atomic<int> cpu{0},gpu{0};
    std::atomic<int> host_encodes{0}, gpu_resource_queries{0};
    int delay_ms=5;bool unsupported=false;
    int fail_prepare_call=0;
    std::uint64_t prepare_host_bytes=100;
    std::vector<int> device_delay_ms;
    std::atomic<int> preparation_attempts{0},active_preparations{0},peak_preparations{0};
    struct Event { const char *kind; ValueId value; std::thread::id thread; };
    std::vector<Event> events;
    std::vector<std::future<void>> preparations;
    std::mutex mutex;
    void event(const char *kind, ValueId id) {
        std::lock_guard<std::mutex> lock(mutex);
        events.push_back({kind, id, std::this_thread::get_id()});
    }
    ResourceUsage value_resources(const ValueDesc &v) {
        ResourceUsage r(devices);if(v.place.kind==PlaceKind::Host)r.host=v.components*100;
        else if(v.kind==ValueKind::Plaintext)r.plain.at(v.place.index)=100;else r.front.at(v.place.index)=200;return r;
    }
    PrepareResources prepare_resources(const ValueDesc &v,PrepareBackend backend) {
        if (backend == PrepareBackend::Gpu) ++gpu_resource_queries;
        auto resident=value_resources(v),peak=resident;peak.host=prepare_host_bytes;peak.pinned=50;peak.raw=20;
        if(backend==PrepareBackend::Gpu)peak.plain.at(v.place.index)+=50;
        return {peak,resident,!unsupported};
    }
    Value encode_plaintext(const ValueDesc &desc, const std::vector<double> &slots) {
        ++host_encodes;
        return MockVecApi::encode_plaintext(desc, slots);
    }
    ResourceUsage instruction_workspace(const Instruction &i) {
        ResourceUsage r(devices);
        if(const auto *op=std::get_if<ComputeOp>(&i.body))r.front.at(op->place.kind==PlaceKind::Device?op->place.index:0)=100;
        return r;
    }
    Value prepare_plaintext_async(const ValueDesc &desc,const std::vector<double> &slots,PrepareBackend backend) {
        event("prepare", desc.id);
        if(backend==PrepareBackend::Cpu)++cpu;else ++gpu;
        const auto attempt=++preparation_attempts;
        if(attempt==fail_prepare_call)throw std::runtime_error("injected Prepare allocation failure at attempt 2");
        auto host=desc;host.place={PlaceKind::Host,0,0};auto source=encode_plaintext(host,slots);
        auto result=Value::pending(source.kind(),source.metadata());
        const int completion_delay=device_delay_ms.empty()?delay_ms:device_delay_ms.at(desc.place.index);
        const int active=++active_preparations;
        int peak=peak_preparations.load();
        while(active>peak && !peak_preparations.compare_exchange_weak(peak,active)) {}
        std::lock_guard<std::mutex> lock(mutex);
        preparations.push_back(std::async(std::launch::async,[=,this] {
            std::this_thread::sleep_for(std::chrono::milliseconds(completion_delay));
            auto payload=source.materialize();
            event("prepare_complete",desc.id);--active_preparations;result.fulfill(std::move(payload));
        }));return result;
    }
    Value compute(const ComputeOp &op, const std::vector<Value> &inputs) {
        event("compute_start", op.output);
        auto result = MockVecApi::compute(op, inputs);
        event("compute_complete", op.output);
        return result;
    }
    bool is_ready(Value &v) {return v.is_ready();}
    void drain() {for(auto &future:preparations)future.get();preparations.clear();MockVecApi::drain();}
};

// Keep this single-rank mock alive after surfacing the fatal error so a second,
// explicitly requested run can check that runtime state was reset. No retry occurs.
class RestartablePrepareApi : public PrepareApi {
public:
    using PrepareApi::PrepareApi;
    [[noreturn]] void abort_all(int,const std::string &reason) {throw ClusterPanic(reason);}
    void preflight(std::string_view digest,bool skip,const TargetConfig &target,
                   const OperatorSpec &spec,const PlanRequirements &requirements) {
        if(preflight_done_) {
            require(digest==digest_ && spec.poly_degree==degree_ && target.world_size==1,
                    "restart fixture changed plan/spec");
            return; // MockCluster itself intentionally permits only one handshake.
        }
        MockVecApi::preflight(digest,skip,target,spec,requirements);
        digest_=digest;degree_=spec.poly_degree;preflight_done_=true;
    }
private:
    bool preflight_done_=false;
    std::string digest_;
    std::uint64_t degree_=0;
};

// As in runtime_release_tests, weak wrappers observe Runtime ownership without
// keeping the underlying mock executor's payloads alive on its behalf.
class BundleFailurePrepareApi {
public:
    using Value = std::shared_ptr<VecValue>;
    static constexpr bool background_communication_issuer = true;
    struct CommHandle {
        MockVecApi::CommHandle inner;
        std::vector<std::optional<Value>> outputs;
    };
    RestartablePrepareApi inner{std::make_shared<MockCluster>(MockClusterConfig{})};
    int aborts=0,drains=0;
    bool expect_failed_cleanup=true;
    std::string name() const {return "BundleFailurePrepareApi";}
    Value observe(ValueId id,VecValue value) {
        auto result=std::make_shared<VecValue>(std::move(value));
        std::lock_guard<std::mutex> lock(mutex_);observed_[id]=result;return result;
    }
    bool expired(ValueId id) {
        std::lock_guard<std::mutex> lock(mutex_);return observed_.at(id).expired();
    }
    Value encode_plaintext(const ValueDesc &desc,const std::vector<double> &slots) {
        return observe(desc.id,inner.encode_plaintext(desc,slots));
    }
    Value prepare_plaintext_async(const ValueDesc &desc,const std::vector<double> &slots,
                                  PrepareBackend backend) {
        return observe(desc.id,inner.prepare_plaintext_async(desc,slots,backend));
    }
    ResourceUsage value_resources(const ValueDesc &desc) {return inner.value_resources(desc);}
    PrepareResources prepare_resources(const ValueDesc &desc,PrepareBackend backend) {
        return inner.prepare_resources(desc,backend);
    }
    ResourceUsage instruction_workspace(const Instruction &instruction) {
        return inner.instruction_workspace(instruction);
    }
    Value compute(const ComputeOp &op,const std::vector<Value> &inputs) {
        return observe(op.output,inner.compute(op,unwrap(inputs)));
    }
    CommHandle communicate_async(const CommAction &action,const std::vector<Value> &inputs,
                                 const std::vector<ValueDesc> &descs) {
        CommHandle handle;
        handle.inner=inner.communicate_async(action,unwrap(inputs),descs);
        auto outputs=inner.posted_outputs(handle.inner);handle.outputs.resize(outputs.size());
        for(std::size_t slot=0;slot<outputs.size();++slot)
            if(outputs[slot])handle.outputs[slot]=observe(action.outputs[slot],std::move(*outputs[slot]));
        return handle;
    }
    std::vector<std::optional<Value>> posted_outputs(CommHandle &handle) {return take_outputs(handle);}
    std::vector<std::optional<Value>> wait(CommHandle &handle) {
        inner.wait(handle.inner);return take_outputs(handle);
    }
    void collect_completed() {inner.collect_completed();}
    bool is_ready(Value &value) {return inner.is_ready(*value);}
    void synchronize(Value &value) {inner.synchronize(*value);}
    void validate_value(const Value &value,const ValueDesc &desc) const {inner.validate_value(*value,desc);}
    void preflight(std::string_view digest,bool skip,const TargetConfig &target,
                   const OperatorSpec &spec,const PlanRequirements &requirements) {
        inner.preflight(digest,skip,target,spec,requirements);
    }
    [[noreturn]] void abort_all(int,const std::string &reason) {++aborts;throw ClusterPanic(reason);}
    void drain() {
        ++drains;
        if(expect_failed_cleanup)
            require(aborts==1 && !expired(2),"bundle failure released the admitted source before drain");
        inner.drain();
        require(inner.active_preparations==0 && inner.preparations.empty(),
                "bundle failure drain left submitted preparation work live");
        if(expect_failed_cleanup)
            require(!expired(2),"bundle failure released the admitted source during drain");
    }
private:
    static std::vector<VecValue> unwrap(const std::vector<Value> &inputs) {
        std::vector<VecValue> result;for(const auto &value:inputs)result.push_back(*value);return result;
    }
    static std::vector<std::optional<Value>> take_outputs(CommHandle &handle) {
        std::vector<std::optional<Value>> result(handle.outputs.size());result.swap(handle.outputs);return result;
    }
    std::mutex mutex_;
    std::unordered_map<ValueId,std::weak_ptr<VecValue>> observed_;
};

class CaptureRuntimeErrors {
public:
    std::ostringstream stream;
    CaptureRuntimeErrors():previous_(std::cerr.rdbuf(stream.rdbuf())) {}
    ~CaptureRuntimeErrors() {std::cerr.rdbuf(previous_);std::cerr<<stream.str();}
private:
    std::streambuf *previous_;
};

class AllocatorLimitedPrepareApi : public PrepareApi {
public:
    using PrepareApi::PrepareApi;
    std::vector<std::uint64_t> allocator_capacity;
    void validate_waterline_resources(const RuntimePlan &) {validated_=true;}
    std::vector<std::uint64_t> waterline_allocator_capacity() const {
        require(validated_,"allocator capacity queried before resource validation");
        return allocator_capacity;
    }
private:
    bool validated_=false;
};

class FixedBufferPrepareApi : public RestartablePrepareApi {
public:
    using RestartablePrepareApi::RestartablePrepareApi;
    ResourceUsage waterline_fixed_resources() const {
        ResourceUsage usage(devices);usage.host=80;usage.pinned=60;return usage;
    }
};

class PrefixBudgetPrepareApi : public PrepareApi {
public:
    PrefixBudgetPrepareApi(bool pinned_limited)
        :PrepareApi(std::make_shared<MockCluster>(MockClusterConfig{}),4),
         pinned_limited_(pinned_limited) {}
    std::atomic<unsigned> blocked_group_queries{0};
    PrepareResources prepare_resources(const ValueDesc &desc,PrepareBackend backend) {
        auto cost=PrepareApi::prepare_resources(desc,backend);
        const std::uint64_t shared_cost[]={600,400,100,100};
        if(pinned_limited_)cost.peak.pinned=shared_cost[desc.place.index];
        else cost.peak.host=shared_cost[desc.place.index];
        if(desc.id==21) {
            std::lock_guard<std::mutex> lock(mutex);
            bool first_started=false,first_completed=false;
            for(const auto &entry:events)if(entry.value==11) {
                first_started|=std::string(entry.kind)=="prepare";
                first_completed|=std::string(entry.kind)=="prepare_complete";
            }
            if(first_started && !first_completed)++blocked_group_queries;
        }
        return cost;
    }
private:
    bool pinned_limited_;
};

void test_fixed_prepare_buffers(RuntimePlan plan,const LoadedOperatorSpec &spec,const std::vector<double> &input) {
    plan.waterline->backend=PrepareBackend::Cpu;
    plan.waterline->device_capacity_bytes={1000};
    for(int boundary=0;boundary<3;++boundary) {
        // External Host input and final Host output coexist (400 bytes),
        // including after all dynamic preparation credit has been returned.
        plan.waterline->host_budget_bytes=boundary==0?450:480;
        plan.waterline->pinned_budget_bytes=boundary==1?100:110;
        FixedBufferPrepareApi api(std::make_shared<MockCluster>(MockClusterConfig{}));
        SequentialRuntime<FixedBufferPrepareApi> runtime(0,1,1,api,DeviceExecutionMode::PerDeviceWorkers);
        const auto run=[&] {return runtime.run({plan,"sha256:"+std::string(64,'1')},{spec,std::nullopt,false},
            {{0,make_cipher(input,spec.spec.context_id,spec.spec.poly_degree,3,30)}});};
        if(boundary<2) {
            CaptureRuntimeErrors capture;
            expect_error(run,"Waterline single-group baseline infeasible");
            require(api.preparation_attempts==0,"fixed buffer budget rejected after Prepare submission");
        } else {
            auto result=run().values.at(plan.final_outputs.front()).value.materialize();
            require(std::abs(result.slots[0]-11)<1e-9 && std::abs(result.slots[1]+22)<1e-9,
                    "fixed buffer exact budgets changed final arithmetic");
            require(api.cpu==20 && api.gpu==0,"fixed buffer test lost rolling CPU preparations");
        }
    }
    std::cout<<"PASS fixed Host/pinned buffers remain charged through rolling completions\n";
}

std::size_t event_index(const PrepareApi &api, const char *kind, ValueId id) {
    for (std::size_t i = 0; i < api.events.size(); ++i)
        if (std::string(api.events[i].kind) == kind && api.events[i].value == id) return i;
    throw std::runtime_error(std::string("missing event '") + kind + "' for ValueId " + std::to_string(id));
}

void number_instructions(RuntimePlan &plan) {
    std::size_t ordinal = 0;
    for (auto *phase : {&plan.initialization, &plan.execution, &plan.finalization})
        for (auto &instruction : *phase) instruction.ordinal = ordinal++;
}

RuntimePlan binary_round_trip(RuntimePlan plan) {
    std::stringstream wire(std::ios::in | std::ios::out | std::ios::binary);
    binary_io::write_plan(wire, plan);
    const auto size = wire.str().size();
    wire.seekg(0);
    return binary_io::read_plan(wire, size);
}

void test_json_and_negative_config(const RuntimePlan &template_plan, const LoadedOperatorSpec &spec) {
    using Json = binary_io::Json;
    auto root = binary_io::metadata(template_plan);
    root["plan_id"] = "18446744073709551615";
    const Json device{{"kind", "device"}, {"rank", 0}, {"index", 0}};
    const Json host{{"kind", "host"}, {"rank", 0}};
    for (unsigned id = 0; id < 4; ++id)
        root["values"].push_back({{"id", std::to_string(id)},
            {"kind", id == 1 ? "plaintext" : "ciphertext"}, {"place", id == 0 ? host : device},
            {"context", spec.spec.context_id}, {"level", 3}, {"scale_log2", 30},
            {"ntt", true}, {"components", id == 1 ? 1 : 2}});
    root["external_inputs"] = {"0"};
    root["initialization"] = {{{"ordinal", 0}, {"kind", "transfer"}, {"transfer_id", "0"},
        {"hint", "point_to_point"}, {"inputs", {"0"}}, {"outputs", {"3"}},
        {"sources", {host}}, {"destinations", {device}}, {"output_kinds", {"ciphertext"}}}};
    root["execution"] = {
        {{"ordinal", 1}, {"kind", "prepare_plaintext"}, {"output", "1"},
            {"payload", {{"kind", "inline"}, {"values", {.5, -1.0}}}}},
        {{"ordinal", 2}, {"kind", "compute"}, {"op", "add_cp"}, {"place", device},
            {"inputs", {"3", "1"}}, {"output", "2"}},
        {{"ordinal", 3}, {"kind", "release"}, {"value", "1"}},
        {{"ordinal", 4}, {"kind", "release"}, {"value", "3"}}};
    root["final_outputs"] = {"2"};

    for (const char *backend : {"cpu", "gpu", "auto"}) {
        root["waterline"]["backend"] = backend;
        auto loaded = RuntimePlanJsonReader::read_text(root.dump());
        PlanVerifier::verify(loaded.plan, spec, false);
        auto parsed = binary_round_trip(loaded.plan);
        PlanVerifier::verify(parsed, spec, false);
        require(binary_io::metadata(parsed) == binary_io::metadata(loaded.plan),
                "V4 JSON/binary round trip changed Waterline metadata");
        require(parsed.plan_id == std::numeric_limits<std::uint64_t>::max() &&
                parsed.external_inputs == loaded.plan.external_inputs &&
                parsed.final_outputs == loaded.plan.final_outputs && parsed.values.size() == 4,
                "V4 round trip changed identifiers or values");
        const auto &prepare = std::get<PreparePlaintextOp>(parsed.execution.at(0).body);
        require(prepare.output == 1 && std::get<InlineEncodePayload>(prepare.payload).values ==
                std::vector<double>({.5, -1.0}), "V4 round trip changed inline Prepare payload");
        const auto &compute = std::get<ComputeOp>(parsed.execution.at(1).body);
        require(compute.kind == ComputeKind::AddCP && compute.inputs == std::vector<ValueId>({3, 1}) &&
                compute.output == 2 && compute.place == Place{PlaceKind::Device, 0, 0} &&
                std::get<ReleaseOp>(parsed.execution.at(2).body).value == 1,
                "V4 round trip changed Compute/Release records");
    }
    // Bundle Prepare uses the binary string table as well as the new instruction tag.
    const auto digest = "sha256:" + std::string(64, 'a');
    auto bundled = root;
    bundled["execution"][0]["payload"] = {{"kind", "bundle"}, {"content", digest}};
    bundled["plaintext_bundle"] = {{"id", "waterline-test"}, {"version", 1}, {"manifest_sha256", digest}};
    auto bundle_plan = RuntimePlanJsonReader::read_text(bundled.dump()).plan;
    PlanVerifier::verify(bundle_plan, spec, false);
    auto bundle_parsed = binary_round_trip(bundle_plan);
    PlanVerifier::verify(bundle_parsed, spec, false);
    require(std::get<BundleEncodePayload>(std::get<PreparePlaintextOp>(bundle_parsed.execution[0].body).payload).content == digest &&
            bundle_parsed.plaintext_bundle->manifest_sha256 == digest,
            "V4 round trip changed bundle Prepare/string table");

    const auto reject_json = [&](const Json &bad, const char *message) {
        expect_error([&] { RuntimePlanJsonReader::read_text(bad.dump()); }, message);
    };
    auto bad = root;bad["waterline"]["backend"] = "fft";reject_json(bad, "expected cpu, gpu or auto");
    bad = root;bad["waterline"]["device_capacity_bytes"] = 1000;reject_json(bad, "device budgets must be arrays");
    bad = root;bad["waterline"]["device_fractions"] = .25;reject_json(bad, "expected array");
    bad = root;bad["waterline"]["device_fractions"] = {true};reject_json(bad, "expected number");
    bad = root;bad["waterline"]["host_budget_bytes"] = -1;reject_json(bad, "outside the supported range");
    bad = root;bad["waterline"]["host_budget_bytes"] = .5;reject_json(bad, "expected integer");
    bad = root;bad["waterline"]["host_budget_bytes"] = (1ULL << 53);reject_json(bad, "outside the supported range");
    bad = root;bad["waterline"]["lookahead_groups"] = 0;reject_json(bad, "outside the supported range");
    bad = root;bad["waterline"]["obsolete_batch_size"] = 4;reject_json(bad, "unknown field");
    bad = root;bad["format_version"] = 3;reject_json(bad, "waterline requires V4");
    bad.erase("waterline");reject_json(bad, "PreparePlaintext requires format version 4");

    const auto valid = RuntimePlanJsonReader::read_text(root.dump()).plan;
    auto legacy = valid;legacy.format_version = 3;legacy.waterline.reset();
    expect_error([&] { binary_round_trip(legacy); }, "binary PreparePlaintext requires plan V4");
    const auto reject_plan = [&](const RuntimePlan &invalid, const char *message) {
        expect_error([&] { PlanVerifier::verify(invalid, spec, false); }, message);
    };
    for (int fault = 0; fault < 3; ++fault) {
        auto invalid = valid;
        if (fault == 0) invalid.waterline->device_capacity_bytes.clear();
        if (fault == 1) invalid.waterline->device_reserve_bytes.clear();
        if (fault == 2) invalid.waterline->device_fractions.clear();
        reject_plan(invalid, "device budget table mismatch");
    }
    for (double fraction : {0.0, 1.0, -.1, std::numeric_limits<double>::infinity(),
                            std::numeric_limits<double>::quiet_NaN()}) {
        auto invalid = valid;invalid.waterline->device_fractions = {fraction};
        reject_plan(invalid, "device fraction");
    }
    for (int fault = 0; fault < 6; ++fault) {
        auto invalid = valid;
        if (fault == 0) invalid.waterline->device_capacity_bytes = {0};
        if (fault == 1) invalid.waterline->device_reserve_bytes = invalid.waterline->device_capacity_bytes;
        if (fault == 2) invalid.waterline->host_budget_bytes = 0;
        if (fault == 3) invalid.waterline->pinned_budget_bytes = 0;
        if (fault == 4) invalid.waterline->raw_budget_bytes = 0;
        if (fault == 5) invalid.waterline->lookahead_groups = 0;
        reject_plan(invalid, fault < 2 ? "capacity/reserve" : "positive budgets");
    }
    auto invalid = valid;invalid.waterline.reset();reject_plan(invalid, "PreparePlaintext requires V4");
    for (int fault = 0; fault < 4; ++fault) {
        invalid = valid;
        if (fault == 0) invalid.values[1].place.kind = PlaceKind::Host;
        if (fault == 1) {invalid.values[1].kind = ValueKind::Ciphertext;invalid.values[1].components = 2;}
        if (fault == 2) invalid.values[1].ntt = false;
        if (fault == 3) invalid.values[1].components = 2;
        reject_plan(invalid, fault == 3 ? "plaintext components" : "Device NTT plaintext");
    }
    invalid = valid;
    std::get<InlineEncodePayload>(std::get<PreparePlaintextOp>(invalid.execution[0].body).payload).values =
        {std::numeric_limits<double>::quiet_NaN()};
    reject_plan(invalid, "non-finite PreparePlaintext payload");
    invalid = valid;invalid.initialization.push_back(invalid.execution.front());
    invalid.execution.erase(invalid.execution.begin());number_instructions(invalid);
    reject_plan(invalid, "PreparePlaintext requires V4 execution");
    std::cout << "PASS V4 JSON/binary inline and bundle round trips, strict config and Prepare rejection\n";
}

void test_host_waiter(const RuntimePlan &rolling, const LoadedOperatorSpec &spec,
                      const std::vector<double> &input) {
    auto plan = rolling;
    plan.waterline->backend = PrepareBackend::Cpu;
    plan.execution.resize(4); // First Prepare/AddCP/Release pair from the rolling graph.
    auto received = plan.values.at(0);received.id = 50;
    auto output = received;output.id = 51;
    plan.values.erase(std::remove_if(plan.values.begin(), plan.values.end(), [](const ValueDesc &value) {
        return value.id != 0 && value.id != 41 && value.id != 1 && value.id != 2;
    }), plan.values.end());
    plan.values.push_back(received);plan.values.push_back(output);
    const Place device{PlaceKind::Device, 0, 0}, host{PlaceKind::Host, 0, 0};
    plan.execution.push_back({0, CommAction{2, CommKind::Transfer, CommHint::PointToPoint,
        {2}, {50}, {device}, {host}, {ValueKind::Ciphertext}}});
    plan.execution.push_back({0, ComputeOp{ComputeKind::Negate, {50}, 51, host, {}}});
    plan.execution.push_back({0, ReleaseOp{50}});
    plan.execution.push_back({0, ReleaseOp{2}});
    plan.finalization.clear();plan.final_outputs = {51};number_instructions(plan);
    // The CPU queue begins with Negate, whose producer needs this same CPU
    // worker to Prepare. Dispatching the undefined Host consumer would deadlock.
    PrepareApi api(std::make_shared<MockCluster>(MockClusterConfig{}));api.delay_ms = 20;
    SequentialRuntime<PrepareApi> runtime(0, 1, 1, api, DeviceExecutionMode::PerDeviceWorkers);
    auto artifact = runtime.run({plan, "sha256:" + std::string(64, '1')}, {spec, std::nullopt, false},
        {{0, make_cipher(input, spec.spec.context_id, spec.spec.poly_degree, 3, 30)}});
    const auto result = artifact.values.at(51).value.materialize();
    require(std::abs(result.slots[0] + 1.5) < 1e-9 && std::abs(result.slots[1] - 3) < 1e-9,
            "Host consumer failed to read prepared result");
    const auto prepare = event_index(api, "prepare", 1), host_compute = event_index(api, "compute_start", 51);
    require(prepare < host_compute && api.events[prepare].thread == api.events[host_compute].thread &&
            api.cpu == 1 && api.gpu == 0 && artifact.timing.fence_calls == 0,
            "Host waiter blocked or bypassed the single CPU preparation worker");
    std::cout << "PASS Host waiter leaves the single CPU worker available for required Prepare\n";
}

void test_small_waterline(const RuntimePlan &rolling, const LoadedOperatorSpec &spec,
                          const std::vector<double> &input) {
    for (const auto backend : {PrepareBackend::Cpu, PrepareBackend::Auto}) {
        auto plan = rolling;plan.waterline->backend = backend;
        plan.waterline->device_fractions = {.1}; // CPU peak fits one plaintext; GPU peak does not.
        PrepareApi api(std::make_shared<MockCluster>(MockClusterConfig{}));api.delay_ms = 10;
        SequentialRuntime<PrepareApi> runtime(0, 1, 1, api, DeviceExecutionMode::PerDeviceWorkers);
        auto artifact = runtime.run({plan, "sha256:" + std::string(64, '1')}, {spec, std::nullopt, false},
            {{0, make_cipher(input, spec.spec.context_id, spec.spec.poly_degree, 3, 30)}});
        for (unsigned i = 1; i < 20; ++i)
            require(event_index(api, "compute_complete", 2 * i) < event_index(api, "prepare", 1 + 2 * i),
                    "small waterline submitted the next group before the current consumer completed");
        auto result = artifact.values.at(42).value.materialize();
        require(std::abs(result.slots[0] - 11) < 1e-9 && std::abs(result.slots[1] + 22) < 1e-9 &&
                api.cpu == 20 && api.gpu == 0 && artifact.timing.fence_calls == 0,
                "small waterline failed to resume all groups after reclaim");
        require(backend != PrepareBackend::Auto || api.gpu_resource_queries > 0,
                "Auto small-waterline test never evaluated an oversized GPU candidate");
        std::cout << "PASS one-plaintext waterline backend=" << int(backend)
                  << " resumes 20 consumers, oversized GPU candidate keeps legal CPU path\n";
    }
}

void test_initialization_preflight(const RuntimePlan &rolling, const LoadedOperatorSpec &spec,
                                   const std::vector<double> &input) {
    auto plan = rolling;plan.waterline->backend = PrepareBackend::Cpu;
    plan.waterline->device_capacity_bytes = {450}; // Later AddCP needs input + output + workspace = 500.
    auto plaintext = plan.values.at(0);plaintext.id = 60;plaintext.kind = ValueKind::Plaintext;plaintext.components = 1;
    plan.values.push_back(plaintext);
    plan.initialization.insert(plan.initialization.begin(), {0, ReleaseOp{60}});
    plan.initialization.insert(plan.initialization.begin(), {0, EncodeOp{InlineEncodePayload{{.5, -1}}, 60}});
    number_instructions(plan);
    PrepareApi api(std::make_shared<MockCluster>(MockClusterConfig{}));
    SequentialRuntime<PrepareApi> runtime(0, 1, 1, api, DeviceExecutionMode::PerDeviceWorkers);
    expect_error([&] {
        runtime.run({plan, "sha256:" + std::string(64, '1')}, {spec, std::nullopt, false},
            {{0, make_cipher(input, spec.spec.context_id, spec.spec.poly_degree, 3, 30)}});
    }, "single-group");
    require(api.host_encodes == 0 && api.cpu == 0 && api.gpu == 0 && api.stats().communicate_calls == 0,
            "infeasible execution submitted initialization Encode/Transfer before full-plan preflight");
    std::cout << "PASS later infeasible group rejects before initialization Encode/Transfer submission\n";
}

void test_prepare_across_fence_and_unused(const RuntimePlan &rolling, const LoadedOperatorSpec &spec,
                                         const std::vector<double> &input) {
    auto plan = rolling;plan.waterline->backend = PrepareBackend::Cpu;
    plan.execution.insert(plan.execution.begin() + 1, {0, FenceOp{}});
    auto unused = *std::find_if(plan.values.begin(), plan.values.end(), [](const ValueDesc &value) {
        return value.id == 1;
    });
    unused.id = 61;plan.values.push_back(unused);
    plan.execution.push_back({0, PreparePlaintextOp{InlineEncodePayload{{7.0, 2.0}}, 61}});
    number_instructions(plan);
    PrepareApi api(std::make_shared<MockCluster>(MockClusterConfig{}));
    SequentialRuntime<PrepareApi> runtime(0, 1, 1, api, DeviceExecutionMode::PerDeviceWorkers);
    auto artifact = runtime.run({plan, "sha256:" + std::string(64, '1')}, {spec, std::nullopt, false},
        {{0, make_cipher(input, spec.spec.context_id, spec.spec.poly_degree, 3, 30)}});
    auto result = artifact.values.at(42).value.materialize();
    require(std::abs(result.slots[0] - 11) < 1e-9 && std::abs(result.slots[1] + 22) < 1e-9 &&
            api.cpu == 20 && api.gpu == 0 && artifact.timing.fence_calls == 1,
            "Prepare registered before Fence was lost or unused Prepare blocked drain");
    require(std::none_of(api.events.begin(), api.events.end(), [](const PrepareApi::Event &event) {
        return std::string(event.kind) == "prepare" && event.value == 61;
    }), "unused Prepare submitted work without a consumer group");
    std::cout << "PASS Prepare crosses explicit Fence and unused Prepare finishes without submission\n";
}

void test_prepare_final_outputs(const RuntimePlan &rolling, const LoadedOperatorSpec &spec,
                                const std::vector<double> &input) {
    auto plan = rolling;
    const auto plaintext = *std::find_if(plan.values.begin(), plan.values.end(),
        [](const ValueDesc &value) { return value.id == 1; });
    plan.initialization.clear();plan.execution.clear();plan.finalization.clear();
    plan.values.clear();plan.external_inputs.clear();plan.final_outputs.clear();
    plan.waterline->device_fractions = {.3};
    // Only one preparation fits the shared budgets at a time, while both
    // returned plaintexts must remain resident together.
    plan.waterline->host_budget_bytes = 100;
    plan.waterline->pinned_budget_bytes = 50;
    plan.waterline->raw_budget_bytes = 20;
    for (ValueId id : {1, 3}) {
        auto output = plaintext;output.id = id;plan.values.push_back(output);
        plan.execution.push_back({0, PreparePlaintextOp{InlineEncodePayload{{double(id), -1}}, id}});
        plan.final_outputs.push_back(id);
    }
    plan.execution.insert(plan.execution.begin() + 1, {0, FenceOp{}});
    number_instructions(plan);
    for (auto backend : {PrepareBackend::Cpu, PrepareBackend::Gpu, PrepareBackend::Auto}) {
        plan.waterline->backend = backend;
        PrepareApi api(std::make_shared<MockCluster>(MockClusterConfig{}));
        SequentialRuntime<PrepareApi> runtime(0, 1, 1, api, DeviceExecutionMode::PerDeviceWorkers);
        auto artifact = runtime.run({plan, "sha256:" + std::string(64, '1')},
            {spec, std::nullopt, false}, {});
        require(artifact.values.size() == 2 && api.preparation_attempts == 2 &&
                api.peak_preparations == 1 && artifact.timing.fence_calls == 1,
                "returned-only Prepare values bypassed admission or repeated preparation");
        for (auto id : plan.final_outputs) {
            const auto result = artifact.values.at(id).value.materialize();
            require(result.slots[0] == double(id) && result.slots[1] == -1,
                    "returned-only Prepare result was not materialized");
        }
    }
    // A returned value which already has a compute consumer is prepared once.
    auto consumed = rolling;consumed.execution.resize(2);consumed.finalization.clear();
    consumed.final_outputs = {1, 2};
    consumed.values.erase(std::remove_if(consumed.values.begin(), consumed.values.end(),
        [](const ValueDesc &v) {return v.id != 0 && v.id != 41 && v.id != 1 && v.id != 2;}),
        consumed.values.end());
    number_instructions(consumed);
    PrepareApi consumed_api(std::make_shared<MockCluster>(MockClusterConfig{}));
    SequentialRuntime<PrepareApi> consumed_runtime(0, 1, 1, consumed_api, DeviceExecutionMode::PerDeviceWorkers);
    auto consumed_artifact = consumed_runtime.run({consumed, "sha256:" + std::string(64, '1')},
        {spec, std::nullopt, false},
        {{0, make_cipher(input, spec.spec.context_id, spec.spec.poly_degree, 3, 30)}});
    require(consumed_api.preparation_attempts == 1 && consumed_artifact.values.size() == 2,
            "final-output demand prepared a consumed plaintext twice");

    // The terminal demand must participate in preflight with its full peak,
    // even though its resident plaintext alone fits the device waterline.
    plan.values.resize(1);plan.final_outputs.resize(1);plan.execution.resize(1);
    for (auto backend : {PrepareBackend::Cpu, PrepareBackend::Gpu}) {
        plan.waterline->backend = backend;
        plan.waterline->host_budget_bytes = backend == PrepareBackend::Cpu ? 99 : 100;
        plan.waterline->device_fractions = {backend == PrepareBackend::Gpu ? .12 : .3};
        PrepareApi api(std::make_shared<MockCluster>(MockClusterConfig{}));
        SequentialRuntime<PrepareApi> runtime(0, 1, 1, api, DeviceExecutionMode::PerDeviceWorkers);
        expect_error([&] { runtime.run({plan, "sha256:" + std::string(64, '1')},
            {spec, std::nullopt, false}, {}); }, "single-group baseline infeasible");
        require(api.preparation_attempts == 0 && api.host_encodes == 0,
                "infeasible returned Prepare submitted work before preflight rejection");
    }
    std::cout << "PASS returned Prepare demands share admission, survive Fence, and preflight full peaks\n";
}

void test_prepare_failure_and_explicit_restart(const RuntimePlan &rolling,const LoadedOperatorSpec &spec,
                                               const std::vector<double> &input) {
    for(const auto backend:{PrepareBackend::Cpu,PrepareBackend::Gpu}) {
        auto plan=rolling;plan.waterline->backend=backend;
        RestartablePrepareApi api(std::make_shared<MockCluster>(MockClusterConfig{}));
        api.fail_prepare_call=2;api.delay_ms=20;
        SequentialRuntime<RestartablePrepareApi> runtime(0,1,1,api,DeviceExecutionMode::PerDeviceWorkers);
        const LoadedRuntimePlan loaded{plan,"sha256:"+std::string(64,'1')};
        const auto make_inputs=[&] {
            return std::unordered_map<ValueId,PrepareApi::Value>{
                {0,make_cipher(input,spec.spec.context_id,spec.spec.poly_degree,3,30)}};
        };
        expect_error([&] {runtime.run(loaded,{spec,std::nullopt,false},make_inputs());},
                     "injected Prepare allocation failure at attempt 2");
        api.drain();
        require(api.preparation_attempts==2 && api.active_preparations==0 && api.preparations.empty() &&
                (backend==PrepareBackend::Cpu?(api.cpu==2 && api.gpu==0):(api.gpu==2 && api.cpu==0)),
                "failed Prepare retried, switched backend, or left live preparation work");
        // The failure was observed above. This separate run explicitly removes
        // the injection and reuses the same API/runtime, with the same fixed backend.
        api.fail_prepare_call=0;
        auto artifact=runtime.run(loaded,{spec,std::nullopt,false},make_inputs());
        const auto result=artifact.values.at(42).value.materialize();
        require(std::abs(result.slots[0]-11)<1e-9 && std::abs(result.slots[1]+22)<1e-9 &&
                artifact.timing.encode_calls==20 && artifact.timing.fence_calls==0 &&
                api.preparation_attempts==22 && api.active_preparations==0 && api.preparations.empty() &&
                (backend==PrepareBackend::Cpu?(api.cpu==22 && api.gpu==0):(api.gpu==22 && api.cpu==0)),
                "explicit restart retained stale credits/dependencies or changed backend");
        std::cout<<"PASS injected second Prepare failure backend="<<int(backend)
                 <<" surfaces exact error, workers converge, explicit same-runtime restart completes\n";
    }
}

void test_lazy_bundle_failure_and_explicit_repair(const RuntimePlan &rolling,const LoadedOperatorSpec &spec,
                                                 const std::vector<double> &input) {
    // There is no replaceable BundleReader in RuntimeResources. A format1
    // manifest is indexed without opening its blobs, so a genuinely missing
    // blob fails in read_encode_slots on the selected Prepare worker.
    for(const auto backend:{PrepareBackend::Cpu,PrepareBackend::Gpu}) {
        struct TemporaryBundle {
            std::filesystem::path path;
            ~TemporaryBundle() {std::error_code error;std::filesystem::remove_all(path,error);}
        } bundle{std::filesystem::temp_directory_path()/
            ("ckks-waterline-bundle-failure-"+std::to_string(int(backend))+"-"+
             std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()))};
        std::filesystem::create_directories(bundle.path/"data");
        const auto slot_bytes=[](double first,double second) {
            std::string bytes(16,'\0');const double values[]={first,second};
            for(std::size_t i=0;i<2;++i) {
                std::uint64_t bits=0;std::memcpy(&bits,&values[i],sizeof(bits));
                for(unsigned byte=0;byte<8;++byte)bytes[i*8+byte]=static_cast<char>(bits>>(byte*8));
            }
            return bytes;
        };
        const auto good_bytes=slot_bytes(.5,-1),missing_bytes=slot_bytes(.25,.75);
        const auto good="sha256:"+sha256_hex(good_bytes),missing="sha256:"+sha256_hex(missing_bytes);
        const auto write=[&](const std::filesystem::path &path,const std::string &bytes) {
            std::ofstream output(path,std::ios::binary);output.write(bytes.data(),bytes.size());
            require(bool(output),"failed to write temporary bundle fixture");
        };
        const binary_io::Json manifest={{"bundle_format_version",1},{"bundle_id","worker-read-failure"},
            {"version",1},{"blobs",binary_io::Json::array({
                {{"content",good},{"byte_length",16}},{{"content",missing},{"byte_length",16}}})}};
        const auto manifest_bytes=manifest.dump();
        write(bundle.path/"manifest.json",manifest_bytes);
        write(bundle.path/"data"/(good.substr(7)+".bin"),good_bytes);
        auto plan=rolling;plan.waterline->backend=backend;
        // One complete group fits exactly within the shared temporary budgets;
        // a second plaintext cannot overlap the first resident plaintext. The
        // first consumer therefore produces an owned source before read #2.
        plan.waterline->device_capacity_bytes={700};
        plan.waterline->host_budget_bytes=400;
        plan.waterline->pinned_budget_bytes=50;plan.waterline->raw_budget_bytes=20;
        plan.plaintext_bundle=PlaintextBundleRef{"worker-read-failure",1,"sha256:"+sha256_hex(manifest_bytes)};
        for(auto &instruction:plan.execution)
            if(auto *prepare=std::get_if<PreparePlaintextOp>(&instruction.body))
                prepare->payload=BundleEncodePayload{prepare->output==3?missing:good};
        BundleFailurePrepareApi api;
        auto external=api.observe(0,make_cipher(input,spec.spec.context_id,spec.spec.poly_degree,3,30));
        SequentialRuntime<BundleFailurePrepareApi> runtime(0,1,1,api,DeviceExecutionMode::PerDeviceWorkers);
        const LoadedRuntimePlan loaded{plan,"sha256:"+std::string(64,'1')};
        CaptureRuntimeErrors captured;
        expect_error([&] {runtime.run(loaded,{spec,bundle.path,false},{{0,external}});},
                     (missing.substr(7)+".bin").c_str());
        require(captured.stream.str().find("cannot open file:")!=std::string::npos && api.aborts==1 && api.drains==1,
                "lazy bundle read failure was not surfaced through abort and cleanup drain");
        require(api.inner.preparation_attempts==1 && api.inner.host_encodes==1 &&
                api.inner.stats().compute_calls==1 && api.inner.active_preparations==0 &&
                api.inner.preparations.empty() &&
                (backend==PrepareBackend::Cpu?(api.inner.cpu==1 && api.inner.gpu==0):
                                             (api.inner.gpu==1 && api.inner.cpu==0)),
                "bundle read failure submitted its consumer, retried, changed backend, or left work live");
        require(event_index(api.inner,"compute_complete",2)>event_index(api.inner,"prepare",1) &&
                std::none_of(api.inner.events.begin(),api.inner.events.end(),[](const PrepareApi::Event &event) {
                    return std::string(event.kind)=="compute_start" && event.value>=4;
                }),"failed bundle consumer or a later consumer was submitted");
        require(api.expired(1) && api.expired(2) && api.expired(41) && !api.expired(0),
                "bundle failure retained owned plaintext/source or destroyed the caller input");

        // Only this explicit second call repairs the real source file. Reusing
        // this runtime and the unchanged tight budgets checks that failed
        // admission/dependency credits do not survive into a subsequent run.
        write(bundle.path/"data"/(missing.substr(7)+".bin"),missing_bytes);
        api.expect_failed_cleanup=false;
        auto artifact=runtime.run(loaded,{spec,bundle.path,false},{{0,external}});
        const auto result=artifact.values.at(42).value->materialize();
        require(std::abs(result.slots[0]-10.75)<1e-9 && std::abs(result.slots[1]+20.25)<1e-9 &&
                artifact.timing.bundle_read_calls==20 && artifact.timing.encode_calls==20 &&
                artifact.timing.fence_calls==0 && api.inner.preparation_attempts==21 &&
                api.inner.active_preparations==0 && api.inner.preparations.empty() && api.drains==2 &&
                (backend==PrepareBackend::Cpu?(api.inner.cpu==21 && api.inner.gpu==0):
                                             (api.inner.gpu==21 && api.inner.cpu==0)),
                "explicit repaired-bundle run retained failed work/credits or changed its selected backend");
        const auto log=captured.stream.str();
        require(log.find("remaining_front=0 remaining_plain=0")!=std::string::npos &&
                log.find("pinned_peak=50 raw_peak=20")!=std::string::npos,
                "repaired-bundle run retained device credits or exceeded the shared temporary budgets");
        require(api.expired(1) && api.expired(2) && api.expired(41) && !api.expired(0),
                "repaired-bundle run retained intermediate runtime ownership");
        std::cout<<"PASS real lazy bundle read failure backend="<<int(backend)
                 <<" before failing consumer, admitted work drained, owned source released, explicit repaired run succeeds\n";
    }
}

void test_four_device_progress_and_shared_budgets(const RuntimePlan &rolling,const LoadedOperatorSpec &spec,
                                                 const std::vector<double> &input) {
    auto plan=rolling;
    plan.target.device_counts={4};plan.values.clear();plan.execution.clear();plan.finalization.clear();plan.final_outputs.clear();
    const Place host{PlaceKind::Host,0,0};
    const auto descriptor=[&](ValueId id,ValueKind kind,Place place) {
        return ValueDesc{id,kind,place,spec.spec.context_id,3,30,true,kind==ValueKind::Plaintext?1:2};
    };
    plan.values.push_back(descriptor(0,ValueKind::Ciphertext,host));
    CommAction replicate{0,CommKind::Replicate,CommHint::PointToPoint,{0},{},{host},{},{}};
    for(int device=0;device<4;++device) {
        const Place place{PlaceKind::Device,0,device};
        const ValueId in=10+10*device,plaintext=in+1,out=in+2,result=100+device;
        plan.values.push_back(descriptor(in,ValueKind::Ciphertext,place));
        plan.values.push_back(descriptor(plaintext,ValueKind::Plaintext,place));
        plan.values.push_back(descriptor(out,ValueKind::Ciphertext,place));
        plan.values.push_back(descriptor(result,ValueKind::Ciphertext,host));
        replicate.outputs.push_back(in);replicate.destinations.push_back(place);replicate.output_types.push_back(ValueKind::Ciphertext);
        plan.execution.push_back({0,PreparePlaintextOp{InlineEncodePayload{{.5,-1}},plaintext}});
        plan.execution.push_back({0,ComputeOp{ComputeKind::AddCP,{in,plaintext},out,place,{}}});
        plan.execution.push_back({0,ReleaseOp{plaintext}});plan.execution.push_back({0,ReleaseOp{in}});
        plan.finalization.push_back({0,CommAction{TransferId(1+device),CommKind::Transfer,CommHint::PointToPoint,
            {out},{result},{place},{host},{ValueKind::Ciphertext}}});
        plan.finalization.push_back({0,ReleaseOp{out}});plan.final_outputs.push_back(result);
    }
    plan.initialization={{0,std::move(replicate)}};number_instructions(plan);
    plan.waterline->backend=PrepareBackend::Gpu;
    plan.waterline->device_capacity_bytes=std::vector<std::uint64_t>(4,1000);
    plan.waterline->device_reserve_bytes=std::vector<std::uint64_t>(4,0);
    plan.waterline->device_fractions=std::vector<double>(4,.4);
    plan.waterline->lookahead_groups=16;plan.waterline->raw_budget_bytes=200;
    for(bool host_limited:{true,false}) {
        // One external Host value uses 200 bytes. Two preparations need another
        // 1000 Host / 100 pinned bytes; all four would need 2000 / 200 instead.
        plan.waterline->host_budget_bytes=host_limited?1200:3000;
        plan.waterline->pinned_budget_bytes=host_limited?500:100;
        PrepareApi api(std::make_shared<MockCluster>(MockClusterConfig{}),4);
        api.prepare_host_bytes=500;api.device_delay_ms={300,5,5,5};
        SequentialRuntime<PrepareApi> runtime(0,1,4,api,DeviceExecutionMode::PerDeviceWorkers);
        CaptureRuntimeErrors captured;
        auto artifact=runtime.run({plan,"sha256:"+std::string(64,'1')},{spec,std::nullopt,false},
            {{0,make_cipher(input,spec.spec.context_id,spec.spec.poly_degree,3,30)}});
        const auto slow_complete=event_index(api,"prepare_complete",11);
        for(int device=1;device<4;++device)
            require(event_index(api,"compute_complete",12+10*device)<slow_complete,
                    "slow GPU0 stopped an admitted independent device");
        for(const auto id:plan.final_outputs) {
            const auto result=artifact.values.at(id).value.materialize();
            require(std::abs(result.slots[0]-1.5)<1e-9 && std::abs(result.slots[1]+3)<1e-9,
                    "four-device preparation changed branch arithmetic");
        }
        require(api.cpu==0 && api.gpu==4 && api.peak_preparations==2 && api.active_preparations==0 &&
                api.preparations.empty() && artifact.timing.fence_calls==0,
                "shared Host/pinned budget exceeded two in-flight preparations or leaked temporary work");
        const auto log=captured.stream.str();
        std::size_t cursor=0,returned=0;
        while((cursor=log.find("remaining_front=0 remaining_plain=0",cursor))!=std::string::npos) {++returned;++cursor;}
        require(returned==4 && log.find("host_peak=1200 pinned_peak=100 raw_peak=40")!=std::string::npos,
                "four-device ledger exceeded shared budgets or retained device credits");
        std::cout<<"PASS four logical GPUs, GPU1/2/3 complete before GPU0, shared "
                 <<(host_limited?"Host":"pinned")<<" limits Prepare concurrency to 2, temporary/device credits returned\n";
    }
    for(bool pinned_limited:{false,true}) {
        // All device groups fit by themselves. While the first Prepare is
        // pending, shared Host has 200 bytes left (or pinned has 200 left),
        // so the second needs too much (400), but either later group (100)
        // would fit. No group may reserve partially or skip the blocked head.
        plan.waterline->device_capacity_bytes=std::vector<std::uint64_t>(4,700);
        plan.waterline->host_budget_bytes=1000;
        plan.waterline->pinned_budget_bytes=pinned_limited?800:500;
        PrefixBudgetPrepareApi api(pinned_limited);
        api.device_delay_ms={300,80,5,5};
        SequentialRuntime<PrefixBudgetPrepareApi> runtime(0,1,4,api,DeviceExecutionMode::PerDeviceWorkers);
        CaptureRuntimeErrors captured;
        auto artifact=runtime.run({plan,"sha256:"+std::string(64,'1')},{spec,std::nullopt,false},
            {{0,make_cipher(input,spec.spec.context_id,spec.spec.poly_degree,3,30)}});
        require(api.blocked_group_queries>0,
                "shared-budget prefix test never encountered the blocked second group during first Prepare");
        const auto first_complete=event_index(api,"prepare_complete",11);
        for(int device=1;device<4;++device)
            require(first_complete<event_index(api,"prepare",11+10*device),
                    "executor partially started the blocked group or skipped it for a later fitting group");
        for(const auto id:plan.final_outputs) {
            const auto result=artifact.values.at(id).value.materialize();
            require(std::abs(result.slots[0]-1.5)<1e-9 && std::abs(result.slots[1]+3)<1e-9,
                    "shared-budget prefix changed branch arithmetic");
        }
        require(api.cpu==0 && api.gpu==4 && api.active_preparations==0 && api.preparations.empty() &&
                artifact.timing.fence_calls==0,"shared-budget prefix leaked, retried or inserted a Fence");
        const auto log=captured.stream.str();
        std::size_t cursor=0,returned=0;
        while((cursor=log.find("remaining_front=0 remaining_plain=0",cursor))!=std::string::npos) {++returned;++cursor;}
        require(returned==4,"blocked shared-budget admission retained partial device credits");
        std::cout<<"PASS executor shared "<<(pinned_limited?"pinned":"Host")
                 <<" head blocks later fitting groups atomically, completion resumes all four without Fence\n";
    }
}

void test_allocator_capacity(const RuntimePlan &rolling,const LoadedOperatorSpec &spec,
                              const std::vector<double> &input) {
    auto plan=rolling;
    auto plaintext=plan.values.front();plaintext.id=60;plaintext.kind=ValueKind::Plaintext;plaintext.components=1;
    plan.values.push_back(plaintext);
    plan.initialization.insert(plan.initialization.begin(),{0,ReleaseOp{60}});
    plan.initialization.insert(plan.initialization.begin(),{0,EncodeOp{InlineEncodePayload{{.5,-1}},60}});
    number_instructions(plan);
    for(const auto backend:{PrepareBackend::Cpu,PrepareBackend::Gpu,PrepareBackend::Auto}) {
        plan.waterline->backend=backend;
        AllocatorLimitedPrepareApi rejected(std::make_shared<MockCluster>(MockClusterConfig{}));
        rejected.allocator_capacity={550}; // M=1000 fits; CPU needs 500+100, GPU 500+150.
        SequentialRuntime<AllocatorLimitedPrepareApi> bad_runtime(0,1,1,rejected,DeviceExecutionMode::PerDeviceWorkers);
        expect_error([&] {
            bad_runtime.run({plan,"sha256:"+std::string(64,'1')},{spec,std::nullopt,false},
                {{0,make_cipher(input,spec.spec.context_id,spec.spec.poly_degree,3,30)}});
        },"allocator_capacity=550");
        require(rejected.host_encodes==0 && rejected.preparation_attempts==0 && rejected.stats().communicate_calls==0,
                "small allocator capacity rejected only after initialization work was submitted");

        const std::uint64_t limit=backend==PrepareBackend::Cpu?600:650;
        AllocatorLimitedPrepareApi accepted(std::make_shared<MockCluster>(MockClusterConfig{}));
        accepted.allocator_capacity={limit};
        SequentialRuntime<AllocatorLimitedPrepareApi> runtime(0,1,1,accepted,DeviceExecutionMode::PerDeviceWorkers);
        auto artifact=runtime.run({plan,"sha256:"+std::string(64,'1')},{spec,std::nullopt,false},
            {{0,make_cipher(input,spec.spec.context_id,spec.spec.poly_degree,3,30)}});
        auto result=artifact.values.at(42).value.materialize();
        require(std::abs(result.slots[0]-11)<1e-9 && std::abs(result.slots[1]+22)<1e-9 &&
                accepted.cpu+accepted.gpu==20 && artifact.timing.encode_calls==21 && artifact.timing.fence_calls==0 &&
                accepted.active_preparations==0,
                "legal small allocator capacity failed to reclaim and advance");
        std::cout<<"PASS allocator cap backend="<<int(backend)<<" rejects 550 before initialization, rolls at "<<limit<<'\n';
    }
    auto config=*plan.waterline;
    ResourceUsage group(1);group.front[0]=500;group.plain[0]=100;
    WaterlineLedger unconstrained(config,{}),exact(config,{600}),too_small(config,{550});
    require(unconstrained.fits(group) && exact.fits(group) && !too_small.fits(group),
            "allocator cap did not constrain foreground plus plaintext together");
    exact.reserve(group);exact.release(group);
    require(exact.used.front[0]==0 && exact.used.plain[0]==0,"allocator-limited ledger retained credits");
    expect_error([&] {WaterlineLedger mismatch(config,{600,600});},"allocator capacity table mismatch");
    AllocatorLimitedPrepareApi no_limit(std::make_shared<MockCluster>(MockClusterConfig{}));
    SequentialRuntime<AllocatorLimitedPrepareApi> empty_runtime(0,1,1,no_limit,DeviceExecutionMode::PerDeviceWorkers);
    auto artifact=empty_runtime.run({plan,"sha256:"+std::string(64,'1')},{spec,std::nullopt,false},
        {{0,make_cipher(input,spec.spec.context_id,spec.spec.poly_degree,3,30)}});
    require(artifact.timing.encode_calls==21 && no_limit.cpu+no_limit.gpu==20,
            "empty allocator-capacity vector changed ordinary Waterline behavior");
    std::cout<<"PASS empty allocator capacity leaves configured budgets unchanged, malformed table rejects\n";
}

void test_parallel_encode_timing(const RuntimePlan &rolling,const LoadedOperatorSpec &spec,
                                 const std::vector<double> &input) {
    auto plan=rolling;plan.waterline->backend=PrepareBackend::Gpu;
    auto plaintext=plan.values.front();plaintext.id=60;plaintext.kind=ValueKind::Plaintext;plaintext.components=1;
    plan.values.push_back(plaintext);
    plan.execution.insert(plan.execution.begin(),{0,ReleaseOp{60}});
    plan.execution.insert(plan.execution.begin(),{0,EncodeOp{InlineEncodePayload{{.5,-1}},60}});
    number_instructions(plan);
    PrepareApi api(std::make_shared<MockCluster>(MockClusterConfig{}));
    SequentialRuntime<PrepareApi> runtime(0,1,1,api,DeviceExecutionMode::PerDeviceWorkers);
    auto artifact=runtime.run({plan,"sha256:"+std::string(64,'1')},{spec,std::nullopt,false},
        {{0,make_cipher(input,spec.spec.context_id,spec.spec.poly_degree,3,30)}});
    require(artifact.timing.encode_calls==21 && artifact.timing.encode_nanoseconds>0 && api.host_encodes==21 &&
            api.gpu==20 && api.cpu==0,"ordinary Encode and GPU Prepare corrupted shared timing counters");
    auto result=artifact.values.at(42).value.materialize();
    require(std::abs(result.slots[0]-11)<1e-9 && std::abs(result.slots[1]+22)<1e-9,
            "parallel ordinary Encode/GPU Prepare changed arithmetic");
    std::cout<<"PASS ordinary CPU Encode and GPU Prepare share protected timing counters\n";
}
int main() {
 try {
    WaterlineConfig c; c.device_capacity_bytes={1000};c.device_reserve_bytes={0};c.device_fractions={.4};
    c.host_budget_bytes=600;c.pinned_budget_bytes=300;c.raw_budget_bytes=100;c.lookahead_groups=8;
    WaterlineLedger ledger(c);ResourceUsage first(1);first.front[0]=400;first.plain[0]=300;ledger.reserve(first);
    ResourceUsage two(1);two.plain[0]=200;two.front[0]=200;two.host=100;
    ResourceUsage one(1);one.plain[0]=100;
    require(ledger.fits(one) && !ledger.fits(two),"two-plaintext complete group was partially admitted");
    expect_error([&] { ledger.reserve(two); }, "complete group admission");
    require(ledger.used.front[0]==400 && ledger.used.plain[0]==300 && ledger.used.host==0,
            "failed complete-group reservation mutated the ledger");
    ledger.release(first);require(ledger.fits(two), "group did not become admissible after completion reclaim");
    ledger.reserve(two);ledger.release(two);
    require(ledger.used.front[0]==0 && ledger.used.plain[0]==0 && ledger.used.host==0,
            "complete group failed to return its resource credits");
    auto spec=OperatorSpecReader::read_file(std::string(CKKS_RUNTIME_SOURCE_DIR)+"/docs/operator-spec/v1/profiles/poseidon-ckks-gpu.v1.json");
    RuntimePlan plan;plan.format_version=4;plan.plan_id=404;plan.target={spec.spec.target_id,1,{1},1,{spec.spec.id,spec.spec.version,spec.source_sha256}};
    const Place d{PlaceKind::Device,0,0};const auto v=[&](ValueId id,ValueKind kind) {return ValueDesc{id,kind,d,spec.spec.context_id,3,30,true,kind==ValueKind::Plaintext?1:2};};
    plan.values.push_back(v(0,ValueKind::Ciphertext));plan.external_inputs={0};
    const Place host{PlaceKind::Host,0,0};
    plan.values.front().place=host;
    auto initial=v(41,ValueKind::Ciphertext);plan.values.push_back(initial);
    plan.initialization={{0,CommAction{0,CommKind::Transfer,CommHint::PointToPoint,{0},{41},{host},{d},{ValueKind::Ciphertext}}}};
    ValueId previous=41;
    for(unsigned i=0;i<20;++i) {
        const auto p=ValueId(1+i*2),out=p+1;plan.values.push_back(v(p,ValueKind::Plaintext));plan.values.push_back(v(out,ValueKind::Ciphertext));
        plan.execution.push_back({0,PreparePlaintextOp{InlineEncodePayload{{.5,-1}},p}});
        plan.execution.push_back({0,ComputeOp{ComputeKind::AddCP,{previous,p},out,d,{}}});
        plan.execution.push_back({0,ReleaseOp{p}});if(previous)plan.execution.push_back({0,ReleaseOp{previous}});previous=out;
    }
    auto result_desc=v(42,ValueKind::Ciphertext);result_desc.place=host;plan.values.push_back(result_desc);
    plan.finalization={{0,CommAction{1,CommKind::Transfer,CommHint::PointToPoint,{previous},{42},{d},{host},{ValueKind::Ciphertext}}},{0,ReleaseOp{previous}}};
    previous=42;plan.final_outputs={previous};
    std::size_t ordinal=0;for(auto *phase:{&plan.initialization,&plan.execution,&plan.finalization})for(auto &i:*phase)i.ordinal=ordinal++;
    c.device_fractions={.25};plan.waterline=c;
    std::stringstream wire(std::ios::in|std::ios::out|std::ios::binary);binary_io::write_plan(wire,plan);const auto size=wire.str().size();wire.seekg(0);
    auto parsed=binary_io::read_plan(wire,size);require(parsed.format_version==4 && parsed.waterline && parsed.execution.size()==plan.execution.size(),"V4 binary round trip changed plan");
    require(std::get<PreparePlaintextOp>(parsed.execution[0].body).output==1,"binary Prepare tag/output corrupted");
    PlanVerifier::verify(parsed,spec,false);
    std::vector<double> input(spec.spec.poly_degree/2);input[0]=1;input[1]=-2;
    test_host_waiter(parsed, spec, input);
    test_small_waterline(parsed, spec, input);
    test_json_and_negative_config(parsed, spec);
    test_initialization_preflight(parsed, spec, input);
    test_prepare_across_fence_and_unused(parsed, spec, input);
    test_prepare_final_outputs(parsed, spec, input);
    test_prepare_failure_and_explicit_restart(parsed,spec,input);
    test_lazy_bundle_failure_and_explicit_repair(parsed,spec,input);
    test_four_device_progress_and_shared_budgets(parsed,spec,input);
    test_allocator_capacity(parsed,spec,input);
    test_fixed_prepare_buffers(parsed,spec,input);
    test_parallel_encode_timing(parsed,spec,input);
    for(auto backend:{PrepareBackend::Cpu,PrepareBackend::Gpu,PrepareBackend::Auto}) {
        parsed.waterline->backend=backend;
        // Auto needs room for two groups including GPU's extra 50-byte workspace.
        parsed.waterline->device_capacity_bytes={backend==PrepareBackend::Auto?1200u:1000u};
        PrepareApi api(std::make_shared<MockCluster>(MockClusterConfig{}));
        SequentialRuntime<PrepareApi> runtime(0,1,1,api,DeviceExecutionMode::PerDeviceWorkers);
        auto artifact=runtime.run({parsed,"sha256:"+std::string(64,'1')},{spec,std::nullopt,false},
            {{0,make_cipher(input,spec.spec.context_id,spec.spec.poly_degree,3,30)}});
        auto result=artifact.values.at(previous).value.materialize();require(std::abs(result.slots[0]-11)<1e-9 && std::abs(result.slots[1]+22)<1e-9,"rolling Prepare changed arithmetic");
        require(api.cpu+api.gpu==20 && artifact.timing.fence_calls==0,"Prepare repeated or inserted Fence");
        require(backend!=PrepareBackend::Cpu || (api.cpu==20 && api.gpu==0), "CPU-only selected a GPU Prepare");
        require(backend!=PrepareBackend::Gpu || (api.gpu==20 && api.cpu==0), "GPU-only selected a CPU Prepare");
        require(backend!=PrepareBackend::Auto || (api.cpu>0 && api.gpu>0), "Auto rolling test did not exercise both backends");
        std::cout<<"PASS rolling backend="<<int(backend)<<" cpu="<<api.cpu<<" gpu="<<api.gpu<<"\n";
    }
    parsed.waterline->device_capacity_bytes={1000};
    for(int fault=0;fault<6;++fault) {
        auto bad=parsed;
        if(fault==0)bad.waterline->device_fractions={.01};
        if(fault==2)bad.waterline->device_capacity_bytes={450}; // Input + output + workspace need 500.
        if(fault==3)bad.waterline->host_budget_bytes=250; // External Host input + Prepare need 300.
        if(fault==4)bad.waterline->pinned_budget_bytes=25;
        if(fault==5)bad.waterline->raw_budget_bytes=10;
        PrepareApi api(std::make_shared<MockCluster>(MockClusterConfig{}));api.unsupported=fault==1;
        SequentialRuntime<PrepareApi> runtime(0,1,1,api,DeviceExecutionMode::PerDeviceWorkers);
        bool rejected=false;
        try {runtime.run({bad,"sha256:"+std::string(64,'1')},{spec,std::nullopt,false},{{0,make_cipher(input,spec.spec.context_id,spec.spec.poly_degree,3,30)}});}
        catch(const ClusterPanic &e) {rejected=std::string(e.what()).find(fault==1?"unsupported complete":"single-group")!=std::string::npos;}
        require(rejected && api.cpu+api.gpu==0,"infeasible/unsupported baseline submitted partial work");
    }
    std::cout<<"PASS complete atomic group, delayed events, continuous reclaim, no partial start on failure, V4 binary\n";
 }catch(const std::exception &e) {std::cerr<<e.what()<<'\n';return 1;}
}
