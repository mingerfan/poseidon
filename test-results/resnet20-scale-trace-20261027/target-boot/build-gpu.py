from pathlib import Path
import shlex
import subprocess
import time

root = Path('/home/xuming/poseidon-manual-boot-20261026')
out = root/'resnet20-optimized-20261027T055506Z/placement-audit/target-scale-experiment/boot-tests/gpu'
out.mkdir(parents=True,exist_ok=True)
source = root/'src/src/poseidon/tests/runtime_api/poseidon_gpu_plan_e2e.cpp'
s = source.read_text()
context_start = s.index('        poseidon::PoseidonContext context =')
input_start = s.index('        std::vector<double> input{')
api_start = s.index('        PoseidonGpuApi api(')
runtime_start = s.index('        fhegpu::SequentialRuntime')
load_start = s.index('        const auto loaded_plan =')
load = s[load_start:context_start].replace('argv[1]','argv[case_index]')
load = load.replace('        const auto loaded_spec = fhegpu::OperatorSpecReader::read_file(argv[2]);\n','')
case = load+s[input_start:api_start]+s[runtime_start:s.index('    catch (const std::exception &error)')]
case = case[:case.index('        return passed ? 0 : 1;')]
case = case.replace('for (std::size_t i = 0; i < input.size(); ++i) input[i] = double(i % 17 + 1) / 32;',
 'const std::vector<double> samples{-0.75,-0.5,-0.25,-0.125,0.0,0.125,0.25,0.5,0.75,0.875};\n            for (std::size_t i = 0; i < input.size(); ++i) input[i] = samples[i % samples.size()];')
case = case.replace('write_report(argv[3], report);','write_report(report_path, report);')
case = case.replace('<< argv[3] <<', '<< report_path.string() <<')
# Each case has its own plan directory, while the context and Boot keys are reused.
text = s[:s.index('int main(')]+'''int main(int argc, char **argv) {
    if (argc < 4) {
        std::cerr << "usage: target_boot_gpu OPERATOR_SPEC REPORT_DIR PLAN...\\n";
        return 2;
    }
    try {
        const auto loaded_spec = fhegpu::OperatorSpecReader::read_file(argv[1]);
'''+s[context_start:input_start]+'''        std::cout << "Preparing native manual Boot resources" << std::endl;
'''+s[api_start:runtime_start]+'''        std::cout << "Native manual Boot resources ready" << std::endl;
        bool all_passed = true;
        for (int case_index = 3; case_index < argc; ++case_index) {
            const auto case_name = std::filesystem::path(argv[case_index]).parent_path().filename().string();
            const auto report_path = std::filesystem::path(argv[2]) / (case_name + "-gpu-report.json");
            std::cout << "Running " << case_name << std::endl;
'''+case+'''            all_passed = all_passed && passed;
        }
        return all_passed ? 0 : 1;
    }
'''+s[s.index('    catch (const std::exception &error)'):]
(out/'target_boot_gpu.cpp').write_text(text)
target = root/'build/src/poseidon/runtime_api/CMakeFiles/poseidon_runtime_gpu_plan_e2e.dir'
flags = {}
for line in (target/'flags.make').read_text().splitlines():
 if line.startswith('CXX_'):
  name,value = line.split(' = ',1)
  flags[name] = shlex.split(value)
command = ['/usr/bin/c++',*flags['CXX_DEFINES'],*flags['CXX_INCLUDES'],*flags['CXX_FLAGS'],'-I'+str(source.parent),'-c',str(out/'target_boot_gpu.cpp'),'-o',str(out/'target_boot_gpu.cpp.o')]
t=time.monotonic()
run=subprocess.run(command,capture_output=True,text=True)
(out/'compile.log').write_text(run.stdout+run.stderr)
assert run.returncode==0,run.stderr
link=shlex.split((target/'link.txt').read_text())
link[link.index('-o')+1]=str(out/'target_boot_gpu')
link=[str(out/'target_boot_gpu.cpp.o') if a.endswith('poseidon_gpu_plan_e2e.cpp.o') else a for a in link]
run=subprocess.run(link,cwd=root/'build/src/poseidon/runtime_api',capture_output=True,text=True)
(out/'link.log').write_text(run.stdout+run.stderr)
assert run.returncode==0,run.stderr
print('Built native-Boot GPU probe in',round(time.monotonic()-t,2),'seconds',flush=True)
