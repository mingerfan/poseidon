当前仓库的整体结构是：
1. DSL：神经网络/科学计算的流程通过DSL描述
2. compiler：编译器对DSL进行编译，然后下降到IR层面，进行一系列优化。生成runtime plan
3. runtime：runtime根据runtime plan执行代码，调用底层的api执行全同态
4. api：大体上分为GPU api和CPU api，分别调用GPU和CPU进行同态计算

基于项目的整体架构，当前的仓库主要开发分支分为四个：
framework分支：开发主分支，包含其他分支的所有修改。会定期拉取其他分支对应的修改。其他分支也需要定期拉取
benchmark分支：用于开发对库进行测试的各种benchmark
gpu-api分支：用于开发库的gpu侧的各种api支持
cpu-api分支：用于开发库的cpu侧的各种api支持
