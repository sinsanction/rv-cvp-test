# RV-CVP Test

RV-CVP 指令集测试程序

## 内容

cnnbench 单个算子的测试程序。cnnapibench 真实网络的测试程序。

其中 base 程序为核心计算采用标准指令的测试程序，用作性能测试的 baseline。perf 程序为核心计算采用 RV-CVP 指令集的测试程序，用于测试 RV-CVP 性能。func 为功能测试程序，用于验证框架实现的正确性。

cnnapibench 目录下的 v1test、v2test 为 CNNAPI 的功能测试程序。lenet 为真实模型 LeNet-5 测试程序，内置 MNIST 数据集。

## 使用

使用 CNNAPI 库一起编译，具体见 [RV-CVP CNNAPI](https://github.com/sinsanction/rv-cvp-cnnapi)。
