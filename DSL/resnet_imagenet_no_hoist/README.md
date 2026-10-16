# ImageNet ResNet no-hoist DSL

- [ResNet18：已验收交付包](resnet18/README.md)：原版与DSL完整GPU运行分类均为65（label65）；同密钥、同输入密文下，1000个最终密文metadata和全部RNS words逐位一致，logits差为0。源码、真实编译产物、固定模型数据、复现脚本和验收证据齐备。
- ResNet50：本目录原有文件保留为历史checkpoint，不作为已验收的新DSL交付。原生多项式最大池化修正及其验证继续在服务器/本地进行，本次没有发布ResNet50更新。

ResNet18结论限于image0单样本，不代表测试集准确率。使用方法、参数、独立加密误差与同密文对照范围见ResNet18子目录。

根目录原有tools/evidence/native源码归档属于早期18/50共同开发记录；当前ResNet18复现以`resnet18/`的冻结快照、manifest和脚本为准，不依赖历史运行状态。
