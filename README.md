# CycleGAN 结构化复现项目

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg) ![License](https://img.shields.io/badge/License-MIT-green.svg)

基于PyTorch的CycleGAN结构化复现与整合项目

## 📝 项目简介

本项目是CycleGAN（Cycle-Consistent Generative Adversarial Networks）的完整PyTorch复现，实现了非配对图像到图像转换的核心功能。CycleGAN能够在没有配对训练数据的情况下，学习两个不同域之间的映射关系，适用于风格迁移、图像转换等多种任务。

## 🎯 功能特性

- **完整的CycleGAN实现**：包含生成器、判别器和损失函数的完整实现
- **非配对训练**：支持无配对数据的图像域转换
- **循环一致性**：确保A→B→A的循环重建能力
- **身份损失**：保持图像的基本结构和内容
- **混合精度训练**：支持AMP加速训练，提升训练效率
- **可视化展示**：实时生成训练过程的样本结果和损失曲线
- **模块化设计**：代码结构清晰，易于理解和修改

## 📊 数据集

本项目**默(夹)认(带)使(私)用(货)** [face2genshin](https://github.com/AiXing-w/face2genshin) 数据集作为示例：

- **训练数据**：
  - 域A图像（trainA）：777张
  - 域B图像（trainB）：777张
- **测试数据**：
  - 域A图像（testA）：100张
  - 域B图像（testB）：100张
- **图像尺寸**：256×256 RGB
- **数据格式**：JPG格式

**注意**：用户可以根据自己的需求替换数据集，只需按照相同的目录结构组织数据即可。

## 🛠️ 环境要求

主要用到PyTorch GPU版本，以及torchvision、tqdm、matplotlib、numpy、PIL等，具体见代码开头。PyTorch请根据[官方文档](https://pytorch.org/get-started/locally/)和自己的CUDA版本，选择合适的版本安装。

## 🚀 快速开始

### 1. 准备数据
1. 下载数据集。示例：[face2genshin数据集](https://github.com/AiXing-w/face2genshin)
```bash
git clone https://github.com/AiXing-w/face2genshin.git
```
2. 将数据解压到项目根目录下的数据集文件夹
3. 确保数据结构如下：
```
your_dataset/
├── trainA/     # 域A训练图像
├── trainB/     # 域B训练图像
├── testA/      # 域A测试图像（可选）
└── testB/      # 域B测试图像（可选）
```

ps：记得修改代码中的数据路径哟(❁´◡`❁)

### 2. 运行训练

按照notebook中的步骤依次执行所有单元格，完成模型训练和测试。

**注意根据实际情况，修改相关参数！**

## 📁 项目结构

```
cycleGAN/
├── CycleGAN.ipynb          # 主训练notebook
├── checkpoints/            # 模型权重保存目录
│   ├── cyclegan_final.pth  # 作者在云GPU训练后的模型权重(可忽略)
│   └── cyclegan_final1.pth # 保存的最终模型权重
├── your_dataset/          # 数据集目录（用户自定义）
│   ├── trainA/            # 域A训练数据
│   ├── trainB/            # 域B训练数据
│   ├── testA/             # 域A测试数据（可选）
│   └── testB/             # 域B测试数据（可选）
├── results/               # 结果输出目录
│   ├── loss_curves.png    # 损失曲线图
│   └── test_results.png   # 测试结果图
└── README.md             # 项目说明文档
```

## ⚙️ 模型参数

### 训练参数
- **批大小**：1
- **训练轮数**：200+
- **学习率**：0.0002
- **优化器**：Adam (β1=0.5)
- **循环一致性损失（$λ_{cyc}$）**：10.0
- **身份损失（$λ_{id}$）**：0.5

### 模型结构
- **生成器**：基于ResNet的9个残差块
- **判别器**：70×70 PatchGAN
- **归一化**：Instance Normalization
- **激活函数**：ReLU（生成器）、LeakyReLU（判别器）

## 🎨 损失函数

CycleGAN包含以下损失：

1. **对抗损失**（GAN Loss）：使用LSGAN损失，生成器试图生成逼真图像欺骗判别器
2. **循环一致性损失**（Cycle Consistency Loss）：确保A→B→A的循环重建，权重λ_cyc=10.0
3. **身份损失**（Identity Loss）：当输入图像已属于目标域时保持不变，权重λ_id=0.5

总损失 = GAN损失 + $λ_{cyc}$ × 循环损失 + $λ_{id}$ × 身份损失

## 📈 训练技巧

- **历史图像缓冲区**：存储50张历史生成图像，稳定判别器训练
- **学习率衰减**：100轮后线性衰减
- **混合精度训练**：使用PyTorch AMP加速训练
- **权重初始化**：正态分布初始化（均值0，标准差0.02）

## 🖼️ 结果展示

训练完成后，可以在 `results/` 目录下查看：

- **损失曲线**：显示生成器损失、判别器损失、循环损失和身份损失的变化趋势
- **测试结果**：展示双向转换和循环重建效果

**模型权重**保存在`checkpoints/`目录下的.pth文件

## 🐛 常见问题

1. **显存不足**：减小批大小或图像尺寸
2. **训练不稳定**：调整学习率或损失权重
3. **模式崩溃**：增加历史缓冲区大小或调整网络结构

## 🤝 贡献

欢迎提交Issue和Pull Request来改进项目！

## 📄 许可证

本项目基于MIT许可证开源

## 🙏 致谢

- [CycleGAN论文](https://arxiv.org/abs/1703.10593)：Jun-Yan Zhu等人的开创性工作
- [face2genshin数据集](https://github.com/AiXing-w/face2genshin)：用于演示的数据集
- [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)：官方实现参考

## 📚 参考资料

- CycleGAN原论文：[Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593)
- PyTorch官方文档：https://pytorch.org/docs/
- 混合精度训练：https://pytorch.org/docs/stable/amp.html

---

⭐ 如果这个项目对你有帮助，请给个Star！