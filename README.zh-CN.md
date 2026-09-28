# OpenJev-Vision

**一张图编码一次，多个问题共享同一个概率分布。**

OpenJev-Vision 是一个研究视觉概率判断的开源项目，提供原创合成数据、公开图像实验、
可下载的训练权重和完整复现流程。原 OpenJev 文本概率决策实验也保留在本仓库中。

[视觉项目说明](docs/VISION.zh-CN.md) · [视觉模型权重](https://huggingface.co/IamBusy/OpenJev-Vision) · [数据集](https://huggingface.co/datasets/IamBusy/OpenJev-Vision-Research-v0.1) · [视觉实验结果](reports/vision-v01/RESULTS.md) · [English](README.md)

![多个视觉判断共享同一个概率分布](reports/vision-v01/demo.png)

## 快速体验视觉模型

当前版本包含 8,192 张原创合成图像、2,960 张 Oxford-IIIT Pet 照片子集和
1,680 张 CLEVR-4 图像。合成场景使用小型 CNN 和额外提供的先验；公开图像实验
使用冻结的 DINOv2 特征和训练后的预测头。两类实验各有固定的事件与类别范围，
查询使用声明好的事件语义和有限英语模板，尚不支持任意图片与自由问答。

```bash
git clone https://github.com/IamBusy/OpenJev-Vision.git
cd OpenJev-Vision
uv sync --frozen --extra vision --extra dev
uv run --no-sync openjev-vision download
uv run --no-sync openjev-vision predict \
  --checkpoint artifacts/openjev-vision-v0.1/synthetic-joint \
  --image examples/vision/scene-0.png \
  --prior examples/vision/scene-0-prior.json \
  --questions examples/vision/questions.json
```

三组训练种子的结果、独立校准和失败案例均已公开：整体后验预测在新依赖结构上
退步；在 CLEVR-4 未见组合上，独立属性基线优于本次整体类别头和低秩交互头。
公开照片推理与完整复现见[视觉项目说明](docs/VISION.zh-CN.md)。

## 文本概率决策实验

[Hugging Face 模型](https://huggingface.co/IamBusy/OpenJev-0.6B) · [English](README.md) · [实验结果](reports/v03/RESULTS.md) · [复现说明](docs/REPRODUCING.md)

仓库名称为 **OpenJev-Vision**，公开文本模型名称为 **OpenJev-0.6B**。训练版本和软件版本
单独管理，详见[命名与版本规则](docs/NAMING.md)。

OpenJev 是一个研究型开源项目：输入状态、问题和候选描述，直接得到结构化概率，
无需生成答案文本。支持 `choice`（候选选择）、`noul`（命题为真的概率）和
`score`（有序评分的期望等级）。

项目受 TypeSafe 的 Jev 启发，与 TypeSafe 无隶属关系。这里使用监督交叉熵训练和
独立校准集，没有声称复现 Jev 未公开的模型结构、权重或 RLCD 算法。

## 快速使用

建议使用 Python 3.12 和 [uv](https://docs.astral.sh/uv/getting-started/installation/)。
实测环境为 Apple M3 Pro；CPU 使用 FP32，本版本尚未实现 CUDA 加速。
基础模型约 1.2 GB，安装和运行需预留数 GB 磁盘及内存空间。

```bash
git clone https://github.com/IamBusy/OpenJev-Vision.git
cd OpenJev-Vision
uv sync --frozen --extra qwen --extra dev
uv run --no-sync openjev-model download
uv run --no-sync openjev-model predict --input examples/refund.json
uv run --no-sync openjev-model serve --port 8081
```

下载命令会取得固定版本的 Qwen 基模，以及 GitHub Release 上的 LoRA、评分头和
校准参数，并检查哈希。推理和默认数据重建都不需要 DeepSeek 密钥。
服务位于 `http://127.0.0.1:8081/v1/decide`，只面向本机实验。

## 当前实验

v0.3 使用 Qwen3-0.6B，同一状态只编码一次，每个问题和候选独立评分，再在每题内
归一化。单题最多 255 个候选，每次请求最多 512 个候选分支。状态前缀上限为
768 tokens，候选分支为 192 tokens，包含提示词开销；超限会明确报错。

- 固定 60 题：v0.2 为 39/60，v0.3 为 45/60，DeepSeek 为 59/60。
- 固定 16 题候选换序一致率：68.75% → 100%。
- 较长合成状态、8 题 × 4 候选：本机热启动耗时约 1.38 秒 → 0.70 秒。
- 短状态单题反而变慢，部分银行分类、权限规则和物流任务退步。

这些结果来自小规模固定实验。结构和训练数据同时发生变化，不能把准确率变化
单独归因于结构；概率校准也不保证跨域有效。详见[完整结果与限制](reports/v03/RESULTS.md)。

## 开发与复现

视觉快速开始的环境可以运行数据及 CPU 单元测试，不依赖 PEFT。
开发所有后端时，安装完整的贡献者环境：

```bash
uv sync --frozen --extra vision --extra dev --extra qwen
uv run --no-sync pytest -q
uv run --no-sync ruff check src tests scripts
uv build
```

[复现指南](docs/REPRODUCING.md)说明了公开数据下载、固定场景快照、训练选择、
校准和评测流程。CI 分别验证不含 PEFT 的 `vision` + `dev` 环境和完整依赖环境，
两者都覆盖微型随机模型的缓存前向及梯度等价检查；需要完整模型和
数据的集成测试，在缺少本地文件时会明确跳过。

代码使用 Apache-2.0。公共数据保留各自许可证，详见[来源与署名](THIRD_PARTY.md)。
[模型说明](docs/MODEL_CARD.md) · [贡献指南](CONTRIBUTING.md) · [安全说明](SECURITY.md)

## 从 Hugging Face 直接加载

```python
from openjev import OpenJevModel
model = OpenJevModel.from_pretrained("IamBusy/OpenJev-0.6B")
```

会自动下载训练后的 LoRA、独立评分头、校准参数和固定版本基模，无需手动安排
文件目录。模型参数与原 v0.3 实验一致；当前公开入口由代码版本 v0.3.2 提供，旧接口继续兼容。
详见[发布形式与加载说明](docs/HUGGING_FACE.md)。

[演示与图文素材源码](marketing/README.md) · [本地清理与环境恢复](docs/LOCAL_STORAGE.md)
