# msModelSlim IterSmooth 分析与迁移

## 分析基线

本次以工作区源码为准：msmodelslim `a18feb70d980b3e119f295d7e6accd524234e9c8`，
llm-compressor `af7967973f9e0928ad8b25e05a79b1af97394bc4`。

主要参考位置：

- [IterSmooth 处理器与统计收集](https://github.com/Ascend/msmodelslim/blob/a18feb70d980b3e119f295d7e6accd524234e9c8/msmodelslim/processor/anti_outlier/iter_smooth/processor.py)
- [各子图的算法分派](https://github.com/Ascend/msmodelslim/blob/a18feb70d980b3e119f295d7e6accd524234e9c8/msmodelslim/processor/anti_outlier/iter_smooth/api.py)
- [缩放系数计算](https://github.com/Ascend/msmodelslim/blob/a18feb70d980b3e119f295d7e6accd524234e9c8/msmodelslim/processor/anti_outlier/common/scale_computation.py)
- [子图处理顺序](https://github.com/Ascend/msmodelslim/blob/a18feb70d980b3e119f295d7e6accd524234e9c8/msmodelslim/processor/anti_outlier/smooth_base.py)

IterSmooth 是量化前的离群值抑制步骤，本身不产生整数权重，也不决定推理端量化格式。
后续还需要 `linear_quant`，迁移后对应 `QuantizationModifier` 或 `GPTQModifier`。
虽然名称包含 Iter，且部分上游说明使用“迭代至收敛”的描述，新版 Python 处理器对每个子图只做一次
统计驱动的解析缩放，没有迭代次数、收敛判据或量化误差搜索。迁移版保留该行为。

## msModelSlim 中的入口

旧 Python API 使用 `AntiOutlierConfig(anti_method="m4", ...)` 创建配置，随后调用
`AntiOutlier(model, calib_data=..., cfg=...).process()`，最后才运行
`Calibrator.run()` 和保存。`m4` 分支调用 `anti_utils.iter_smooth`；工作区中这一模块只有
Linux 平台的 `.so` 文件，没有可审查的 Python 实现。因此本次迁移以新版处理器公开源码为
算法基线，不承诺与所有旧版 CANN/二进制实现逐位一致。
旧 `AntiOutlierConfig` 初始化的 `alpha=0.5`、`a_sym=False` 也不同于新版处理器默认值，
不能仅凭同名参数认定配置等价。

新版使用 `apiversion: modelslim_v1`，在 `spec.process` 中先配置 `type: iter_smooth`，
再配置 `type: linear_quant`，在 `spec.save` 中选择保存器。完整配方例子位于
msmodelslim 的 `test/st_pr/modelslim_v1/modelslim_v1_w8a8_itersmooth/`。
模型适配器通过 `IterSmoothInterface.get_adapter_config_for_subgraph()` 提供 source/targets
及融合布局。迁移后这些信息改为 `IterSmoothMapping`，或复用 llm-compressor 的架构映射。

```bash
msmodelslim quant --model_path /path/to/model --save_path /path/to/output \
  --model_type Qwen3-8B --device npu --config /path/to/quant.yaml
```

| msModelSlim 配置/入口 | llm-compressor 对应项 |
| --- | --- |
| `type: iter_smooth` / 旧接口 `anti_method="m4"` | `IterSmoothModifier` |
| `alpha`, `scale_min`, `symmetric` | 同名参数；默认值采用新版处理器 |
| `enable_subgraph_type` | 同名子图开关 |
| `include`, `exclude` | 同名 source 名称 glob 过滤 |
| `AdapterConfig.mapping.source/targets` | `IterSmoothMapping.source/targets` |
| 连续 QKV / gate-up fusion 配置 | `fused=True`，OV 同时指定或推断头数 |
| `linear_quant` | `QuantizationModifier`，也可选择 GPTQ |
| `ascendv1_saver` | `save_pretrained(save_compressed=True)`，输出 compressed-tensors |

下面 W8A8 示例采用 llm-compressor 预设。上游实际配方可能使用非对称激活、其他 observer
或跳过 MLP，不能把迁移后的算法与完整量化配方的精度直接等同。复现实验时应显式对齐权重/
激活的位数、对称性、粒度、observer、回退层与校准数据。

## 算法与使用

收集第一个目标 Linear 的输入，展平除最后一维以外的维度，跨校准批次累积每个输入通道的
最小值和最大值。对称模式使用 `A = max(abs(x_min), abs(x_max))`。
对共享输入的所有目标 Linear，取每列权重的绝对值最大值，得到 `W`：

```text
W_j = max(all target weight rows at input channel j, abs), lower bounded by 1e-5
s_j = max(A_j ** alpha / W_j ** (1 - alpha), scale_min)
```

默认 `alpha=0.9`、`scale_min=1e-5`、`symmetric=True`。
增大 alpha 会更多地把激活离群值转移到权重。应使用真实校准数据评估精度，不保证 alpha
越大越好。迁移版用 FP32 计算统计和缩放，再将更新后的参数转换回原始 dtype。

| 子图 | 收集统计的位置 | 参数变换 |
| --- | --- | --- |
| norm-linear | 首个目标 Linear 的输入 | norm 的 weight/bias 除以 s，所有目标权重的列乘以 s |
| linear-linear | 第二个 Linear 的输入 | 第一个 Linear 的权重行和 bias 除以 s，第二个权重列乘以 s |
| up-down | down_proj 输入，即激活函数与门控乘法之后 | 只缩放 up 分支，不改变 gate 分支；down 权重列乘以 s |
| ov | o_proj 输入，即注意力聚合之后 | V 权重行和 bias 除以共享 s，O 权重列乘以共享 s |

执行顺序与源码一致：`up-down → ov → linear-linear → norm-linear`。
顺序影响后续子图看到的权重，因此不能任意交换。

GQA/MQA 中多个 query head 共享一个 KV head。先计算 O 输入通道的缩放，再按
`[kv_heads, query_heads/kv_heads, head_dim]` 分组，对共享组求平均；V 使用组平均值，
O 使用重复展开后的相同值。分别缩放共享 V 和各 O head 会破坏等价性。

非对称模式只对 norm-linear 生效，其他类型仍使用对称平滑：

```text
c = (x_max + x_min) / 2
A = (x_max - x_min) / 2
gamma' = gamma / s
beta' = (beta - c) / s
W' = W * s
b' = b + W_original @ c
```

这里的 symmetric 控制是否中心化平滑，不会自动设置后续量化器的 symmetric。

## llm-compressor 接入

实现位于 `src/llmcompressor/modifiers/transform/itersmooth/`，无需安装 msmodelslim、
torch_npu 或 CANN。继承原生 Modifier，使用现有钩子生命周期、配方工厂和
`update_offload_parameter` 更新参数。结束后移除统计钩子，推理无需平滑钩子。

```python
from llmcompressor import oneshot
from llmcompressor.modifiers.transform.itersmooth import IterSmoothModifier
from llmcompressor.modifiers.quantization import QuantizationModifier

recipe = [
    IterSmoothModifier(alpha=0.9),
    QuantizationModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"]),
]
oneshot(
    model=model,
    processor=tokenizer,
    dataset=calibration_dataloader,
    recipe=recipe,
    pipeline="independent",
)
model.save_pretrained("model-itersmooth-w8a8", save_compressed=True)
tokenizer.save_pretrained("model-itersmooth-w8a8")
```

`independent` 会为 IterSmooth 和量化器分别运行校准过程，量化器因此观察平滑后的输入。
不要把它们放在同一轮 basic/sequential 校准中，否则激活量化统计可能仍属于平滑前的张量。
也可以分两次调用 oneshot，第一轮仅做 IterSmooth，第二轮量化。
GPTQ 的用法是将后一个 Modifier 替换为 `GPTQModifier`。

完整命令行示例见 [itersmooth_example.py](../examples/quantization_w8a8_int8/itersmooth_example.py)。

默认复用 SmoothQuant 的架构感知 norm 映射，并为独立的常规
`gate_proj/up_proj/down_proj`、`q_proj/k_proj/v_proj/o_proj` 增加内部映射。
不自动猜测 fused 投影、自定义门控、V normalization、交错 head 布局或其他特殊拓扑。
对于新架构，应显式提供完整消费者映射并验证浮点输出等价性。

```python
from llmcompressor.modifiers.transform.itersmooth import (
    IterSmoothMapping,
    IterSmoothModifier,
)

modifier = IterSmoothModifier(
    mappings=[
        IterSmoothMapping(
            source="re:.*input_layernorm$",
            targets=["re:.*q_proj$", "re:.*k_proj$", "re:.*v_proj$"],
            subgraph_type="norm-linear",
        ),
        IterSmoothMapping(
            source="re:.*v_proj$",
            targets=["re:.*o_proj$"],
            subgraph_type="ov",
            num_attention_heads=32,
            num_key_value_heads=8,
        ),
    ],
    include=["model.layers.*"],
    exclude=["model.layers.0.*"],
)
```

`include/exclude` 按 source 的完整名称做 glob 匹配，exclude 优先。映射名称使用
llm-compressor 的名称/`re:` 正则规则。一次映射必须包含 source 的所有消费者，尤其是
MoE router 和共享分支。不能只补偿部分消费者，然后继续缩放共享 norm。
所有映射成员需要处于同一个顺序校准子图中，通常选择 DecoderLayer 为
`sequential_targets`；跨子图映射会明确报错。

连续 `[Q, K, V]` fused projection 或 `[gate, up]` projection 可显式设置
`fused=True`，仅修改 V 或 up 对应的行；不支持交错存储。
OV 头数可在映射中指定，或从 attention/model 的 config 读取。

配方也可以使用 YAML：

```yaml
default_stage:
  default_modifiers:
    IterSmoothModifier:
      alpha: 0.9
      scale_min: 0.00001
      symmetric: true
      enable_subgraph_type: [norm-linear, linear-linear, ov, up-down]
    QuantizationModifier:
      targets: [Linear]
      scheme: W8A8
      ignore: [lm_head]
```

将该 YAML 作为 recipe 传给 oneshot，并设置 `pipeline="independent"`。

## 与上游的差异和边界

| 项目 | 迁移行为与原因 |
| --- | --- |
| Linear/OV/up-down 权重统计 | 统一使用 absmax。上游这些分支将原始带符号权重传给 `max(dim=0)`，与 norm-linear 分支及文档公式不同；迁移版避免全负列被错误夹到 1e-5 |
| 非对称统计 | 根据跨批次最终 min/max 计算中心化后的精确幅度。上游按不断更新的 shift 再累积 absmax，结果可能依赖校准批次划分 |
| 数值保护 | 拒绝非有限激活，并在参数修改前检查前驱权重/bias 的逆缩放是否超出 dtype 范围；FP16 的极小缩放可通过提高 scale_min 或使用 FP32 校准处理 |
| 非对称无 bias RMSNorm | 提前报错。上游替换为自定义 RMSNormBias 并新增 Linear bias；普通 HF 架构重载不能保证读取新增参数。迁移版仅支持 norm 与所有 Linear 已有 bias 的 norm-linear 映射 |
| 非融合 source=None | 本次不支持。上游依赖运行时 pre-hook；直接复制该 hook 不足以让 compressed-tensors/vLLM 保存和恢复变换。必须先实现推理端可序列化的输入变换协议 |
| Offset norm | oneshot 自带的 norm_calibration_context 负责已注册的 offset norm；直接使用底层生命周期时必须使用相同上下文 |
| 分布式 | 支持模型副本的数据并行 min/max 合并；未命中专家的 rank 也参与相同顺序的 collective。张量并行与专家并行不在当前支持范围 |
| 模型精度 | 等价性测试验证浮点变换；实际量化质量还需要在目标模型上评估困惑度和任务指标 |

## 验证

新增测试覆盖解析公式、零通道、非连续输入、跨批次统计、对称/非对称 bias 补偿、
FP32/FP16/BF16、门控输入采集、MHA/GQA/MQA、fused 行选择、未命中专家、顺序回调幂等性、
分布式缺少本地样本的路径、参数校验、名称过滤，以及配方工厂和 YAML 往返。
另有本地随机初始化的小型 Llama 测试，用于检查真实 oneshot、量化和模型保存重载，
无需下载权重。

```bash
python -m pytest tests/llmcompressor/modifiers/transform/itersmooth -q
```

本次工作区的实际验证结果（2026-10-08）：新增代码通过 Ruff 检查、格式检查、
Python 编译检查、CUDA 调用规范检查和 `git diff --check`。
动态测试尚未执行：Windows 隔离环境中的 PyTorch 2.14.1 在 `import torch` 时报告
`OSError: [WinError 1114] ... torch/lib/c10.dll`，pytest 在加载 conftest 时退出，
没有进入测试收集。Visual C++ 运行库可单独加载；尚未确定 PyTorch DLL 初始化失败的根因。
因此不能将上述测试覆盖项视为已经通过，需要在可正常导入 PyTorch 的环境中运行命令。

这些测试不能替代真实大模型的精度、内存与吞吐评估。
