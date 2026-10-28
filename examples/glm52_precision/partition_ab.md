# GLM-5.2 单层分区 A/B（服务器）

目标：判断一个实际 MoE Decoder Layer 能否在单卡上按整层校准，以及细粒度分区的
显存收益是否值得其调度成本。**不重新跑完整 L5，不加载 78 层，不保存量化 checkpoint。**
现有 L5 已完成量化和保存；它没有 Decoder Layer 对照组，本实验补这个证据。

## 不需要旧 L4 cache 的运行方法

在这份 checkout 的仓库根目录、原有 L5 Python/NPU 环境执行。权重只读路径为
`/mnt/weight/GLM-5.2`。脚本只读取第 3 层（0-based）的参数，保留全部 routed/shared
experts。必须是原始 BF16、显式 2D expert 布局；若布局不符，会在 header 检查阶段
退出，不尝试读取完整模型。

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -m pip install --no-deps --no-build-isolation -e .

# 先验证服务器环境可执行这套 harness，不读取真实权重。
python tools/glm52_partition_ab.py ab \
  --tiny --device npu:0 --dtype bfloat16 --offload disk \
  --samples 2 --tokens 8 --pairs 1 \
  --output /mnt/weight/glm52-partition-tiny-ab

# 真实单层：A=DecoderLayer，B=当前 Expert targets；顺序 AB / BA。
# 输出目录必须是新目录；也可换到有足够空间的本地 NVMe。
python tools/glm52_partition_ab.py ab \
  --model-dir /mnt/weight/GLM-5.2 --layer 3 --synthetic-inputs \
  --device npu:0 --dtype bfloat16 --offload disk \
  --samples 5 --tokens 2688 --threads 4 --pairs 2 \
  --output /mnt/weight/glm52-partition-layer3-ab
```

不要使用 torchrun。本轮只占一张 NPU；替换 `npu:0` 可选择其他空闲卡。
`--pairs 2` 启动四个独立进程 ABBA，每次重新读取未变换的单层权重；若先做最小
一组可用 `--pairs 1`，但单组时间不能证明稳定速度差。终端输出每个 worker 的退出码；
运行详情实时写入对应 `0-layer.log` / `0-expert.log` 等文件，可另开终端 `tail -f`。

`--synthetic-inputs` 固定 seed，生成 5 组不同的 `[1,2688,hidden_size]` 标准正态输入、
causal mask 和真实配置的 RoPE。shared indexer 层用固定 causal recent-token 索引
代替前一层输出。**它用于容量/调度测试，不代表真实激活分布、真实路由比例或精度评测。**
不需要找到旧 L4 文件，也不会生成或加载完整模型。每个 trial 会校验相同初始权重和
输入的 SHA256，避免把不同输入的时间误作 A/B。

## 比较什么，哪些条件保持一致

两组都使用现有 `mixed_mxfp.yaml`、QuaRot block32/seed1234/FP32，Flex max_tokens=None，
BF16 storage，routed 模式，batch1、propagate_error=True、CPU intermediate cache。
运行真实 `session.initialize → SequentialPipeline → session.finalize`，沿用 oneshot 的
norm calibration context；不是手写模拟子图，也不会调用 AutoModel 加载全模型。

唯一调度变量：

- A：`sequential_targets=["GlmMoeDsaDecoderLayer"]`。
- B：`input_layernorm / ExpertMLP / GlmMoeDsaMLP`，每子图 1 target。

边界复用 L4 的 identity embedding/head 和 unit final norm，保持 QuaRot 坐标变换。
它们也有内存/计算开销，双方相同；不能将本实验耗时直接乘 78 预测生产耗时。
第 3 层保留原始 indexer/layer 配置，重编号为 harness layer0，不更改层内 forward。

`--offload disk` 为所有层参数建立**本次运行私有的 CT DiskCache**。原始 checkpoint
只读；临时 backing 默认在 trial 输出目录内，也可 `--scratch /existing/scratch/parent`。
每轮结束自动清理其私有临时目录，不触碰 checkpoint。需要至少一个 BF16 层的 backing
空间（当前配置约 19 GiB，再留临时写入余量）；建议准备 50 GiB 可用空间。硬杀进程
可能留下该轮 `glm52-ab-*` 私有 scratch，确认进程结束后再清理。

这与生产的 **8-rank shared CPU/tmpfs + disk auto_offload** 不完全相同：本轮没有共享
backing/HCCL，也没有自动 CPU/disk 配额分配。结果只证明当前单层、单卡、该 backing
条件。若要对照 CPU backing，可另用新 output 跑相同命令并改 `--offload cpu`，
不能把两种 backing 的数据混作同一组 A/B。

## 采集与时间口径

每轮 `report.json` 记录源码 commit/状态、harness/recipe 哈希、版本、输入形状和哈希、
所有初始/最终参数与 buffer 哈希、Flex alpha/beta/loss、以下计时与计数：

- `selected_layer_load`：选层读取与私有 backing 创建。
- `initialize`：三个 Modifier 初始化与 qparams 注册。
- `trace_partition`：真实 FX trace 和图分区，附每个子图成员。
- `calibration_start_including_quarot`：QuaRot 全单层变换及后续 hook 安装；与分区循环分开。
- `sequential_execution`：激活 cache 创建、校准 forward、Flex search/apply、weight observer、
  传播与 calibration_end；这是主要比较阶段。
- `finalize`：生命周期收尾。

以上阶段边界同步设备并 reset allocator peak，记录同步 wall time、末端等待时间、
peak allocated/reserved、250ms 采样的 RSS 最大值及 OS process I/O 增量。
reserved 可包含前一阶段留下的空闲 allocator 块；allocated 更适合比较活跃张量峰值。
allocator 统计不包含全部驱动占用；RSS 不是 PSS，采样不能保证抓住瞬时尖峰。
需要整卡物理峰值时，请另保留该实验时间窗口的现有 NPU/系统监控记录。

高频操作只累计次数和 host time，不逐调用同步：每个子图的 calibration/propagation、
Flex search/apply、quant epoch、intermediates fetch/update/delete、每个 expert 的调用数
与 token 数。CPUCache/DiskCache 另记录按阶段/模块的 onload/update 次数和逻辑 bytes。
这些 **host time 不是完整设备时间，逻辑 bytes 不是物理磁盘或 PCIe 流量**。
没有每个子图的独立同步 kernel 时间；主阶段同步计时用于 A/B。

不主动清空 OS page cache，不更改 mount/shm，不宣称冷盘实验；选层加载、初始哈希和
QuaRot 都会预热访问。ABBA 用于检查顺序偏差，不能替代重复测量。数值审计和
post-finalize forward 在测量阶段之外，避免其权重读取污染校准计数/峰值。

## 返回文件与判定

把整个 `glm52-partition-layer3-ab` 输出目录（JSON、log、numerics.safetensors）打包返回；
不需要发送层权重或 checkpoint。`numerics.safetensors` 仅包含 Flex scale、单层输出和
top-k。如失败，仍返回日志及 partial report；SIGKILL/其他错误不能自动当作 OOM。

`comparison.json` 给出两组的阶段指标、输出 max_abs/relative_L2、Flex scale 对比、
最终参数/qparams 哈希差异及 `exact_numerical_match`。**不放宽任何已有 L4 tolerance**：
完全一致时可直接比较性能；数值不同时保留差异，由我们分析，不能仅因更快就选它。
这没有验证 packed checkpoint、save/reload 或任务精度。退出非零也可能只是 A/B 不一致，
不等同于某算法已证实错误。

判断假设：A 在目标卡上有明确峰值余量、数值对齐且稳定更快，则支持继续验证整层策略；
若 A OOM 且日志明确、B 完成，则当前输入/offload 条件下细分有实际容量收益。
即使 A 成功，也仅完成单层单卡容量门槛，不能立即宣称八卡全模型默认已验证。

如果之后需要真实激活分布：将 `--synthetic-inputs --samples ... --tokens ...` 换为
`--activation-cache cache1.safetensors cache2.safetensors ...`；输入文件必须是既有
`glm52_layer_l4_capture` 的完整层格式，脚本会核对选层权重和 config 指纹。
不要求为了本轮性能测试重新采集它。

## 本地测试（无 NPU / checkpoint）

```bash
python -m pytest tests/flex_smooth/test_partition_ab.py -q
python tools/glm52_partition_ab.py ab --tiny --device cpu --dtype bfloat16 \
  --offload disk --threads 1 --pairs 2 --output /tmp/glm52-tiny-partition-ab
```

覆盖 CPU/disk × FP32/BF16 的三 Modifier 真生命周期、真实子图/专家/缓存操作计数、
合成 checkpoint 的仅选层加载、shared indexer、真实 cache 读取模式、原文件不变、
局部执行设备固定与恢复、边界同步次数和失败报告。CPU 结果不是 NPU 加速证据。
