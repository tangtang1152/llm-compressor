# FlexSmooth E2：复用 QuaRot E2 的多 rank 数据采集

## 范围与已验证边界

用户最新选择：最多 8 rank，偶数 rank 采 Calibration，奇数 rank 采 Validation；
**仅按完整原始 prefill 分片，序列不足时减少有效计算 rank**。不切 token 区间，
不重放前缀，不增加 token 预算。2 rank 仍对应 rank0 Calibration / rank1 Validation。
没有更改生产 LLMC/CT、QuaRot 实验、原 capture/search/evaluate 或已有结果。

本地开发分支实际为 `flexsmooth-experiments`，开发起点 `23c65185`。
服务器提到的 `flexsmooth-experiments-wt` 需现场 `git branch --show-current` 核实；
不要因名称不同切换/覆盖服务器工作。远端 fetch 因 SSH/DNS 失败，未声称同步。
本机 Windows/CPU，无真实 GLM 权重、NPU 或服务器结果目录访问；下述服务器步骤
尚未执行。P1 已完成，不在本流程重复。

## 复用边界与数据语义

对照 `quarot-experiments` 分支的 `e2_collect.py`、`e2_sequential.py`、
`e2_capture.py`、`E2_IMPLEMENTATION.md`：移植其 torchrun / LOCAL_RANK、
混合 Gloo/HCCL 进程组、共享 auto_offload、all-rank 阶段错误共识和 shard 校验设计。
直接复用 FlexSmooth `offload.SequentialRunner`（此前已从 QuaRot E2 适配）和
LLMC 的 trace_subgraphs / IntermediatesCache / disable_offloading；不另写执行引擎。
QuaRot Expert 捕获 hook 不适用，因此保留 `glm_mappings()` 和首 consumer pre-hook。

`distributed_capture.py` 将操作拆为 prepare/partition、单 rank capture_split、
merge_shards。原 `capture.py` 保持不变，继续支持 E1 单进程和数值对照。

- 原 split_records：相同 tokenizer、seed、文档去重和偶/奇划分。
- 每 split 先按原 context 和 token 截断规则准备完整输入，再把整条输入分给该 split
  当前 token 负载最少的 rank（同负载优先低 rank）。全组总共 2048/2048，不是每 rank 2048。
- rank shard 记录原 token 位置，合并恢复原始顺序。逐 mapping 校验 consumer 顺序、
  原始 dtype、权重 SHA256、激活 SHA256、token identities、无重合、无缺失。
- 若只有一条 2048-token 输入/每 split，有效计算 rank 就是 0、1。其余 rank 参与共享
  加载及状态共识，记录 `active=false`、0 tokens/0 subgraphs，完全不做模型前向。
  `plan` 不加载权重，先给出实际并行度；可据此用新 RUN_ROOT 和 NPROC=2/4/6 重新规划，
  减少空闲 worker 的加载/通信开销。不得宣称固定 8 倍加速。
- Sequential 只执行到最高目标层 Attention，保留原始前面层的必要 MoE 计算。
  不执行最高目标层 Attention 后的 MoE、后续层或 LM Head。
- 原 BF16 加载及 checkpoint 的 FP32 例外由 from_pretrained 保留，不调用全模型 `.bfloat16()`。
  同一 group 的拼接 dtype 提升、激活存储 dtype 与原 E1 相同。
- 最终仍为 schema=1 `capture/cache.pt`，可直接交给原 E1 search/evaluate。

## Shared backing 和资源预算

本地实际 CT 为 `0.19.1a20260919`。检查其 load.py、dist_cpu.py、dist_disk.py 后确认：
rank0 加载，其他 rank 从 meta 构建；CPU storage 通过共享句柄连接，disk index 由源 rank
广播。启动时检查安装的 loader 合约、记录上述文件 SHA 和依赖版本；加载后核对各 rank
的全部共享 storage 句柄 / disk 索引、dtype、shape。遇到其他 backing 类型拒绝继续。
服务器版本没有未经验证地视为相同；load probe 是必要验证。

仍使用标准 `load_context()` + `device_map="auto_offload"`。
仅传 `max_memory={"cpu": shared_bytes}` **限制共享 CPU backing**，没有任何 NPU key，
不是此前六卡权重放置。这个上限也避免 CT 默认估计只看 `/dev/shm` 总容量而忽略可用量。
所有 worker 按 LOCAL_RANK 选择自己的执行 NPU，只有前向时分子图 onload。

必须显式提供四项预算（GiB），不从未知服务器容量猜值：

| 参数 | 含义与检查 |
|---|---|
| shared-cpu-gib | 全任务共享 backing 上限；加 tensor/page 预留后不得超过空闲 SHM |
| private-cpu-gib | 每 rank 工作内存；rank0 合并需能容纳所有 shard 和最终 cache |
| npu-gib | 每卡 allocator 上限；必须小于该卡空闲内存，另为 HCCL/runtime 留余量 |
| output-reserve-gib | shard/cache/结果磁盘预留；offload 另保守预留两倍 checkpoint 字节 |

主机可用 RAM 至少覆盖 `2 × shared + world_size × private + page overhead`，
其中第二份 shared 为源 rank 加载临时开销。另检查 mmap 数量和输出/offload 文件系统。
运行时每 0.5s 监控非 SHM PSS（Pss_Anon + Pss_File），源 rank 加载阶段允许 shared
大小的临时额度，其余阶段按 private 额度；同时记录 RSS/USS 峰值。USS 不能直接当成
独占内存：其他 rank 尚未触碰的共享页可能暂时被记入 USS。监控是采样边界，不能防止
采样间瞬时峰值/外部任务抢占；预检不是内存预留，也不修改服务器系统配置。

加载超过 `--max-load-seconds`（默认 600s）会中止，保留诊断文件；不会拖着异常加载
继续长时间 Capture。任一 rank 异常通过阶段共识或 torchrun 故障处理停止任务。
只接受全新 output/offload 路径，不删除任何旧目录。最终 `complete.json` 是成功标志，
仅所有 rank 完成、合并验证和 Smoke 比较通过后写出。失败保留 rank shard/tmp/log。

## 本地与服务器验证

新增测试覆盖 2/8 rank 分片、完整序列约束、空闲 rank、恢复原始 token 顺序、缺失/重复
rank、错误 split、缺失 mapping、consumer 次序/dtype/权重不一致、重复/跨 split token、
激活 hash、gate 和 peer 失败。tiny BF16 GLM 实际运行 pre-hook 和 Sequential 路径，
比较旧 capture；测试目标层后 MoE/后续层/LM Head 不执行。合并缓存还进入旧搜索/评价
做小矩阵接口验证，不是完整 GLM 搜索。

另有真实两进程 Gloo + CT shared loader 的 tiny GLM 测试，仅 Linux 执行；Windows
明确 skip，不能据本地 CPU 测试宣称 HCCL、NPU 或多进程共享加载已验证。

2026-11-20 本地验证：新增 23 passed / 1 skipped；原 core/offload/next_stage 回归
22 passed。tiny 2/8 逻辑 rank 的激活与旧入口零容差比较通过；Ruff、format、入口
`--help`、`bash -n` 均通过。尚无真实分布式/NPU或 GLM-5.2 Smoke 结果。

```bash
python -m pytest experiments/flexsmooth_proxy/test_distributed_capture.py \
  experiments/flexsmooth_proxy/test_offload.py -q -p no:cacheprovider
```

服务器最小假设：当前 CT 支持共享 backing；完整序列数据分片不改变捕获语义；多 rank
可并行处理不同 prefills。必须通过下列阶段，不能越过 gate：

1. **plan**：不加载模型。2048/2048 原 IDs 与旧 E1 完全相同；看 partition.csv 的有效 rank。
2. **load**：加载原始模型但不 forward。各 rank backing 签名一致、预算通过、加载时间正常。
3. **smoke**：L3 input_layernorm，每 split 128 tokens。完整序列太少时允许空闲 rank。
   与旧单进程 cache 的相同有序前缀比：权重/IDs 必须完全相同；激活保存实际 MSE、NMSE、
   max_abs、bitwise_equal，默认 NMSE ≤ 1e-5 且 max_abs ≤ 0.125。该容差是待检查的验收设置，
   不是已观测误差；不因不同 NPU 就假设逐位一致。超限失败，先诊断，不能静默放宽。
4. **formal**：L6/16/60/77，两 mapping，2048/2048。要求相同 world_size、数据、模型索引、
   runtime、源码和预算的 load/smoke gate。不重新捕获 L3/L40。
5. **offline-smoke → search → evaluate → analyze**：复用现有 next_stage；正式缓存自己的
   E0/QDQ smoke 通过后才能原六臂搜索。只改 capture，并未将 P1 优化接入搜索。

默认 Smoke reference 是既有 `/full/capture/cache.pt` 的 L3/128 前缀；若其来源不是旧
单进程 capture，必须显式 `REFERENCE_CACHE` 指向相同数据/权重的旧单进程小缓存。
无需为比较重跑完整 2048 capture；不能拿不同 seed/context 的缓存替代。

## 服务器命令（本机未执行）

先将本次变更整合到服务器实际分支，保留服务器已有修复；不要 reset/覆盖工作。
核对 commit 和依赖。只选择空闲卡；若没有八张空闲卡，使用可用偶数卡数/新 RUN_ROOT。

```bash
cd /mnt/share/t50065065/flexsmooth/llm-compressor
git branch --show-current
git status --short
npu-smi info
free -h
df -h /dev/shm /mnt/share/t50065065/results
cat /proc/sys/vm/max_map_count

export EXPECTED_COMMIT=REPLACE_WITH_VERIFIED_INTEGRATED_FULL_SHA
export RUN_ROOT=/mnt/share/t50065065/results/flexsmooth/next/layer_extension/distributed/$(date +%Y%m%d-%H%M%S)
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # 仅在八张都空闲时
export NPROC=8
export MODEL=/mnt/weight/GLM-5.2
export DATA=/mnt/share/datasets/qwen3_cot_w4a4.json  # 必须是 E1 同一份数据
export SHARED_CPU_GIB=REPLACE_WITH_SHARED_BUDGET
export PRIVATE_CPU_GIB=REPLACE_WITH_PER_RANK_WORKING_BUDGET
export NPU_GIB=REPLACE_WITH_PER_DEVICE_BUDGET
export OUTPUT_RESERVE_GIB=REPLACE_WITH_OUTPUT_DISK_BUDGET
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"

# 先运行 Linux 两进程 tiny 测试，然后检查真实数据的有效并行度。
python -m pytest experiments/flexsmooth_proxy/test_distributed_capture.py -q -p no:cacheprovider
bash experiments/flexsmooth_proxy/run_distributed.sh plan
cat "$RUN_ROOT/plan/partition.csv"

# 分阶段执行；每个命令结束检查 complete.json / timings / logs。
bash experiments/flexsmooth_proxy/run_distributed.sh load
bash experiments/flexsmooth_proxy/run_distributed.sh smoke
cat "$RUN_ROOT/smoke/comparison.csv"
bash experiments/flexsmooth_proxy/run_distributed.sh formal
bash experiments/flexsmooth_proxy/run_distributed.sh offline-smoke
bash experiments/flexsmooth_proxy/run_distributed.sh search
bash experiments/flexsmooth_proxy/run_distributed.sh evaluate
bash experiments/flexsmooth_proxy/run_distributed.sh analyze
```

各阶段使用独立 offload 目录；可占用较多磁盘，不自动复用或删除。每次运行使用新
RUN_ROOT，失败后保留日志，修复后另起目录。调整 NPROC/runtime/budget/source 后必须
重新 load/smoke。脚本只执行指定阶段，且禁用 torchrun 自动重启，不会重新跑 P1。
离线搜索/评价沿用原单进程实现；八卡数据并行仅用于 capture，不暗改搜索分布。

## 产物与跨层解释

每阶段：`partition.json`、`status-rankN.json`、torchrun 每 rank 日志；成功后有
`complete.json`、`timings.csv`、`report.md`。Capture 保留 `rankN.pt` 和
`capture/cache.pt`，Smoke 另有 comparison.json/csv。
时间分为加载、权重快照、trace、前向采集；内存有各 rank RSS/USS/非 SHM PSS 与 NPU
allocator 峰值。SHM 不应把各 rank RSS 相加后宣称独占总内存。

正式完成后仍生成原 `search/search.pt`、`evaluation/evaluation.json`、summary.csv、
候选 Loss 曲线。现有 analyze 分开 input_layernorm / q_a_layernorm，报告匹配 Proxy
相对 INT8、W+A 相对 W-only、No Smooth、后层退化与参数边界。

本次增加 capture 源文件，`capture_source_sha256` 与旧 E1 不相同。现有 analyzer 会
如实标为 separate experimental batches，而不会伪造旧 hash 以绕过检查。检查 token
身份、runtime 和 Smoke 的实测等价性之后，可以据此讨论方法可比性；单个 L3/128
Smoke 不能证明所有层/不同 runtime 严格等价。没有服务器结果前，不给六层趋势或
Selective FlexSmooth 的结论。
