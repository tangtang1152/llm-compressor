# FlexSmooth E2：两 rank、一次加载

2026-11-20 用户纠正后的执行方案，替代原 plan/load/smoke/formal 多加载流程。
当前每个 split 一条完整 prefill：固定 NPROC=2，rank0 Calibration、rank1 Validation，
各 2048 tokens。原数据划分、context、seed、token identity、norm-linear consumer
pre-hook、权重顺序、存储 dtype 和 QDQ/search/evaluate 不变。

## 与 QuaRot E2 对照

查看 quarot-experiments 分支 e2_collect.py/e2_sequential.py 后确认：历史默认是
load_context + device_map=auto_offload + BF16/eager + 两 rank 共用的新 offload_folder，
默认不传 max_memory；显式参数才覆盖它。通信 timeout 是 3 小时。
本次恢复这些默认参数，删除强制 CPU max_memory、600 秒 watchdog、独立 load 阶段及
load/smoke/formal Gate。直接复用已适配 QuaRot 的 SequentialRunner/IntermediatesCache，
不修改 QuaRot 原始代码或生产 LLMC/CT。

40 it/s 是用户报告的 Source Rank Dispatch 速度，1.3 分钟是此前单进程加载时间；
二者不是同一个指标，不能据此计算总加速比或确定退化原因。真实服务器需要比较
load_seconds + smoke/capture_seconds + wall_seconds。不要继续调试六卡权重放置。

## 一次执行

每 rank 仅调用一次 from_pretrained。先执行 L3/input_layernorm、128-token Smoke，
比较旧单进程 cache 相同前缀：权重及 identities 完全一致，激活记录 MSE/NMSE/max_abs
和 bitwise_equal，默认 NMSE <= 1e-5 且 max_abs <= 0.125。超限停止，不自动放宽。
通过后在同一模型上捕获 L6/16/60/77 的 input_layernorm、q_a_layernorm，2048/2048。
到 L77 Attention 即止。原始 MoE 前缀是必要计算，但不执行 QuaRot/量化 Modifier。

保留必要检查：原始 checkpoint 配置、旧 E1 完整 token identities、rank backing 一致、
每 mapping consumer 次序/dtype/权重 hash、activation hash、finite、token 无缺失/重合。
沿用 QuaRot 的 all-rank 错误共识。资源保护只保留空闲 RAM/SHM 检查与每子图工作内存
检查（默认留 16 GiB，可用 --reserve-cpu-gib 调整）；无后台 watchdog/加载截止时间。
共享默认估计不能安全容纳时停止，不私自复制两份 CPU 模型。原生 allocator OOM 仍会
报错并由 torchrun 终止 peers；这些检查不保证并发任务抢占下永不 OOM。

所有输出/offload 必须是新目录；失败日志不清理。源文件 hash 保留，跨批分析不会
伪造与旧捕获相同源码。加载性能依赖服务器安装版本，必须以实际日志为准。

## 服务器运行

本地实际分支 flexsmooth-experiments；服务器先核对实际分支并整合本次提交，不覆盖
服务器未提交修复。本机是 Windows/CPU，没有可执行的服务器连接，以下未在真实 NPU 执行。

```bash
cd /mnt/share/t50065065/flexsmooth/llm-compressor
 git branch --show-current
 git status --short
npu-smi info
export EXPECTED_COMMIT=REPLACE_WITH_VERIFIED_INTEGRATED_FULL_SHA
export RUN_ROOT=/mnt/share/t50065065/results/flexsmooth/next/layer_extension/distributed/$(date +%Y%m%d-%H%M%S)
export ASCEND_RT_VISIBLE_DEVICES=0,1  # 必须选择两张空闲卡
export MODEL=/mnt/weight/GLM-5.2
export DATA=/mnt/share/datasets/qwen3_cot_w4a4.json  # E1 原数据
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
python -m pytest experiments/flexsmooth_proxy/test_distributed_capture.py -q -p no:cacheprovider
bash experiments/flexsmooth_proxy/run_distributed.sh all
```

all 顺序执行 capture（一次加载内 Smoke+Formal）→ 原 offline-smoke → search → evaluate
→ analyze。也可用同一脚本分别执行这五个阶段，失败后不得覆盖已有目录。
正式 search 仍为 INT8/MXFP4/MXFP8 × W-only/W+A，统一评价 W4A4/W8A8。
旧 full/L3/L40 只读，未重跑 P1/QuaRot/GPQA。

输出 formal/capture-smoke/comparison.csv、formal/capture/cache.pt、各 rank shard、
status-rankN.json、timings.csv、complete.json 及 torchrun 日志。后续原 search/evaluation/
analysis 子目录保存六臂结果、候选曲线和分 mapping 跨层收益；capture 源码不同会明确
标为不同批次，结合 Smoke 证据判断可比性，不直接声明严格同口径。

若真实两 rank 共享加载持续很慢，应比较完整墙钟后使用已经验证的原单进程 E1
capture auto_offload/sequential 入口，目标改为四个新层，使用另一个新输出/offload
目录，不传 max_memory。不得在缺少计时证据时宣称两 rank 更快，也不自动重复加载
做无限性能调参。本地无法提供真实加载/Capture 时间或六层搜索结果。
