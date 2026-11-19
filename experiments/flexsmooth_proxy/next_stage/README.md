# P1 性能 + E2 扩大层采样（E3 已暂缓）

用户已明确暂缓 Block/Attention 实验。本目录只提供 P1/E2，不包含 Block
捕获或量化入口。生产 FlexSmooth、QuaRot、CT、旧 core.py/search/capture 均未修改。

## 状态与边界

本地分支为 flexsmooth-experiments，起点 c183b144。服务器 /full/ 文件本机不可见，
现有 8/8、6/8 与平均 4.76% 等结论属于用户提供的结果，未重新读取验证。
本机没有 torch_npu；不得将 CPU benchmark 当成 NPU 实测，也不自动连接服务器。
fetch 仍因 SSH/DNS 限制失败；不能宣称本地与远端已同步。

每个阶段要求 --expected-commit 为本次交付的完整 Git SHA；检查 HEAD 和实验目录
已跟踪文件的未提交差异。status.json 同时记录运行参数、seed、依赖版本和代码哈希。
旧 cache/search 的 provenance 不要求 Git HEAD 与新入口相同，而是比较实际搜索、
量化文件哈希及运行库版本：新增 orchestration 不等于改了搜索算法。
服务器若有自己的未提交修复，先保存并与本次代码整合到明确提交，不要覆盖或 reset。
旧服务器代码/产物 schema 若不同，本入口会报错；不静默修改或猜测其语义。

所有输出目录必须是新路径；/full/ 始终只读。失败写 status.json、failure.md、
failure.csv；不自动重试、缩小分块或清理 offload 文件。OOM 后保留记录，用新目录
和显式 --batch/--rows 重试。缓存容量不足需要显式提高 --cache-mib；缩小 batch
不会减少整份候选激活缓存的 payload。无完整 PTQ、权重导出、QuaRot 或 GPQA。

## P1

- 读取原 /full/capture/cache.pt 的 L3 input_layernorm 和 self_attn.q_a_layernorm。
- 使用原 benchmark_reuse 的两种 assess，3 个候选 (0,1)、(.5,.5)、(1,0)。
- 每个 Proxy × objective：预热一遍候选后，至少 3 次稳态重复，版本顺序交替，
  取每候选平均时间的中位数。端到端计时与分段同步计时分开。
- 原 dtype 保持不变（含被多个 consumer 拼接提升为 FP32 的旧缓存）；两版本一致。
- 保留原严格 NMSE 容差、候选排名检查、实际 QDQ/GEMM 次数、payload 容量检查。
- 每个 mapping 开始时 reset NPU peak；峰值包含 reference 和两个版本，不伪称单版本峰值。
- 每个 mapping 单独落盘，不必等 E2。根 summary.csv 给 speedup 和 W+A/W-only 比例，
  子目录 report.md/results.json 给各分段耗时、调用次数、数值差异和内存。

## E2

仅新捕获 L6/L16/L60/L77，每层两个完整 norm-linear consumer group；不捕获 MTP。
沿用 load_context + auto_offload + sequential：每个 split 一条前缀同时取得四层，
停在 L77 attention 后，不为每个目标层分别运行前缀。前面各层原始 MoE 计算是取得
真实 hidden state 必需的；它们不会被平滑或量化。

捕获前从原 cache 读取 seed、model_path、按序 calibration/validation token IDs。
使用相同本地 JSON/JSONL、tokenizer、seed 和 context，重新 tokenization 的身份序列
必须与旧 cache 完全一致（128 smoke 时为同一序列前缀）。不足/重合/改变就报错。
默认不改变 context=2048；不要为了 128-token smoke 改成另一种文档切分。

成本较低的推荐流程：一次捕获 2048/2048 全预算；随后只用前 128 个校准 tokens、
3 个 scale 做离线 smoke，验证捕获完整性、三种 QDQ finite、E0。只有该 **同一缓存**
的 smoke 通过后才能启动正式搜索。这样不用为 128-token 和 2048-token 各加载一次
巨大模型。若先要单独做 128-token 捕获排查，可显式 --tokens 128，使用另外的输出和
offload 目录；它不能作为正式 2048 搜索的 gate，正式缓存仍须 smoke。

L77 需要更长前缀，不能按 L40 资源消耗承诺可运行。复用此前验证过的 max-memory
JSON（可选 --max-memory），预留 attention workspace、CPU intermediates、原始缓存
与离线 reference 内存。磁盘需独立容纳 offload；本工具不猜容量、不删除旧 offload。
先确认 /dev/shm、CPU/NPU RAM 和磁盘空间。当前仅单进程，用 python，不用 torchrun。

正式 search/evaluate 直接调用旧实验函数：21 alpha + 21 beta、later-tie-wins、
三 Proxy × 两 objective 不变。本轮不把性能优化默认接进正式搜索，以保持旧结果可比性。

汇总要求旧 root 和新 root 都有 capture/cache.pt、search/search.pt、
evaluation/evaluation.json。每批先检查 cache/search 哈希、六臂完整性、参数一致性、
E0 gate 与目标 NMSE finite。跨批比较 ordered token identities、seed、capture source、
CT/Torch 版本、量化/搜索来源哈希与 loss policy。缺失证据或不一致时：明确标注不同
批次，分开绘图，不能当成严格配对的六层趋势。路径差异本身不算源码差异。

输出 raw_layers.csv（各臂 NMSE、No Smooth、绝对/相对改善、alpha/beta、scale 统计、
loss、耗时），gains.csv（匹配 Proxy vs INT8、W+A vs W-only 的两种收益），以及：
matched_proxy.png、activation_aware.png、no_smooth.png；都区分 mapping、目标格式和批次。
results.json 和 report.md 保留比较条件与退化条目。原始数据比拟合线更重要；不因百分比
下降宣称深层失效，不从六个采样层直接推断单调性或设计 Selective FlexSmooth。

## 服务器命令（未在本机执行）

从 work/llm-compressor 根目录，先核对本次回复给出的提交号，再填写 REV。
不能只把服务器当前 HEAD 当成正确版本。以下路径沿用已有服务器布局。

```bash
set -euo pipefail
REV=REPLACE_WITH_DELIVERED_COMMIT
EXPECTED_COMMIT=$(git rev-parse "$REV^{commit}")
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
git status --short
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
python -m pytest experiments/flexsmooth_proxy/next_stage/test_next.py \
  experiments/flexsmooth_proxy/test_benchmark_reuse.py -q -p no:cacheprovider

OLD=/mnt/share/t50065065/results/flexsmooth/full
NEXT=/mnt/share/t50065065/results/flexsmooth/next
RUN_ID=$(date +%Y%m%d-%H%M%S)
P1="$NEXT/npu_benchmark/$RUN_ID"
E2="$NEXT/layer_extension/$RUN_ID"

# P1：不 capture，不加载模型。完成即有独立报告。
python -m experiments.flexsmooth_proxy.next_stage.run p1 \
  --expected-commit "$EXPECTED_COMMIT" --device npu:0 \
  --cache "$OLD/capture/cache.pt" --batch 128 --rows 256 --cache-mib 64 \
  --output "$P1"

# E2：只捕获四个新层。model-path 必须与旧 cache 中的路径一致。
python -m experiments.flexsmooth_proxy.next_stage.run e2-capture \
  --expected-commit "$EXPECTED_COMMIT" --device npu:0 \
  --old-cache "$OLD/capture/cache.pt" \
  --model-path /mnt/weight/GLM-5.2 \
  --data /mnt/share/datasets/qwen3_cot_w4a4.json \
  --tokens 2048 --context 2048 --seed 42 \
  --offload-dir "$NEXT/layer_extension/$RUN_ID-offload" --output "$E2"

# 小规模离线检查，原始缓存不改变。
python -m experiments.flexsmooth_proxy.next_stage.run e2-smoke \
  --expected-commit "$EXPECTED_COMMIT" --device npu:0 \
  --cache "$E2/capture/cache.pt" --output "$E2/smoke"

# gate 同时校验缓存 SHA、Git 版本和 runtime，防止误用其他 smoke。
python -m experiments.flexsmooth_proxy.next_stage.run e2-search \
  --expected-commit "$EXPECTED_COMMIT" --device npu:0 \
  --cache "$E2/capture/cache.pt" --smoke "$E2/smoke" --output "$E2/search"
python -m experiments.flexsmooth_proxy.next_stage.run e2-evaluate \
  --expected-commit "$EXPECTED_COMMIT" --device npu:0 \
  --cache "$E2/capture/cache.pt" --search "$E2/search/search.pt" \
  --output "$E2/evaluation"

# 汇总可离线 CPU 执行；matplotlib 仅用于静态 PNG。
python -m experiments.flexsmooth_proxy.next_stage.run e2-analyze \
  --expected-commit "$EXPECTED_COMMIT" --old "$OLD" --new "$E2" \
  --output "$E2/analysis"
```

如 P1 OOM，用新 RUN_ID 和显式较小 batch/rows 重跑两版本，不复用半途输出；若是
activation cache 上限报错，先检查 payload 与 NPU 可用空间再选择更高上限。不同分块
可能改变 GEMM 细节，新分块的 Baseline/Optimized 必须重新共同测量。

E3 已按用户要求暂缓：不补捕获 Block 激活，不运行 Attention/Decoder 输出比较。
当前只能准备并运行 P1/E2，不能回答真实 NPU 加速、六层泛化及 Selective 策略的结论，
直到服务器的真实结果返回。
