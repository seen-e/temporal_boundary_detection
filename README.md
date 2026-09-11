# temporal_boundary_detection

基于机器人轨迹参数进行时间边界检测、夹爪相位切分、关键帧提取和 VLM 二次过滤的处理框架。当前版本主要面向 ABC-130K / LeRobot v3 数据集，输入为统一的 gripper trajectory index JSON，输出为传统算法关键帧、VLM 三阶段审核结果和可视化图像。

## 1. 框架目标

本框架处理的是一条 robot episode 中随时间变化的左右夹爪状态轨迹。核心目标是从连续轨迹中提取可用于视频切分、动作阶段分析和后续人工/VLM 审核的关键时间点。

融合后的主流程是：

1. 从索引 JSON 定位 episode 对应的 parquet 行范围和左右夹爪字段。
2. 读取左右夹爪轨迹。
3. 对轨迹做预处理和平滑。
4. 提取候选关键帧：局部极大值、局部极小值、平台期左端点、平台期右端点。
5. 使用传统规则过滤噪声关键帧。
6. 对保留的极值点做左右山脚点检测。
7. 使用山脚信息做第二轮传统过滤。
8. 将传统算法结果输入三阶段 VLM 审核。
9. 输出最终清洗后的关键帧结果和对比可视化。

## 2. 目录结构

```text
/mnt/workspace/temporal_boundary_detection
├── config.yaml
├── README.md
├── keyframe_pipeline
│   ├── run_pipeline.py
│   ├── index_io.py
│   ├── traditional
│   │   ├── extractor.py
│   │   ├── filters
│   │   │   ├── gripper_keypoint_filter.py
│   │   │   ├── extrema_base_detector.py
│   │   │   └── post_base_filter.py
│   │   └── phase_segment
│   │       └── gripper_phase_segment
│   │           ├── segmenter.py
│   │           ├── preprocess.py
│   │           ├── trend.py
│   │           ├── plateau.py
│   │           ├── velocity.py
│   │           ├── dual.py
│   │           └── config.py
│   └── vlm_review
│       ├── runner
│       ├── slicing
│       ├── prompts
│       ├── vlm
│       ├── refinement
│       ├── validation
│       └── visualization
```

各部分作用：

- `config.yaml`：统一配置文件，所有路径、传统算法参数、VLM 参数、运行范围都在这里设置。
- `keyframe_pipeline/run_pipeline.py`：融合入口，负责串起传统算法和 VLM 三阶段过滤。
- `keyframe_pipeline/index_io.py`：读取 gripper trajectory index JSON，并根据索引读取 parquet 中对应 episode 的轨迹。
- `keyframe_pipeline/traditional/extractor.py`：传统关键帧提取主逻辑。
- `keyframe_pipeline/traditional/filters`：传统候选点去噪、极值点左右山脚检测、山脚检测后二次传统过滤。
- `keyframe_pipeline/traditional/phase_segment/gripper_phase_segment`：夹爪轨迹预处理、平滑、极值检测、平台期检测和双夹爪相位切分模块。
- `keyframe_pipeline/vlm_review`：三阶段 VLM 审核框架，包含切片、提示词、请求构造、结果解析、后处理和可视化。

## 3. 输入数据

### 3.1 数据集

默认数据集路径：

```text
/mnt/data/yuluo/data/abc_130k_v3_train
```

该目录是 LeRobot v3 格式，通常包含：

```text
abc_130k_v3_train
├── meta
│   ├── info.json
│   ├── tasks.parquet
│   └── episodes
├── data
│   └── chunk-xxx/file-xxx.parquet
└── videos
    ├── observation.images.top
    ├── observation.images.left_wrist
    └── observation.images.right_wrist
```

当前 ABC-130K v3 的夹爪字段来自 `meta/info.json` 中的 `observation.state.names`：

- 左夹爪：`observation.state[6]`
- 右夹爪：`observation.state[13]`

### 3.2 索引 JSON

融合 pipeline 的入口不是直接扫描全量 parquet，而是读取已经生成好的索引：

```text
/mnt/data/chachaxu/save/abc_130k_v3/gripper_trajectory_index.json
```

索引中每个 episode 至少需要包含：

```json
{
  "dataset_root": "/mnt/data/yuluo/data/abc_130k_v3_train",
  "episodes": {
    "5": {
      "episode_id": "abc_130k_v3_train__episode_000005",
      "episode_index": 5,
      "fps": 30.0,
      "frame_count": 3000,
      "left_gripper": {
        "path": "data/chunk-000/file-000.parquet",
        "row_start": 10000,
        "row_end": 13000,
        "column": "observation.state",
        "indices": [6]
      },
      "right_gripper": {
        "path": "data/chunk-000/file-000.parquet",
        "row_start": 10000,
        "row_end": 13000,
        "column": "observation.state",
        "indices": [13]
      },
      "video_segments": {
        "observation.images.top": {
          "path": "videos/observation.images.top/chunk-000/file-000.mp4",
          "start_frame": 10000,
          "end_frame": 13000,
          "fps": 30.0
        }
      }
    }
  }
}
```

注意：

- `row_end` 是 exclusive。
- `path` 是相对 `dataset_root` 的路径。
- pipeline 只把轨迹索引、关键帧和审核结果写入输出目录，不复制完整轨迹数值。

## 4. 配置文件

默认配置：

```text
/mnt/workspace/temporal_boundary_detection/config.yaml
```

### 4.1 路径配置

```yaml
paths:
  index_json: /mnt/data/chachaxu/save/abc_130k_v3/gripper_trajectory_index.json
  dataset_root: /mnt/data/yuluo/data/abc_130k_v3_train
  phase_module_root: /mnt/workspace/temporal_boundary_detection
  output_root: /mnt/workspace/temporal_boundary_detection/outputs/abc_130k_v3_pipeline
  keyframe_root: /mnt/workspace/temporal_boundary_detection/outputs/abc_130k_v3_pipeline/traditional_keyframes
  vlm_output_root: /mnt/workspace/temporal_boundary_detection/outputs/abc_130k_v3_pipeline/vlm_review
```

含义：

- `index_json`：统一索引文件。
- `dataset_root`：真实 LeRobot v3 数据集根目录。
- `phase_module_root`：当前工程根目录，用于导入 `keyframe_pipeline` 下的传统 phase segmentation 模块。
- `output_root`：pipeline 总输出目录。
- `keyframe_root`：传统算法关键帧 JSON 输出目录。
- `vlm_output_root`：VLM 审核和可视化输出目录。

### 4.2 总流程开关

```yaml
pipeline:
  stages:
    - traditional
    - vlm
    - visualize
```

可选值：

- `traditional`：只跑传统关键帧提取和传统过滤。
- `vlm`：基于传统关键帧结果跑 VLM 审核。
- `visualize`：绘制传统结果、Pass2、Pass3 的 episode 级对比图。
- `all`：等价于 `traditional + vlm + visualize`。

示例：

```yaml
pipeline:
  stages:
    - traditional
```

表示只生成传统关键帧，不调用 VLM。

### 4.3 VLM 内部阶段开关

```yaml
vlm:
  stages:
    - pass1
    - pass2
    - pass3
```

支持三种前缀流程：

```yaml
vlm:
  stages: [pass1]
```

只跑 Pass 1 分段审核。此模式下不会做 Pass 2 删除/合并，也不会做 Pass 3 补点；最终事件默认保留。

```yaml
vlm:
  stages: [pass1, pass2]
```

跑 Pass 1 和 Pass 2。Pass 2 会根据 Pass 1 的 segment 结构对关键帧做 KEEP / REMOVE / MERGE / RELABEL / RELOCALIZE 等审核，不跑 Pass 3。

```yaml
vlm:
  stages: [pass1, pass2, pass3]
```

完整三阶段流程。Pass 3 会在 Pass 2 清洗后的结果中检查是否缺少关键帧，并尝试补点。

不支持跳阶段，例如 `[pass2]` 或 `[pass1, pass3]` 都是不完整流程。

### 4.4 运行范围

```yaml
run:
  dry_run: false
  arms:
    - left
    - right
  workers: 32
  episode_index: null
  episode_start: null
  episode_end: null
  limit_episodes: null
  limit_slices: null
```

含义：

- `dry_run`：不调用真实 VLM，使用占位响应验证流程。
- `arms`：处理左手、右手或两者。
- `workers`：并发数。传统阶段按 data parquet 文件并发；VLM 阶段按 slice 并发。
- `episode_index`：只处理单个 episode。
- `episode_start` / `episode_end`：处理一个 episode index 区间，`episode_end` 为 exclusive。
- `limit_episodes`：最多处理多少个 episode。
- `limit_slices`：每个 arm 最多处理多少个 VLM slice，适合调试。

### 4.5 传统算法参数

传统算法相关配置包含四块：

- `gripper_phase_segmentation`
- `keyframe_filter`
- `extrema_base_detection`
- `post_base_filter`

`gripper_phase_segmentation` 控制轨迹预处理、平滑、极值检测和平台检测。例如：

```yaml
gripper_phase_segmentation:
  smoothing:
    method: savgol
    window_sec: 0.30
    polyorder: 2
  extrema:
    enabled: true
    prominence_mode: auto
    auto_prominence_ratio: 0.03
    min_distance_sec: 0.05
  plateau:
    enabled: true
    window_sec: 0.25
    range_ratio: 0.03
    normalized_slope_threshold: 0.03
    min_duration_sec: 0.50
    suppress_same_direction_internal_plateaus: true
```

`keyframe_filter` 控制候选关键帧的传统去噪。例如短平台、伪平台、近邻事件、弱极值等：

```yaml
keyframe_filter:
  enabled: true
  max_iterations: 4
  event_cluster_sec: 0.25
  min_plateau_duration_sec: 0.35
  min_extrema_prominence_ratio: 0.02
```

`extrema_base_detection` 控制极值点左右山脚检测：

```yaml
extrema_base_detection:
  enabled: true
  method: monotonic_walk
  min_extrema_prominence_ratio: 0.02
  max_base_duration_sec: 12.0
  guard_neighbor_extrema: true
```

当前山脚检测逻辑：

- 对极大值：从极值点分别向左右走，只要 smoothed value 继续下降就继续；遇到不再下降或边界就停止。
- 对极小值：等价于先把曲线取负，再使用极大值同样的规则，也就是向左右寻找上升停止点。
- 输出 `left_base` 和 `right_base`，包含山脚时间、frame、value、宽度和状态。

`post_base_filter` 控制山脚检测后的二次过滤：

```yaml
post_base_filter:
  enabled: true
  remove_failed_bases: true
  enable_adjacent_extrema_filter: true
  adjacent_extrema_sec: 0.30
```

## 5. 数据处理流程

### 5.1 传统关键帧提取

入口：

```text
keyframe_pipeline/traditional/extractor.py
```

处理单个 episode 时执行：

1. 根据 index JSON 找到 episode 对应 parquet 文件。
2. 根据 `row_start:row_end` 切出该 episode。
3. 从 `observation.state` 中取出 `left_gripper.indices` 和 `right_gripper.indices`。
4. 分别对左右夹爪轨迹调用 `segment_gripper_trajectory(...)`。
5. `segment_gripper_trajectory` 内部执行输入校验、缺失值插值、outlier 处理、Savitzky-Golay 平滑、velocity 计算、局部极值检测、平台期检测和同趋势内部平台抑制。
6. 将极值点转成 `local_maximum` / `local_minimum`。
7. 将平台期左右边界转成 `plateau_left_endpoint` / `plateau_right_endpoint`。
8. 按时间排序候选点。
9. 执行 `filter_keyframes` 传统去噪。
10. 执行 `annotate_extrema_bases` 山脚检测。
11. 执行 `filter_after_extrema_bases` 二次过滤。
12. 保存每个 episode 的传统关键帧 JSON。

### 5.2 传统关键帧 JSON 输出

默认输出位置：

```text
/mnt/workspace/temporal_boundary_detection/outputs/abc_130k_v3_pipeline/traditional_keyframes/chunk-xxx/episode_xxxxxx.json
```

核心结构：

```json
{
  "status": "complete",
  "schema_version": "2.0",
  "episode_id": "abc_130k_v3_train__episode_000005",
  "episode_index": 5,
  "fps": 30.0,
  "frame_count": 3000,
  "data_file": "/mnt/data/yuluo/data/abc_130k_v3_train/data/chunk-000/file-000.parquet",
  "left_gripper": {
    "keyframes": [],
    "cleaned_keyframes": [],
    "filtering": {},
    "extrema_base_detection": {},
    "post_base_filtering": {}
  },
  "right_gripper": {
    "keyframes": [],
    "cleaned_keyframes": [],
    "filtering": {},
    "extrema_base_detection": {},
    "post_base_filtering": {}
  },
  "video_segments": {}
}
```

其中：

- `keyframes`：传统算法初始候选点。
- `cleaned_keyframes`：传统过滤 + 山脚检测 + 二次过滤后的候选点，也是 VLM 的输入点。
- `filtering`：第一轮传统过滤统计和删除原因。
- `extrema_base_detection`：极值点山脚检测统计。
- `post_base_filtering`：山脚后处理统计。

### 5.3 VLM 三阶段审核

入口：

```text
keyframe_pipeline/vlm_review/runner/review_runner.py
```

VLM 输入来自传统关键帧 JSON 中每个 arm 的 `cleaned_keyframes`。

#### Pass 1：轨迹分段理解

Pass 1 输入：

- 全局轨迹图。
- 局部 target region 图。
- 当前 slice 的 metadata。
- target 区域和 overlap/context 区域说明。

Pass 1 目标：

- 不直接删除关键帧。
- 只把局部轨迹划分成若干稳定状态/动作趋势 segment。
- 输出 frozen segments，供 Pass 2 使用。

#### Pass 2：关键帧审核

Pass 2 输入：

- Pass 1 输出的 frozen segments。
- 局部候选关键帧图。
- 每个 keyframe 的 id、类型、时间和值。
- target + overlap/context 区域内的关键帧结构。

Pass 2 目标：

- 判断 target 内关键帧是否保留。
- 支持 `KEEP`、`REMOVE`、`MERGE`、`RELABEL`、`RELOCALIZE`。
- overlap/context 中的点用于判断上下文，但最终写回只使用 target owned 的审核结果。

#### Pass 3：Missing Keyframe Completion

Pass 3 输入：

- Pass 2 后的 cleaned keyframes。
- frozen segment boundary。
- 当前 slice 中未被已有关键帧覆盖的内部边界。

Pass 3 目标：

- 不负责删除点。
- 只检查是否缺少关键帧。
- 对明显缺失的位置提出新增点。

### 5.4 VLM 输出目录

默认输出位置：

```text
/mnt/workspace/temporal_boundary_detection/outputs/abc_130k_v3_pipeline/vlm_review
```

单个 episode 结构：

```text
vlm_review
└── abc_130k_v3_train__episode_000005
    ├── left
    │   ├── slice_000.review.json
    │   ├── slice_000
    │   │   ├── pass1
    │   │   │   ├── global.png
    │   │   │   ├── local.png
    │   │   │   ├── request.json
    │   │   │   ├── response.json
    │   │   │   └── review.json
    │   │   └── pass2
    │   │       ├── local_candidates.png
    │   │       ├── request.json
    │   │       ├── response.json
    │   │       └── review.json
    │   └── final_review.json
    ├── right
    │   └── ...
    ├── summary.json
    └── three_stage_before_after_comparison.png
```

重要文件：

- `slice_xxx/pass1/global.png`：Pass 1 全局轨迹输入图。
- `slice_xxx/pass1/local.png`：Pass 1 局部轨迹输入图。
- `slice_xxx/pass2/local_candidates.png`：Pass 2 候选关键帧输入图。
- `slice_xxx.review.json`：该 slice 的综合审核结果。
- `left/final_review.json` / `right/final_review.json`：单臂最终审核结果。
- `three_stage_before_after_comparison.png`：episode 级三阶段前后对比图。

## 6. 常用命令

### 6.1 跑完整流程

```bash
cd /mnt/workspace/temporal_boundary_detection
python3 -m keyframe_pipeline.run_pipeline --config config.yaml
```

完整流程由 `config.yaml` 中这两处共同决定：

```yaml
pipeline:
  stages: [traditional, vlm, visualize]

vlm:
  stages: [pass1, pass2, pass3]
```

### 6.2 只跑传统算法

```bash
python3 -m keyframe_pipeline.run_pipeline \
  --config config.yaml \
  --stage traditional
```

### 6.3 只跑 VLM

前提：`paths.keyframe_root` 下已经有传统关键帧 JSON。

```bash
python3 -m keyframe_pipeline.run_pipeline \
  --config config.yaml \
  --stage vlm
```

### 6.4 只处理单个 episode

```bash
python3 -m keyframe_pipeline.run_pipeline \
  --config config.yaml \
  --episode-index 5 \
  --force
```

### 6.5 处理一个 episode 区间

```bash
python3 -m keyframe_pipeline.run_pipeline \
  --config config.yaml \
  --episode-start 0 \
  --episode-end 100 \
  --workers 32
```

这里 `episode_end` 是 exclusive，即处理 `[0, 100)`。

### 6.6 调试时限制 slice 数量

```bash
python3 -m keyframe_pipeline.run_pipeline \
  --config config.yaml \
  --episode-index 5 \
  --limit-slices 1 \
  --workers 1 \
  --dry-run \
  --force
```

### 6.7 临时指定 VLM 阶段

只跑 Pass 1：

```bash
python3 -m keyframe_pipeline.run_pipeline \
  --config config.yaml \
  --stage traditional vlm \
  --episode-index 5 \
  --vlm-stage pass1 \
  --force
```

跑 Pass 1 + Pass 2：

```bash
python3 -m keyframe_pipeline.run_pipeline \
  --config config.yaml \
  --stage traditional vlm \
  --episode-index 5 \
  --vlm-stage pass1 pass2 \
  --force
```

跑完整三阶段：

```bash
python3 -m keyframe_pipeline.run_pipeline \
  --config config.yaml \
  --stage traditional vlm visualize \
  --episode-index 5 \
  --vlm-stage pass1 pass2 pass3 \
  --force
```

### 6.8 Dry Run

```bash
python3 -m keyframe_pipeline.run_pipeline \
  --config config.yaml \
  --episode-index 5 \
  --dry-run \
  --force
```

`dry_run: true` 时不调用真实 VLM，只生成占位审核结果。适合检查索引读取、传统关键帧生成、slice 切分、输入图像、message 结构和输出目录。

## 7. 输出汇总文件

完整运行后会生成：

```text
outputs/abc_130k_v3_pipeline
├── pipeline_summary.json
├── traditional_summary.json
├── vlm_summary.json
├── visualization_summary.json
├── traditional_keyframes
└── vlm_review
```

说明：

- `pipeline_summary.json`：总流程摘要。
- `traditional_summary.json`：传统关键帧阶段处理数量、成功失败数、速度。
- `vlm_summary.json`：VLM 审核阶段摘要。
- `visualization_summary.json`：可视化生成结果。

## 8. Resume、Overwrite 和 Force

传统阶段：

- `traditional.overwrite: false` 时，会扫描 `keyframe_root`，跳过已经存在且 `status == "complete"` 的 episode。
- `traditional.overwrite: true` 时，会覆盖已有传统关键帧 JSON。

VLM 阶段：

- `vlm_trajectory_review.resume: true` 时，会复用已完成的 slice review。
- `vlm_trajectory_review.force: true` 时，会重新跑 slice，覆盖旧结果。

命令行 `--force` 会同时设置：

- `traditional.overwrite = true`
- `vlm_trajectory_review.force = true`

## 9. 可视化说明

主要可视化有三类：

1. VLM 输入图：Pass 1 global/local、Pass 2 local candidates、Pass 3 missing boundary 检查图。
2. slice 级结果图：Pass 1 segment 可视化、Pass 2 过滤前后对比。
3. episode 级结果图：`three_stage_before_after_comparison.png`。

当前默认配置为：

```yaml
vlm_trajectory_review:
  show_raw_trajectory: false
  show_smoothed_trajectory: true
```

也就是 VLM 输入和最终可视化默认只画 smoothed 曲线，保证检测关键帧和模型看到的曲线一致。

## 10. 常见问题

### 10.1 为什么只画 smoothed？

当前关键帧检测基于平滑后的夹爪轨迹，因此 VLM 输入图也只画平滑曲线。这样可以避免 raw 曲线遮挡或干扰模型判断。

### 10.2 为什么 VLM stages 只能是前缀？

Pass 2 依赖 Pass 1 的 frozen segments，Pass 3 依赖 Pass 2 的清洗结果。因此合法流程只能是：

- `[pass1]`
- `[pass1, pass2]`
- `[pass1, pass2, pass3]`

### 10.3 Pass 1 会删除关键帧吗？

不会。Pass 1 只做轨迹局部分段理解，输出 segment 结构。

### 10.4 Pass 2 会删除关键帧吗？

会。Pass 2 是主要的 VLM 关键帧审核阶段，会输出 KEEP / REMOVE / MERGE / RELABEL / RELOCALIZE。

### 10.5 Pass 3 会删除关键帧吗？

不会。Pass 3 只做 missing keyframe completion，用于补充缺失点。

### 10.6 左右夹爪怎么处理？

传统阶段和 VLM 阶段都会分别处理左夹爪和右夹爪。输出中分别保存到：

```text
left/final_review.json
right/final_review.json
```

当前融合 pipeline 没有把左右夹爪的结果做并集融合成单一时间线；它保留左右手各自的关键帧结果，便于后续根据任务需要再做双手合并。

### 10.7 matplotlib 报中文字体 warning 怎么办？

如果看到类似：

```text
UserWarning: Glyph xxxx missing from font(s) DejaVu Sans
```

这只是可视化字体缺字警告，不影响 JSON 输出和核心算法。需要消除时可以在环境中安装中文字体，或把图中的中文标签换成英文。

## 11. 推荐调试顺序

第一次跑新数据或新配置时建议：

1. 先 dry-run 单个 episode：

```bash
python3 -m keyframe_pipeline.run_pipeline \
  --config config.yaml \
  --episode-index 5 \
  --limit-slices 1 \
  --workers 1 \
  --dry-run \
  --force
```

2. 检查传统关键帧 JSON：

```text
outputs/abc_130k_v3_pipeline/traditional_keyframes/chunk-000/episode_000005.json
```

3. 检查 VLM 输入图：

```text
outputs/abc_130k_v3_pipeline/vlm_review/abc_130k_v3_train__episode_000005/left/slice_000/pass1/global.png
outputs/abc_130k_v3_pipeline/vlm_review/abc_130k_v3_train__episode_000005/left/slice_000/pass1/local.png
outputs/abc_130k_v3_pipeline/vlm_review/abc_130k_v3_train__episode_000005/left/slice_000/pass2/local_candidates.png
```

4. 再跑真实 VLM 单个 episode：

```bash
python3 -m keyframe_pipeline.run_pipeline \
  --config config.yaml \
  --episode-index 5 \
  --limit-slices 1 \
  --workers 1 \
  --force
```

5. 最后批量运行：

```bash
python3 -m keyframe_pipeline.run_pipeline \
  --config config.yaml \
  --episode-start 0 \
  --episode-end 1000 \
  --workers 32
```

## 12. 关键入口速查

- 总入口：

```bash
python3 -m keyframe_pipeline.run_pipeline --config config.yaml
```

- 传统关键帧提取：

```text
keyframe_pipeline/traditional/extractor.py
```

- 夹爪轨迹相位切分：

```text
keyframe_pipeline/traditional/phase_segment/gripper_phase_segment/segmenter.py
```

- VLM 三阶段审核：

```text
keyframe_pipeline/vlm_review/runner/review_runner.py
```

- VLM 提示词：

```text
keyframe_pipeline/vlm_review/prompts/three_stage_prompts.py
keyframe_pipeline/vlm_review/prompts/three_stage_image_instructions.py
```

- VLM 请求构造：

```text
keyframe_pipeline/vlm_review/vlm/request_builder.py
```

- VLM 输出解析：

```text
keyframe_pipeline/vlm_review/vlm/parser.py
keyframe_pipeline/vlm_review/validation/validator.py
```

- 最终对比可视化：

```text
keyframe_pipeline/vlm_review/visualization/three_stage_comparison.py
```
