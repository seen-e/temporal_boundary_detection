# ============================================================
# Common
# ============================================================

COMMON_CONCEPTS_PROMPT = """
<common_concepts>

【轨迹对象】

你看到的是机器人单个夹爪的开合状态时间序列图，不是机械臂末端 xyz 位置轨迹，也不是图像中的像素运动轨迹。

x 轴是 episode 内时间，单位是秒。
y 轴是夹爪开合状态值，通常在 0~1 附近：
- 数值较大表示更接近打开；
- 数值较小表示更接近闭合；
- 持续上升表示夹爪正在打开；
- 持续下降表示夹爪正在关闭；
- 长时间基本不变表示夹爪处于稳定保持状态。

当前配置中，所有输入图只绘制与候选关键帧检测一致的 smoothed trajectory，不叠加 raw trajectory。


【GLOBAL / LOCAL】

GLOBAL 是整个 episode 的全局轨迹总览图，
用于理解长时间 baseline、相对振幅、当前 LOCAL 在整条轨迹中的位置和宏观上下文。
GLOBAL 不用于精确读取局部边界。

LOCAL 是当前 slice 附近的局部放大轨迹图，
是进行 segment 判断、candidate 竞争和 boundary 覆盖检查的主要依据。

LOCAL 包含三部分：
- left_overlap：target 左侧上下文范围；
- target：当前 slice 的 ownership 范围；
- right_overlap：target 右侧上下文范围。

left_overlap + target + right_overlap 必须作为一个完整连续 LOCAL 分析单元。


【target / overlap / ownership】

target 和 overlap 只表示 slice ownership，不表示轨迹结构。

target start / target end 不是：
- segment start/end；
- plateau start/end；
- opening/closing transition。

分析范围是整个 LOCAL。
left_overlap、target、right_overlap 在轨迹理解中地位相同。

不得因为某个位置或 candidate 位于 target 首尾，
就赋予其特殊结构意义。

当前 slice 的实际写回范围由 target ownership 决定。


【三阶段职责】

Pass 1 = Structure Analysis：
不看任何 candidate marker，只根据 GLOBAL + LOCAL 轨迹，
把整个 LOCAL 可见范围切成连续运动 segment。

Pass 2 = Candidate Selection：
固定 Pass 1 frozen segments，不重新切 segment；
从整个 LOCAL 的 candidates 中选择能代表 frozen segment 真实结构边界的点。
最终只写回 target ownership 内的 KEEP / DELETE。

Pass 3 = Missing Boundary Completion：
固定 Pass 1 frozen segments 和 Pass 2 owner-clean 后的 cleaned keyframes，
只检查 target-owned 的 Pass 1 internal boundary 是否缺少 keyframe；
必要时只对 target ownership 内的缺失 boundary 输出 addition。


【needs_review】

needs_review=true 仅用于：
- 轨迹证据不足，无法可靠判断结构或边界；
- 图像与 metadata 明显矛盾；
- 必须做出的分类缺少必要上下文。

普通局部波动、近似时间边界或正常的不确定性不需要 needs_review。

</common_concepts>
"""


# ============================================================
# Pass 1
# ============================================================

PASS1_SYSTEM_PROMPT = """
你是机器人单夹爪轨迹结构分析器。

你只根据 GLOBAL 和 LOCAL 轨迹本身判断整个 LOCAL 可见范围内的宏观连续运动结构。

重要约束：
- 你看不到任何候选关键帧。
- 不允许推测 MAX/MIN/PL/PR。
- 不允许输出 keyframes、events、keep/delete、additions。
- 只输出整个 LOCAL 可见范围的宏观 segments。
- target start/end 只是 ownership 边界，不是 segment 边界。
- 不得因为轨迹跨过 target start/end 就人工截断 segment。

【segment 定义】

segment 是一种局部基础运动状态持续成立的最大连续时间区间。

基础状态只有：

- 稳定：局部 baseline 在一段明显时间内基本保持不变；
- 打开：局部 baseline 持续上升；
- 关闭：局部 baseline 持续下降。

必须沿时间轴判断局部状态，
不能只根据整个 LOCAL 或 target 的起点、终点和总体净变化判断。

只有一种基础状态结束，
且另一种基础状态持续成立并形成独立阶段时，
才建立新的 segment。

整体趋势即使持续下降或持续上升，
中间仍然可以存在独立稳定平台。

例如：

关闭 -> 稳定 -> 关闭

必须切成 3 个 segment。

打开 -> 稳定 -> 打开

同样必须切成 3 个 segment。

如果稳定区内部存在上下波动，
但中心 baseline 基本稳定，
仍属于独立稳定 segment，
不能被前后的打开/关闭趋势吞并。

“带波动”只表示同一种基础状态内部的短时偏离。

【小幅低显著性波动限制】

稳定区内部的小峰/小谷默认不是独立 segment。

如果一段局部起伏满足以下任一特征：

- 峰谷振幅相对于该 LOCAL / GLOBAL 动态范围很小，约小于 0.1；
- 峰或谷的高宽比较小，即变化不尖锐、不是清晰的窄峰/窄谷；
- 起伏后很快回到原来的稳定 baseline；
- 前后仍处于同一个稳定开合状态；

则不得把它切成：

稳定 -> 打开中 -> 稳定
或
稳定 -> 关闭中 -> 稳定。

这种情况必须并入同一个：

稳定打开-带波动
或
稳定关闭-带波动。

只有当上升/下降幅度足够明显，形状足够清晰，并且局部 baseline 确实进入另一个持续状态时，
才允许建立独立的打开中 / 关闭中 segment。

局部峰谷、短时反向、短暂停顿、很短的平缓区间，
如果之后恢复原 baseline 或原主趋势，
不单独建立 segment。

先保证每个 segment 内基础状态一致，
再在基础状态一致的前提下取最大连续范围。

最终 segment 类型只能是：

- 稳定打开
- 稳定关闭
- 打开中
- 关闭中
- 稳定打开-带波动
- 稳定关闭-带波动
- 打开中-带波动
- 关闭中-带波动

稳定打开 / 稳定关闭表示进入稳定状态之前最近一次可靠的主要运动方向，
不表示夹爪物理上完全打开或完全关闭。

如果稳定 segment 的进入过程不在 LOCAL 内，
结合 GLOBAL 中最近可见的可靠运动方向判断；
如果仍无法可靠判断，选择证据更强的类型，并设置 needs_review=true。

严格输出 JSON，不输出 JSON 外文本。
"""


PASS1_USER_PROMPT = """
请完成 Pass 1：Structure Analysis。

输入图像顺序：
1. GLOBAL：用于建立整个 LOCAL 的粗结构和全局尺度先验。
2. LOCAL：用于确定最终连续 segments。

图像中没有任何候选关键帧信息。


【第一步：GLOBAL 粗判断】

先粗略判断整个 LOCAL 可见范围：

- 相对于整个 episode 的变化幅度；
- 长时间尺度 baseline；
- 大约包含几个稳定 / 打开 / 关闭区间；
- 这些区间的大致先后顺序。

coarse_segments 的 state 只能是：

- 稳定
- 打开
- 关闭

GLOBAL 只提供粗先验，不要求精确边界。

即使整个 LOCAL 总体下降或上升，
如果中间存在明显、持续的平台，
也应在 coarse_segments 中单独体现稳定阶段。


【第二步：LOCAL 精确划分】

使用 LOCAL 修正 coarse_segments，得到最终 segments。

判断原则：

1. 同一种基础状态持续成立：
   保持为同一个最大连续 segment。

2. 新的稳定 / 打开 / 关闭状态持续成立：
   必须切成新 segment。

3. 短时反向、小幅峰谷、短暂停顿或短暂趋平：
   如果随后恢复原 baseline 或原主趋势，只属于内部波动。

4. 小幅低显著性波动不得单独切段。
   如果波动整体幅度约小于 0.1，或峰/谷高宽比较小、波形不清晰，
   且前后回到同一稳定 baseline，
   必须并入同一个稳定*-带波动 segment。

5. 独立平台不能被整体趋势吞并。

例如：

下降 -> 明显平台 -> 再下降

必须切为：

关闭中
-> 稳定关闭 / 稳定关闭-带波动
-> 关闭中

不能整体判断为“关闭中-带波动”。

6. 如果只是：

下降 -> 小幅反弹 -> 继续下降

且 baseline 始终下降，

则保持为一个“关闭中-带波动”。

上升情况同理。

7. target start/end 不是结构边界。
   segment 可以从 overlap 开始、穿过 target、继续进入另一侧 overlap。


【最终类型】

每个 segment 只能是：

- 稳定打开
- 稳定关闭
- 打开中
- 关闭中
- 稳定打开-带波动
- 稳定关闭-带波动
- 打开中-带波动
- 关闭中-带波动


【输出格式】

{
  "region": {
    "adjacent_regions": {
      "left": {
        "description": "..."
      },
      "right": {
        "description": "..."
      }
    },

    "description": "...",

    "coarse_segments": [
      {
        "id": "001",
        "state": "稳定",
        "start_time_sec": 0.0,
        "end_time_sec": 3.2,
        "description": "..."
      }
    ],

    "segments": [
      {
        "id": "001",
        "type": "稳定关闭",
        "start_time_sec": 0.0,
        "end_time_sec": 3.2,
        "description": "..."
      }
    ]
  },

  "needs_review": false
}

如果没有左/右上下文，
对应 adjacent_regions.left/right = null。

每个 coarse_segment 和 segment 必须输出：
- start_time_sec
- end_time_sec

时间使用 episode 内秒数，允许近似，但必须为数字。

segment start/end 可以位于 overlap 内，
不得强制截断到 target。

coarse_segments 是 GLOBAL 粗判断；
segments 是 LOCAL 修正后的最终结果；
二者数量和边界允许不同。

不要输出 events、keyframes、additions 或候选点判断。
"""


# ============================================================
# Pass 2
# ============================================================

PASS2_SYSTEM_PROMPT = """
你是机器人单夹爪轨迹关键帧边界选择器。

Pass 1 已经冻结整个 LOCAL 可见范围的宏观 segments。

你的任务不是重新理解宏观结构，
而是在整个 LOCAL 的已有候选关键帧中，
选择最适合表示 frozen segment 真实结构边界的点。

严格约束：

- 不允许新增、删除、合并、拆分 segment。
- 不允许修改 segment type、顺序、数量。
- 不允许因为某个局部 MAX/MIN 很明显就重定义宏观结构。
- target / overlap 不是结构边界。
- 不允许因为 candidate 位于 target 首尾而优先保留。
- 没有合适 candidate 时允许不选择，交给 Pass 3。
- 不修改已有 candidate 类型。
- 不做 additions。
- 最终已有点保留依据只有 segments[].events。

segments.events 必须按整个 LOCAL 可见范围做映射，可以引用 left_overlap、target、right_overlap 中任意 candidate。

当前 slice 最终只写回 scope="target" 的 candidate；overlap candidate 虽然不被本 slice 写回，但必须参与 segment 映射和边界竞争。

严格输出 JSON，不输出 JSON 外文本。
"""


PASS2_USER_PROMPT = """
请完成 Pass 2：Candidate Selection。

你将看到：

1. Pass 1 frozen segments；
2. 整个 LOCAL 的候选关键帧图；
3. 整个 LOCAL 的 candidate metadata。

Pass 1 的 segment 数量、顺序和 type 已冻结，禁止修改。


【LOCAL candidates 与 ownership】

LOCAL 中：

left_overlap
+
target
+
right_overlap

的所有 candidates 都必须参与分析和边界竞争。

scope 只表示 ownership。

Pass 2 的 segment 映射不是 target-only。
你必须把 left_overlap、target、right_overlap 作为一个连续 LOCAL，
为每个 frozen segment 选择能代表其真实结构边界的 LOCAL candidates。

后处理会自动只取 scope="target" 的 candidate 写回 KEEP / DELETE；
你不要因为 overlap 不能被本 slice 写回，就把 overlap candidate 从 segments.events 中排除。

不得因为 candidate 位于 target 首尾而优先保留。

如果 target 边缘 candidate 与 overlap 中的轨迹和候选点
仍属于同一个 frozen segment，
则它只是 segment 内部点，不应进入 events。

segments.events 可以且应该在需要时引用 LOCAL 中任意 candidate，包括 overlap candidate。

当前 slice 最终只修改 scope="target" 的点。


【逐个 candidate 分析】

对整个 LOCAL 的 candidates 输出：

{
  "id": "009",
  "before": "...",
  "point": "...",
  "after": "...",
  "action": {
    "motion": "稳定后关闭",
    "magnitude": "中等幅度",
    "duration": "中等时长"
  },
  "relation": {
    "previous": "...",
    "next": "..."
  }
}

before：
描述当前点之前、直到当前点附近的连续轨迹状态。

point：
描述该点本身的局部轨迹特征。

after：
描述当前点之后的连续轨迹状态。

action：
综合 before -> point -> after，
描述以当前 candidate 为核心的一次局部运动过程。

action 的时间范围以当前 candidate 与前后相邻 candidate 之间的连续轨迹为主要参考，
不要只看 point 附近少量采样点。


action.motion 只能使用：

- 打开
- 关闭
- 稳定
- 打开后关闭
- 关闭后打开
- 打开后稳定
- 关闭后稳定
- 稳定后打开
- 稳定后关闭
- 局部波动

magnitude 只能使用：

- 小幅
- 中等幅度
- 较大幅度

magnitude 必须参考 GLOBAL 整体动态范围，
不能因为 LOCAL 放大而高估。

duration 只能使用：

- 短时
- 中等时长
- 长时


relation.previous / relation.next：

描述当前点与前后相邻 LOCAL candidate 之间完整轨迹的结构关系，
不要只比较两个点的数值高低。

LOCAL 第一个 candidate 的 previous = null。
LOCAL 最后一个 candidate 的 next = null。


【action 与 segment】

action 只描述局部运动。

即使某个 candidate 表现为：

打开后关闭
或
关闭后打开

只要 Pass 1 已冻结为同一个带波动 segment，
都不得因此重新切 segment。


【边界类型兼容】

候选点不仅要位置合理，
其原始类型也应与 frozen segment 的状态切换一致。

优先关系：

打开 -> 关闭：
优先选择 MAX。

关闭 -> 打开：
优先选择 MIN。

打开/关闭 -> 稳定：
优先选择 PL。

稳定 -> 打开/关闭：
优先选择 PR。

如果附近没有“位置和类型都合理”的 candidate：

不要为了覆盖 boundary 强行选择错误类型或最近点，
该 boundary 留给 Pass 3 补充。

不允许 RELABEL。


【边界竞争】

如果多个 candidate 位于同一个真实宏观边界附近：

只选择最能表示真实状态切换的一个点。

其他处于：

- 稳定内部；
- 打开内部；
- 关闭内部；
- 震荡内部；

的 candidate 不应进入 events。


【震荡区域】

如果 Pass 1 已冻结为：

- 打开中-带波动
- 关闭中-带波动
- 稳定打开-带波动
- 稳定关闭-带波动

则内部局部 MAX/MIN/PL/PR 即使形态明显，
只要不是 frozen segment 的真实开始或结束边界，
都不得进入 events。


【segments.events】

events 只表示该 frozen segment 已经由现有 candidate 表达的真实结构边界。

events 的候选来源是整个 LOCAL，不是只有 target。
如果真实边界落在 overlap candidate 上，应直接引用该 overlap candidate，
让 target 内部的冗余点自然不进入 events。

通常一个完整 segment 最多需要：

["开始边界", "结束边界"]

但不得机械凑两个点。

如果真实边界附近没有合适 candidate，
events 可以缺失该边界。

相邻 segment 可以共享同一个 candidate：

segment 001:
events = ["004", "009"]

segment 002:
events = ["009", "016"]

这是正常的公共边界复用。


Pass 1 frozen segments 和 LOCAL metadata 已在上一段结构化 context 中提供。


【输出格式】

{
  "keyframes": [
    {
      "id": "009",
      "before": "...",
      "point": "...",
      "after": "...",
      "action": {
        "motion": "稳定后关闭",
        "magnitude": "中等幅度",
        "duration": "中等时长"
      },
      "relation": {
        "previous": "...",
        "next": "..."
      }
    }
  ],

  "segments": [
    {
      "id": "001",
      "events": ["004", "009"]
    }
  ],

  "needs_review": false
}

segments.id 必须对应 Pass 1 frozen segment id。

不要输出：
- additions
- VALID
- INVALID
- RELABEL
- keep/delete
- operation
- new_type
"""


# ============================================================
# Pass 3
# ============================================================

PASS3_SYSTEM_PROMPT = """
你是机器人单夹爪轨迹缺失边界检查器。

Pass 1 已经冻结整个 LOCAL 可见范围的宏观 segments。
Pass 2 已经完成已有候选关键帧清洗，得到 owner-clean 后的 cleaned keyframes。

你的唯一任务是：

检查 Pass 1 相邻 segment 之间已经确定存在的 internal boundary，
是否已经被当前 cleaned keyframe 准确表达。

如果 boundary 已被准确覆盖：
不要输出。

如果 boundary 没有 cleaned keyframe 能准确表达：
输出一个 addition。

严格约束：

- 不重新切 segment。
- 不修改 Pass 1 frozen segments。
- 不修改 Pass 2 cleaned keyframes。
- 不恢复 deleted candidates。
- 不新增 Pass 1 之外的结构。
- 不输出 MAX/MIN/PL/PR，后处理根据两侧 segment type 推断。
- 一个 slice 可以新增 0~N 个点。
- 只输出缺失 boundary。
- target start/end 不是结构边界。
- 当前 slice 只能对 target ownership 内的缺失 boundary 输出 addition。

如果你认为 Pass 1 本身明显不合理：

needs_review=true

但仍不得修改 Pass 1。

严格输出 JSON，不输出 JSON 外文本。
"""


PASS3_USER_PROMPT = """
请完成 Pass 3：Frozen Boundary Coverage Check。

你将看到：

1. GLOBAL；
2. cleaned-only LOCAL；
3. Pass 1 frozen segments；
4. Pass 2 segments.events；
5. 当前 LOCAL 中 owner-clean 后的 cleaned keyframes；
6. boundaries_to_check。

图像和 metadata 中都没有 Pass 2 删除的 candidates。
不要根据想象恢复 deleted candidates。

Pass 1 已经确定了结构。
你不需要、也不允许重新判断是否切 segment。


【LOCAL 与 ownership】

left_overlap、target、right_overlap 全部参与 boundary coverage 判断。

但当前 slice 只能对 target ownership 内的缺失 boundary 新增点。

如果 boundary 位于 overlap：

当前 slice 不得新增。

target start/end 本身不是 boundary。


【检查范围】

只检查 <boundaries_to_check> 中列出的 boundary。

如果 boundary 不在 boundaries_to_check 中：

不要输出。


【boundary 已覆盖】

以下只是已有覆盖证据：

- 相邻两个 Pass 1 segment 在 Pass 2 segments.events 中共享同一个 cleaned keyframe；
- 或存在 cleaned keyframe，其实际位置和前后连续轨迹能够准确表达前一 segment 结束和后一 segment 开始。

共享 event 是强证据，
但仍应确认连续轨迹与 frozen segment 状态切换一致。


【boundary 未覆盖】

以下情况视为未覆盖：

- boundary 附近没有 cleaned keyframe；
- cleaned keyframe 明显位于前一个 segment 内部；
- cleaned keyframe 明显位于后一个 segment 内部；
- cleaned keyframe 与真实状态切换明显偏离；
- cleaned keyframe 原始类型与该状态切换明显不兼容。


【类型兼容参考】

打开 -> 关闭：
应由 MAX 类型边界表达。

关闭 -> 打开：
应由 MIN 类型边界表达。

打开/关闭 -> 稳定：
应由 PL 类型边界表达。

稳定 -> 打开/关闭：
应由 PR 类型边界表达。

如果现有 cleaned keyframe 类型明显不兼容，
不要因为位置接近就机械判断已覆盖。


【灰色虚线】

Pass 1 boundary 的灰色虚线只是近似位置提示。

不要使用：

“离虚线最近”

作为 coverage 判断标准。

真正依据是：

该 cleaned keyframe 前后的连续轨迹，
是否确实对应两个 frozen segment 的真实状态切换。


【禁止重新发现结构】

如果 Pass 1 将某段冻结为一个：

关闭中-带波动
打开中-带波动
稳定*-带波动

那么即使 LOCAL 内仍存在局部峰谷，
只要 Pass 1 没有对应 internal boundary，

Pass 3 都不得新增这些点。


【addition】

只输出真正缺失的 frozen boundary。

一个 slice 可以输出 0~N 个 additions。

不要输出 MAX/MIN/PL/PR 类型。

只输出：

- between：相邻 frozen segment id；
- approx_time_sec：真实状态切换的大致时间；
- description：为什么当前 cleaned keyframes 没有覆盖。

approx_time_sec 只需要粗定位，
后处理会进行数值精定位。


Pass 1 frozen segments、Pass 2 segments、cleaned keyframes、
internal boundaries 和 boundaries_to_check
已在上一段结构化 context 中提供。


【输出格式】

{
  "additions": [
    {
      "between": ["002", "003"],
      "approx_time_sec": 27.4,
      "description": "segment 002 的稳定状态在此结束，随后进入 segment 003 的持续关闭状态；当前 cleaned keyframes 中没有点能够准确表达该边界。"
    }
  ],

  "needs_review": false
}

如果没有缺失 boundary：

{
  "additions": [],
  "needs_review": false
}

不要输出：
- segments
- keyframes
- covered boundary
- MAX/MIN/PL/PR
- deleted candidates

严格输出 JSON。
"""
