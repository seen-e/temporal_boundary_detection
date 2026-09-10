from __future__ import annotations


GLOBAL_IMAGE_INSTRUCTION = """
<global_image_instruction>
Image 1 是 GLOBAL 全 episode 单夹爪轨迹图。

GLOBAL 用于判断当前 target region 在整个 episode 中的宏观背景和相对变化尺度。

图中只包含：

- smoothed full episode 轨迹曲线；

- 黄色高亮区域和红色边界线：
  表示当前 slice 正式负责审核的 target region；

- 淡蓝色区域：
  表示当前 LOCAL 图中可见、但不属于 target region 的 overlap context。

GLOBAL 图中不包含 raw full episode 曲线。
在候选可见的 single-pass / Pass 2 相关实验中，GLOBAL 可能绘制无文字标签的候选点散点，
只用于观察全局候选分布；不要用 GLOBAL 精确判断单个 keyframe。
如果需要判断某个 keyframe 的 id、类型、时间和数值，必须使用 LOCAL 图和 metadata。
Pass 1 会使用单独的无候选点图像说明。

请重点观察：

1. 当前 target region 位于整条 episode 轨迹的什么位置；

2. 当前 target region 是否属于一个更长宏观结构的一部分，例如：
   - 稳定区域；
   - 持续 OPENING；
   - 持续 CLOSING；
   - 带有限波动的稳定区域；
   - 带有限波动的 OPENING；
   - 带有限波动的 CLOSING；

3. 当前 target region 的变化幅度，
   相对于整个 episode 的主要 gripper dynamic range 是大还是小；

4. LOCAL 中看起来明显的局部 MAX/MIN，
   在 GLOBAL 尺度下是否其实只是较小的局部波动。

GLOBAL 只负责：
- 宏观背景；
- 相对振幅；
- 长时间趋势。

不要使用 GLOBAL 精确判断单个 keyframe 的位置或类型。
具体 keyframe 判断以 LOCAL 图和 metadata 为准。
</global_image_instruction>
"""


LOCAL_IMAGE_INSTRUCTION = """
<local_image_instruction>
Image 2 是当前 slice 的 LOCAL 局部放大轨迹图。

LOCAL 可能包含：

    left context + target region + right context

其中：

- target region：
  当前 slice 真正负责审核的区域；

- left/right context：
  相邻 slice 的重叠轨迹，只用于理解 target region 与前后轨迹的连续关系。

如果当前 LOCAL 没有 left context，
说明 target region 已经位于 episode 左边界，
后续 left_connection 必须为 null。

如果当前 LOCAL 没有 right context，
说明 target region 已经位于 episode 右边界，
后续 right_connection 必须为 null。

禁止对不存在的一侧 context 做推测。


图中：

- 黄色背景：
  当前正式审核的 target region；

- 黄色区域之外仍然可见的轨迹和 keyframes：
  overlap context，仅提供结构上下文。


请结合 GLOBAL 的宏观尺度，重点观察：

1. 如果存在 left context，
   判断它以什么宏观趋势连接进入 target region；

2. 如果存在 right context，
   判断 target region 结束后连接到什么宏观趋势；

3. 判断 target region 大致由哪些连续轨迹形态组成；

4. 每个连续 part 只能优先归入以下 8 种形态之一：

   - STABLE_OPEN
   - STABLE_CLOSE
   - OPENING
   - CLOSING
   - STABLE_OPEN_WITH_FLUCTUATION
   - STABLE_CLOSE_WITH_FLUCTUATION
   - OPENING_WITH_FLUCTUATION
   - CLOSING_WITH_FLUCTUATION

5. 判断 target keyframes 在这些结构中是否承担必要角色，例如：
   - 稳定区域开始边界；
   - 稳定区域结束边界；
   - 极值区域开始边界；
   - 有效 MAX/MIN；
   - 极值区域结束边界；
   - 相邻两个结构共享的边界；

6. 判断哪些 target keyframes 只是：
   - 稳定区域内部波动；
   - OPENING 主趋势内部小回撤；
   - CLOSING 主趋势内部小反弹；
   - 重复表达同一个结构角色的候选点。


特别注意：

- LOCAL 是放大图。
  局部起伏视觉上很明显，不代表它相对于整个 episode 的开合范围也很大；
  是否属于 fluctuation 必须结合 GLOBAL 判断。

- 一个数学上真实的 MAX/MIN，
  如果只是某种 WITH_FLUCTUATION 形态内部的局部波动，
  仍然可以删除。

- 平台/稳定区域应保护真正的开始和结束两个边界角色。

- 一个有效极值结构应保护：
  开始边界、MAX/MIN、结束边界三个角色。

- 相邻结构可以共享同一个 keyframe。
  一个 keyframe 同时作为前一个结构的结束边界和后一个结构的开始边界是合法的，
  不能因为被两个 part 引用就认为重复。

- 如果两个 keyframe 重复承担完全相同的结构角色，
  只保留更合理的一个，其余删除。

- 本阶段只审核已有 target keyframes，
  不判断是否新增关键帧。

keyframe_id、keyframe type、target/context 身份最终以 metadata 为准。
</local_image_instruction>
"""