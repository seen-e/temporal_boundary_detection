
PASS1_GLOBAL_IMAGE_INSTRUCTION = """
<pass1_global_image_instruction>

Image 1 是 Pass 1 GLOBAL 全 episode 平滑轨迹图。

黄色区域是当前 slice 拥有写回权限的 target ownership 范围，淡蓝色区域是 LOCAL 中可见的 overlap/context，红色虚线是 ownership 边界。

图中没有任何候选关键帧 marker、id、MAX/MIN/PL/PR 或文字标签。

只用它判断整个 LOCAL 可见范围在 episode 中的长时间尺度 baseline、相对振幅和大致宏观结构，不用于精确确定 segment 边界。

</pass1_global_image_instruction>
"""


PASS1_LOCAL_IMAGE_INSTRUCTION = """
<pass1_local_image_instruction>

Image 2 是 Pass 1 LOCAL 局部轨迹图。

黄色区域是当前 slice 拥有写回权限的 target ownership 范围，淡蓝色区域是左右 overlap/context，红色虚线是 ownership 边界。

图中没有任何候选关键帧 marker、id、MAX/MIN/PL/PR 或文字标签。

请对整个 LOCAL 可见范围切分宏观连续 segment；target 边界只是 ownership 标记，不是 segment 边界，segment 可以跨 target 边界连续存在。

只根据轨迹形状切分宏观连续 segment，不要推测候选点位置。

</pass1_local_image_instruction>
"""


PASS2_LOCAL_IMAGE_INSTRUCTION = """
<pass2_local_image_instruction>

Image 是 Pass 2 LOCAL 候选关键帧图。

黄色区域是当前 target ownership，淡蓝色区域是 left/right overlap。

图中显示整个 LOCAL 内的全部 candidates，并用三位数字 id 标出；候选类型、时间、scope 以 metadata 为准。

Pass 1 的 segments 已冻结；本图用于让 left_overlap、target、right_overlap candidates 共同竞争 frozen segment 的真实结构边界。

target/overlap 只表示当前 slice 是否拥有写回权限，不是轨迹结构边界。

segments.events 必须按整个 LOCAL 映射，可以引用 overlap candidate；后处理只会把 target candidate 的 KEEP / DELETE 写回当前 slice。

如果某个 frozen boundary 附近没有合适 candidate，可以不选择，不要为了覆盖边界强行选择最近点。

</pass2_local_image_instruction>
"""


PASS3_GLOBAL_IMAGE_INSTRUCTION = """
<pass3_global_image_instruction>

Image 1 是 Pass 3 GLOBAL 全 episode 轨迹图。

图中只使用与候选关键帧检测一致的单一轨迹源；当前配置为 smoothed-only，不叠加 raw。

黄色区域是当前 slice 拥有写回权限的 target ownership 范围，淡蓝色区域是 overlap/context。

GLOBAL 只用于判断相对振幅、长期 baseline 和全局尺度。

</pass3_global_image_instruction>
"""


PASS3_LOCAL_IMAGE_INSTRUCTION = """
<pass3_local_image_instruction>

Image 2 是 Pass 3 cleaned-only LOCAL 局部轨迹图。

图中只使用与候选关键帧检测一致的单一轨迹源；当前配置为 smoothed-only，不叠加 raw。

黄色区域是当前 slice 拥有写回权限的 target ownership 范围，淡蓝色区域是 overlap/context。

轨迹曲线正常显示；候选点 marker 只显示 Pass 2 owner-clean 后的 cleaned keyframes 和三位数字 id。

灰色轻虚线是当前 slice 需要检查的 target-owned Pass 1 boundary 近似位置，只是待检查位置提示，不是正确答案。

判断 boundary 是否已覆盖时，应观察该位置前后的真实轨迹状态变化，不能仅根据 cleaned keyframe 与灰色虚线的距离判断。

图中不显示 deleted candidates 或传统候选点。

请用整个 LOCAL 中的 cleaned keyframes 判断覆盖情况；当前 slice 只允许对 target ownership 内的缺失 boundary 新增。

</pass3_local_image_instruction>
"""
