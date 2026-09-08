# 从第一视角清扫视频到可执行机器人场景

## 交付范围

输入是 6 秒、1280×720、24 fps 的桌面视频。左手定位簸箕，右手刷子向左推进彩色方块，最后撤刷。本次实现包含视频观测、程序化视觉场景、商用 TIAGo++ 模型、MuJoCo 接触仿真、独立审计、Blender 三视角、实际工作窗口录制和公开博客。

这是根据视频理解任务后编写技能与 IK 控制器的完整制作过程。工具尺寸来自估计，未标定单目相机。机器人从已握持工具的状态开始；刷毛是刚性碰撞代理，外观另用细丝表现。

## 1. 先确定视频中的任务和可相信的观测

用 ffprobe 读取视频元数据，用 ffmpeg 抽取每秒两帧的接触表和几个完整分辨率关键帧。原片首帧清楚可见三块，末段可见六块，不能直接当作数量守恒的物理状态序列。因此场景在开始就放置六个自由刚体，整个仿真不增删方块。

手动抽帧命令需要 `ffmpeg` 和 `ffprobe` 在 PATH 中；主复现脚本会自动使用已安装的 imageio-ffmpeg。

```bash
mkdir -p agent/out/source
ffprobe -v quiet -show_format -show_streams inputs/deskclean.mp4
ffmpeg -i inputs/deskclean.mp4 -vf 'fps=2,scale=480:-1,tile=4x3' -frames:v 1 agent/out/source/contact-sheet.jpg
```

关键建模设定是 22 mm 方块、280 mm 簸箕开口、230 mm 簸箕深度和 730 mm 台面高度。它们是推定值，保存于 `agent/out/scene_spec.json`，不宣称由单目视频测得。

## 2. 统一坐标，再分别表达外观和接触

全局采用米制、Z-up；机器人朝 +X，左侧为 +Y。簸箕局部原点在开口中心，内腔朝 +Y；刷子局部原点在刷毛底面中心，长边沿 X，清扫方向为 +Y。机器人可达性规划将整组工作物体平移 `[-0.06,-0.12,0]` m，最终坐标另保存到 `resolved-scene-spec.json`。

`create_assets.py` 在 Blender 中生成簸箕塑料壳、把手、入口薄唇，以及刷头、把手和 624 组可见刷毛细丝。视觉工具与仿真碰撞使用相同原点和尺寸。

簸箕不能直接使用整个凹形表面的凸包碰撞，否则开口会被封住。MuJoCo 模型由薄底、入口坡面、两侧和后挡板构成。底面局部 y/z 轮廓为 `[[-0.015,-0.001],[0,0.002],[0.23,0.009]]`，前缘厚度为零，后续厚度为 2 mm。Blender 视觉也采用此入口轮廓。

来源：[MuJoCo 碰撞说明](https://mujoco.readthedocs.io/en/3.4.0/computation.html#collision-detection)。

## 3. 使用商用机器人原模型

采用 PAL Robotics TIAGo++ 的双臂全向轮式模型。模型来自 MuJoCo Menagerie，原始来源为 PAL 的 URDF，固定 revision 为 `8161bba264d7fa7c99ca301e91e7fb44737676ad`。仅下载约 8 MB 的所需文件，并逐文件核对 Git blob SHA。保留 Apache-2.0 许可。

场景层移除底座 free joint，使底座固定；保留真实机器人网格、双臂关节结构、原始关节力限和执行器力限。工具固定在原夹爪腕部，手指保持夹持开度。工具预抓持是明确起始条件，本次不包含自主接近与抓取过程。

来源：[TIAGo 产品](https://pal-robotics.com/robot/tiago/)、[模型说明与许可](https://github.com/google-deepmind/mujoco_menagerie/tree/8161bba264d7fa7c99ca301e91e7fb44737676ad/pal_tiago_dual)。

## 4. IK 生成目标，物理积分产生运动

`robot_control.py` 用独立 MuJoCo 状态计算 FK，并使用 SciPy `least_squares` 求解工具位姿。腕部姿态保留壳体碰撞余量。目标动作包含簸箕定位、刷子推进、退刷和抬起簸箕。

实际仿真步长为 0.002 s。运行路径只设置 actuator `ctrl`，随后调用 `mj_step`。所有方块均为 freejoint；初始化之后不改写方块姿态，不给它们 mocap、关键帧或外力。控制记录保存真实发送给引擎的指令。

初态与每一步后的状态保留为 `state[T+1]`，控制为 `ctrl[T]`。接触记录包含求解时刻、接触物体、距离、位置和接触坐标系中的六维力。为导出渲染，独立前向计算刷新与保存 qpos 对应的几何姿态。

## 5. 让成功可以检查

一次早期运行虽然把六块推入簸箕，但薄底默认接触参数造成约 6.1 mm 下沉。移除机器人、只保留薄底和单方块后仍出现同样问题，定位到接触柔度。将任务接触时间常数设为 0.006 s 后，独立单块测试穿透约 0.35 mm，完整动作全程最大穿透 0.668 mm。

最终审计检查五个方面：

1. 六块的全部八个角点位于簸箕内腔，底面接触容差为 1 mm。
2. 从初态重放保存的 ctrl，复现完整状态，最大 qpos 误差为零。
3. 状态、控制和渲染帧契约一致。
4. 运行态仅通过 ctrl 与物理积分推进。
5. 原始机器人关节和执行器力限逐项一致。

最终结果为六块全部收集；最大机器人自身碰撞穿透为零；IK 工具原点最大位置误差 0.232 mm。刷子—方块法向冲量为 0.6191124 N·s，簸箕—方块为 1.8636634 N·s。只有独立审计通过后才写入 `success=true`。

附加对照在独立模型中复用完全相同的控制：禁用刷子—方块接触后收集数为 0/6；两次 ±3 mm 初始位置扰动和一次方块 geom 摩擦参数 +20% 的检查均为 6/6。这些是本次有限对照，输出和原始轨迹分开保存。

## 6. 从同一物理轨迹得到三个视角

使用官方 `mujoco.usd.exporter.USDExporter` 导出机器人与方块。将 FPS 显式设为 24，单位为米，up-axis 为 Z，并修正官方导出器默认额外声明的隐藏终帧。

Blender 导入 USD 后，将刚体世界变换烘焙为关键帧，移除运行时 USD 依赖；簸箕和刷子的高质量外观由相同 body pose 驱动。检查确认 `.blend` 不依赖远程 USD 或外部纹理，工具位置转存最大误差约 `2.86e-8 m`。

三个相机分别是固定全局相机、挂在机器人头部的 ego 相机、与右夹爪刚性关联的局部相机。夹爪相机选择能看到刷毛前方接触面的安装位置，避免刷头完全遮住方块。

物理动作时长为 7.2 s；174 张 24 fps 帧封装后的视频长 7.25 s。网页按两段视频各自的归一化时间同步对照，不假设它们逐帧时间完全相同。

来源：[MuJoCo 3.4 USD 导出器](https://mujoco.readthedocs.io/en/3.4.0/python.html#usd-exporter)、[Blender 4.2 Python API](https://docs.blender.org/api/4.2/)。

## 7. 记录真实过程，控制资源

本机只抽帧、编辑代码和轻量网页预览。rl-1 使用独立项目目录。Blender GUI 在独占 Xvfb 显示 `:117` 中运行，ffmpeg 从窗口启动前开始以 3 fps 持续录制。后续短片从该真实录屏剪辑和加速，展示场景载入、检查与动画回放；后台执行的建模、仿真和渲染另保留源码、命令与日志。

GUI 的控制通过主线程 timer 读取原子替换的 Python 命令文件；每次命令完成后才提交下一次。一次加载文件后误用 `bpy.context.screen` 导致 timer 退出，原 traceback 被保留；只重启本任务 Blender，持续录屏没有中断，随后改用明确窗口上下文。后续在同一个 timer tick 内切换工作区并查找文本编辑器，也因界面尚未更新而失败；工作区切换需要等待事件循环。两次均为界面审阅控制问题，未改动物理结果。

自行记录时，先运行 `mkdir -p agent/out/process`，并确保 `ffmpeg` 在 PATH 中。可在安装 Xvfb 的 Linux 上选用空闲显示号，以 `Xvfb :117 -screen 0 1280x720x24` 启动虚拟屏幕，再运行 `ffmpeg -f x11grab -framerate 3 -video_size 1280x720 -i :117.0 -c:v libx264 -threads 2 agent/out/process/workflow-live.mkv`。记录开始后，以 `DISPLAY=:117` 启动 Blender GUI。工具包的 `agent/scripts/blender_control.py` 使用环境变量 `DESKCLEAN_CONTROL_DIR` 定位控制目录；目录内预先写好 `command.py`（初始内容可为 `pass`），新命令先写临时文件再原子替换，确认 `status.json` 的 `done` 状态及命令时间戳匹配后继续。录制结束用 SIGINT 让 ffmpeg 完成封装。该界面录制通道与后台仿真、正式渲染彼此独立。

渲染使用 GPU 7 一张 A100，两个进程分别渲染前后半段，共享固定的八个逻辑 CPU，48 samples、1280×720。首次 CUDA 内核 JIT 约占六分钟；之后同样配置的样帧约需数秒，无需改动系统驱动或重启服务器。

## 8. 复现与输出

解压 `reproduce.zip`，用 Python 3.12 安装 `requirements.txt`，设置 Blender 4.2.1 路径和可用 CUDA 设备，运行 `bash scripts/reproduce.sh`。代码和第三方资产齐备，输入视频保存在 `inputs/deskclean.mp4`。

归档中的主要输出：`agent/out/simulation/` 的 MJCF、状态、控制、接触和原始指标；`agent/out/usd/` 的官方轨迹导出；`agent/out/deskclean.blend`。复现脚本每次创建独立的 `agent/out/reproductions/run-XXXXXX/`，其中保留重新生成的 `agent/out/videos/{global,ego,gripper}.mp4` 和其他阶段输出，不覆盖归档原始结果。每一步命令日志保存在本次运行的 `agent/log/`，附加对照保存在归档的 `agent/out/validation/`。
