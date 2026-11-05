# Ubuntu CPU 摄像头物体识别参考

本项目供 Ubuntu 无 NVIDIA 独显笔记本参考学习：用 OpenCV 读取摄像头，使用 YOLO26n 预训练模型检测常见物品，显示检测框、类别和置信度，支持截图和正常退出。默认使用 CPU，不需要安装 CUDA 或 NVIDIA 驱动。

建议先按默认轻量配置运行：640×480 摄像头画面、模型输入 416、请求采集 15 FPS、最多 4 个 PyTorch CPU 线程。实际帧率取决于 CPU、温度、供电方式和摄像头；本项目没有针对收件人的笔记本实测帧率。

## 准备环境

以下以普通 x86_64 Ubuntu 22.04 或 24.04 桌面系统为目标，要求已安装 Miniconda 且终端可以使用 `conda`。如果尚未安装 Miniconda，请按照官方 Linux 安装说明选择对应 CPU 架构的安装包：https://www.anaconda.com/docs/getting-started/miniconda/install

解压本压缩包，在项目目录打开终端，执行：

```bash
conda env create -f environment.yml
conda activate perception-cpu
python -m pip install -r requirements.txt
python -m pip check
```

依赖文件固定使用 PyTorch 2.10.0 与 Torchvision 0.25.0 的 CPU 构建，另含 Ultralytics 8.4.194 与 OpenCV 5.0.0.93。必须等待安装成功再运行程序，不要使用发送者 Windows 项目里的 CUDA 依赖文件。

若安装过程中访问 PyTorch CPU 索引超时，需要先排查网络后重试。若终端提示 Conda 尚未初始化，在 Miniconda 安装目录的 `bin/conda` 能运行的前提下执行 `conda init bash`，关闭并重新打开终端。

若运行时缺少 `libGL.so.1`：

```bash
sudo apt update
sudo apt install libgl1
```

若提示缺少 `libgthread-2.0.so.0`，Ubuntu 22.04 安装 `libglib2.0-0`，Ubuntu 24.04 安装 `libglib2.0-0t64`。本项目需要有本地桌面会话的 OpenCV 窗口，不要用 `opencv-python-headless` 替代窗口版本。

## 先验证图片

压缩包已附带 `models/yolo26n.pt` 和公开图片 `assets/bus.jpg`，不需要先连摄像头：

```bash
python check_environment.py
```

成功后应显示 `inference_device: cpu` 和 `cpu_inference: passed`，并把检测结果与版本信息保存到 `runs/environment-check/`。CPU 模式下没有 `cuda:0` 是正常现象。

## 再运行摄像头

```bash
python detect_camera.py
```

也可以在已激活的环境中运行 `bash start.sh`。启动时先加载模型，随后显示画面。

- 点击预览窗口获得键盘焦点，按 Q 或 Esc 退出，也可以关闭窗口或在终端按 Ctrl+C。
- 按 S 保存当前检测截图到 `runs/camera/`。默认不保存摄像头照片，不录制视频。
- 按 H 隐藏或显示信息栏。默认是紧凑半透明背景，`--hud-opacity 0.15` 可调得更透明，`--hide-info` 可默认隐藏。
- 显示英文类别及置信度。模型置信度不等于经过验证的识别准确率。

常见调整：

```bash
# 速度优先，可能增加小目标漏检
python detect_camera.py --imgsz 320 --threads 2

# 细节优先，需要更强的 CPU，帧率可能下降
python detect_camera.py --imgsz 640

# 切换摄像头编号
python detect_camera.py --source 1

# 指定摄像头采集格式请求
python detect_camera.py --width 640 --height 480 --fps 15

# 用本地图片或视频理解流程
python detect_camera.py --source assets/bus.jpg
python detect_camera.py --source example.mp4
```

`--width/--height` 是摄像头采集分辨率，`--imgsz` 是模型缩放输入尺度，两者独立。输入尺度需为 32 的正整数倍。提高采集分辨率后，图像仍可能被缩小送入模型，不保证检测效果提升。

`--fps` 只请求采集帧率，设备可能忽略，CPU 也可能处理不到该帧率。信息栏中 processing FPS 只统计推理与绘制；loop FPS 包含帧间读取、处理与显示等待；camera reported 是驱动报告值。它们都不是相机到屏幕的延迟。`--threads` 调整 CPU 线程数，线程越多不一定越快，可以根据自己的 CPU 比较 2 和 4。

## VS Code 运行

用 VS Code 打开整个解压后的项目文件夹，安装推荐的 Python 和 Pylance 扩展。按 Ctrl+Shift+P，执行 `Python: Select Interpreter`，选择 `perception-cpu` 环境，再新建终端，确认提示符有 `(perception-cpu)`。

在终端执行 `python detect_camera.py`，或在运行与调试中选择 `Ubuntu CPU camera`。配置没有写死发送者电脑的 Python 路径。

## 摄像头问题排查

Ubuntu 下优先使用 V4L2 采集。检查摄像头物理遮挡开关，关闭浏览器视频会议、系统相机等占用程序。如果失败，尝试 `--source 1` 或显式使用 `--backend v4l2`。

以下为只读检查命令；缺少工具时可先用 `sudo apt install v4l-utils` 安装：

```bash
ls -l /dev/video*
v4l2-ctl --list-devices
v4l2-ctl -d /dev/video0 --list-formats-ext
```

并非每个 `/dev/videoN` 都是可用画面设备，应以设备列表和实际采集为准。若出现 Permission denied，检查 `id -nG` 及设备文件权限；若设备属于 `video` 组且当前用户无访问权限，可以执行 `sudo usermod -aG video "$USER"`，然后注销并重新登录。不要用 sudo 启动整个 Python 程序来绕过环境问题。

程序在初始化时会过滤连续近乎全黑的画面，并阻止同一用户重复启动本程序占用同一编号的摄像头；其他应用仍可能占用摄像头。在纯 SSH、服务器或没有桌面的环境下，窗口无法正常显示，可先用图片加 `--no-display` 验证推理。

## 理解主要流程

程序先准备 CPU 线程与预训练模型，再打开摄像头。每次读取一帧 BGR 图像，交给 YOLO 完成缩放、归一化、网络推理与检测结果后处理，再绘制检测框、类别、置信度和信息栏，最后显示并读取下一帧。通过 `finally` 保证异常退出时也释放摄像头、关闭窗口。

YOLO26n 的 n 表示 nano，适合先在 CPU 上进行轻量实践。当前使用 COCO 预训练检测权重，覆盖 80 类常见目标，没有自定义训练，因此不能识别任意自定义物品。此项目没有使用集显加速；CPU 是默认计算设备。

## 文件与提交

- `detect_camera.py`：摄像头、图片及视频识别程序。
- `check_environment.py`：无需摄像头的 CPU 图片检查。
- `environment.yml`、`requirements.txt`：环境与依赖说明。
- `.vscode/`：可移植的编辑器配置。
- `models/yolo26n.pt`、`assets/bus.jpg`：预训练权重与公开测试图片。
- `examples/bus-detection.jpg`：公开图片的 CPU 检测示例，不是收件人的实拍结果。

`.gitignore` 会排除模型权重、缓存与运行输出。上传 GitHub 时在 README 保留模型下载地址，并添加自己实际运行的截图、系统及版本记录。学习后应能解释关键参数与完整处理流程。

此参考包的 CPU 图片推理已在发送者 Windows 环境中执行；该环境安装的是支持 CPU 运行的 CUDA 构建 PyTorch。收件人的 Ubuntu CPU 专用依赖安装、Linux 摄像头与桌面窗口仍需在她的电脑上验证，不能把附带示例当作她的实机运行证明。

## 资源来源

- YOLO26n 权重：https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt
- 公开示例图：https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/bus.jpg
- Ultralytics 文档：https://docs.ultralytics.com/models/yolo26/
- Ultralytics 许可说明：https://www.ultralytics.com/license
- PyTorch CPU 安装源：https://download.pytorch.org/whl/cpu

若权重缺失，请从上述官方链接下载为 `models/yolo26n.pt`；下载失败时排查网络，文件名保持一致。
1
