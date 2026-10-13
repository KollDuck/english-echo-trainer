# 🎧 English Echo Trainer（英语母语级精听与回声训练助手）

输入一个 YouTube 链接，自动完成：**音频下载 → Whisper 逐句时间戳识别 → 逐句切片播放/循环/慢速回声训练 → 麦克风跟读录音对比 → LLM 知识点总结**。所有解析结果自动存入本地历史库，随时回顾。

> 本 README 面向人类用户与 **AI Agent** 双受众：所有安装步骤均为可直接执行的幂等命令，并附自检验证方法，Agent 可逐步执行并核对输出。

## 功能特性

- 🔊 **逐句精听**：每句独立 ▶ 播放 / 🔁 单句循环（回声训练法）/ 🐢 0.75x 慢速（变速不变调）
- 🙈 **盲听优先**：原文默认打码，👁 按句揭示或全局显示
- 🎙️ **跟读对比**：每行内录音（自动统一转 WAV，兼容 Chrome/Safari 录音格式），🔊 回放对比原声
- 🤖 **AI 助教**：地道表达/动词词组、连读弱读发音难点、一句话总结（兼容任意 OpenAI 格式 API）
- 📚 **本地历史库**：自动保存解析记录与音频，搜索/一键回顾/删除
- ⚡ **GPU 加速**：faster-whisper `large-v3` INT8 量化（约 4GB 显存），RTX 50 系 Blackwell 实测可用；无 GPU 自动降级 CPU
- 💾 **智能缓存**：同链接同模型直接从历史库秒开，不重复下载与识别

## 工作原理

```
YouTube URL
   │  yt-dlp（走系统代理）
   ▼
mp3（播放/切片）+ 16kHz wav（识别）
   │  faster-whisper large-v3, word_timestamps=True
   ▼
词级时间戳 → 按句末标点(. ! ?)合并成句（缩写保护：Mr. / J. K. 不误切）
   │  pydub 按句切片（毫秒精度）
   ▼
Streamlit UI：逐句播放器 + 录音对比 + LLM 分析
   │
   ▼
data/library/{video_id}.json 本地历史库
```

## 环境要求

| 项目 | 要求 |
|---|---|
| 操作系统 | Windows 10/11（Linux/macOS 可参考手动步骤改造） |
| GPU | NVIDIA ≥ 6GB 显存（推荐，非必需；纯 CPU 可运行） |
| 网络 | 访问 YouTube 需系统级代理（应用自动读取，无需在应用内配置） |
| LLM | 任意 OpenAI 兼容 API Key（OpenAI / DeepSeek / Qwen / Ollama 等） |
| 磁盘 | ≥ 8GB（conda 环境 ~3GB + 模型 ~3GB + 音频缓存） |

## 快速开始（AI Agent 一键部署）

```powershell
# 国内网络（走 hf-mirror.com 镜像下载模型，推荐）
powershell -ExecutionPolicy Bypass -File setup.ps1 -Mirror

# 可直连 HuggingFace 的网络
powershell -ExecutionPolicy Bypass -File setup.ps1
```

脚本幂等，可重复执行。完成后用 `start.bat`（或见下方第 5 步命令）启动。

## 分步安装（每步可独立验证）

以下命令均在 **PowerShell** 中执行，`$envPy` 为环境内 Python 的绝对路径。

### 1. 安装 Miniconda

```powershell
winget install --id=Anaconda.Miniconda3 -e --silent --accept-package-agreements --accept-source-agreements
```

验证：

```powershell
Test-Path "$env:USERPROFILE\miniconda3\Scripts\conda.exe"   # 期望 True
```

### 2. 创建 conda 环境（Python 3.11 + ffmpeg）

```powershell
& "$env:USERPROFILE\miniconda3\Scripts\conda.exe" create -n english python=3.11 ffmpeg --override-channels -c conda-forge -y
```

> ffmpeg 由 conda-forge 装进环境内（`<env>\Library\bin`），应用启动时会自动解析并前置 PATH，**无需全局安装**。

验证：

```powershell
Test-Path "$env:USERPROFILE\miniconda3\envs\english\Library\bin\ffmpeg.exe"   # 期望 True
```

### 3. 安装 Python 依赖

```powershell
$envPy = "$env:USERPROFILE\miniconda3\envs\english\python.exe"
& $envPy -m pip install -r requirements.txt
# 网络慢可换国内镜像：
# & $envPy -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

验证：

```powershell
& $envPy -c "import streamlit, faster_whisper, yt_dlp, pydub, openai; print('deps OK')"
```

### 4. 预下载 Whisper 模型（约 3GB，一次性）

```powershell
# 国内（推荐）：
& $envPy scripts\download_model.py large-v3 --mirror
# 海外：
& $envPy scripts\download_model.py large-v3
```

模型缓存于 `%USERPROFILE%\.cache\huggingface\hub`，应用启动时自动加载。

### 5. 启动

```powershell
# 双击 start.bat，或：
cd <项目目录>
& "$env:USERPROFILE\miniconda3\envs\english\python.exe" -m streamlit run app.py
```

浏览器自动打开 `http://localhost:8501`。

### 6. Agent 自检验证清单

```powershell
# a) GPU 是否可用（无 GPU 属正常结果，应用会降级 CPU）
& $envPy scripts\check_gpu.py

# b) ffmpeg 是否就位
& "$env:USERPROFILE\miniconda3\envs\english\Library\bin\ffmpeg.exe" -version

# c) 应用是否可服务（预期输出 HTTP 200）
Start-Process "$envPy" -ArgumentList "-m","streamlit","run","app.py","--server.headless","true","--server.port","8501"
Start-Sleep 15
(Invoke-WebRequest -Uri "http://localhost:8501" -UseBasicParsing).StatusCode
```

## 应用内配置（侧边栏，点左上角 » 展开）

| 配置 | 说明 |
|---|---|
| 🤖 LLM Base URL | 默认 `https://api.openai.com/v1`，可换成 `https://api.deepseek.com/v1` 等 |
| 🤖 API Key / 模型名 | 任意 OpenAI 兼容端点；也可用环境变量 `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `LLM_MODEL` 预填 |
| 🌐 hf-mirror 镜像 | 勾选后模型下载走镜像（自动设置 `HF_HUB_DISABLE_XET=1`，见下方排障 #1） |
| 🎙️ Whisper 模型 | large-v3（默认，最准）/ medium / small / base |

> **代理**：yt-dlp 在 Windows 上自动读取系统代理设置，应用内无需配置。若下载失败请确认系统代理 / TUN 已开启。

## 使用说明

1. 粘贴 YouTube 链接 → **🚀 开始处理**（下载 → 识别，长视频请耐心等待）
2. **逐句精听**：▶ 播放 / 🔁 循环 / 🐢 慢速；先盲听，再 👁 揭示原文对照
3. **跟读对比**：🎤 录下你的跟读（重录自动覆盖）→ 🔊 回放对比（按钮有录音时变色高亮）
4. **AI 助教** Tab：生成地道表达、连读弱读难点、总结
5. **历史记录**：页面底部，支持搜索、📂 回顾、🗑 删除

## 项目结构

```
├── app.py                    # Streamlit UI + 交互逻辑 + 历史库
├── downloader.py             # yt-dlp 封装（错误映射为中文提示、双格式转码）
├── transcriber.py            # faster-whisper 封装（词级时间戳→句子切分、GPU→CPU 自动降级）
├── llm_analyzer.py           # OpenAI 兼容 API（JSON 结构化输出 + 容错解析）
├── requirements.txt
├── setup.ps1                 # 一键环境部署（幂等，供 Agent 执行）
├── start.bat                 # 便携启动器（自动定位 conda 环境）
├── scripts/
│   ├── download_model.py     # 模型预下载（支持 --mirror）
│   └── check_gpu.py          # GPU 可用性自检
└── data/                     # 运行时数据（不入库）
    ├── library/              # 历史记录 {video_id}.json
    └── *.mp3 / *_16k.wav     # 音频缓存
```

## 故障排查（实测踩坑记录）

| # | 症状 | 原因 | 解决 |
|---|---|---|---|
| 1 | HF 下载大文件报 `401 Unauthorized`（`cas-server.xethub.hf.co`） | HF 新 Xet 存储，镜像无法代理 | 设置 `HF_HUB_DISABLE_XET=1`（应用勾选镜像时自动设置） |
| 2 | pydub 报 `Couldn't find ffprobe` | pydub 只查 PATH | 应用启动时自动解析 conda 环境内 ffmpeg 并前置 PATH；请通过 `start.bat` 或环境内 python 启动 |
| 3 | RTX 50 系 GPU 报错或无输出 | Blackwell(sm_120) 需新版 CUDA 内核 | `ctranslate2>=4.5`（实测 4.8.3 + cudnn-cu12 9.7 可用）；应用内置失败自动降级 CPU |
| 4 | 录音回放无声 | `st.audio_input` 实为 WebM/Opus，文件名却叫 .wav | 应用已自动转码为 WAV（`_to_wav`） |
| 5 | .bat 中文乱码/命令错位 | cmd 代码页与 UTF-8 解析冲突 | `start.bat` 保持纯 ASCII，勿添加中文 |
| 6 | 首次识别很慢 | 模型冷加载 + cudnn 预热 | 属正常；第二次起秒级。可先跑 `scripts/download_model.py` 预下载 |
| 7 | winget 安装 Git/工具要求管理员 | 安装器需 UAC 授权 | 可用便携版免管理员部署（本项目 git/gh 即便携安装） |

## License

[MIT](LICENSE)
