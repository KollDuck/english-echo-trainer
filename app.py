"""英语母语级精听与回声训练助手 - Streamlit 主应用。"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import streamlit as st

from downloader import download_audio, DownloadError
from transcriber import Transcriber, setup_cuda_libs
from llm_analyzer import analyze

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
LIB_DIR = DATA_DIR / "library"
LIB_DIR.mkdir(exist_ok=True)


def _find_ffmpeg():
    """显式解析 ffmpeg：优先 conda 环境内（<env>\Library\bin），其次 PATH。"""
    exe_dir = Path(sys.executable).parent
    for d in (exe_dir / "Library" / "bin", exe_dir):
        f = d / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
        if f.exists():
            return f
    return shutil.which("ffmpeg")


FFMPEG = _find_ffmpeg()
if FFMPEG:
    ffmpeg_dir = Path(FFMPEG).parent
    # 前置到 PATH：pydub 的 ffprobe 探测、yt-dlp、subprocess 都依赖 PATH 查找
    if str(ffmpeg_dir) not in os.environ.get("PATH", ""):
        os.environ["PATH"] = str(ffmpeg_dir) + os.pathsep + os.environ.get("PATH", "")
    try:
        os.add_dll_directory(str(ffmpeg_dir))
    except Exception:
        pass
    # 双保险：显式绑定 pydub 的 converter
    from pydub import AudioSegment
    AudioSegment.converter = str(FFMPEG)
    AudioSegment.ffmpeg = str(FFMPEG)

_MODE_LABEL = {"once": "▶ 播放一次", "loop": "🔁 单句循环", "slow": "🐢 0.75x 慢速", "consumed": ""}
_TEXT_MASK = "█ █ █ █ █ █"


def _text_visible(idx: int) -> bool:
    """原文默认隐藏：全局开关打开、或该句被 👁 揭示过才显示。"""
    if st.session_state.get("show_all_text"):
        return True
    return idx in st.session_state.setdefault("revealed", set())


# ================= 工具与业务函数（先定义，后执行 UI 流程） =================
def _fmt_t(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    return f"{m:02d}:{s:02d}"


def _save_to_library(video: dict, model: str, url: str):
    """处理完成后把完整记录（含句子与音频路径）写入历史库。"""
    entry = {**video, "url": url, "model": model, "added_at": time.time()}
    path = LIB_DIR / f"{video['video_id']}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(entry, f, ensure_ascii=False)
    return entry


def _load_library() -> list:
    """读取历史库全部记录，按加入时间倒序。"""
    entries = []
    for p in LIB_DIR.glob("*.json"):
        try:
            with open(p, encoding="utf-8") as f:
                entries.append(json.load(f))
        except Exception:
            continue
    entries.sort(key=lambda e: e.get("added_at", 0), reverse=True)
    return entries


def _find_in_library(url: str):
    for entry in _load_library():
        if entry.get("url") == url:
            return entry
    return None


def _delete_library_entry(video_id: str):
    """删除历史记录及对应的音频文件。"""
    path = LIB_DIR / f"{video_id}.json"
    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                entry = json.load(f)
            for k in ("mp3_path", "wav_path"):
                if entry.get(k) and Path(entry[k]).exists():
                    os.remove(entry[k])
        except Exception:
            pass
        path.unlink(missing_ok=True)


def _migrate_old_cache():
    """把旧版 transcript_*.json 缓存导入历史库（仅库为空时执行一次）。"""
    if any(LIB_DIR.glob("*.json")):
        return
    for p in DATA_DIR.glob("transcript_*.json"):
        try:
            with open(p, encoding="utf-8") as f:
                v = json.load(f)
            if not v.get("sentences") or not Path(v.get("mp3_path", "")).exists():
                continue
            v["url"] = v.get("url", "")
            v["model"] = v.get("model", "large-v3")
            v["added_at"] = v.get("added_at", p.stat().st_mtime)
            with open(LIB_DIR / f"{v['video_id']}.json", "w", encoding="utf-8") as f:
                json.dump(v, f, ensure_ascii=False)
        except Exception:
            continue


def _reset_play_state():
    st.session_state.pop("play", None)
    st.session_state.pop("slices", None)
    st.session_state.pop("llm_result", None)
    st.session_state.pop("revealed", None)
    st.session_state.pop("rec_active", None)
    st.session_state.pop("play_mine", None)
    for k in [k for k in list(st.session_state.keys()) if k.startswith("rec_")]:
        st.session_state.pop(k, None)


def _process(url: str, whisper_model: str):
    # 历史库命中（同链接 + 同模型 + 音频文件还在）则直接打开
    entry = _find_in_library(url)
    if (entry and entry.get("model") == whisper_model
            and entry.get("sentences") and Path(entry.get("mp3_path", "")).exists()):
        _reset_play_state()
        st.session_state["video"] = entry
        st.toast("已从历史记录加载，无需重复下载与识别 ✅")
        return

    try:
        with st.status("正在处理…", expanded=True) as status:
            status.update(label="📥 步骤 1/3：下载音频（yt-dlp）…")
            video = download_audio(url, DATA_DIR)
            st.write(f"视频: **{video['title']}**")

            status.update(label="🎙️ 步骤 2/3：Whisper 识别中（首次运行需下载模型，请耐心等待）…")
            setup_cuda_libs()
            transcriber = Transcriber(whisper_model)
            progress_bar = st.progress(0.0, text="识别进度")
            sentences, _info, device = transcriber.transcribe(
                video["wav_path"], language="en",
                progress_cb=lambda f: progress_bar.progress(f, text=f"Whisper 识别中 {int(f * 100)}%"),
            )
            progress_bar.empty()

            if not sentences:
                raise RuntimeError("未识别到任何语音内容（视频可能没有对白/全是音乐）。")
            video["sentences"] = sentences
            video["asr_device"] = device
            status.update(label=f"✅ 完成：共识别 {len(sentences)} 句（{device.upper()} 加速）",
                          state="complete", expanded=False)

        _reset_play_state()
        st.session_state["video"] = video
        _save_to_library(video, whisper_model, url)
        st.rerun()
    except DownloadError as e:
        st.error(f"❌ {e}")
    except Exception as e:
        st.error(f"❌ 处理失败: {e}")


@st.cache_resource(show_spinner="加载音频…")
def _load_audio(mp3_path: str):
    from pydub import AudioSegment
    return AudioSegment.from_file(mp3_path)


def _slow_down(clip, tempo: float = 0.75):
    """用 ffmpeg atempo 做变速不变调的慢速处理。"""
    if not FFMPEG:
        raise RuntimeError("未找到 ffmpeg，无法生成慢速音频。")
    with tempfile.TemporaryDirectory() as td:
        fin = os.path.join(td, "in.wav")
        fout = os.path.join(td, "out.wav")
        clip.export(fin, format="wav")
        subprocess.run(
            [str(FFMPEG), "-y", "-loglevel", "error", "-i", fin, "-filter:a", f"atempo={tempo}", fout],
            check=True, capture_output=True,
        )
        from pydub import AudioSegment
        return AudioSegment.from_file(fout)


def _to_wav(raw: bytes) -> bytes:
    """
    浏览器麦克风录音实际是 WebM/Opus（Chrome 的 MediaRecorder 首选格式，
    文件名虽叫 .wav 但内容不是），统一用 ffmpeg 转成标准 WAV 供 st.audio 播放。
    转换失败时原样返回（尽力而为）。
    """
    if not FFMPEG:
        return raw
    with tempfile.TemporaryDirectory() as td:
        fin = os.path.join(td, "in.bin")      # ffmpeg 按内容探测格式，扩展名无关
        fout = os.path.join(td, "out.wav")
        Path(fin).write_bytes(raw)
        try:
            subprocess.run([str(FFMPEG), "-y", "-loglevel", "error", "-i", fin, fout],
                           check=True, capture_output=True)
            out = Path(fout).read_bytes()
            return out if out else raw
        except Exception:
            return raw


def _get_slice(video: dict, idx: int, variant: str) -> bytes:
    """按句切片并缓存。variant: normal / slow"""
    key = f"slice_{idx}_{variant}"
    cache = st.session_state.setdefault("slices", {})
    if key not in cache:
        full = _load_audio(video["mp3_path"])
        seg = video["sentences"][idx]
        start_ms = max(int(seg["start"] * 1000), 0)
        end_ms = min(int(seg["end"] * 1000), len(full) - 1)
        clip = full[start_ms:end_ms]
        if variant == "slow":
            clip = _slow_down(clip)
        cache[key] = clip.export(format="wav").read()
    return cache[key]


def _render_player(video: dict, llm_api_key: str, llm_base_url: str, llm_model: str):
    st.success(f"**{video['title']}** ｜ 时长 {_fmt_t(video['duration'])} ｜ "
               f"共 {len(video['sentences'])} 句 ｜ ASR 设备: {video.get('asr_device', '?').upper()}")

    tab_listen, tab_ai = st.tabs(["🔊 逐句精听（回声训练）", "🤖 AI 助教分析"])

    # ---------- 逐句精听 ----------
    with tab_listen:
        col_audio, col_play = st.columns([3, 2])
        with col_audio:
            st.subheader("完整音频")
            with open(video["mp3_path"], "rb") as f:
                st.audio(f.read(), format="audio/mpeg")
        with col_play:
            st.subheader("正在播放")
            play = st.session_state.get("play")
            if play:
                idx, mode = play["index"], play["mode"]
                seg = video["sentences"][idx]
                st.markdown(f"**#{idx + 1}** [{_fmt_t(seg['start'])} - {_fmt_t(seg['end'])}] "
                            f"{_MODE_LABEL.get(mode, '')}")
                if _text_visible(idx):
                    st.markdown(f"> {seg['text']}")
                else:
                    st.caption(f"{_TEXT_MASK}\n\n（原文已隐藏，点列表中的 👁 查看）")
                if mode != "consumed":
                    data = _get_slice(video, idx, "slow" if mode == "slow" else "normal")
                    st.audio(data, format="audio/wav", autoplay=True, loop=(mode == "loop"))
                    # 一次性消费，避免其他交互触发的 rerun 反复自动播放
                    st.session_state["play"] = {"index": idx, "mode": "consumed"}
                st.caption("💡 跟读录音在每个句子的行内：点 🎤 录制，🔊 回放对比")
            else:
                st.info("点击下方任意句子的按钮开始训练：\n\n"
                        "▶ 播放 ｜ 🔁 循环 ｜ 🐢 慢速 ｜ 👁 显示原文 ｜ 🎤 录音跟读")

        st.divider()
        st.subheader("逐句列表")
        st.toggle("显示全部原文", key="show_all_text", value=False,
                  help="默认隐藏原文，先盲听；需要对照时再打开")
        for i, seg in enumerate(video["sentences"]):
            c_time, c_text, c_ctrl, c_mic = st.columns([1.0, 4.6, 2.0, 2.0])
            with c_time:
                st.caption(f"**#{i + 1}**\n{_fmt_t(seg['start'])}")
            with c_text:
                if _text_visible(i):
                    st.markdown(seg["text"])
                else:
                    st.markdown(_TEXT_MASK)
            with c_ctrl:
                b1, b2, b3, b4 = st.columns(4)
                if b1.button("▶", key=f"play_{i}", use_container_width=True, help="播放一次"):
                    st.session_state["play"] = {"index": i, "mode": "once"}
                    st.rerun()
                if b2.button("🔁", key=f"loop_{i}", use_container_width=True, help="单句循环（回声训练）"):
                    st.session_state["play"] = {"index": i, "mode": "loop"}
                    st.rerun()
                if b3.button("🐢", key=f"slow_{i}", use_container_width=True, help="0.75 倍慢速播放一次"):
                    st.session_state["play"] = {"index": i, "mode": "slow"}
                    st.rerun()
                if b4.button("👁", key=f"show_{i}", use_container_width=True, help="显示 / 隐藏本句原文"):
                    revealed = st.session_state.setdefault("revealed", set())
                    if i in revealed:
                        revealed.discard(i)
                    else:
                        revealed.add(i)
                    st.rerun()
            with c_mic:
                has_rec = st.session_state.get(f"rec_saved_{i}") is not None
                m1, m2 = st.columns(2)
                if m1.button("🎤", key=f"mic_{i}",
                             type="primary" if st.session_state.get("rec_active") == i else "secondary",
                             use_container_width=True, help="录音跟读（每次保存最新一条）"):
                    st.session_state["rec_active"] = None if st.session_state.get("rec_active") == i else i
                    st.rerun()
                # 有录音时按钮高亮（primary 变色），无录音时禁用
                if m2.button("🔊", key=f"playrec_{i}",
                             type="primary" if has_rec else "secondary",
                             disabled=not has_rec,
                             use_container_width=True,
                             help="播放我的最新录音" if has_rec else "暂无录音"):
                    st.session_state["play_mine"] = None if st.session_state.get("play_mine") == i else i
                    st.rerun()

            # 行内录音 / 回放扩展区
            if st.session_state.get("rec_active") == i or st.session_state.get("play_mine") == i:
                with st.container(border=True):
                    if st.session_state.get("rec_active") == i:
                        # 录音控件挂载期间把字节收割到 rec_saved_，避免收起后状态丢失
                        rec_new = st.audio_input(f"🎙️ 第 {i + 1} 句跟读录音（重新录制会覆盖上一条）",
                                                 key=f"rec_input_{i}")
                        if rec_new is not None:
                            raw = rec_new.getvalue()
                            if st.session_state.get(f"rec_raw_{i}") != raw:
                                st.session_state[f"rec_raw_{i}"] = raw
                                st.session_state[f"rec_saved_{i}"] = _to_wav(raw)
                    if st.session_state.get("play_mine") == i:
                        my_rec = st.session_state.get(f"rec_saved_{i}")
                        if my_rec is not None:
                            st.caption(f"🎧 第 {i + 1} 句我的最新录音（对比上方原声）")
                            st.audio(my_rec, format="audio/wav", autoplay=True)
                            # 播放完自动收起，避免后续交互反复自动播放
                            st.session_state["play_mine"] = None

        st.download_button(
            "⬇️ 导出完整转录 (JSON)",
            data=json.dumps(video, ensure_ascii=False, indent=2),
            file_name=f"{video['video_id']}_transcript.json",
            mime="application/json",
        )

    # ---------- AI 助教 ----------
    with tab_ai:
        st.subheader("AI 助教分析")
        st.caption(f"模型: {llm_model} ｜ Base URL: {llm_base_url or '(OpenAI 默认)'}")
        if st.button("✨ 生成分析", type="primary"):
            try:
                with st.spinner("AI 助教分析中…"):
                    full_text = " ".join(s["text"] for s in video["sentences"])
                    st.session_state["llm_result"] = analyze(full_text, llm_api_key, llm_base_url, llm_model)
            except ValueError as e:
                st.error(f"❌ {e}")

        result = st.session_state.get("llm_result")
        if result:
            st.markdown("#### 📝 一句话总结")
            st.info(result["summary"] or "（无）")

            st.markdown("#### 🌟 地道表达 / 动词词组")
            for item in result["expressions"]:
                st.markdown(
                    f"- **{item.get('phrase', '')}** — {item.get('meaning', '')}\n\n"
                    f"  > 例: {item.get('example', '')}"
                )

            st.markdown("#### 👄 连读 / 弱读难点")
            if result["pronunciation_tips"]:
                for item in result["pronunciation_tips"]:
                    st.markdown(f"- **{item.get('text', '')}** — {item.get('tip', '')}")
            else:
                st.write("（未识别到明显难点）")


def _render_history():
    entries = _load_library()
    st.divider()
    st.subheader(f"📚 历史记录（{len(entries)}）")

    query = st.text_input("搜索", placeholder="🔍 输入标题关键词过滤...",
                          label_visibility="collapsed").strip()
    if query:
        entries = [e for e in entries if query.lower() in (e.get("title") or "").lower()]

    if not entries:
        st.caption("暂无记录。处理过的视频会自动保存到这里，随时可以回顾。")
        return

    for e in entries:
        c_info, c_open, c_del = st.columns([6.6, 1.2, 1.2])
        with c_info:
            added = time.strftime("%Y-%m-%d %H:%M", time.localtime(e.get("added_at", 0)))
            st.markdown(f"**{e.get('title') or '(未知标题)'}**")
            st.caption(f"{added} ｜ {_fmt_t(e.get('duration') or 0)} ｜ "
                       f"{len(e.get('sentences') or [])} 句 ｜ 模型 {e.get('model', '?')}")
        with c_open:
            if st.button("📂 打开", key=f"open_{e['video_id']}", use_container_width=True):
                if Path(e.get("mp3_path", "")).exists():
                    _reset_play_state()
                    st.session_state["video"] = e
                    st.rerun()
                else:
                    st.toast("⚠️ 该记录的音频文件已丢失，请重新用链接处理", icon="⚠️")
        with c_del:
            if st.button("🗑 删除", key=f"del_{e['video_id']}", use_container_width=True,
                         help="删除记录及音频文件"):
                _delete_library_entry(e["video_id"])
                if st.session_state.get("video", {}).get("video_id") == e["video_id"]:
                    st.session_state.pop("video", None)
                st.rerun()


# ================= UI 主流程 =================
st.set_page_config(page_title="英语精听与回声训练助手", page_icon="🎧", layout="wide",
                   initial_sidebar_state="collapsed")

# ---------------- 侧边栏配置（默认收起，点左上角 » 展开） ----------------
with st.sidebar:
    st.header("⚙️ 设置")

    st.subheader("🌐 网络")
    st.caption("YouTube 下载自动使用系统代理，无需额外配置。"
               "若下载失败，请确认系统代理 / TUN 模式已开启。")
    use_hf_mirror = st.checkbox("使用 hf-mirror.com 镜像下载 Whisper 模型（国内推荐）", value=False)
    if use_hf_mirror:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        # hf-mirror 不代理 Xet CAS 存储，必须禁用 Xet 才能下载大文件
        os.environ["HF_HUB_DISABLE_XET"] = "1"
        try:
            import huggingface_hub.constants as _hfc
            _hfc.HF_ENDPOINT = "https://hf-mirror.com"
        except Exception:
            pass

    st.subheader("🤖 LLM 助教（OpenAI 兼容）")
    llm_base_url = st.text_input(
        "Base URL",
        value=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        placeholder="https://api.openai.com/v1 或 https://api.deepseek.com/v1",
    )
    llm_api_key = st.text_input("API Key", type="password", value=os.environ.get("OPENAI_API_KEY", ""))
    llm_model = st.text_input("模型名", value=os.environ.get("LLM_MODEL", "gpt-4o-mini"))

    st.subheader("🎙️ 识别模型")
    whisper_model = st.selectbox(
        "Whisper 模型",
        ["large-v3", "medium", "small", "base"],
        index=0,
        help="首次使用会自动下载模型（large-v3 约 3GB，精度最高）",
    )

if not FFMPEG:
    st.warning("⚠️ 未检测到 ffmpeg。请通过 conda 环境启动本应用（见 start.bat），否则音频处理将失败。")

st.title("🎧 英语母语级精听与回声训练助手")
st.caption("输入 YouTube 链接 → 自动下载音频 → 逐句精听 / 回声循环训练 → AI 助教总结地道表达与连读难点")

url = st.text_input("YouTube 视频链接", placeholder="https://www.youtube.com/watch?v=...")
if st.button("🚀 开始处理", type="primary", use_container_width=True):
    if not url.strip():
        st.error("请先输入 YouTube 链接。")
    elif not ("youtube.com" in url or "youtu.be" in url):
        st.error("请输入有效的 YouTube 链接。")
    else:
        _process(url.strip(), whisper_model)

if "video" in st.session_state:
    _render_player(st.session_state["video"],
                    llm_api_key=llm_api_key, llm_base_url=llm_base_url, llm_model=llm_model)

# 旧缓存迁移（每个会话执行一次）+ 历史记录
if "migrated" not in st.session_state:
    _migrate_old_cache()
    st.session_state["migrated"] = True
_render_history()
