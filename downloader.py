"""YouTube 音频下载模块：yt-dlp 封装，含错误映射与音频转码。"""
import re
from pathlib import Path

import yt_dlp
from pydub import AudioSegment


class DownloadError(Exception):
    """附带用户友好中文提示的下载异常"""


# 已知错误 -> 中文提示
_FRIENDLY_ERRORS = [
    (re.compile(r"private video", re.I), "该视频是私有视频，无法下载。"),
    (re.compile(r"sign in to confirm your age", re.I), "该视频需要登录确认年龄，无法下载。"),
    (re.compile(r"members[- ]only", re.I), "该视频是会员专属内容，无法下载。"),
    (re.compile(r"video unavailable", re.I), "视频不可用（可能已被删除或受地区限制）。"),
    (re.compile(r"not a valid url", re.I), "请输入有效的 YouTube 链接。"),
    (re.compile(r"unsupported url", re.I), "仅支持 YouTube 视频链接。"),
    (re.compile(r"live event will begin", re.I), "该视频是正在等待的直播，暂不支持。"),
    (re.compile(r"is live", re.I), "该视频是正在进行的直播，暂不支持。"),
    (re.compile(r"copyright", re.I), "该视频因版权原因不可用。"),
]


def _friendly_message(raw: str) -> str:
    for pat, msg in _FRIENDLY_ERRORS:
        if pat.search(raw):
            return msg
    low = raw.lower()
    if any(k in low for k in ("timed out", "getaddrinfo failed", "unable to download",
                              "connection", "reset by peer", "resolve", "proxy")):
        return "网络连接失败。请检查网络，并确认系统代理 / TUN 模式已开启且能访问 YouTube。"
    return f"下载失败: {raw[:300]}"


def download_audio(url: str, out_dir) -> dict:
    """
    下载 YouTube 视频的最佳音频并转码。

    返回:
        {
            "video_id": str,
            "title": str,
            "duration": float,   # 秒
            "mp3_path": str,     # 原始音质 mp3，用于播放与逐句切片
            "wav_path": str,     # 16kHz 单声道 wav，用于 Whisper 识别
        }
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    opts = {
        "format": "bestaudio/best",
        "outtmpl": str(out_dir / "%(id)s.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
    }
    # 不手动配置代理：yt-dlp 在 Windows 上会自动读取系统代理设置

    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
    except Exception as e:
        raise DownloadError(_friendly_message(str(e))) from e

    video_id = info["id"]
    title = info.get("title", video_id)
    duration = float(info.get("duration") or 0)
    mp3_path = out_dir / f"{video_id}.mp3"
    if not mp3_path.exists():
        # 极少情况下后处理器保留了原容器，做兜底查找
        candidates = [p for p in out_dir.glob(f"{video_id}.*")
                      if p.suffix.lower() in (".mp3", ".m4a", ".webm", ".opus", ".ogg", ".wav")]
        if not candidates:
            raise DownloadError("音频下载后未找到输出文件，请重试。")
        mp3_path = candidates[0]

    # 生成 16kHz 单声道 wav 供 Whisper 使用（与 mp3 时间轴一致）
    wav_path = out_dir / f"{video_id}_16k.wav"
    if not wav_path.exists():
        try:
            (AudioSegment.from_file(str(mp3_path))
             .set_frame_rate(16000)
             .set_channels(1)
             .export(str(wav_path), format="wav"))
        except Exception as e:
            raise DownloadError(f"音频转码失败（ffmpeg 异常）: {e}") from e

    return {
        "video_id": video_id,
        "title": title,
        "duration": duration,
        "mp3_path": str(mp3_path),
        "wav_path": str(wav_path),
    }
