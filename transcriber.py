"""faster-whisper 语音识别模块：词级时间戳 -> 句子级切分，GPU 自动降级 CPU。"""
import os
import sys

_SENTENCE_END = (".", "!", "?", "。", "！", "？")
# 常见缩写：以句号结尾但不是句子结束
_ABBR = {"mr.", "mrs.", "ms.", "dr.", "prof.", "st.", "vs.", "etc.", "e.g.", "i.e.", "jr.", "sr."}


def _is_sentence_end(token: str) -> bool:
    """判断一个词是否为句子结尾（句号/问号/感叹号），排除缩写与首字母缩写（Mr. / J.）。"""
    if not token.endswith(_SENTENCE_END):
        return False
    if token.lower() in _ABBR:
        return False
    if len(token) == 2 and token[0].isalpha() and token.endswith("."):
        return False
    return True


def setup_cuda_libs():
    """
    Windows 下 CTranslate2 的 pip 包不自带 CUDA 运行库 DLL，
    尝试把 pip 安装的 nvidia-cublas / nvidia-cudnn 的 DLL 目录加入搜索路径。
    必须在第一次 import faster_whisper / ctranslate2 之前调用。
    任何失败都静默忽略（不影响 CPU 模式）。
    """
    if sys.platform != "win32":
        return
    import importlib.util
    dirs = []
    for pkg_name in ("nvidia.cudnn", "nvidia.cublas"):
        try:
            spec = importlib.util.find_spec(pkg_name)
            if spec and spec.submodule_search_locations:
                base = list(spec.submodule_search_locations)[0]
                for sub in ("bin", "lib"):
                    d = os.path.join(base, sub)
                    if os.path.isdir(d):
                        dirs.append(d)
        except Exception:
            continue
    for d in dirs:
        try:
            os.add_dll_directory(d)
        except Exception:
            pass
        os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")


def _flush(cur_words, out):
    """把当前累积的词列表收尾成一句，追加到 out。"""
    if not cur_words:
        return
    text = " ".join("".join(w.word for w in cur_words).split())
    if text and text not in ("...", ".", "!", "?"):
        out.append({
            "start": round(cur_words[0].start, 3),
            "end": round(cur_words[-1].end, 3),
            "text": text,
        })


class Transcriber:
    """Whisper 转写器。模型按 (model_size) 进程级缓存；GPU 不可用时自动降级 CPU。"""

    _model = None
    _model_key = None
    _force_cpu = False

    def __init__(self, model_size: str = "large-v3"):
        self.model_size = model_size
        self.device = "cpu"

    def _get_model(self):
        if Transcriber._model is not None and Transcriber._model_key == self.model_size:
            self.device = "cpu" if Transcriber._force_cpu else "cuda"
            return Transcriber._model

        from faster_whisper import WhisperModel

        attempts = [("cpu", "int8")] if Transcriber._force_cpu else \
                   [("cuda", "int8_float16"), ("cpu", "int8")]
        last_err = None
        for device, compute in attempts:
            try:
                Transcriber._model = WhisperModel(self.model_size, device=device, compute_type=compute)
                Transcriber._model_key = self.model_size
                self.device = device
                return Transcriber._model
            except Exception as e:
                last_err = e
        raise RuntimeError(f"Whisper 模型加载失败: {last_err}")

    def transcribe(self, audio_path: str, language: str = "en", progress_cb=None):
        """
        识别音频并切分为句子。

        返回:
            sentences: [{"start": float, "end": float, "text": str}, ...]
            info: faster-whisper 的 TranscriptionInfo
            device: 实际使用的设备 ("cuda" / "cpu")
        """
        for attempt in (1, 2):
            model = self._get_model()
            try:
                segments, info = model.transcribe(
                    audio_path,
                    language=language,
                    word_timestamps=True,
                    vad_filter=True,
                    beam_size=5,
                )
                total = float(info.duration) or 1.0
                sentences = self._to_sentences(segments, total, progress_cb)
                return sentences, info, self.device
            except Exception as e:
                # GPU 上首次推理才暴露的兼容性问题（如 Blackwell 内核缺失）-> 强制降级 CPU 重试
                if self.device == "cuda" and attempt == 1:
                    Transcriber._model = None
                    Transcriber._model_key = None
                    Transcriber._force_cpu = True
                    continue
                raise RuntimeError(f"语音识别失败: {e}") from e

    @staticmethod
    def _to_sentences(segments, total_duration, progress_cb=None):
        """将词级时间戳按句末标点合并为句子；无标点的长段按停顿/时长兜底切分。"""
        words = []
        for seg in segments:
            if progress_cb is not None and total_duration:
                try:
                    progress_cb(min(seg.end / total_duration, 1.0))
                except Exception:
                    pass
            words.extend(seg.words or [])

        sentences = []
        cur = []
        for w in words:
            # 仅作兜底：超长停顿（>1.5s）几乎必然是句子边界；
            # 句内逗号处的自然停顿一般 <1s，不做切分，保证句子只按句末标点划分
            if cur and (w.start - cur[-1].end) > 1.5:
                _flush(cur, sentences)
                cur = []
            cur.append(w)
            if _is_sentence_end(w.word.strip()):
                _flush(cur, sentences)
                cur = []
        _flush(cur, sentences)
        return sentences
