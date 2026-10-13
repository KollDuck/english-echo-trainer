"""LLM 助教分析模块：OpenAI 兼容 API，输出结构化 JSON 知识点。"""
import json
import re

from openai import OpenAI

SYSTEM_PROMPT = """你是一位资深英语听力与发音教练。请分析用户提供的英文音频转录文本，严格输出 JSON（不要 markdown 代码块标记），结构如下：
{
  "expressions": [
    {"phrase": "原文中的地道表达或动词词组", "meaning": "中文解释", "example": "一个简短英文例句"}
  ],
  "pronunciation_tips": [
    {"text": "原文中出现连读/弱读的片段", "tip": "用中文说明这里发生了什么连读(linking)/弱读(weak form)/失去爆破等，听力练习时该注意什么"}
  ],
  "summary": "全文内容的一句话中文总结"
}
要求：
1. expressions 提取 3-5 个最地道、最值得学习的表达或动词词组，phrase 必须来自原文；
2. pronunciation_tips 指出 2-4 处典型连读、弱读、缩读、失去爆破等发音现象，text 必须来自原文；
3. 所有解释用中文。"""

MAX_CHARS = 12000


def analyze(text: str, api_key: str, base_url: str, model: str) -> dict:
    """
    调用 LLM 分析转录文本。
    返回: {"expressions": [...], "pronunciation_tips": [...], "summary": str}
    出错抛出 ValueError（含中文提示）。
    """
    if not api_key:
        raise ValueError("请先在左侧边栏填写 LLM API Key。")
    text = (text or "").strip()
    if not text:
        raise ValueError("转录文本为空，无法分析。")

    truncated = len(text) > MAX_CHARS
    if truncated:
        text = text[:MAX_CHARS]

    client = OpenAI(api_key=api_key, base_url=(base_url or None))
    prefix = "以下是音频的英文转录（文本过长，已截断）:\n\n" if truncated else "以下是音频的英文转录:\n\n"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prefix + text},
    ]

    try:
        resp = client.chat.completions.create(
            model=model, messages=messages, temperature=0.3,
            response_format={"type": "json_object"},
        )
    except Exception:
        # 部分兼容端点不支持 response_format，降级为普通调用
        try:
            resp = client.chat.completions.create(model=model, messages=messages, temperature=0.3)
        except Exception as e:
            raise ValueError(f"LLM 调用失败（请检查 Base URL / API Key / 模型名）: {e}") from e

    return _parse_json(resp.choices[0].message.content or "")


def _parse_json(content: str) -> dict:
    """容错解析 LLM 返回的 JSON（剥掉可能的 ```json``` 包裹）。"""
    m = re.search(r"\{.*\}", content, re.S)
    raw = m.group(0) if m else content
    try:
        data = json.loads(raw)
    except Exception as e:
        raise ValueError(f"LLM 返回内容无法解析为 JSON:\n{content[:500]}") from e
    return {
        "expressions": data.get("expressions") or [],
        "pronunciation_tips": data.get("pronunciation_tips") or [],
        "summary": data.get("summary", ""),
    }
