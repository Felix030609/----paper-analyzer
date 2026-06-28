from __future__ import annotations

import os
from typing import Any

from openai import APITimeoutError, OpenAI


DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-flash"
DEFAULT_DEEPSEEK_TIMEOUT_SECONDS = 120


class MissingAPIKeyError(RuntimeError):
    pass


class DeepSeekCallError(RuntimeError):
    pass


class DeepSeekTimeoutError(DeepSeekCallError):
    pass


def _read_streamlit_secret(name: str) -> str | None:
    try:
        import streamlit as st

        value = st.secrets.get(name)
    except Exception:
        return None
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def _read_raw_deepseek_api_key() -> str | None:
    key = _read_streamlit_secret("DEEPSEEK_API_KEY")
    if key:
        return key

    key = os.getenv("DEEPSEEK_API_KEY")
    return key.strip() if key and key.strip() else None


def validate_deepseek_api_key(key: str | None) -> tuple[bool, str]:
    if not key:
        return False, "未配置 DeepSeek API Key，无法生成报告。"
    if not key.isascii() or any(char.isspace() for char in key):
        return False, "DEEPSEEK_API_KEY 格式不正确，请重新配置真实 API Key。"
    if not key.startswith("sk-"):
        return False, "DEEPSEEK_API_KEY 格式不正确，DeepSeek API Key 通常以 sk- 开头。"
    if len(key) < 20:
        return False, "DEEPSEEK_API_KEY 长度异常，请重新配置真实 API Key。"
    return True, "DeepSeek API 已配置。"


def get_deepseek_key_status() -> tuple[str, str]:
    key = _read_raw_deepseek_api_key()
    if not key:
        return "missing", "未配置 DeepSeek API Key，无法生成报告。"
    is_valid, message = validate_deepseek_api_key(key)
    if not is_valid:
        return "invalid", message
    return "configured", message


def get_deepseek_api_key() -> str | None:
    key = _read_raw_deepseek_api_key()
    is_valid, _ = validate_deepseek_api_key(key)
    return key if is_valid else None


def get_deepseek_model(model_name: str | None = None) -> str:
    if model_name and str(model_name).strip():
        return str(model_name).strip()

    model = _read_streamlit_secret("DEEPSEEK_MODEL")
    if model:
        return model

    model = os.getenv("DEEPSEEK_MODEL")
    if model and model.strip():
        return model.strip()
    return DEFAULT_DEEPSEEK_MODEL


def get_deepseek_client() -> OpenAI:
    raw_key = _read_raw_deepseek_api_key()
    is_valid, message = validate_deepseek_api_key(raw_key)
    if not is_valid:
        raise MissingAPIKeyError(message)
    api_key = raw_key
    return OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)


def call_deepseek_chat(
    messages: list[dict[str, str]],
    *,
    model_name: str | None = None,
    json_mode: bool = False,
    thinking: bool = True,
    reasoning_effort: str = "high",
    temperature: float = 0.2,
    max_tokens: int = 2200,
    timeout: int = DEFAULT_DEEPSEEK_TIMEOUT_SECONDS,
) -> str:
    kwargs: dict[str, Any] = {
        "model": get_deepseek_model(model_name),
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    if thinking:
        kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
        kwargs["reasoning_effort"] = reasoning_effort

    try:
        response = get_deepseek_client().chat.completions.create(**kwargs, timeout=timeout)
        content = response.choices[0].message.content
    except MissingAPIKeyError:
        raise
    except APITimeoutError as exc:
        raise DeepSeekTimeoutError(f"DeepSeek 请求超时（>{timeout} 秒）。") from exc
    except Exception as exc:
        if "timed out" in str(exc).lower() or "timeout" in str(exc).lower():
            raise DeepSeekTimeoutError(f"DeepSeek 请求超时（>{timeout} 秒）。") from exc
        raise DeepSeekCallError(f"DeepSeek 调用失败：{exc}") from exc

    if not content:
        raise DeepSeekCallError("DeepSeek 返回了空内容。")
    return content
