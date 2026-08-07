"""LLM 供应商抽象层（策略模式）

BaseLLMProvider 定义统一接口；ZhipuProvider / DeepSeekProvider 为两个具体实现；
FallbackLLMProvider 实现"主备降级"策略。所有 invoke 统一走
"超时重试 + 指数退避"（tenacity），主供应商重试耗尽后自动切换备用。
"""
import logging
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Union

import tenacity

from config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    LLM_MAX_RETRIES,
    LLM_PROVIDER,
    LLM_TIMEOUT,
    ZHIPU_API_KEY,
    ZHIPU_BASE_URL,
)

logger = logging.getLogger("policy_qa.llm")


class LLMProviderError(RuntimeError):
    """LLM 调用在重试耗尽后仍失败（供上层决定降级或终止）"""


try:
    from langchain_zhipu import ChatZhipuAI as _ChatZhipuAI

    class ZhipuJudgeChatZhipuAI(_ChatZhipuAI):
        """评测（judge）用智谱模型：ragas 会把 temperature 设为 1/n（长小数），
        而智谱 API 限制小数点后 2 位，这里统一截断。"""

        def get_model_kwargs(self):
            kwargs = super().get_model_kwargs()
            temp = kwargs.get("temperature")
            if temp is not None:
                kwargs["temperature"] = round(float(temp), 2)
            return kwargs

except ImportError:
    ZhipuJudgeChatZhipuAI = None


class BaseLLMProvider(ABC):
    """LLM 供应商统一接口"""

    name: str = "base"

    def __init__(self, max_retries: int = LLM_MAX_RETRIES):
        self.max_retries = max_retries

    @abstractmethod
    def _invoke_once(self, prompt: Union[str, List[Any]], **kwargs) -> Any:
        """单次调用模型（子类实现，不负责重试）"""

    @property
    @abstractmethod
    def llm(self) -> Any:
        """返回 LangChain 兼容模型实例（供检索链构建使用）"""

    def judge_llm(self) -> Any:
        """评测（judge）用模型实例；默认与 llm 相同，子类可按需覆盖"""
        return self.llm

    def invoke(self, prompt: Union[str, List[Any]], **kwargs) -> Any:
        """调用模型并返回响应（统一内置超时重试 + 指数退避）"""
        return self._call_with_retry(self._invoke_once, prompt=prompt, **kwargs)

    def _call_with_retry(self, func, **kwargs) -> Any:
        """超时重试 + 指数退避：1s -> 2s -> 4s ... 至多 max_retries 次"""

        @tenacity.retry(
            stop=tenacity.stop_after_attempt(self.max_retries),
            wait=tenacity.wait_exponential(multiplier=1, min=1, max=10),
            retry=tenacity.retry_if_exception_type(
                (TimeoutError, ConnectionError, OSError)
            ),
            reraise=True,
            before_sleep=tenacity.before_sleep_log(logger, logging.WARNING),
        )
        def _retried():
            return func(**kwargs)

        try:
            return _retried()
        except Exception as e:
            raise LLMProviderError(
                f"{self.name} 调用失败（重试 {self.max_retries - 1} 次后仍失败）: {e}"
            ) from e


class ZhipuProvider(BaseLLMProvider):
    """智谱 GLM-4-flash（主供应商）"""

    name = "zhipu"

    def __init__(
        self,
        api_key: str = ZHIPU_API_KEY,
        model: str = "glm-4-flash",
        base_url: str = ZHIPU_BASE_URL,
        timeout: int = LLM_TIMEOUT,
        max_retries: int = LLM_MAX_RETRIES,
    ):
        super().__init__(max_retries=max_retries)
        self.api_key = api_key
        self.model = model
        # langchain_zhipu 4.1.x 的 _ask_remote 会把 "api/paas/v4/chat/completions"
        # 拼到 base_url 之后，因此 base_url 必须是裸域名；兼容旧的带后缀配置
        self.base_url = _normalize_zhipu_base_url(base_url)
        self.timeout = timeout
        if not api_key:
            raise LLMProviderError("缺少 ZHIPU_API_KEY")
        from langchain_zhipu import ChatZhipuAI

        self._chat = ChatZhipuAI(api_key=api_key, model=model, base_url=self.base_url)

    @property
    def llm(self) -> Any:
        return self._chat

    def judge_llm(self) -> Any:
        if ZhipuJudgeChatZhipuAI is None:
            return self._chat
        return ZhipuJudgeChatZhipuAI(
            api_key=self.api_key,
            model=self.model,
            base_url=self.base_url,
        )

    def _invoke_once(self, prompt: Union[str, List[Any]], **kwargs) -> Any:
        return self._chat.invoke(prompt, **kwargs)


class DeepSeekProvider(BaseLLMProvider):
    """DeepSeek（备用供应商，OpenAI 兼容接口）"""

    name = "deepseek"

    def __init__(
        self,
        api_key: str = DEEPSEEK_API_KEY,
        model: str = DEEPSEEK_MODEL,
        base_url: str = DEEPSEEK_BASE_URL,
        timeout: int = LLM_TIMEOUT,
        max_retries: int = LLM_MAX_RETRIES,
    ):
        super().__init__(max_retries=max_retries)
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        if not api_key:
            raise LLMProviderError("缺少 DEEPSEEK_API_KEY（备用供应商不可用）")
        from langchain_openai import ChatOpenAI

        self._chat = ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=0,  # 重试由本层指数退避策略接管
        )

    @property
    def llm(self) -> Any:
        return self._chat

    def _invoke_once(self, prompt: Union[str, List[Any]], **kwargs) -> Any:
        return self._chat.invoke(prompt, **kwargs)


class FallbackLLMProvider(BaseLLMProvider):
    """主备降级链：主供应商重试耗尽后切换到备用"""

    def __init__(self, primary: BaseLLMProvider, fallback: BaseLLMProvider):
        super().__init__(max_retries=primary.max_retries)
        self.primary = primary
        self.fallback = fallback
        self.name = f"fallback({primary.name}->{fallback.name})"

    @property
    def llm(self) -> Any:
        # 检索链使用主供应商的 LangChain 模型实例
        return self.primary.llm

    def judge_llm(self) -> Any:
        return self.primary.judge_llm()

    def _invoke_once(self, prompt: Union[str, List[Any]], **kwargs) -> Any:
        # Fallback 层整体覆写 invoke，此处仅满足抽象接口
        return self.primary._invoke_once(prompt, **kwargs)

    def invoke(self, prompt: Union[str, List[Any]], **kwargs) -> Any:
        try:
            return self.primary.invoke(prompt, **kwargs)
        except LLMProviderError as e:
            logger.error(
                f"主供应商 {self.primary.name} 失败，切换到备用 {self.fallback.name}: {e}"
            )
            return self.fallback.invoke(prompt, **kwargs)


_llm_provider: Optional[BaseLLMProvider] = None


def _normalize_zhipu_base_url(base_url: str) -> str:
    """兼容新旧两种 ZHIPU_BASE_URL 配置（裸域名 或 .../api/paas/v4 后缀）"""
    url = (base_url or "").strip().rstrip("/")
    suffix = "/api/paas/v4"
    if url.endswith(suffix):
        url = url[: -len(suffix)]
    return url or "https://open.bigmodel.cn"


def get_llm_provider() -> BaseLLMProvider:
    """返回全局 LLM 供应商（主备降级链），惰性构建并缓存"""
    global _llm_provider
    if _llm_provider is not None:
        return _llm_provider

    if LLM_PROVIDER == "deepseek":
        primary: BaseLLMProvider = DeepSeekProvider()
        try:
            fallback: Optional[BaseLLMProvider] = ZhipuProvider()
        except LLMProviderError:
            fallback = None
    else:
        primary = ZhipuProvider()
        try:
            fallback = DeepSeekProvider()
        except LLMProviderError:
            fallback = None

    _llm_provider = FallbackLLMProvider(primary, fallback) if fallback else primary
    logger.info(f"LLM 供应商初始化: {_llm_provider.name}")
    return _llm_provider
