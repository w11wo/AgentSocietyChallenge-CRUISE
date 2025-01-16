from typing import List, Dict, Optional, Union
import requests
import logging

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from langchain_core.embeddings import Embeddings

from websocietysimulator.llm import LLMBase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("websocietysimulator")


class vLLMEmbeddings(Embeddings):
    def __init__(
        self,
        api_key: str,
        model: str = "BAAI/bge-m3",
        infinity_api_url: str = "https://pd-wilso-vllm-embed-c50233c2459e405ba911f0861edd8b7c.nvidia-oci.saturnenterprise.io/v1",
    ):
        self.api_key = api_key
        self.model = model
        self.infinity_api_url = infinity_api_url

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=10, max=60),  # 等待时间从10秒开始，指数增长，最长60秒
        stop=stop_after_attempt(5),  # 最多重试5次
    )
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents into vectors"""
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}

        payload = {"model": self.model, "input": texts}

        response = requests.post(f"{self.infinity_api_url}/embeddings", headers=headers, json=payload)

        if response.status_code == 200:
            return [data["embedding"] for data in response.json()["data"]]
        else:
            raise ValueError(f"API call failed: {response.text}")

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text into a vector"""
        embeddings = self.embed_documents([text])
        return embeddings[0]


class vLLM(LLMBase):
    def __init__(self, api_key: str, model: str = "Qwen/Qwen2.5-72B-Instruct-AWQ"):
        super().__init__(model)
        self.client = OpenAI(
            base_url="https://pd-wilso-vllm-api-ec0de7122faa4abdac4d5c48074b9870.nvidia-oci.saturnenterprise.io/v1/",
            api_key=api_key,
        )
        self.embedding_model = vLLMEmbeddings(api_key=api_key)

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=10, max=300),
        stop=stop_after_attempt(10),
    )
    def __call__(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 500,
        stop_strs: Optional[List[str]] = None,
        n: int = 1,
    ) -> Union[str, List[str]]:
        try:
            response = self.client.chat.completions.create(
                model=model or self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop_strs,
                n=n,
            )

            if n == 1:
                return response.choices[0].message.content
            else:
                return [choice.message.content for choice in response.choices]
        except Exception as e:
            logger.error(f"LLM Error: {e}")
            raise e

    def get_embedding_model(self):
        return self.embedding_model
