"""
Falcon local OpenAI-compatible Responses API server (CPU-first).

Endpoints:
- GET  /health
- POST /v1/responses

Environment variables:
- FALCON_LOCAL_MODEL  (default: tiiuae/Falcon-H1-7B-Instruct)
- FALCON_DEVICE       (default: cpu)
- FALCON_DTYPE        (default: float32)
- FALCON_MAX_NEW_TOKENS (default: 512)
- FALCON_TEMPERATURE  (default: 0.2)
- FALCON_TOP_P        (default: 0.9)
"""

from __future__ import annotations

import os
import time
import threading
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


MODEL_DEFAULT = os.getenv("FALCON_LOCAL_MODEL", "tiiuae/Falcon-H1-7B-Instruct")
DEVICE = os.getenv("FALCON_DEVICE", "cpu")
DTYPE_NAME = os.getenv("FALCON_DTYPE", "float32").lower()
MAX_NEW_TOKENS = int(os.getenv("FALCON_MAX_NEW_TOKENS", "512"))
TEMPERATURE = float(os.getenv("FALCON_TEMPERATURE", "0.2"))
TOP_P = float(os.getenv("FALCON_TOP_P", "0.9"))


def _resolve_dtype(name: str):
    import torch

    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float32


DTYPE = _resolve_dtype(DTYPE_NAME)


class ResponsesRequest(BaseModel):
    model: Optional[str] = None
    input: str
    max_output_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None


class LocalModelRuntime:
    def __init__(self) -> None:
        self._tokenizer = None
        self._model = None
        self._loaded_model_name: Optional[str] = None
        self._lock = threading.Lock()

    def _load(self, model_name: str) -> None:
        if self._model is not None and self._loaded_model_name == model_name:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model_kwargs: Dict[str, Any] = {"trust_remote_code": True}

        if DEVICE == "cpu":
            model_kwargs["torch_dtype"] = DTYPE
            self._model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
            self._model = self._model.to("cpu")
        else:
            model_kwargs["torch_dtype"] = DTYPE
            model_kwargs["device_map"] = "auto"
            self._model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

        self._loaded_model_name = model_name

    def generate(self, prompt: str, model_name: str, max_new_tokens: int, temperature: float, top_p: float) -> str:
        with self._lock:
            self._load(model_name)
            assert self._tokenizer is not None and self._model is not None

            messages = [{"role": "user", "content": prompt}]
            text = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self._tokenizer([text], return_tensors="pt")

            if DEVICE == "cpu":
                inputs = {k: v.to("cpu") for k, v in inputs.items()}
            else:
                inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

            do_sample = temperature > 0
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=max(temperature, 1e-5),
                top_p=top_p,
            )
            gen_ids = output_ids[0][inputs["input_ids"].shape[1]:]
            return self._tokenizer.decode(gen_ids, skip_special_tokens=True)


app = FastAPI(title="Falcon Local Responses API", version="1.0.0")
runtime = LocalModelRuntime()


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "model_default": MODEL_DEFAULT,
        "device": DEVICE,
        "dtype": DTYPE_NAME,
        "loaded_model": runtime._loaded_model_name,
    }


@app.post("/v1/responses")
def responses(req: ResponsesRequest) -> Dict[str, Any]:
    if not req.input or not req.input.strip():
        raise HTTPException(status_code=400, detail="input must not be empty")

    model_name = req.model or MODEL_DEFAULT
    max_new_tokens = req.max_output_tokens or MAX_NEW_TOKENS
    temperature = TEMPERATURE if req.temperature is None else req.temperature
    top_p = TOP_P if req.top_p is None else req.top_p

    started = time.time()
    text = runtime.generate(
        prompt=req.input,
        model_name=model_name,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    elapsed = time.time() - started

    return {
        "id": f"resp_local_{int(time.time() * 1000)}",
        "object": "response",
        "created": int(time.time()),
        "model": model_name,
        "output_text": text,
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": text,
                    }
                ],
            }
        ],
        "usage": {
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
        },
        "meta": {
            "elapsed_seconds": elapsed,
            "device": DEVICE,
            "dtype": DTYPE_NAME,
        },
    }


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("FALCON_LOCAL_HOST", "127.0.0.1")
    port = int(os.getenv("FALCON_LOCAL_PORT", "8080"))
    uvicorn.run(app, host=host, port=port)
