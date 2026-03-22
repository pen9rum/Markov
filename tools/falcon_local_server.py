"""
Falcon local OpenAI-compatible Responses API server.

Endpoints:
- GET  /health
- POST /v1/responses

Environment variables:
- FALCON_LOCAL_MODEL        (default: tiiuae/Falcon-H1-7B-Instruct)
- FALCON_DTYPE              (default: bfloat16)
- FALCON_MAX_NEW_TOKENS     (default: 128)
- FALCON_LOCAL_HOST         (default: 127.0.0.1)
- FALCON_LOCAL_PORT         (default: 8080)

Optional:
- FALCON_INPUT_MAX_LENGTH   (default: 0, meaning no manual truncation)
    If > 0, tokenizer will truncate input to this length.

Usage:
- GPU mode (default): python tools/falcon_local_server.py
- CPU mode:           python tools/falcon_local_server.py --cpu
"""

from __future__ import annotations

import os
import sys
import time
import threading
import traceback
from typing import Any, Dict, Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer


if "--cpu" in sys.argv:
    DEVICE = "cpu"
else:
    DEVICE = "cuda"

MODEL_DEFAULT = os.getenv("FALCON_LOCAL_MODEL", "tiiuae/Falcon-H1-7B-Instruct")
DTYPE_NAME = os.getenv("FALCON_DTYPE", "bfloat16").lower()
MAX_NEW_TOKENS = int(os.getenv("FALCON_MAX_NEW_TOKENS", "128"))
INPUT_MAX_LENGTH = int(os.getenv("FALCON_INPUT_MAX_LENGTH", "0"))


def _resolve_dtype(name: str):
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


class LocalModelRuntime:
    def __init__(self) -> None:
        self._tokenizer = None
        self._model = None
        self._loaded_model_name: Optional[str] = None
        self._lock = threading.Lock()

    def _load(self, model_name: str) -> None:
        if self._model is not None and self._loaded_model_name == model_name:
            return

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        model_kwargs: Dict[str, Any] = {
            "trust_remote_code": True,
            "torch_dtype": DTYPE,
        }

        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs,
        )

        if DEVICE == "cpu":
            self._model = self._model.to("cpu")
        else:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available, but DEVICE is set to cuda.")
            self._model = self._model.to("cuda")

        self._model.eval()
        self._loaded_model_name = model_name

    def generate(
        self,
        prompt: str,
        model_name: str,
        max_new_tokens: int,
    ) -> str:
        with self._lock:
            self._load(model_name)
            assert self._tokenizer is not None and self._model is not None

            messages = [{"role": "user", "content": prompt}]
            text = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            tokenizer_kwargs = {
                "return_tensors": "pt",
                "padding": True,
            }

            # 只有使用者明確要求時才做 truncation
            if INPUT_MAX_LENGTH > 0:
                tokenizer_kwargs["truncation"] = True
                tokenizer_kwargs["max_length"] = INPUT_MAX_LENGTH

            inputs = self._tokenizer([text], **tokenizer_kwargs)

            input_token_count = int(inputs["input_ids"].shape[1])
            total_requested = input_token_count + int(max_new_tokens)

            print("=" * 80)
            print(f"[Falcon] model={model_name}")
            print(f"[Falcon] device={DEVICE}, dtype={DTYPE_NAME}")
            print(f"[Falcon] input_tokens={input_token_count}")
            print(f"[Falcon] requested_max_new_tokens={max_new_tokens}")
            print(f"[Falcon] requested_total_tokens={total_requested}")
            if INPUT_MAX_LENGTH > 0:
                print(f"[Falcon] manual_input_truncation=ON (max_length={INPUT_MAX_LENGTH})")
            else:
                print("[Falcon] manual_input_truncation=OFF")
            print("=" * 80)

            if DEVICE == "cpu":
                inputs = {k: v.to("cpu") for k, v in inputs.items()}
            else:
                inputs = {k: v.to("cuda") for k, v in inputs.items()}

            with torch.no_grad():
                output_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self._tokenizer.eos_token_id,
                    use_cache=True,
                )

            gen_ids = output_ids[0][inputs["input_ids"].shape[1]:]
            return self._tokenizer.decode(gen_ids, skip_special_tokens=True)


app = FastAPI(title="Falcon Local Responses API", version="1.0.3")
runtime = LocalModelRuntime()


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "model_default": MODEL_DEFAULT,
        "device": DEVICE,
        "dtype": DTYPE_NAME,
        "loaded_model": runtime._loaded_model_name,
        "input_max_length": INPUT_MAX_LENGTH,
    }


@app.post("/v1/responses")
def responses(req: ResponsesRequest) -> Dict[str, Any]:
    if not req.input or not req.input.strip():
        raise HTTPException(status_code=400, detail="input must not be empty")

    model_name = req.model or MODEL_DEFAULT
    max_new_tokens = req.max_output_tokens or MAX_NEW_TOKENS

    started = time.time()
    try:
        text = runtime.generate(
            prompt=req.input,
            model_name=model_name,
            max_new_tokens=max_new_tokens,
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

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
            "input_max_length": INPUT_MAX_LENGTH,
        },
    }


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("FALCON_LOCAL_HOST", "127.0.0.1")
    port = int(os.getenv("FALCON_LOCAL_PORT", "8080"))
    uvicorn.run(app, host=host, port=port)