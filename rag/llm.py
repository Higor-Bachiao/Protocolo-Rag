"""Cliente de LLM com três provedores: openai, gemini (endpoint compatível com OpenAI) e anthropic.

Configuração por variáveis de ambiente (veja .env.example):
  LLM_PROVIDER  openai | gemini | anthropic | eco
  LLM_MODEL     nome exato do modelo no provedor
  OPENAI_API_KEY / GEMINI_API_KEY / ANTHROPIC_API_KEY

O provedor "eco" não chama API nenhuma: devolve o primeiro trecho do contexto.
Serve só para testar o pipeline sem custo; nunca use os números dele no relatório.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
CACHE_DIR = Path(os.getenv("LLM_CACHE_DIR", ".cache_llm"))


@dataclass
class RespostaLLM:
    texto: str
    segundos: float
    tokens_entrada: int | None = None
    tokens_saida: int | None = None


class LLM:
    def __init__(self, provedor: str | None = None, modelo: str | None = None, temperatura: float = 0.0):
        self.provedor = (provedor or os.getenv("LLM_PROVIDER", "openai")).lower()
        self.modelo = modelo or os.getenv("LLM_MODEL", "")
        self.temperatura = temperatura
        if self.provedor != "eco" and not self.modelo:
            raise SystemExit("Defina LLM_MODEL no .env (nome exato do modelo do provedor).")

        if self.provedor == "openai":
            from openai import OpenAI
            self._cli = OpenAI()
        elif self.provedor == "gemini":
            from openai import OpenAI
            self._cli = OpenAI(api_key=os.environ["GEMINI_API_KEY"], base_url=GEMINI_BASE_URL)
        elif self.provedor == "anthropic":
            from anthropic import Anthropic
            self._cli = Anthropic()
        elif self.provedor != "eco":
            raise ValueError(f"provedor desconhecido: {self.provedor}")

    def gerar(self, sistema: str, usuario: str, max_tokens: int = 600) -> RespostaLLM:
        """Chamada com cache em disco: rodar de novo não paga a API de novo.
        A latência guardada é a da chamada original, não a do cache."""
        chave = hashlib.sha256(
            json.dumps([self.provedor, self.modelo, self.temperatura, sistema, usuario, max_tokens]).encode()
        ).hexdigest()
        arq = CACHE_DIR / f"{chave}.json"
        if self.provedor != "eco" and arq.exists():
            return RespostaLLM(**json.loads(arq.read_text(encoding="utf-8")))
        r = self._gerar(sistema, usuario, max_tokens)
        if self.provedor != "eco":
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            arq.write_text(json.dumps(asdict(r), ensure_ascii=False), encoding="utf-8")
        return r

    def _gerar(self, sistema: str, usuario: str, max_tokens: int) -> RespostaLLM:
        t0 = time.perf_counter()
        if self.provedor == "eco":
            trecho = usuario.split("[1]", 1)[-1].split("\n[2]", 1)[0].strip()
            return RespostaLLM(trecho[:400] or "Não encontrei essa informação nos documentos.", time.perf_counter() - t0)

        if self.provedor == "anthropic":
            r = self._cli.messages.create(
                model=self.modelo, system=sistema, max_tokens=max_tokens, temperature=self.temperatura,
                messages=[{"role": "user", "content": usuario}],
            )
            texto = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
            return RespostaLLM(texto, time.perf_counter() - t0, r.usage.input_tokens, r.usage.output_tokens)

        r = self._cli.chat.completions.create(
            model=self.modelo, temperature=self.temperatura, max_tokens=max_tokens,
            messages=[{"role": "system", "content": sistema}, {"role": "user", "content": usuario}],
        )
        u = getattr(r, "usage", None)
        return RespostaLLM(
            r.choices[0].message.content or "", time.perf_counter() - t0,
            getattr(u, "prompt_tokens", None), getattr(u, "completion_tokens", None),
        )
