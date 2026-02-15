# TitanSDK (LMStudioTitan) Turkce Dokumantasyon

Bu dokuman, `TitanSDK.py` dosyasindaki `LMStudioTitan` sinifini (ve ilgili yardimci tipleri) uctan uca aciklar. Hedef: OpenAI-uyumlu `/v1/chat/completions` ve `/v1/embeddings` endpoint'lerine baglanabilen, tool-calling destekli, gercek streaming yapabilen ve Genkit tarzi "registry" yaklasimlarini (prompt/flow/retriever/evaluator) barindiran pratik bir Python SDK.

> Notlar
> - SDK, OpenAI uyumlu bir API semasini hedefler (LM Studio gibi).
> - Streaming + tool-calling birlikte calisir (SSE stream'inden `tool_calls` delta'lari toplanir).
> - Bu bir "tek dosya" SDK oldugu icin bazi gelismis alanlar (tam tracing exporter, dataset runner, kalici vector store vb.) minimal tutulur.

---

## 1) Hizli Baslangic

### 1.1) Kurulum

Bu repo icinde `TitanSDK.py` dosyasi dogrudan import edilerek kullanilir.

Gereksinimler:
 - Python 3.9+
 - `httpx`

Ornek:

```py
from TitanSDK import LMStudioTitan

titan = LMStudioTitan({
    "base_url": "http://localhost:1234/v1",
    "model": "local-model",
    "debug": True,
})
```

### 1.2) En Basit Soru-Cevap

```py
import asyncio
from TitanSDK import LMStudioTitan

async def demo():
    titan = LMStudioTitan({"debug": False})
    chat = titan.session("demo")
    result = await chat.ask("Merhaba! Kimsin?", {"stream": False})
    print(result["text"])

asyncio.run(demo())
```

### 1.3) Streaming (Gercek Zamanli Cikti)

```py
import asyncio
from TitanSDK import LMStudioTitan

async def demo():
    titan = LMStudioTitan({"debug": False})
    chat = titan.session("stream")

    async for chunk in chat.ask("Bana kisa bir hikaye yaz.", {"stream": True}):
        print(chunk, end="", flush=True)

    print()

asyncio.run(demo())
```

Bu kullanimda:
 - `chat.ask(...)` bir nesne dondurur.
 - Bu nesne hem `await` edilebilir (tum sonucu dict olarak verir), hem de `async for` ile stream edilebilir (parca parca `str` verir).

---

## 2) Temel Kavramlar

### 2.1) LMStudioTitan

SDK'nin ana giris noktasi `LMStudioTitan` sinifidir:
 - HTTP uzerinden model endpoint'lerine istek atar (`generate`, `embed`).
 - Tool registry tutar (`register_tool`).
 - Prompt/Flow/Retriever/Evaluator registry tutar.
 - Interceptor (hook) mekanizmasi sunar.

### 2.2) Session (Chat Oturumu)

`titan.session(session_id)` size bir chat oturumu verir:
 - `history` (mesaj gecmisi) tutar.
 - `ask()` ile soru sorarsiniz.
 - Tool-calling kullaniliyorsa, modelin istedigi tools calistirilir ve tool sonuc mesajlari `history` icine eklenir.

History formati OpenAI uyumludur:
 - `{"role": "system"|"user"|"assistant"|"tool", "content": "...", ...}`
 - Tool mesajlarinda ek alanlar: `tool_call_id`, `name`

---

## 3) Konfigurasyon (LMStudioTitan __init__)

`LMStudioTitan(config)` ile alinabilen alanlar:

 - `base_url` (str): Varsayilan `http://localhost:1234/v1`
 - `model` (str): Varsayilan `local-model`
 - `debug` (bool): Debug loglari

HTTP/Dayaniklilik:
 - `api_key` (str|None): Bearer token gerekiyorsa
 - `headers` (dict): Ek header'lar
 - `timeout` (httpx timeout veya None): Varsayilan `None` (sinirsiz)
 - `max_retries` (int): Varsayilan `2`
 - `retry_backoff_s` (float): Varsayilan `0.5` (exponential backoff tabani)

Ornek:

```py
titan = LMStudioTitan({
    "base_url": "http://localhost:1234/v1",
    "model": "qwen2.5",
    "api_key": "xxx",             # gerekiyorsa
    "headers": {"X-App": "demo"}, # opsiyonel
    "timeout": 120.0,
    "max_retries": 3,
    "retry_backoff_s": 0.4,
    "debug": True,
})
```

---

## 4) Core LLM: generate()

### 4.1) Non-stream (tek seferde sonuc)

`await titan.generate(options)` bir dict dondurur:
 - `text`: modelin yaniti
 - `tool_calls`: model tool cagirdiysa list
 - `usage`: token kullanim bilgisi (varsa)
 - `latency`: ms cinsinden sure
 - `model`: server'in dondurdugu model adi

Kullanim:

```py
result = await titan.generate({
    "messages": [{"role": "user", "content": "2+2?"}],
    "temperature": 0.0,
})
print(result["text"])
```

### 4.2) Stream (AsyncIterator[str])

`await titan.generate({"stream": True, ...})` bir async iterator dondurur. Bu iterator:
 - Her parcayi `str` olarak `yield` eder (SSE `delta.content`)
 - Stream sirasinda `state` (internal) uzerinden tool_calls da biriktirilebilir

Genellikle `Session.ask(stream=True)` uzerinden kullanmaniz onerilir.

### 4.3) Retries

`generate()` transient HTTP status code'larda retry yapar:
 - 408, 409, 425, 429, 500, 502, 503, 504
Backoff: `retry_backoff_s * 2^(attempt-1)`

---

## 5) Session.ask(): Tool-calling + Streaming

`chat.ask(text, gen_options)` iki sekilde kullanilir:

1) Tam sonucu almak:

```py
resp = await chat.ask("Merhaba", {"stream": False})
print(resp["text"])
```

2) Streaming almak:

```py
async for chunk in chat.ask("Merhaba", {"stream": True}):
    print(chunk, end="")
```

### 5.1) Tool-calling akisi

Tool kullanimi icin `{"use_tools": True}` verilir.

Akis:
1. User mesaji `history`'e eklenir
2. Model cagirilir (`generate`)
3. Model `tool_calls` dondururse:
   - `assistant` mesajina `tool_calls` eklenir (content `None`)
   - Her tool call icin ilgili handler calisir
   - Tool sonucu `role="tool"` olarak `history`'e eklenir
4. Model tekrar cagirilir (bir sonraki adim)
5. Tool derinligi `max_tool_depth=5` ile sinirlidir

### 5.2) Streaming + Tool-calling birlikte nasil calisir?

Streaming modunda model cevabi parcali gelir. SDK:
 - `delta.content` parcalarini aninda yazar/yield eder
 - `delta.tool_calls` alanlarini parca parca biriktirir (arguments concat)

Stream bittikten sonra:
 - Eger `tool_calls` biriktiyse tools calistirilir ve yeni tur baslar
 - Eger tool yoksa stream'den biriken `text` asistan mesaji olarak history'ye eklenir

---

## 6) Tools (Fonksiyon Cagirma)

Tool kayit:

```py
def my_tool(args):
    return {"ok": True, "echo": args}

titan.register_tool(
    name="echo",
    description="Gelen argumanlari geri dondurur",
    parameters={"text": {"type": "string"}},
    fn=my_tool,
)
```

Handler sync veya async olabilir:

```py
async def slow_tool(args):
    await asyncio.sleep(0.1)
    return {"done": True}
```

Tool sonucu otomatik olarak `role="tool"` mesajina JSON string olarak eklenir.

---

## 7) Interceptors (Hook Mekanizmasi)

SDK birkac hook noktasi saglar:
 - `before_request`: payload olusmadan sonra, request oncesi
 - `after_response`: non-stream sonuc dict'i olustuktan sonra
 - `on_error`: hata yakalandiginda
 - `before_tool`: tool calismadan hemen once
 - `after_tool`: tool calistikten hemen sonra

Ekleme:

```py
def log_payload(payload):
    print("payload keys:", list(payload.keys()))

titan.add_interceptor("before_request", log_payload)
```

Tool hook ornegi:

```py
def tool_start(ctx):
    print("tool:", ctx["name"], "args:", ctx["args"])

titan.add_interceptor("before_tool", tool_start)
```

---

## 8) Prompts (Template Registry)

Prompt kaydetme:

```py
titan.register_prompt("welcome", "Merhaba {name}, bugun nasil yardim edeyim?")
text = titan.render_prompt("welcome", name="Arda")
```

Callable template de olabilir:

```py
def tmpl(**v):
    return f"Soru: {v['q']}\nCevap:"

titan.register_prompt("qa", tmpl)
```

---

## 9) Flows (Genkit benzeri is akisleri)

Flow decorator:

```py
@titan.flow("setup_vm")
async def setup_vm_flow(vm_name: str):
    chat = titan.session(f"flow:{vm_name}")
    return await chat.ask("VM hazir mi?", {"stream": False})

result = await titan.run_flow("setup_vm", "win10")
```

Bu katman:
 - Flow'lari isimle kaydedip cagirmayi kolaylastirir
 - Gelismis orkestrasyon icin temel bir iskelet saglar

---

## 10) Retrievers ve RAG

### 10.1) Document tipi

`Document`:
 - `content` (str): dokuman metni
 - `metadata` (dict): ek bilgiler (kaynak, url, sayfa, skor vb.)
 - `id` (str|None)
 - `embedding` (list[float]|None)

### 10.2) Retriever kaydi

Retriever: `fn(query, k, **kwargs) -> List[Document]` (sync veya async)

```py
from TitanSDK import Document

def simple_retriever(query: str, k: int = 5, **kwargs):
    return [
        Document(content="Python bir programlama dilidir.", metadata={"source": "note"}),
    ][:k]

titan.register_retriever("simple", simple_retriever)
docs = await titan.retrieve("Python nedir?", retriever="simple", k=3)
```

### 10.3) rag_answer() yardimcisi

```py
resp = await titan.rag_answer(
    "Python nedir?",
    retriever="simple",
    k=3,
    system="Kisa ve dogru yanit ver.",
    temperature=0.2,
)
print(resp["text"])
```

Bu helper:
 - Retriever'dan docs alir
 - Docs'u "Context" olarak modele verir
 - Yaniti `generate()` ile uretir

---

## 11) Evaluators

Evaluator kaydi:

```py
def length_eval(input, output, **kwargs):
    return {"length": len(str(output))}

titan.register_evaluator("len", length_eval)
score = await titan.evaluate("len", input="x", output="hello")
print(score)  # {"length": 5}
```

Evaluators genellikle:
 - kalibrasyon (uzunluk, format uyumu)
 - regressor/similarity (embedding ile)
 - policy checks
icin kullanilir.

---

## 12) Embeddings ve Similarity

Embedding:

```py
vecs = await titan.embed(["merhaba", "selam"])
```

Cosine benzerlik:

```py
sim = titan.cosine_similarity(vecs[0], vecs[1])
```

Bu, basit bir RAG / semantic search altyapisi icin temel saglar.

---

## 13) Vision

`vision(prompt, image_source, options)`:
 - `image_source` base64 data-uri ile baslayabilir (`data:image...`)
 - degilse `data:image/jpeg;base64,{image_source}` seklinde paketler

```py
resp = await titan.vision("Bu resimde ne var?", image_b64, {"stream": False})
print(resp["text"])
```

---

## 14) Hatalar ve Teshis

### 14.1) Hata siniflari

 - `TitanError`: taban
 - `TitanHTTPError`: HTTP status/bodysi ile
 - `TitanToolError`: tool tarafinda kullanilabilir (su an handler hatalari JSON icine yazilir)

### 14.2) Diagnostics

```py
print(titan.get_diagnostics())
```

Icerik:
 - toplam istek sayisi
 - toplam token
 - hata sayisi
 - uptime
 - aktif session sayisi
 - kayitli tools

---

## 15) Tasarim Notlari ve Sinirlar

Bu SDK, pratik bir cekirdek sunar; "tam Genkit seviyesine" cikarmak icin tipik eksikler:
 - Tracing/telemetry exporter (OTel spans, JSON trace export)
 - Dataset + eval runner (senaryolar, regression testleri)
 - Vector store / indexer (FAISS, sqlite, chroma vb.) entegrasyonlari
 - Structured output (JSON schema ile katı dogrulama)
 - Memory/checkpointing (kalici conversation store)
 - Plugin sistemi (paket/discovery)

Istersen bunlari moduler olarak ekleyebiliriz (tek dosyayi buyutmeden `titan/plugins/*` gibi).

---

## 16) main.py ile Ornek (Tools + Streaming)

Asagidaki desen, `main.py` icin tipik kullanimdir:

```py
async def main():
    chat = titan.session("test_vm_session")
    async for chunk in chat.ask(
        "VM icinde 'whoami' calistir ve sonucu yaz.",
        {"use_tools": True, "stream": True},
    ):
        print(chunk, end="", flush=True)
```

