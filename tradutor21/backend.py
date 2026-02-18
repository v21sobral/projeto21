#!/usr/bin/env python3
"""
PDF Translator Pro — Backend FastAPI
Roda localmente e serve a tradução direta no navegador.

Instalação:
    pip install fastapi uvicorn python-multipart pypdf pdfplumber \
                pdf2image pytesseract reportlab deep-translator tqdm pillow

Execução:
    python backend.py

Acesse: http://localhost:8000
"""

import os
import uuid
import shutil
import asyncio
import logging
import tempfile
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

# ── FastAPI ──────────────────────────────────────────────────────────────────
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# ── Translator core (mesmo arquivo que já temos) ─────────────────────────────
# Importa as classes do pdf_translator.py que está na mesma pasta
import sys
sys.path.insert(0, str(Path(__file__).parent))

try:
    from pdf_translator import TranslationJob, PDFTranslator
    TRANSLATOR_AVAILABLE = True
except ImportError as e:
    TRANSLATOR_AVAILABLE = False
    print(f"[AVISO] pdf_translator.py não encontrado: {e}")
    print("        Coloque pdf_translator.py na mesma pasta que backend.py")

# ── Config ────────────────────────────────────────────────────────────────────
WORK_DIR = Path(tempfile.gettempdir()) / "pdf_translator_jobs"
WORK_DIR.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ── Lifespan para limpeza de jobs ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    yield
    # Shutdown
    import time
    for d in WORK_DIR.iterdir():
        if d.is_dir():
            age = time.time() - d.stat().st_mtime
            if age > 7200:
                shutil.rmtree(d, ignore_errors=True)


app = FastAPI(title="PDF Translator Pro", version="1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Jobs em memória ───────────────────────────────────────────────────────────
jobs: dict[str, dict] = {}


# ── Rota principal: serve o index.html ───────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def root():
    html_path = Path(__file__).parent / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>index.html não encontrado</h1>", status_code=404)


# ── Upload + início de tradução ───────────────────────────────────────────────
@app.post("/translate")
async def start_translation(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    source_lang: str = Form("auto"),
    target_lang: str = Form("pt"),
    engine: str = Form("google"),
    workers: int = Form(4),
    page_start: int = Form(1),
    page_end: Optional[str] = Form(None),
    use_ocr: bool = Form(True),
    save_original_pdf: bool = Form(False),
    resume: bool = Form(False),
):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(400, "Apenas arquivos PDF são aceitos.")

    if not TRANSLATOR_AVAILABLE:
        raise HTTPException(500, "pdf_translator.py não encontrado na mesma pasta.")

    # Cria pasta do job
    job_id = str(uuid.uuid4())
    job_dir = WORK_DIR / job_id
    job_dir.mkdir()

    # Salva PDF enviado
    input_path = job_dir / file.filename
    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    stem = Path(file.filename).stem
    output_path = job_dir / f"{stem}_traduzido_{target_lang}.pdf"

    end_page = None
    if page_end and page_end.strip().isdigit():
        end_page = int(page_end)

    job = TranslationJob(
        input_path=str(input_path),
        output_path=str(output_path),
        source_lang=source_lang,
        target_lang=target_lang,
        engine=engine,
        workers=workers,
        start_page=page_start,
        end_page=end_page,
        use_ocr=use_ocr,
        save_original_pdf=save_original_pdf,
        resume=resume,
        cache_file=str(job_dir / ".cache.json"),
    )

    jobs[job_id] = {
        "status": "running",
        "progress": 0,
        "total": 0,
        "log": [],
        "output_path": str(output_path),
        "output_name": output_path.name,
    }

    background_tasks.add_task(_run_translation, job_id, job)

    return JSONResponse({"job_id": job_id})


# ── Status do job ─────────────────────────────────────────────────────────────
@app.get("/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job não encontrado.")
    return JSONResponse(jobs[job_id])


# ── Download do PDF traduzido ─────────────────────────────────────────────────
@app.get("/download/{job_id}")
async def download(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job não encontrado.")
    info = jobs[job_id]
    if info["status"] != "done":
        raise HTTPException(400, "Tradução ainda não concluída.")
    path = Path(info["output_path"])
    if not path.exists():
        raise HTTPException(404, "Arquivo de saída não encontrado.")
    return FileResponse(
        path=str(path),
        media_type="application/pdf",
        filename=info["output_name"],
    )


# ── Worker assíncrono ─────────────────────────────────────────────────────────
async def _run_translation(job_id: str, job: TranslationJob):
    info = jobs[job_id]
    try:
        log.info(f"[{job_id}] Iniciando tradução: {job.input_path}")
        info["log"].append("🔄 Iniciando tradução…")

        # Conta total de páginas para progresso
        try:
            from pypdf import PdfReader
            reader = PdfReader(job.input_path)
            total = len(reader.pages)
            start = job.start_page
            end = job.end_page or total
            info["total"] = end - start + 1
        except Exception:
            info["total"] = 1

        # Roda a tradução em thread separada (é síncrono/CPU bound)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _sync_translate, job_id, job)

    except Exception as e:
        log.error(f"[{job_id}] Erro: {e}", exc_info=True)
        info["status"] = "error"
        info["error"] = str(e)
        info["log"].append(f"❌ Erro: {e}")


def _sync_translate(job_id: str, job: TranslationJob):
    """Roda no executor de threads — atualiza progresso via patch no cache."""
    info = jobs[job_id]

    # Monkey-patch para capturar progresso via cache
    original_put = None
    try:
        translator = PDFTranslator(job)

        # Intercepta o cache.put para atualizar progresso
        original_put = translator.cache.put

        def patched_put(result):
            original_put(result)
            info["progress"] = len(translator.cache.data)
            status = "ok" if not result.error else "err"
            ocr = " [OCR]" if result.used_ocr else ""
            info["log"].append(
                f"{'✅' if not result.error else '❌'} Página {result.page_num}{ocr} traduzida"
            )
            # Mantém log com máximo de 200 entradas
            if len(info["log"]) > 200:
                info["log"] = info["log"][-200:]

        translator.cache.put = patched_put

        translator.run()

        info["status"] = "done"
        info["progress"] = info["total"]
        info["log"].append("✅ Tradução concluída!")
        log.info(f"[{job_id}] Concluído → {job.output_path}")

    except Exception as e:
        info["status"] = "error"
        info["error"] = str(e)
        info["log"].append(f"❌ Erro: {e}")
        log.error(f"[{job_id}] Falha: {e}", exc_info=True)


# ── Entrypoint ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("backend:app", host="0.0.0.0", port=port, reload=False)
