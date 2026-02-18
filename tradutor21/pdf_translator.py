#!/usr/bin/env python3
"""
PDF Translator Pro - Tradutor de PDF com OCR
Suporta tradução de livros inteiros com milhares de páginas.

Dependências:
    pip install pypdf pdfplumber pdf2image pytesseract reportlab deep-translator tqdm pillow

Sistema:
    - Tesseract OCR: https://github.com/tesseract-ocr/tesseract
      Ubuntu/Debian: sudo apt install tesseract-ocr tesseract-ocr-por tesseract-ocr-eng
      macOS:         brew install tesseract
      Windows:       https://github.com/UB-Mannheim/tesseract/wiki

    - Poppler (para pdf2image):
      Ubuntu/Debian: sudo apt install poppler-utils
      macOS:         brew install poppler
      Windows:       https://github.com/oschwartz10612/poppler-windows
"""

import os
import sys
import time
import json
import logging
import argparse
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from typing import Optional

# ──────────────────────────────────────────────
# Verificação de dependências
# ──────────────────────────────────────────────
MISSING = []
try:
    from pypdf import PdfReader
except ImportError:
    MISSING.append("pypdf")

try:
    import pdfplumber
except ImportError:
    MISSING.append("pdfplumber")

try:
    from pdf2image import convert_from_path
    PDF2IMAGE_OK = True
except ImportError:
    PDF2IMAGE_OK = False
    MISSING.append("pdf2image")

try:
    import pytesseract
    from PIL import Image
    TESSERACT_OK = True
except ImportError:
    TESSERACT_OK = False
    MISSING.append("pytesseract / pillow")

try:
    from deep_translator import GoogleTranslator, DeeplTranslator, LibreTranslator
    TRANSLATOR_OK = True
except ImportError:
    TRANSLATOR_OK = False
    MISSING.append("deep-translator")

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, PageBreak, HRFlowable
    )
    from reportlab.lib import colors
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    REPORTLAB_OK = True
except ImportError:
    REPORTLAB_OK = False
    MISSING.append("reportlab")

try:
    from tqdm import tqdm
    TQDM_OK = True
except ImportError:
    TQDM_OK = False

if MISSING:
    print("⚠️  Dependências faltando. Instale com:\n")
    print(f"    pip install {' '.join(MISSING)}\n")
    if "pypdf" in MISSING or "deep-translator" in MISSING or "reportlab" in MISSING:
        sys.exit(1)

# ──────────────────────────────────────────────
# Configuração de logging
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("pdf_translator.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Idiomas suportados
# ──────────────────────────────────────────────
LANGUAGES = {
    "pt": "Português",
    "en": "English",
    "es": "Español",
    "fr": "Français",
    "de": "Deutsch",
    "it": "Italiano",
    "ja": "日本語",
    "zh-CN": "中文(简体)",
    "zh-TW": "中文(繁體)",
    "ko": "한국어",
    "ru": "Русский",
    "ar": "العربية",
    "hi": "हिन्दी",
    "nl": "Nederlands",
    "pl": "Polski",
    "sv": "Svenska",
    "tr": "Türkçe",
    "uk": "Українська",
}

TESSERACT_LANG_MAP = {
    "pt": "por",
    "en": "eng",
    "es": "spa",
    "fr": "fra",
    "de": "deu",
    "it": "ita",
    "ja": "jpn",
    "zh-CN": "chi_sim",
    "zh-TW": "chi_tra",
    "ko": "kor",
    "ru": "rus",
    "ar": "ara",
    "hi": "hin",
    "nl": "nld",
    "pl": "pol",
    "sv": "swe",
    "tr": "tur",
    "uk": "ukr",
}


# ──────────────────────────────────────────────
# Estruturas de dados
# ──────────────────────────────────────────────
@dataclass
class PageResult:
    page_num: int
    original_text: str = ""
    translated_text: str = ""
    used_ocr: bool = False
    error: Optional[str] = None
    char_count: int = 0


@dataclass
class TranslationJob:
    input_path: str
    output_path: str
    source_lang: str = "auto"
    target_lang: str = "pt"
    engine: str = "google"
    use_ocr: bool = True
    ocr_dpi: int = 300
    workers: int = 4
    chunk_size: int = 4500   # chars por chunk (Google limite ~5000)
    start_page: int = 1
    end_page: Optional[int] = None
    preserve_layout: bool = True
    save_original_pdf: bool = False
    resume: bool = True      # retomar tradução interrompida
    cache_file: str = ""

    def __post_init__(self):
        if not self.cache_file:
            stem = Path(self.input_path).stem
            self.cache_file = f".cache_{stem}_translation.json"


# ──────────────────────────────────────────────
# Cache de progresso (para retomar traduções)
# ──────────────────────────────────────────────
class TranslationCache:
    def __init__(self, cache_path: str):
        self.path = cache_path
        self._lock = threading.Lock()
        self.data: dict[int, dict] = {}
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                self.data = {int(k): v for k, v in raw.items()}
                log.info(f"Cache carregado: {len(self.data)} páginas já traduzidas")
            except Exception:
                self.data = {}

    def save(self):
        with self._lock:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)

    def has(self, page_num: int) -> bool:
        return page_num in self.data

    def get(self, page_num: int) -> Optional[PageResult]:
        if page_num in self.data:
            d = self.data[page_num]
            return PageResult(**d)
        return None

    def put(self, result: PageResult):
        with self._lock:
            self.data[result.page_num] = asdict(result)
        self.save()

    def clear(self):
        self.data = {}
        if os.path.exists(self.path):
            os.remove(self.path)


# ──────────────────────────────────────────────
# Extração de texto
# ──────────────────────────────────────────────
class TextExtractor:
    def __init__(self, pdf_path: str, job: TranslationJob):
        self.pdf_path = pdf_path
        self.job = job

    def extract_page(self, page_num: int) -> tuple[str, bool]:
        """Retorna (texto, usou_ocr)"""
        text = self._extract_digital(page_num)

        if len(text.strip()) < 50 and self.job.use_ocr and TESSERACT_OK and PDF2IMAGE_OK:
            log.debug(f"Página {page_num}: texto digital insuficiente, usando OCR")
            text = self._extract_ocr(page_num)
            return text, True

        return text, False

    def _extract_digital(self, page_num: int) -> str:
        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                if page_num - 1 >= len(pdf.pages):
                    return ""
                page = pdf.pages[page_num - 1]
                text = page.extract_text() or ""
                return text
        except Exception as e:
            log.warning(f"Erro extração digital pág {page_num}: {e}")
            return ""

    def _extract_ocr(self, page_num: int) -> str:
        try:
            images = convert_from_path(
                self.pdf_path,
                dpi=self.job.ocr_dpi,
                first_page=page_num,
                last_page=page_num,
            )
            if not images:
                return ""

            lang_code = TESSERACT_LANG_MAP.get(self.job.source_lang, "por+eng")
            if self.job.source_lang == "auto":
                lang_code = "por+eng"

            config = "--oem 3 --psm 6"
            text = pytesseract.image_to_string(images[0], lang=lang_code, config=config)
            return text
        except Exception as e:
            log.warning(f"Erro OCR pág {page_num}: {e}")
            return ""


# ──────────────────────────────────────────────
# Motor de tradução
# ──────────────────────────────────────────────
class TranslationEngine:
    def __init__(self, job: TranslationJob):
        self.job = job
        self._lock = threading.Lock()
        self._request_count = 0
        self._last_request = 0.0
        self._min_interval = 0.3  # segundos entre requests

    def _get_translator(self):
        engine = self.job.engine.lower()
        src = self.job.source_lang
        tgt = self.job.target_lang

        if engine == "google":
            return GoogleTranslator(source=src, target=tgt)
        elif engine == "deepl":
            api_key = os.environ.get("DEEPL_API_KEY", "")
            if not api_key:
                raise ValueError("DEEPL_API_KEY não definida. Export DEEPL_API_KEY=sua_chave")
            return DeeplTranslator(api_key=api_key, source=src, target=tgt)
        elif engine == "libre":
            url = os.environ.get("LIBRE_URL", "https://libretranslate.com")
            key = os.environ.get("LIBRE_KEY", "")
            return LibreTranslator(source=src, target=tgt, base_url=url, api_key=key)
        else:
            raise ValueError(f"Motor desconhecido: {engine}")

    def _rate_limit(self):
        with self._lock:
            elapsed = time.time() - self._last_request
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)
            self._last_request = time.time()
            self._request_count += 1

    def translate_text(self, text: str, retries: int = 3) -> str:
        if not text.strip():
            return text

        chunks = self._split_chunks(text, self.job.chunk_size)
        translated_chunks = []

        for chunk in chunks:
            translated = self._translate_chunk(chunk, retries)
            translated_chunks.append(translated)

        return "\n".join(translated_chunks)

    def _split_chunks(self, text: str, max_size: int) -> list[str]:
        """Divide texto em chunks por parágrafo, respeitando o limite de tamanho."""
        paragraphs = text.split("\n")
        chunks = []
        current = []
        current_len = 0

        for para in paragraphs:
            if current_len + len(para) + 1 > max_size and current:
                chunks.append("\n".join(current))
                current = []
                current_len = 0

            # Parágrafo maior que o limite? Divide por sentença
            if len(para) > max_size:
                sentences = para.replace(". ", ".|").replace("? ", "?|").replace("! ", "!|").split("|")
                for sent in sentences:
                    if current_len + len(sent) > max_size and current:
                        chunks.append("\n".join(current))
                        current = []
                        current_len = 0
                    current.append(sent)
                    current_len += len(sent) + 1
            else:
                current.append(para)
                current_len += len(para) + 1

        if current:
            chunks.append("\n".join(current))

        return [c for c in chunks if c.strip()]

    def _translate_chunk(self, chunk: str, retries: int) -> str:
        for attempt in range(retries):
            try:
                self._rate_limit()
                translator = self._get_translator()
                result = translator.translate(chunk)
                return result or chunk
            except Exception as e:
                wait = 2 ** attempt
                log.warning(f"Erro tradução (tentativa {attempt+1}/{retries}): {e}. Aguardando {wait}s...")
                time.sleep(wait)

        log.error("Falha na tradução após todas tentativas. Retornando original.")
        return chunk


# ──────────────────────────────────────────────
# Geração do PDF de saída
# ──────────────────────────────────────────────
class PDFWriter:
    def __init__(self, output_path: str, job: TranslationJob):
        self.output_path = output_path
        self.job = job

    def write(self, results: list[PageResult], metadata: dict):
        doc = SimpleDocTemplate(
            self.output_path,
            pagesize=A4,
            rightMargin=2 * cm,
            leftMargin=2 * cm,
            topMargin=2.5 * cm,
            bottomMargin=2.5 * cm,
            title=metadata.get("title", "Documento Traduzido"),
            author=f"PDF Translator Pro | {metadata.get('author', '')}",
            subject=f"Traduzido de {self.job.source_lang} para {self.job.target_lang}",
        )

        styles = getSampleStyleSheet()
        body_style = ParagraphStyle(
            "Body",
            parent=styles["Normal"],
            fontSize=11,
            leading=16,
            spaceAfter=8,
        )
        page_header_style = ParagraphStyle(
            "PageHeader",
            parent=styles["Normal"],
            fontSize=8,
            textColor=colors.grey,
            spaceAfter=4,
        )
        title_style = ParagraphStyle(
            "Title",
            parent=styles["Title"],
            fontSize=18,
            spaceAfter=20,
        )

        story = []

        # Capa
        story.append(Spacer(1, 3 * cm))
        title = metadata.get("title") or Path(self.job.input_path).stem
        story.append(Paragraph(self._esc(title), title_style))
        story.append(Spacer(1, 0.5 * cm))

        info = (
            f"Traduzido de <b>{LANGUAGES.get(self.job.source_lang, self.job.source_lang)}</b> "
            f"para <b>{LANGUAGES.get(self.job.target_lang, self.job.target_lang)}</b><br/>"
            f"Motor: {self.job.engine.upper()} | "
            f"Páginas: {len(results)} | "
            f"Gerado por PDF Translator Pro"
        )
        story.append(Paragraph(info, styles["Normal"]))
        story.append(PageBreak())

        # Páginas traduzidas
        for result in results:
            if result.error:
                story.append(
                    Paragraph(
                        f"[Página {result.page_num}: Erro na tradução - {result.error}]",
                        page_header_style,
                    )
                )
                story.append(PageBreak())
                continue

            # Cabeçalho da página
            ocr_tag = " [OCR]" if result.used_ocr else ""
            story.append(
                Paragraph(
                    f"── Página {result.page_num}{ocr_tag} ──",
                    page_header_style,
                )
            )
            story.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey))
            story.append(Spacer(1, 0.2 * cm))

            # Texto traduzido
            paragraphs = result.translated_text.split("\n")
            for para in paragraphs:
                para = para.strip()
                if para:
                    story.append(Paragraph(self._esc(para), body_style))

            story.append(PageBreak())

        doc.build(story)
        log.info(f"PDF gerado: {self.output_path}")

    @staticmethod
    def _esc(text: str) -> str:
        """Escapa caracteres especiais para ReportLab."""
        return (
            text.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
        )


# ──────────────────────────────────────────────
# Orquestrador principal
# ──────────────────────────────────────────────
class PDFTranslator:
    def __init__(self, job: TranslationJob):
        self.job = job
        self.cache = TranslationCache(job.cache_file)
        self.extractor = TextExtractor(job.input_path, job)
        self.engine = TranslationEngine(job)

    def run(self):
        start_time = time.time()
        log.info("=" * 60)
        log.info("  PDF Translator Pro")
        log.info("=" * 60)
        log.info(f"Arquivo:  {self.job.input_path}")
        log.info(f"Saída:    {self.job.output_path}")
        log.info(f"Idioma:   {self.job.source_lang} → {self.job.target_lang}")
        log.info(f"Motor:    {self.job.engine}")
        log.info(f"OCR:      {'Ativado' if self.job.use_ocr else 'Desativado'}")
        log.info(f"Workers:  {self.job.workers}")

        # Lê metadados e total de páginas
        reader = PdfReader(self.job.input_path)
        total_pages = len(reader.pages)
        meta = reader.metadata or {}
        metadata = {
            "title": meta.get("/Title", ""),
            "author": meta.get("/Author", ""),
        }

        start = max(1, self.job.start_page)
        end = min(total_pages, self.job.end_page or total_pages)
        pages_to_process = list(range(start, end + 1))

        log.info(f"Páginas:  {start}–{end} de {total_pages} total")
        log.info("=" * 60)

        if not self.job.resume:
            self.cache.clear()

        # Processa páginas
        results = self._process_pages(pages_to_process)

        # Ordena por número de página
        results.sort(key=lambda r: r.page_num)

        # Gera PDF
        if REPORTLAB_OK:
            writer = PDFWriter(self.job.output_path, self.job)
            writer.write(results, metadata)
        else:
            log.error("reportlab não instalado. Não é possível gerar PDF.")

        # Salva PDF com texto original se solicitado
        if self.job.save_original_pdf:
            orig_path = Path(self.job.output_path).with_stem(
                Path(self.job.output_path).stem + "_original"
            )
            self._save_original_pdf(results, metadata, str(orig_path))

        elapsed = time.time() - start_time
        ok = sum(1 for r in results if not r.error)
        ocr_count = sum(1 for r in results if r.used_ocr)
        total_chars = sum(r.char_count for r in results)

        log.info("=" * 60)
        log.info(f"✅ Concluído em {elapsed:.1f}s")
        log.info(f"   Páginas traduzidas: {ok}/{len(results)}")
        log.info(f"   Páginas com OCR:    {ocr_count}")
        log.info(f"   Total de caracteres: {total_chars:,}")
        log.info(f"   Arquivo de saída:   {self.job.output_path}")
        if self.job.save_original_pdf:
            orig_stem = Path(self.job.output_path).stem + "_original"
            orig_path = Path(self.job.output_path).with_stem(orig_stem)
            log.info(f"   PDF original:       {orig_path}")
        log.info("=" * 60)

        return results

    def _process_pages(self, pages: list[int]) -> list[PageResult]:
        results = []

        # Separa páginas já no cache
        to_translate = []
        for page_num in pages:
            if self.job.resume and self.cache.has(page_num):
                results.append(self.cache.get(page_num))
                log.debug(f"Página {page_num} carregada do cache")
            else:
                to_translate.append(page_num)

        if not to_translate:
            log.info("Todas as páginas carregadas do cache.")
            return results

        log.info(f"Traduzindo {len(to_translate)} páginas ({len(results)} do cache)...")

        # Processa em paralelo com barra de progresso
        if TQDM_OK:
            pbar = tqdm(total=len(to_translate), desc="Traduzindo", unit="pág")

        with ThreadPoolExecutor(max_workers=self.job.workers) as executor:
            futures = {
                executor.submit(self._process_single_page, pn): pn
                for pn in to_translate
            }
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                self.cache.put(result)
                if TQDM_OK:
                    pbar.update(1)
                    pbar.set_postfix({"pág": result.page_num, "ocr": result.used_ocr})
                else:
                    log.info(f"Pág {result.page_num}/{max(to_translate)} {'[OCR]' if result.used_ocr else ''}")

        if TQDM_OK:
            pbar.close()

        return results

    def _process_single_page(self, page_num: int) -> PageResult:
        result = PageResult(page_num=page_num)
        try:
            # Extração
            text, used_ocr = self.extractor.extract_page(page_num)
            result.original_text = text
            result.used_ocr = used_ocr

            if not text.strip():
                result.translated_text = ""
                return result

            # Tradução
            translated = self.engine.translate_text(text)
            result.translated_text = translated
            result.char_count = len(translated)

        except Exception as e:
            log.error(f"Erro na página {page_num}: {e}", exc_info=True)
            result.error = str(e)
            result.translated_text = result.original_text  # fallback

        return result

    def _save_original_pdf(self, results: list[PageResult], metadata: dict, path: str):
        """Gera um PDF separado contendo o texto original extraído (antes da tradução)."""
        if not REPORTLAB_OK:
            log.error("reportlab não instalado. Não é possível gerar PDF original.")
            return

        doc = SimpleDocTemplate(
            path,
            pagesize=A4,
            rightMargin=2 * cm,
            leftMargin=2 * cm,
            topMargin=2.5 * cm,
            bottomMargin=2.5 * cm,
            title=f"[ORIGINAL] {metadata.get('title', 'Documento')}",
            author=f"PDF Translator Pro | {metadata.get('author', '')}",
            subject=f"Texto original extraído – idioma: {self.job.source_lang}",
        )

        styles = getSampleStyleSheet()
        body_style = ParagraphStyle(
            "OrigBody",
            parent=styles["Normal"],
            fontSize=11,
            leading=16,
            spaceAfter=8,
        )
        page_header_style = ParagraphStyle(
            "OrigPageHeader",
            parent=styles["Normal"],
            fontSize=8,
            textColor=colors.grey,
            spaceAfter=4,
        )
        title_style = ParagraphStyle(
            "OrigTitle",
            parent=styles["Title"],
            fontSize=18,
            spaceAfter=20,
        )

        story = []

        # Capa
        story.append(Spacer(1, 3 * cm))
        title = metadata.get("title") or Path(self.job.input_path).stem
        story.append(Paragraph(PDFWriter._esc(f"[ORIGINAL] {title}"), title_style))
        story.append(Spacer(1, 0.5 * cm))
        info = (
            f"Texto original extraído do arquivo <b>{Path(self.job.input_path).name}</b><br/>"
            f"Idioma de origem: <b>{LANGUAGES.get(self.job.source_lang, self.job.source_lang)}</b> | "
            f"Páginas: {len(results)} | "
            f"Gerado por PDF Translator Pro"
        )
        story.append(Paragraph(info, styles["Normal"]))
        story.append(PageBreak())

        # Páginas com texto original
        for result in results:
            ocr_tag = " [OCR]" if result.used_ocr else ""
            story.append(
                Paragraph(
                    f"── Página {result.page_num}{ocr_tag} ──",
                    page_header_style,
                )
            )
            story.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey))
            story.append(Spacer(1, 0.2 * cm))

            text = result.original_text or "[Sem conteúdo extraído]"
            for para in text.split("\n"):
                para = para.strip()
                if para:
                    story.append(Paragraph(PDFWriter._esc(para), body_style))

            story.append(PageBreak())

        doc.build(story)
        log.info(f"PDF original salvo: {path}")


# ──────────────────────────────────────────────
# Interface de linha de comando
# ──────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="PDF Translator Pro – Traduz PDFs com OCR para qualquer idioma",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  # Traduzir livro completo inglês → português (Google, padrão)
  python pdf_translator.py livro.pdf -o livro_pt.pdf -t pt

  # Com OCR e DPI alto para scan de baixa qualidade
  python pdf_translator.py scan.pdf -o scan_pt.pdf -t pt --ocr --dpi 400

  # Traduzir só páginas 10-50 usando DeepL
  python pdf_translator.py livro.pdf -o saida.pdf -t pt -e deepl --pages 10-50

  # Tradução em espanhol, salvar também PDF original, 8 workers paralelos
  python pdf_translator.py doc.pdf -o doc_es.pdf -s en -t es --workers 8 --save-original-pdf

  # Retomar tradução interrompida
  python pdf_translator.py livro.pdf -o livro_pt.pdf -t pt --resume

Idiomas suportados:
""" + "\n".join(f"  {k:8} {v}" for k, v in LANGUAGES.items()),
    )
    p.add_argument("input", help="Arquivo PDF de entrada")
    p.add_argument("-o", "--output", default="", help="Arquivo PDF de saída")
    p.add_argument("-s", "--source", default="auto", help="Idioma fonte (padrão: auto)")
    p.add_argument("-t", "--target", default="pt", help="Idioma destino (padrão: pt)")
    p.add_argument(
        "-e", "--engine",
        default="google",
        choices=["google", "deepl", "libre"],
        help="Motor de tradução (padrão: google)",
    )
    p.add_argument("--ocr", action="store_true", help="Forçar OCR em todas as páginas")
    p.add_argument("--no-ocr", action="store_true", help="Desativar OCR completamente")
    p.add_argument("--dpi", type=int, default=300, help="DPI para OCR (padrão: 300)")
    p.add_argument("--pages", default="", help="Intervalo de páginas, ex: 10-50")
    p.add_argument("--workers", type=int, default=4, help="Threads paralelas (padrão: 4)")
    p.add_argument("--chunk-size", type=int, default=4500, help="Caracteres por chunk (padrão: 4500)")
    p.add_argument("--save-original-pdf", action="store_true", help="Salvar também PDF com texto original extraído")
    p.add_argument("--resume", action="store_true", default=True, help="Retomar tradução do cache (padrão: True)")
    p.add_argument("--no-resume", action="store_true", help="Ignorar cache e retraduzir tudo")
    p.add_argument("--list-langs", action="store_true", help="Listar idiomas disponíveis")
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.list_langs:
        print("\nIdiomas disponíveis:\n")
        for code, name in LANGUAGES.items():
            print(f"  {code:10} {name}")
        return

    if not os.path.exists(args.input):
        print(f"❌ Arquivo não encontrado: {args.input}")
        sys.exit(1)

    output = args.output or Path(args.input).stem + f"_traduzido_{args.target}.pdf"

    start_page, end_page = 1, None
    if args.pages:
        parts = args.pages.split("-")
        start_page = int(parts[0])
        if len(parts) > 1:
            end_page = int(parts[1])

    use_ocr = True
    if args.no_ocr:
        use_ocr = False

    resume = args.resume and not args.no_resume

    job = TranslationJob(
        input_path=args.input,
        output_path=output,
        source_lang=args.source,
        target_lang=args.target,
        engine=args.engine,
        use_ocr=use_ocr,
        ocr_dpi=args.dpi,
        workers=args.workers,
        chunk_size=args.chunk_size,
        start_page=start_page,
        end_page=end_page,
        save_original_pdf=args.save_original_pdf,
        resume=resume,
    )

    translator = PDFTranslator(job)
    translator.run()


# ──────────────────────────────────────────────
# Uso programático (importar como módulo)
# ──────────────────────────────────────────────
def translate_pdf(
    input_path: str,
    output_path: str = "",
    source_lang: str = "auto",
    target_lang: str = "pt",
    engine: str = "google",
    use_ocr: bool = True,
    workers: int = 4,
    pages: str = "",
    save_original_pdf: bool = False,
    resume: bool = True,
) -> list[PageResult]:
    """
    API simplificada para uso programático.

    Exemplo:
        from pdf_translator import translate_pdf
        results = translate_pdf("livro.pdf", target_lang="pt")
        # Para salvar também um PDF com o texto original extraído:
        results = translate_pdf("livro.pdf", target_lang="pt", save_original_pdf=True)
    """
    if not output_path:
        output_path = Path(input_path).stem + f"_traduzido_{target_lang}.pdf"

    start_page, end_page = 1, None
    if pages:
        parts = pages.split("-")
        start_page = int(parts[0])
        if len(parts) > 1:
            end_page = int(parts[1])

    job = TranslationJob(
        input_path=input_path,
        output_path=output_path,
        source_lang=source_lang,
        target_lang=target_lang,
        engine=engine,
        use_ocr=use_ocr,
        workers=workers,
        start_page=start_page,
        end_page=end_page,
        save_original_pdf=save_original_pdf,
        resume=resume,
    )

    translator = PDFTranslator(job)
    return translator.run()


if __name__ == "__main__":
    main()