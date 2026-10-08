from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


PDF_STRUCTURE_ERROR_MARKERS = (
    "object out of range",
    "non-page object in page tree",
    "cannot find object in xref",
    "expected object number",
    "xref stream",
    "repairing pdf",
    "cannot load object",
)


@dataclass(frozen=True)
class PreparedPdf:
    original_path: Path
    parsing_path: Path
    repaired: bool = False
    warnings: str = ""


def prepare_pdf_for_parsing(pdf_path: Path) -> PreparedPdf:
    """Return a readable PDF path without ever overwriting the source file."""
    pdf_path = pdf_path.resolve()
    if not _env_bool("PDF_AUTO_REPAIR", True):
        return PreparedPdf(pdf_path, pdf_path)

    healthy, warnings = _inspect_with_pymupdf(pdf_path)
    if healthy:
        return PreparedPdf(pdf_path, pdf_path, warnings=warnings)

    cache_dir = Path(os.getenv("PDF_REPAIR_CACHE_DIR", "./.state/repaired_pdfs"))
    cache_dir = cache_dir.expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    digest = _sha256(pdf_path)[:16]
    repaired_path = cache_dir / f"{digest}-{_safe_name(pdf_path.stem)}.pdf"

    if repaired_path.exists():
        cached_healthy, cached_warnings = _inspect_with_pymupdf(repaired_path)
        if cached_healthy:
            print(f"  🩹 使用已有 PDF 修复副本：{repaired_path}")
            return PreparedPdf(pdf_path, repaired_path, True, cached_warnings)
        repaired_path.unlink(missing_ok=True)

    print("  🩹 检测到 PDF xref/page tree 异常，正在创建只读修复副本...")
    errors: list[str] = []
    for repairer in (_repair_with_pymupdf, _repair_with_qpdf, _repair_with_ghostscript):
        temp_path = repaired_path.with_suffix(f".{os.getpid()}.tmp.pdf")
        temp_path.unlink(missing_ok=True)
        try:
            if not repairer(pdf_path, temp_path):
                continue
            repaired_healthy, repaired_warnings = _inspect_with_pymupdf(temp_path)
            if not repaired_healthy:
                raise RuntimeError(repaired_warnings or "修复副本仍无法逐页读取")
            temp_path.replace(repaired_path)
            print(f"  ✅ PDF 修复副本已生成：{repaired_path}")
            return PreparedPdf(pdf_path, repaired_path, True, repaired_warnings)
        except Exception as exc:
            errors.append(f"{repairer.__name__}: {exc}")
        finally:
            temp_path.unlink(missing_ok=True)

    detail = "; ".join(errors[-3:]) or warnings or "没有可用的 PDF 修复工具"
    print(f"  ⚠️ PDF 自动修复失败，将使用原文件并逐页隔离异常：{detail}")
    return PreparedPdf(pdf_path, pdf_path, warnings=warnings)


def _inspect_with_pymupdf(pdf_path: Path) -> tuple[bool, str]:
    try:
        import fitz
    except ImportError:
        return True, ""

    document = None
    errors: list[str] = []
    try:
        fitz.TOOLS.mupdf_display_errors(False)
        fitz.TOOLS.reset_mupdf_warnings()
        document = fitz.open(pdf_path)
        repaired_on_open = bool(document.is_repaired)
        seen_xrefs: set[int] = set()
        xref_length = int(document.xref_length())
        for page_index in range(len(document)):
            page = document.load_page(page_index)
            page.get_text("text")
            for image_info in page.get_images(full=True):
                xref = int(image_info[0])
                if xref <= 0 or xref in seen_xrefs:
                    continue
                seen_xrefs.add(xref)
                if xref >= xref_length:
                    raise RuntimeError(
                        f"page {page_index + 1}: image xref {xref} "
                        f"out of range (xref size {xref_length})"
                    )
                # Validate the object table without decoding every image. Decoding all
                # image bytes here duplicated much of the real parser's work on large PDFs.
                document.xref_object(xref, compressed=False)
        warnings = fitz.TOOLS.mupdf_warnings() or ""
        if repaired_on_open:
            errors.append("PyMuPDF repaired the document while opening it")
        if _has_structure_warning(warnings):
            errors.append(warnings)
    except Exception as exc:
        warnings = fitz.TOOLS.mupdf_warnings() or ""
        errors.append(str(exc))
        if warnings:
            errors.append(warnings)
    finally:
        if document is not None:
            document.close()
        fitz.TOOLS.mupdf_display_errors(True)
    detail = "\n".join(part.strip() for part in errors if part.strip())
    return not errors, detail


def _repair_with_pymupdf(source: Path, target: Path) -> bool:
    import fitz

    document = None
    fitz.TOOLS.mupdf_display_errors(False)
    try:
        document = fitz.open(source)
        document.save(target, garbage=4, clean=True, deflate=True)
    finally:
        if document is not None:
            document.close()
        fitz.TOOLS.mupdf_display_errors(True)
    return target.exists() and target.stat().st_size > 0


def _repair_with_qpdf(source: Path, target: Path) -> bool:
    executable = shutil.which("qpdf")
    if not executable:
        return False
    completed = subprocess.run(
        [executable, str(source), str(target)],
        capture_output=True,
        text=True,
        timeout=int(os.getenv("PDF_REPAIR_TIMEOUT", "600")),
    )
    if completed.returncode not in {0, 3}:
        raise RuntimeError((completed.stderr or completed.stdout).strip())
    return target.exists() and target.stat().st_size > 0


def _repair_with_ghostscript(source: Path, target: Path) -> bool:
    executable = shutil.which("gs")
    if not executable:
        return False
    completed = subprocess.run(
        [
            executable,
            "-q",
            "-dSAFER",
            "-dBATCH",
            "-dNOPAUSE",
            "-sDEVICE=pdfwrite",
            "-dCompatibilityLevel=1.7",
            f"-sOutputFile={target}",
            str(source),
        ],
        capture_output=True,
        text=True,
        timeout=int(os.getenv("PDF_REPAIR_TIMEOUT", "600")),
    )
    if completed.returncode != 0:
        raise RuntimeError((completed.stderr or completed.stdout).strip())
    return target.exists() and target.stat().st_size > 0


def _has_structure_warning(warnings: str) -> bool:
    normalized = warnings.lower()
    return any(marker in normalized for marker in PDF_STRUCTURE_ERROR_MARKERS)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_name(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in "-_" else "_" for char in value)
    return safe.strip("_")[:80] or "repaired"


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}
