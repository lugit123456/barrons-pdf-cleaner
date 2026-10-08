import inspect
import os
import time

from .base import ParseContext, ParseResult


class VectorPdfStrategy:
    """Vector/mixed PDF parser backed by MinerU UNIPipe."""

    engine_name = "Strategy_A_MinerU"

    def parse(self, context: ParseContext) -> ParseResult:
        context.target_output_dir.mkdir(parents=True, exist_ok=True)
        context.image_dir.mkdir(parents=True, exist_ok=True)

        with context.pdf_path.open("rb") as file_obj:
            pdf_bytes = file_obj.read()

        md_content = parse_pdf_bytes_to_markdown(pdf_bytes, context.image_dir)
        body = md_content.strip() if md_content and md_content.strip() else "_未提取到有效文本。_"
        return ParseResult(body_markdown=body, engine_name=self.engine_name)


def parse_pdf_bytes_to_markdown(pdf_bytes: bytes, image_dir) -> str:
    """Run MinerU for PDF bytes so the hybrid strategy can reuse it per page."""
    _configure_mineru_model()

    from magic_pdf.pipe.UNIPipe import UNIPipe
    from magic_pdf.rw.DiskReaderWriter import DiskReaderWriter

    _patch_mineru_progress_logging()
    image_writer = DiskReaderWriter(str(image_dir))
    pipe = _create_unipipe(UNIPipe, pdf_bytes, image_writer)
    try:
        pipe.pipe_classify()
        pipe.pipe_analyze()
        pipe.pipe_parse()
    except SystemExit as exc:
        raise RuntimeError(_mineru_dependency_hint()) from exc

    return _make_markdown(pipe)


def _create_unipipe(unipipe_class, pdf_bytes: bytes, image_writer):
    """Create UNIPipe across MinerU versions while preserving the demo flow."""
    signature = inspect.signature(unipipe_class.__init__)
    params = signature.parameters

    if "jso_useful_key" in params:
        useful_key = {"_pdf_type": "", "model_list": []}
        return unipipe_class(
            pdf_bytes,
            jso_useful_key=useful_key,
            image_writer=image_writer,
        )

    if "jso_elem_list" in params:
        return unipipe_class(
            pdf_bytes,
            jso_elem_list=[],
            image_writer=image_writer,
        )

    return unipipe_class(pdf_bytes, {"_pdf_type": "", "model_list": []}, image_writer)


def _configure_mineru_model() -> None:
    import magic_pdf.model as model_config

    model_config.__use_inside_model__ = _env_bool("MINERU_USE_INSIDE_MODEL", True)
    model_config.__model_mode__ = os.getenv("MINERU_MODEL_MODE", "lite")


def _patch_mineru_progress_logging() -> None:
    if not _env_bool("MINERU_SHOW_PROGRESS", True):
        return

    import magic_pdf.model as model_config
    import magic_pdf.pipe.UNIPipe as unipipe_module
    from magic_pdf.model.model_list import MODEL
    from magic_pdf.model.doc_analyze_by_custom_model import load_images_from_pdf

    if getattr(unipipe_module.doc_analyze, "_codex_progress_patch", False):
        return

    def doc_analyze_with_progress(pdf_bytes: bytes, ocr: bool = False, show_log: bool = False):
        model = None
        if model_config.__model_mode__ == "lite":
            model = MODEL.Paddle
        elif model_config.__model_mode__ == "full":
            model = MODEL.PEK

        if not model_config.__use_inside_model__:
            raise RuntimeError("MinerU inside model 未启用。")

        model_init_start = time.time()
        if model == MODEL.Paddle:
            from magic_pdf.model.pp_structure_v2 import CustomPaddleModel

            custom_model = CustomPaddleModel(ocr=ocr, show_log=show_log)
        elif model == MODEL.PEK:
            from magic_pdf.libs.config_reader import get_device, get_local_models_dir
            from magic_pdf.model.pdf_extract_kit import CustomPEKModel

            custom_model = CustomPEKModel(
                ocr=ocr,
                show_log=show_log,
                models_dir=get_local_models_dir(),
                device=get_device(),
            )
        else:
            raise RuntimeError(f"MinerU 不支持的模型模式：{model_config.__model_mode__}")

        print(f"  ⏱️ MinerU 模型初始化完成：{time.time() - model_init_start:.1f}s")
        images = load_images_from_pdf(pdf_bytes)
        print(f"  🧩 MinerU 开始逐页版面分析：共 {len(images)} 页")

        model_json = []
        analyze_start = time.time()
        for index, img_dict in enumerate(images, start=1):
            page_start = time.time()
            print(f"  🔬 MinerU 分析第 {index}/{len(images)} 页...")
            result = custom_model(img_dict["img"])
            page_info = {
                "page_no": index - 1,
                "height": img_dict["height"],
                "width": img_dict["width"],
            }
            model_json.append({"layout_dets": result, "page_info": page_info})
            print(f"  ✅ 第 {index}/{len(images)} 页完成：{time.time() - page_start:.1f}s")

        print(f"  ⏱️ MinerU 版面分析完成：{time.time() - analyze_start:.1f}s")
        return model_json

    doc_analyze_with_progress._codex_progress_patch = True
    unipipe_module.doc_analyze = doc_analyze_with_progress


def _mineru_dependency_hint() -> str:
    mode = os.getenv("MINERU_MODEL_MODE", "lite")
    if mode == "full":
        return (
            "MinerU full 模式缺少依赖。请安装："
            "pip install 'magic-pdf[full-cpu]' detectron2 "
            "--extra-index-url https://myhloli.github.io/wheels/；"
            "或将 MINERU_MODEL_MODE=lite 使用轻量模式。"
        )
    return (
        "MinerU lite 模式缺少依赖。请安装：pip install 'magic-pdf[cpu]'；"
        "如果已经安装仍失败，请检查 cv2 是否能 import。常见原因是 NumPy 2.x "
        "与旧版 OpenCV wheel ABI 不兼容，可执行："
        "python3 -m pip install --force-reinstall 'numpy<2' "
        "'opencv-python==4.6.0.66' 'opencv-contrib-python==4.6.0.66'。"
    )


def _make_markdown(pipe) -> str:
    signature = inspect.signature(pipe.pipe_mk_markdown)
    params = signature.parameters

    if "image_dir_name" in params:
        return pipe.pipe_mk_markdown(image_dir_name="images")

    return pipe.pipe_mk_markdown("images")


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}
