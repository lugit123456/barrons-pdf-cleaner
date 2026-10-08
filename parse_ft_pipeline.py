import io
import json
import os
import re
import time
import cv2
from dotenv import load_dotenv
import numpy as np
from openai import OpenAI
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
import cv2
import numpy as np

load_dotenv()

PDF_PATH = "ft.pdf"
OUTPUT_DIR = "./ft_output"
CACHE_DIR = os.path.join(OUTPUT_DIR, "cache_json")
RAW_IMG_DIR = os.path.join(OUTPUT_DIR, "images")
ARTICLE_DIR = os.path.join(OUTPUT_DIR, "articles")

client = None
MODEL_NAME = os.environ.get("OPENAI_MODEL", "gpt-4o")


def get_openai_client():
  global client
  if client is None:
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_API_URL", "https://sub2.yz.rs/v1"),
        timeout=60.0,
    )
  return client

def is_ft_paper_background(hsv_crop):
  """检测是否属于 FT 报纸的粉色纸张背景"""
  h, s, v = hsv_crop[:, :, 0], hsv_crop[:, :, 1], hsv_crop[:, :, 2]
  # FT 鲑鱼粉 HSV 范围：Hue 偏红橙 (0~25 或 165~180)，低中饱和度 (10~80)，高明度 (>140)
  paper_mask = (
      ((h < 25) | (h > 165)) & (s >= 10) & (s <= 80) & (v > 140)
  )
  return paper_mask


def is_valid_graphic_or_photo(crop_bgr):
  """精准识别：大图、人物小头像、流程图/图表

  彻底解决大字标题误抓问题：
  1. 优先提取彩色/流程图特征。
  2. 利用“中间阶调占比 (midtone_ratio)”剔除纯文字（含大字标题）。
  3. 最后对通过中间调校验的区域应用小图保护与照片判定。
  """
  h, w, _ = crop_bgr.shape
  total_pixels = float(h * w)
  if total_pixels == 0:
    return False

  # 1. 宽高比初筛：过滤极度狭长的长条文字列（高/宽 > 3.2）
  aspect_ratio = h / float(w)
  if aspect_ratio > 3.2:
    return False

  hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
  v_chan = hsv[:, :, 2]

  # -------------------------------------------------------------
  # 🎯 特征 A：流程图 / 结构图 / 图表识别（优先放行）
  # -------------------------------------------------------------
  h_chan, s_chan = hsv[:, :, 0], hsv[:, :, 1]
  # 统计 FT 标志性蓝色/青色等非纸色像素 (H: 80~140, S > 25)
  distinct_color_mask = (h_chan >= 80) & (h_chan <= 140) & (s_chan > 25)
  distinct_color_ratio = np.count_nonzero(distinct_color_mask) / total_pixels

  if distinct_color_ratio > 0.02:
    return True

  # -------------------------------------------------------------
  # 🎯 特征 B：中间阶调占比检测（杀手锏：彻底干掉大字标题）
  # -------------------------------------------------------------
  # 统计明度 V 在 55 ~ 195 之间的中间过渡像素
  # 文字（含大标题）：只有边缘抗锯齿，中间调占比极低 (< 12%)
  # 照片/头像：光影、肌肤、背景过渡丰富，中间调占比很高 (> 20%)
  midtone_mask = (v_chan >= 55) & (v_chan <= 195)
  midtone_ratio = np.count_nonzero(midtone_mask) / total_pixels

  if midtone_ratio < 0.12:
    print(
        f"  🚫 过滤文本/大字标题 (缺乏中间光影阶调, midtone_ratio="
        f"{midtone_ratio:.1%})"
    )
    return False

  # -------------------------------------------------------------
  # 🎯 特征 C：粉色纸张背景与行间距分析
  # -------------------------------------------------------------
  paper_mask = is_ft_paper_background(hsv)
  paper_ratio = np.count_nonzero(paper_mask) / total_pixels

  gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
  _, binary = cv2.threshold(
      gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
  )
  fg_ratio = np.count_nonzero(binary) / total_pixels

  # 计算文本行间距
  row_fg = np.sum(binary > 0, axis=1)
  empty_rows = np.count_nonzero(row_fg < (w * 0.05))
  empty_row_ratio = empty_rows / float(h)

  # -------------------------------------------------------------
  # 🎯 综合判定逻辑（修正了执行顺序）
  # -------------------------------------------------------------
  # 规则 1：正文段落剔除（纸张背景多 >60% 且 空行明显 >9%）
  if paper_ratio > 0.60 and empty_row_ratio > 0.09:
    return False

  # 规则 2：小头像保护机制（必须先通过上述 midtone_ratio >= 0.12 的光影校验）
  is_small_crop = w < 250 and h < 250
  if is_small_crop and midtone_ratio > 0.20:
    return True

  # 规则 3：真实照片判定（前景丰富或背景被遮挡）
  if fg_ratio > 0.35 or paper_ratio < 0.45 or empty_row_ratio < 0.06:
    return True

  return False


def crop_images_from_scan_page(page_img_pil, page_idx, output_dir):
  """双通道 ROI 提取：色彩通道（抓流程图） + 形态学通道（抓照片/头像）"""
  img_cv = cv2.cvtColor(np.array(page_img_pil), cv2.COLOR_RGB2BGR)
  h_page, w_page, _ = img_cv.shape
  page_area = h_page * w_page

  # -------------------------------------------------------------
  # 通道 1：提取非粉色背景区域（精准定位蓝色流程图、彩色插图）
  # -------------------------------------------------------------
  hsv_page = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
  paper_mask = is_ft_paper_background(hsv_page)
  # 非纸色区域（色彩/图表/照片区域）
  non_paper_mask = cv2.bitwise_not(
      (paper_mask * 255).astype(np.uint8)
  )

  # -------------------------------------------------------------
  # 通道 2：常规灰度形态学（定位黑白/低饱和度照片及头像）
  # -------------------------------------------------------------
  gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
  blurred = cv2.GaussianBlur(gray, (5, 5), 0)
  _, thresh = cv2.threshold(
      blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
  )

  # 混合通道掩膜
  combined_mask = cv2.bitwise_or(non_paper_mask, thresh)

  # 闭运算连接相邻区域（缩小核尺寸至 12x12，防止流程图与周围文字粘连）
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 12))
  closed = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

  contours, _ = cv2.findContours(
      closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
  )

  cropped_images = []
  img_count = 1

  for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    crop_area = w * h

    # 🛑 1. 放宽尺寸筛选：支持捕捉小头像与流程图
    # 最小面积降至 0.2% (0.002)，最小宽高降至 70px
    if not (page_area * 0.002 < crop_area < page_area * 0.75):
      continue
    if w < 70 or h < 70:
      continue

    crop_bgr = img_cv[y : y + h, x : x + w]

    # 🎯 2. 核心校验：精准判定大图、流程图与小头像
    if not is_valid_graphic_or_photo(crop_bgr):
      continue

    # 保存判定通过的图片
    img_name = f"page_{page_idx}_fig_{img_count}.jpg"
    save_path = os.path.join(output_dir, img_name)
    cv2.imwrite(save_path, crop_bgr)

    cropped_images.append({
        "rel_path": f"images/{img_name}",
        "bbox_ratio": {
            "top": round(y / h_page, 2),
            "left": round(x / w_page, 2),
        },
    })
    img_count += 1

  print(
      f"  🖼️ 第 {page_idx} 页精准提取出 {len(cropped_images)} 张图片/图表。"
  )
  return cropped_images


def slice_text_by_anchors(full_text, start_anchor, end_anchor):
  """Python 本地切片抽取完整正文"""
  if not start_anchor or not end_anchor:
    return ""

  clean_start = re.sub(r"\s+", " ", start_anchor).strip()[:20]
  clean_end = re.sub(r"\s+", " ", end_anchor).strip()[-20:]
  normalized_text = re.sub(r"\s+", " ", full_text)

  start_pos = normalized_text.find(clean_start)
  end_pos = normalized_text.find(clean_end)

  if start_pos != -1 and end_pos != -1 and end_pos >= start_pos:
    return normalized_text[start_pos : end_pos + len(clean_end)].strip()
  return ""


def process_single_page(page_idx, page_img):
  cache_file = os.path.join(CACHE_DIR, f"page_{page_idx}.json")
  if os.path.exists(cache_file):
    print(f"⚡ 第 {page_idx} 页已读本地缓存，跳过。")
    with open(cache_file, "r", encoding="utf-8") as f:
      return json.load(f)

  # 1. 自动切出本页的插图
  page_figs = crop_images_from_scan_page(page_img, page_idx, RAW_IMG_DIR)

  # 2. 本地 Tesseract 提取英文
  print(f"  🔍 正在对第 {page_idx} 页进行本地 OCR 识别...")
  page_text = pytesseract.image_to_string(page_img, lang="eng")

  if not page_text.strip():
    return {"page": page_idx, "articles": []}

  # 3. 让 LLM 分组文章 + 挂载插图
  prompt = f"""
Analyze Page {page_idx} of Financial Times:

--- OCR TEXT START ---
{page_text}
--- OCR TEXT END ---

Cropped Figures on this page: {json.dumps(page_figs)}

Task:
1. Identify all distinct articles.
2. For each article, provide `title`, `category`, `start_anchor` (first 10 words), `end_anchor` (last 10 words).
3. Assign matching image `rel_path` from Cropped Figures to the correct article (if applicable).

Return JSON ONLY:
{{
  "articles": [
    {{
      "title": "Article Title",
      "category": "Section",
      "start_anchor": "first 10 words...",
      "end_anchor": "last 10 words...",
      "images": ["images/page_{page_idx}_fig_1.jpg"]
    }}
  ]
}}
"""

  try:
    response = get_openai_client().chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=1500,
    )
    content = response.choices[0].message.content
    if "```" in content:
      match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content)
      if match:
        content = match.group(1)

    data = json.loads(content.strip())
    articles = data.get("articles", [])

    for art in articles:
      full_body = slice_text_by_anchors(
          page_text, art.get("start_anchor"), art.get("end_anchor")
      )
      art["content_markdown"] = (
          full_body
          if full_body
          else f"{art.get('start_anchor')} ... {art.get('end_anchor')}"
      )
      art.pop("start_anchor", None)
      art.pop("end_anchor", None)

    result = {"page": page_idx, "articles": articles}
    with open(cache_file, "w", encoding="utf-8") as f:
      json.dump(result, f, ensure_ascii=False, indent=2)

    return result
  except Exception as e:
    print(f"⚠️ 第 {page_idx} 页解析异常: {e}")
    return {"page": page_idx, "articles": []}


def save_articles_to_markdown(all_results, article_dir=ARTICLE_DIR):
  os.makedirs(article_dir, exist_ok=True)
  written_paths = []
  article_index = 1
  used_names = set()

  for page_result in all_results:
    page_idx = page_result.get("page")
    for article in page_result.get("articles", []):
      title = article.get("title") or f"Article {article_index}"
      filename = unique_article_filename(article_index, title, used_names)
      output_path = os.path.join(article_dir, filename)

      with open(output_path, "w", encoding="utf-8") as f:
        f.write(render_article_markdown(article, title, article_index, page_idx))

      written_paths.append(output_path)
      article_index += 1

  print(f"📰 已导出单篇文章：{len(written_paths)} 篇，目录：{article_dir}")
  return written_paths


def render_article_markdown(article, title, article_index, page_idx):
  category = article.get("category")
  images = article.get("images") or []
  content = article.get("content_markdown") or ""

  sections = [
      f"# {title}",
      f"> **文章序号**：`{article_index}`  \n> **页码**：`{page_idx}`",
  ]
  if category:
    sections.append(f"> **分类**：`{category}`")

  sections.append("---")
  for image in images:
    sections.append(f"![{title}]({article_image_path(image)})")
  if content.strip():
    sections.append(rewrite_article_image_refs(content.strip()))

  return "\n\n".join(sections).strip() + "\n"


def unique_article_filename(index, title, used_names):
  slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:80]
  if not slug:
    slug = "article"

  filename = f"{index:03d}-{slug}.md"
  suffix = 2
  while filename in used_names:
    filename = f"{index:03d}-{slug}-{suffix}.md"
    suffix += 1

  used_names.add(filename)
  return filename


def article_image_path(image_path):
  if image_path.startswith("../"):
    return image_path
  if image_path.startswith("images/"):
    return f"../{image_path}"
  if image_path.startswith("raw_images/"):
    return f"../{image_path}"
  return image_path


def rewrite_article_image_refs(markdown):
  markdown = markdown.replace("(images/", "(../images/")
  markdown = markdown.replace("(raw_images/", "(../raw_images/")
  return markdown


def main():
  if not os.path.exists(PDF_PATH):
    print(f"❌ 找不到 {PDF_PATH}")
    return

  os.makedirs(OUTPUT_DIR, exist_ok=True)
  os.makedirs(CACHE_DIR, exist_ok=True)
  os.makedirs(RAW_IMG_DIR, exist_ok=True)

  print("🚀 渲染 PDF 页面...")
  pages_images = convert_from_path(PDF_PATH, dpi=150)

  all_results = []
  for idx, page_img in enumerate(pages_images, start=1):
    res = process_single_page(idx, page_img)
    all_results.append(res)

  save_articles_to_markdown(all_results)
  print("🎉 全部处理完毕！")


if __name__ == "__main__":
  main()
