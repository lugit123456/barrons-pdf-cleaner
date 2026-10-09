# Barron's PDF Cleaner

从指定目录获取最新的 Barron's PDF，拆分正文文章、清理广告/目录/行情数据页，生成单篇 Markdown、图片资源和与 `economist_weekly_archiver_skill/output_results` 相同结构的 `database.js` / `database_index.js`。

## 处理方式

- native PDF：读取 PyMuPDF 文字块、字号、坐标和嵌入图片，再由 LLM 判断分栏文章边界。
- scanned PDF：使用 Poppler + Tesseract OCR，必要时调用视觉模型识别页面。
- 文章正文：修复断词和视觉换行；`original` 模式只保留英文原文，`bilingual` 模式再生成中文标题、中文结构化摘要和逐段中英对照。
- Barron's 确定性清理：跳过 `Contents`、`Index` 和 `DATA` 页面；广告、service listing、短 page-reference teaser 继续使用现有过滤器剔除。
- 跨页文章：按标题和 continuation 信息合并；`source_pages` 仅在清洗流程内部使用，不写入 Barron's 最终数据库。
- 去重与续跑：页面结果写入 `cache_json`，SQLite 记录完成的 PDF；失败后复用成功页面继续处理。

## 安装

建议 Python 3.9-3.11。系统需要 Poppler 和 Tesseract。

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

在 `.env` 中配置 `OPENAI_API_KEY`、`OPENAI_API_URL` 和 `OPENAI_MODEL`。不要提交 `.env`。

## 配置

本地和服务器均从 `config.example.json` 复制出各自的 `config.json` 后修改。`config.json` 已被 Git 忽略，所有相对路径都以配置文件所在目录为基准。

- `input_dir`：PDF 输入目录。
- `output_dir`：输出根目录。
- `state_db`：去重状态库。
- `dotenv_path`：可选的外部 `.env` 路径，适合服务器把密钥放在项目外。
- `pattern`：文件匹配规则。
- `recursive`：是否递归扫描子目录。
- `selection_mode`：`latest` 每轮只检查出版日期最新的一份；`all_unprocessed` 按新到旧检查所有文件，由状态库跳过已完成文件。
- `content_mode`：`original` 只清洗英文原文且不调用 LLM 编译；`bilingual` 启用 `.env` 中的翻译、摘要、图片解读和 glossary 设置。
- `poll_interval_seconds`：常驻模式的轮询间隔。
- `stable_seconds`：文件最后修改后至少静置多久，避免读取尚未复制完成的 PDF。

`content_mode=original` 会强制关闭文章翻译、图片解读和 glossary，并启用 Barron's 本地坐标清洗，不需要 API key。`bilingual` 模式需要在 `.env` 中配置可用的 LLM API。

`bilingual` 模式采用两个独立请求：先按英文段落逐段翻译，并强制译文与原文一一对应；再单独生成中文标题和中文解读。英文正文始终由本地清洗结果提供，不接受模型改写。中文解读会根据英文篇幅动态控制在 420-1000 个汉字，并检查段落结构、列表式输出和过长句子。

## 运行

先验证配置和文件选择：

```bash
python3 scripts/watch_and_process.py --once --dry-run
```

运行一轮：

```bash
python3 scripts/watch_and_process.py --once
```

按 `poll_interval_seconds` 常驻轮询：

```bash
python3 scripts/watch_and_process.py
```

服务器也可以用进程管理器定时调用 `--once`。项目本身不依赖本机绝对路径，部署时只修改配置文件。

## 输出

每期写入：

```text
output_results/
├── database_index.js
└── BARRONS/
    └── YYYY-MM-DD/
        ├── database.js
        ├── cover.jpg
        ├── articles/
        ├── images/
        └── cache_json/
```

`database.js` 使用 `window.paper_databases[id] = {...}`。Barron's 的 issue、article、paragraph、glossary 和 `database_index.js` 字段集合及顺序与 Economist weekly 归档输出一致；文章 ID 和文件名使用 `art_YYYY-MM-DD_NNN`。

仓库中的 `verification_output/BARRONS/2026-10-05/database.js` 是由测试 PDF 第 7 页真实清洗结果生成的 schema 验证样例，仅含 1 篇文章，不代表完整 56 页期刊。

## 测试

```bash
PYTHONPYCACHEPREFIX=/tmp/barrons-pycache python3 -m unittest discover -s tests -v
python3 -m compileall main.py database_writer.py processing_state.py strategies scripts
```
