# Paper Analyzer

人文社科论文思想谱系分析工具 V0。

当前版本不训练模型，不写分类器训练脚本，也不微调 RoBERTa。V0 使用“RAG 证据召回 + DeepSeek 自动评分 + DeepSeek 报告生成”的方式，对用户上传的中文人文社科论文 PDF/TXT 做自动分析。

## 当前能力

- 上传 PDF 或 TXT
- 自动提取论文正文
- 按段落切分并生成 embedding
- 使用 `BAAI/bge-small-zh-v1.5` 召回各标签相关证据段落
- 使用 DeepSeek 为 19 个思想谱系标签打 0-3 分
- 生成 Markdown 结构化分析报告
- 下载 Markdown、PDF 报告和 JSON 分析结果

## 可视化能力

当前 V0 页面包含：

- P001 示例思想谱系图谱
- 核心判断卡
- 19 标签矩阵
- 标签雷达图
- 标签强度分布图
- 论文—维度—标签网络图
- 核心标签证据链
- 上传论文后的自动分析报告

## 前端信息结构

当前前端采用“核心判断卡 + 标签矩阵 + 空间定位图 + 证据链”的结构：

- 核心判断卡负责给出一句话总览。
- 标签矩阵负责直观展示 19 个标签强弱。
- 空间定位图负责展示论文在二维学术谱系空间中的大致位置。
- 证据链负责解释每个判断的原文依据。

## 本地测试与进度观察

上传页默认开启“快速测试模式”，只分析 5 个核心标签，用于快速验证 API、证据召回和报告生成链路。关闭快速测试模式后，系统才会分析完整 19 个标签。

分析过程中页面会显示分阶段进度，包括文本清洗、段落切分、embedding、逐标签证据召回、逐标签 DeepSeek 分析和最终报告生成。命令行也会打印当前标签、召回证据数量、DeepSeek 请求开始/结束和失败状态。

每次 DeepSeek 请求设置超时保护。单个标签超时会降级为该标签 `score=0`、`confidence=1`，并继续分析后续标签，不会让整个页面无限等待。

## 模型选择与耗时

上传页支持选择 DeepSeek 模型：

- V4 Flash：速度快、成本低，适合快速预览。
- V4 Pro：分析更细，耗时更长，适合正式报告。

API key 仍由站点服务端环境变量或 Streamlit Secrets 配置，用户无需输入 API key，也不能在页面中查看或修改 API key。

耗时会随模型、文本长度、网络状态和标签数量变化。通常：

- 快速测试模式：约 1—3 分钟。
- 完整分析：约 3—10 分钟。
- 超过 30,000 字的长文本可能额外增加 1—3 分钟。

## 证据链说明

系统会先将论文切分为较完整的语义段落，再用 embedding 召回与各标签定义最相关的段落。DeepSeek 基于召回证据进行标签评分与报告生成。

为控制 prompt 长度，传给 DeepSeek 的证据会做截断；页面“证据链”会展示比模型输入更完整的原文上下文，方便人工复核标签分数和判断理由。

## 学术谱系空间图

空间图用二维坐标呈现论文在结构化标签体系中的位置。横轴表示“文本形式 ←→ 社会历史”，纵轴表示“个体经验 ←→ 结构秩序”。系统会根据高分标签的固定坐标和标签分数加权计算论文位置，并连向相关核心标签。

这张图用于帮助用户理解一篇论文处于何种方法论与思想传统交叉区域，例如更靠近“现代性批判 / 思想史研究 / 社会历史批评”，还是更靠近“审美自治 / 文本细读 / 形式实验”。

## 当前 V0 能力边界

- 当前版本没有训练自己的分类模型。
- 当前版本依赖标签定义、证据召回和 DeepSeek 推理。
- 当前结果适合学术阅读辅助，不适合当作最终学术判断。
- 系统只分析论文文本呈现出的思想倾向，不判断作者本人真实政治立场。
- 用户上传论文后，建议人工复核标签分数和证据段落。

当前只有 P001 样本，不适合训练分类模型。后续当人工标注样本达到 30-100 篇后，可以训练小分类模型；当样本达到 300-1000 篇后，再考虑微调 RoBERTa。

## 报告导出

当前支持三种导出：

- Markdown 报告下载
- JSON 分析结果下载
- PDF 报告下载

PDF 由本地 Python 根据已经生成的 Markdown 报告转换生成，DeepSeek 不负责生成 PDF。系统会优先使用 HTML 转 PDF 的方式生成排版更完整的 PDF；如果部署环境中的 PDF 依赖不可用，页面不会崩溃，仍可下载 Markdown 报告和 JSON 分析结果。

## 本地运行

在 PowerShell 中进入项目目录并激活虚拟环境：

```powershell
cd path\to\paper
.\paper-analyzer\Scripts\Activate.ps1
```

安装依赖：

```powershell
python -m pip install -r requirements.txt
```

启动网页 Demo：

```powershell
streamlit run app/streamlit_app.py
```

## 配置 DeepSeek API Key

网站上线后，所有用户使用服务端统一配置的 DeepSeek API。用户不需要、也不应该在页面里输入 API key。

本地临时配置：

```powershell
$env:DEEPSEEK_API_KEY="替换为你的服务端key"
$env:DEEPSEEK_MODEL="deepseek-v4-flash"
```

本地长期配置：

```powershell
setx DEEPSEEK_API_KEY "替换为你的服务端key"
setx DEEPSEEK_MODEL "deepseek-v4-flash"
```

Streamlit Community Cloud 部署时，在 Secrets 中添加：

```toml
DEEPSEEK_API_KEY = "替换为你的服务端key"
DEEPSEEK_MODEL = "deepseek-v4-flash"
```

API key 只从服务端环境变量或 Streamlit Secrets 读取。不要把真实 API key 写入代码、README、`.env`、`.streamlit/secrets.toml`，也不要提交到 GitHub。

本地测试建议使用 `deepseek-v4-flash`，速度更快、成本更低。正式高质量分析可以在 `DEEPSEEK_MODEL` 中切换为 `deepseek-v4-pro`。

## Streamlit Cloud 部署

1. 创建 GitHub 仓库
2. 推送项目代码
3. 登录 Streamlit Community Cloud
4. 选择 GitHub repo
5. Main file path 填 `app/streamlit_app.py`
6. 在 Streamlit Secrets 中添加 `DEEPSEEK_API_KEY` 和 `DEEPSEEK_MODEL`
7. Deploy 后获得公开网址

## 上线部署

部署到 Streamlit Community Cloud：

1. 推送项目到 GitHub。
2. 在 Streamlit Community Cloud 新建 App。
3. 选择 GitHub repo。
4. Main file path 填 `app/streamlit_app.py`。
5. 在 Secrets 中添加：

```toml
DEEPSEEK_API_KEY = "你的 key"
DEEPSEEK_MODEL = "deepseek-v4-flash"
```

6. Deploy 后获得公网链接。

## 使用统计

当前统计分两层：

1. Streamlit Community Cloud 自带 Analytics：用于查看网站访问人数。
2. 项目内部 `usage_logger`：记录上传、分析开始、分析成功、失败、模型选择、耗时和下载事件。

内部统计只保存必要元数据，不保存论文全文、不保存完整文件名、不保存用户隐私文本，也不保存 API key。

默认 fallback 会写入 `outputs/usage_events.jsonl`。本地可用，但 Streamlit Cloud 上 `outputs` 不保证长期持久。正式统计建议配置 Supabase，并在 Secrets 中添加：

```toml
SUPABASE_URL = "你的 Supabase URL"
SUPABASE_SERVICE_ROLE_KEY = "你的 Supabase service role key"
SUPABASE_USAGE_TABLE = "usage_events"
```

如果不配置 Supabase，统计会继续 fallback 到本地 jsonl。访问 `?admin=1` 可以查看隐藏统计面板。

## 依赖说明

V0 的主流程直接使用 `streamlit`、`plotly`、`pdfplumber`、`pandas`、`openpyxl`、`sentence-transformers`、`numpy`、`openai`。PDF 导出使用 `markdown`、`weasyprint`，并用 `reportlab` 作为简化兜底方案。`torch` 和 `transformers` 是 `sentence-transformers` 的底层依赖，仍需保留。

`chromadb` 当前上传分析 V0 尚未直接使用。如果部署包体积压力很大，可以后续确认后临时移除它，以减轻 Streamlit Cloud 安装压力。

## 命令行脚本

检查训练数据：

```powershell
python scripts/01_check_data.py
```

生成 P001 全文 embedding：

```powershell
python scripts/02_build_embeddings.py
```

检索 P001 标签相关证据：

```powershell
python scripts/03_retrieve_evidence.py 现代性批判
```

为 P001 生成 V0 自动分析报告：

```powershell
python scripts/04_generate_report.py
```

如果未配置 `DEEPSEEK_API_KEY`，报告生成脚本和网页都会给出清晰提示。当前 V0 为控制成本，上传文件最大 10MB，只分析提取正文的前 60,000 字。上传页默认开启快速测试模式，仅分析 5 个核心标签；正式分析可关闭快速测试模式。
