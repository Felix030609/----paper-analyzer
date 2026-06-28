# Deploy Checklist

上线前请逐项确认：

- 本地运行 `streamlit run app/streamlit_app.py` 是否成功。
- 是否已重新生成并安全配置 `DEEPSEEK_API_KEY`。
- GitHub 仓库中是否没有 API key、`.env`、`.streamlit/secrets.toml`。
- GitHub 仓库中是否没有 `outputs/`。
- GitHub 仓库中是否没有 `models/`。
- Streamlit Cloud 的 Main file path 是否为 `app/streamlit_app.py`。
- Streamlit Secrets 是否添加 `DEEPSEEK_API_KEY`。
- Streamlit Secrets 是否添加 `DEEPSEEK_MODEL`。
- GitHub 中是否没有真实 API key。
- 页面是否显示当前模型名。
- 用户上传论文时是否不需要输入 API key。
- V4 Flash / V4 Pro 是否可选择。
- 部署后是否测试 PDF 上传。
- 部署后是否测试 TXT 上传。
- 部署后是否测试报告生成。
- 部署后是否测试下载 Markdown 报告。
- 部署后是否测试下载 PDF 报告；如果 PDF 依赖不可用，页面是否仍能下载 Markdown。
- 部署后是否测试下载 JSON 分析结果。
- 是否记录 `app_open` 事件。
- 是否记录 `file_uploaded` 事件。
- 是否记录 `analysis_started` 事件。
- 是否记录 `analysis_completed` 事件。
- `?admin=1` 是否能看到统计面板。
- 超大文件是否会被限制。
- 超长正文是否会被截断并提示用户。

提醒：当前 V0 不训练模型，不微调 RoBERTa。输出结果适合作为学术阅读辅助，正式使用前应人工复核标签分数和证据段落。
