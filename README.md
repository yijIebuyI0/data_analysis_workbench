# DataLab：AI 数据分析工作台

一个面向数据分析初学者和业务同学的 Streamlit MVP：上传 CSV、TXT、Excel 或 JSON，先评估并逐步确认清洗动作，再完成描述性统计与自定义聚合，最后让大模型基于结构化分析结果输出业务策略。

## 为什么做

实际的数据分析经常把大量时间花在重复的清洗与口径确认上。若直接让 AI 自动修改数据，用户又很难判断它做了什么。DataLab 用三个产品约束解决这个问题：

1. 每个清洗动作先预览影响，再由用户确认执行。
2. 原文件只读，整个流程操作会话内工作副本，下载时生成新文件名。
3. AI 只读取经过确认的统计摘要和聚合结果，不直接接收完整明细，也不能自行修改数据。

## 三段式体验

```mermaid
flowchart LR
    A["上传 CSV / TXT / XLSX / JSON"] --> B["自动数据评估"]
    B --> C["配置一个清洗步骤"]
    C --> D["预览命中数与前后样例"]
    D -->|"确认"| E["更新工作副本"]
    D -->|"取消"| C
    E --> F["确认清洗完成"]
    F --> G["描述统计与自定义聚合"]
    G --> H["确认分析结果"]
    H --> I["统计摘要作为工具消息交给 AI"]
    I --> J["输出业务策略并另存结果"]
```

### 1. 数据评估与分步清洗

- 自动展示行列数、空值、完全重复行、字段类型、唯一值和示例值。
- 支持空值处理、重复值处理、类型转换、文本去空格/大小写/显式值映射。
- 每次只配置一个步骤；先展示影响行数、结果行数、空值变化及前后样例，再确认执行。
- 已执行步骤形成会话内清洗日志，后续数据发生变化时会使分析和 AI 结论失效，避免旧结论继续被使用。

### 2. 描述统计与聚合

- 数值字段自动计算有效值数、求和、平均值、最小值和最大值。
- 文本字段展示唯一值数和 Top 3 分布。
- 通过下拉框选择指标、求和/平均值/最大值/最小值/计数/去重计数，以及可选分组字段。
- 用户确认后生成结构化分析上下文，才允许进入 AI 阶段。

### 3. AI 业务策略

- 系统提示词把模型限定为严谨的数据分析和业务策略顾问，要求区分数据事实、推断与待验证假设。
- 使用 DeepSeek Chat Completions 工具调用：模型先请求 `get_analysis_summary`，应用再用 `tool` 消息返回第二阶段的结构化结果。
- 完整原始明细不会进入模型上下文；默认只传汇总统计、最多 50 行聚合结果和清洗记录。
- 用户可以输入自己的业务问题，策略输出包含核心发现、业务解释、行动建议、验证指标和局限性。

## 快速开始

建议使用 Python 3.11 或 3.12。

```powershell
cd "E:\python\data_analysis_workbench"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run datalab.py
```

打开页面后，可以先上传 `sample_data/customer_orders.csv` 体验完整流程。前两个板块不需要 API Key。

若要使用 AI 策略，可任选一种配置方式：

```powershell
$env:DEEPSEEK_API_KEY="your-key"
$env:DEEPSEEK_MODEL="deepseek-v4-flash"
python -m streamlit run datalab.py
```

也可以把 `.streamlit/secrets.toml.example` 复制为 `.streamlit/secrets.toml` 后填写 Key，或只在页面密码框中临时输入。Key 不会写入清洗日志或下载文件。

## 测试

```powershell
python -m unittest discover -s tests -v
```

当前自动化测试覆盖：CSV/TXT/Excel/JSON 读取、空值/去重/类型/口径清洗、源数据不变、聚合计算、另存文件名、AI 工具消息传递，以及 Streamlit 首屏启动。

## 项目结构

```text
ai-data-analysis-workbench/
├── datalab.py              # Streamlit 页面与步骤状态控制
├── src/
│   ├── data_io.py          # 三种文件读取、编码/工作表处理与校验
│   ├── profiling.py        # 数据质量画像
│   ├── cleaning.py         # 清洗规则、执行和前后预览
│   ├── analytics.py        # 描述统计、自由聚合和 AI 上下文
│   ├── llm.py              # 系统提示词与 DeepSeek 工具调用
│   └── exporting.py        # 新文件与审计记录导出
├── tests/                  # 单元测试与 Streamlit 冒烟测试
├── sample_data/            # 可演示样例
└── docs/product-notes.md   # PRD-lite 与面试讲述材料
```

## 产品边界

- 单次最多 500,000 行，所有数据在当前 Python 进程内存中处理。
- Excel 一次分析一个工作表；支持 `.xlsx`，暂不支持旧版 `.xls`。
- TXT 按分隔符表格读取，自动识别逗号、制表符、分号或竖线及 UTF-8/GB18030 编码。
- JSON 支持对象数组、JSON Lines，以及包含 `data` 数组的对象；复杂嵌套结构会被扁平化。
- 暂不包含登录、权限、云存储、协作编辑、定时任务、数据库连接或模型成本管理。
- 当前清洗规则只在本次会话中生效；生产化时再加入可复用规则模板和持久化。

这些限制是 MVP 的主动取舍。下一步优先验证用户是否真的需要“每步确认”和“AI 策略”，再决定是否增加规则复用、图表推荐和数据库接入。

## 技术参考

- [Streamlit Session State](https://docs.streamlit.io/develop/api-reference/caching-and-state/st.session_state)
- [Streamlit file uploader](https://docs.streamlit.io/develop/api-reference/widgets/st.file_uploader)
- [DeepSeek Chat Completion](https://api-docs.deepseek.com/api/create-chat-completion)
- [DeepSeek Tool Calls](https://api-docs.deepseek.com/guides/tool_calls)
- [DeepSeek Models & Pricing](https://api-docs.deepseek.com/quick_start/pricing)
