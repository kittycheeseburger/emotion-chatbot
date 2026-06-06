# Emotion Chatbot

一个前后端分离的情绪分析聊天机器人网页应用。当前版本已接入后端聊天接口，默认使用 GLM-4.7-Flash 生成对话回复，并使用基于 `hfl/chinese-roberta-wwm-ext` 训练的情绪分析模型返回当前用户情绪类型和情绪指数。本地 RAG 负责情绪画像和复盘，OpenAI ChatGPT 保留为可选回复模式。

## 项目结构

```text
.
├── backend/              # FastAPI 后端
│   ├── app/
│   │   ├── api/          # API 路由
│   │   ├── core/         # 配置
│   │   ├── ml/           # 情绪模型训练与推理预留
│   │   ├── schemas/      # 请求/响应模型
│   │   └── services/     # GLM 客户端等业务服务
│   ├── .env.example
│   └── requirements.txt
└── frontend/             # Vite + React 前端
    ├── src/
    │   ├── components/
    │   ├── data/
    │   ├── styles/
    │   └── App.jsx
    └── package.json
```

## 已完成内容

- 前后端分离目录结构
- README、依赖清单、环境变量示例
- FastAPI `/api/chat` 聊天接口
- GLM-4.7-Flash 聊天回复
- 本地 RAG 快速回复，可作为在线模型不可用时的兜底
- OpenAI ChatGPT 后端客户端，可通过配置切换
- React 前端聊天界面
- 当前用户情绪展示
- RAG 情绪画像：基于本地情绪策略知识库做向量检索，展示触发点、对话目标、建议语气、下一步行动
- RAG 复盘：不依赖 GLM，按需基于聊天上下文和检索策略生成本轮情绪摘要
- 基于 `mg1094/sentiment_analysis` 模型头结构的 Transformer 情绪分类器
- 训练脚本与模型推理服务

## 情绪模型

当前模型：

```text
hfl/chinese-roberta-wwm-ext
```

训练数据：

```text
backend\external\BERT_SMP2020-EWECT\data\clean\usual_train.txt
backend\external\BERT_SMP2020-EWECT\data\clean\virus_train.txt
backend\data\emotion_tix007\data\train.csv
```

标签体系：

```text
悲伤 / 快乐 / 爱 / 愤怒 / 恐惧 / 惊讶 / 平静
```

训练命令：

```bash
cd backend
.venv\Scripts\python.exe -m app.ml.train_sentiment --smp-dir external\BERT_SMP2020-EWECT\data\clean --tix007-train data\emotion_tix007\data\train.csv --output-dir models\sentiment_smp_roberta --epochs 2 --batch-size 16 --max-length 140 --validation-ratio 0.1 --max-per-label 10000 --no-include-weibo --include-tix007 --include-smp
```

模型输出：

```text
backend\models\sentiment_smp_roberta\model.pt
backend\models\sentiment_smp_roberta\tokenizer\
backend\models\sentiment_smp_roberta\metrics.json
```

本次验证集指标：

- Accuracy: `0.8402`
- Macro F1: `0.8540`
- 悲伤 F1: `0.7744`
- 快乐 F1: `0.8664`
- 爱 F1: `0.9871`
- 愤怒 F1: `0.8412`
- 恐惧 F1: `0.8540`
- 惊讶 F1: `0.8343`
- 平静 F1: `0.8206`

说明：SMP2020-EWECT 是真实微博情绪数据，作为当前主训练数据；TIX007 用于补充“爱”等细粒度标签。真实数据指标低于模板扩增数据是正常现象，但更接近实际聊天场景。后端还保留了轻量关键词后处理，用于修正“考砸了”“没考好”等明确受挫短句。

## RAG 情绪画像

右侧情绪画像使用本地 RAG，不会额外调用 GLM：

```text
backend\app\data\emotion_strategies.json
```

实现方式：

- 使用情绪分类模型得到当前情绪标签和指数。
- 根据用户输入识别可能触发点，例如学业受挫、工作压力、人际关系。
- 使用 `TfidfVectorizer` 将策略知识库向量化。
- 根据“用户消息 + 当前情绪 + 触发点 + 最近上下文”做相似度检索。
- 返回最匹配策略的对话目标、建议语气、下一步行动和复盘摘要。

这个模块是本地运行的，速度快，不受在线大模型速率限制影响。后续可以把 TF-IDF 替换成本地 embedding 模型，提升语义检索能力。

### 聊天回复模式

默认配置：

```text
CHAT_REPLY_MODE=glm
```

可选值：

```text
glm             # 默认，调用 GLM-4.7-Flash
openai/chatgpt  # 调用 OpenAI Responses API
auto            # 先调用 OpenAI，失败或限流时自动使用 RAG 兜底
rag             # 本地 RAG 回复，不调用外部 API，速度最快
```

需要在 `backend/.env` 中配置：

```text
GLM_API_KEY=你的 GLM API Key
GLM_API_BASE_URL=https://open.bigmodel.cn/api/paas/v4
GLM_MODEL=glm-4.7-flash
```

如果切换到 OpenAI 模式，再配置 `OPENAI_API_KEY`、`OPENAI_API_BASE_URL` 和 `OPENAI_MODEL`。如果你担心在线模型限流，可以把 `CHAT_REPLY_MODE` 改成 `rag`。

如果需要 GPU 训练，Windows 环境下建议安装 CUDA 版 PyTorch，例如：

```bash
.venv\Scripts\python.exe -m pip install --force-reinstall torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

## 本地运行

### 一键启动（推荐）

Windows 下直接双击项目根目录的：

```text
start-local.bat
```

脚本会自动检查后端虚拟环境、前端依赖，并分别启动：

```text
前端：http://127.0.0.1:5173/
后端：http://127.0.0.1:8000/
```

如果需要从 PowerShell 手动启动：

```powershell
.\start-local.ps1
```

停止本地服务：

```text
stop-local.bat
```

如果只想单独启动某一端，也可以使用：

```text
start-backend.bat
start-frontend.bat
```

可选参数：

```powershell
.\start-local.ps1 -NoBrowser      # 启动服务但不自动打开浏览器
.\start-local.ps1 -SkipInstall    # 跳过依赖检查，直接启动
.\start-local.ps1 -Hidden         # 后台启动，并把输出写到日志
```

如果 `backend/.env` 不存在，脚本会从 `backend/.env.example` 复制一份。你仍然需要在 `backend/.env` 中配置 `GLM_API_KEY`。

默认启动会打开后端和前端两个命令窗口，关闭窗口或运行 `stop-local.bat` 即可停止服务。使用 `-Hidden` 时，启动日志会写入：

```text
backend\logs\server.out.log
backend\logs\server.err.log
frontend\logs\server.out.log
frontend\logs\server.err.log
```

### 后端

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
uvicorn app.main:app --reload --port 8000
```

把你的 GLM API Key 放到 `backend/.env` 的 `GLM_API_KEY` 中。只有切换到 `CHAT_REPLY_MODE=openai` 时才需要配置 `OPENAI_API_KEY`。不要把 `.env` 提交到仓库。

### 前端

```bash
cd frontend
npm install
npm run dev
```

默认访问 `http://localhost:5173`。

如果需要修改后端地址，在 `frontend/.env` 中设置：

```bash
VITE_API_BASE_URL=http://127.0.0.1:8000/api
```

## 后续开发计划

1. 增加会话持久化和历史会话列表。
2. 增加情绪趋势折线图。
3. 针对中性/轻负面表达补充更多数据，进一步提升中性分类稳定性。
4. 完成部署配置和端到端测试。

## 可选增强建议

- 情绪趋势折线图：展示最近多轮对话的情绪变化。
- 会话摘要：自动生成本次聊天的情绪总结与行动建议。
- 风险提示：当连续多轮出现强烈负向情绪时展示温和提醒。
- 历史会话：保存不同聊天记录，便于用户复盘。
