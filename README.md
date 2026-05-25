# NLP 課程 QA 系統

基於 BM25 檢索 + LLM 生成的課程文件問答系統。讀取 CSV 問題，從課程教材中檢索相關段落，透過 LLM 生成簡短精確的答案。

## 系統架構

```
問題 CSV → BM25 檢索 Top-k 段落 → LLM 生成答案 → 答案 CSV
```

**答案產生流程（優先順序）：**

1. **Cache** — 命中快取直接回傳
2. **規則層** — 高信心度的日期/數值/定義直接抽取（跳過 LLM）
3. **LLM** — Groq（主要）→ Gemini（備援），自動切換
4. **句子選取** — 所有 LLM 失敗時，從段落中選最相關句子

## 專案結構

```
├── qa.py                 # 主程式：批次 CSV → CSV
├── build_corpus.py       # 建立 BM25 索引
├── run.bat               # 一鍵執行
├── settings.json         # 系統設定
├── requirements.txt      # Python 套件
├── .env                  # API 金鑰
│
├── corpus/               # 課程教材（PDF/PPTX/DOCX/TXT/SRT）
│   ├── slides/
│   ├── handouts/
│   └── transcripts/
│
├── input/
│   └── questions.csv     # 輸入問題
├── output/
│   └── answers.csv       # 輸出答案
├── data/
│   ├── bm25_cache.pkl    # BM25 索引快取
│   └── llm_cache.json    # LLM 回應快取
│
└── src/
    ├── answerer.py       # 答案產生管線
    ├── retriever.py      # BM25 + WAND 檢索
    ├── bm25_store.py     # BM25 索引儲存
    ├── chunker.py        # 文本切段（size=300, overlap=60）
    ├── tokenizer.py      # jieba + 英文分詞
    ├── parser.py         # 多格式文件解析
    ├── llm_client.py     # LLM 客戶端（Groq/Gemini）
    ├── cache.py          # LLM 快取
    ├── post_process.py   # 答案後處理
    └── extractors/       # 規則抽取器
        ├── factoid.py    # 日期/數值/人名
        ├── definition.py # 定義題
        ├── enumeration.py# 列舉題
        └── sentence_picker.py # 句子選取
```

## 快速開始

### 一鍵執行

將課程教材放入 `corpus/` 資料夾後，雙擊 `run.bat` 即可。
首次執行會自動建立 venv、安裝套件、提示填入 API 金鑰、建立索引並執行問答。

### 手動執行

#### 1. 安裝

```bash
python -m venv venv
venv\Scripts\activate        # Windows CMD
# source venv/bin/activate   # Linux/macOS

pip install -r requirements.txt
```

#### 2. 設定 API 金鑰

```bash
cp .env.example .env
```

編輯 `.env`，填入 API 金鑰：

```
GROQ_API_KEY=...
GEMINI_API_KEY=...
```

- **Groq**（主要）：到 [console.groq.com](https://console.groq.com) 免費申請
- **Gemini**（備援）：到 [aistudio.google.com](https://aistudio.google.com/apikey) 免費申請

#### 3. 建立索引

將課程教材放入 `corpus/` 資料夾後：

```bash
python build_corpus.py --corpus ./corpus
```

#### 4. 執行問答

```bash
python qa.py --input input/questions.csv --output output/answers.csv --workers 4
```

## 設定說明

`settings.json` 主要參數：

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `chunk_size` | 300 | 文本切段大小（字元） |
| `chunk_overlap` | 60 | 段落重疊區域 |
| `top_k` | 5 | BM25 檢索段落數 |
| `bm25_min_score` | 0.5 | BM25 最低分數門檻 |
| `answer_max_chars` | 200 | 答案最大字數 |
| `workers` | 4 | 並行處理線程數 |

## 技術細節

- **檢索**：BM25 + WAND 剪枝，jieba 中文斷詞 + 英文 lowercase
- **LLM**：Groq `llama-3.3-70b-versatile`（主要）/ Gemini `gemini-2.5-flash`（備援）
- **速率限制**：Gemini 內建 4 RPM 限速 + 429 自動重試
- **快取**：LLM 回應以 SHA-256 hash 快取，避免重複 API 呼叫
- **評分標準**：80% 答案正確性 + 20% 效率/簡潔度
