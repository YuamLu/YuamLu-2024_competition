# AI-CUP_2024 Winter
- TEAM: 6113
- Author: Yu-An, Lu
- Rank: 4th / 487 teams（0.926174 Precision@1）

## TL;DR

以下是我們的方法的創新與亮點：

- **最大語意的 Chunking**：以sentence為單位，合併相鄰的句子，並預留了足夠的overlap上下文。
- **由淺入深的 Retrieval**：先用小chunk找出關鍵線索，再用大chunk整合線索並進行rerank。
- **對結果做 Stacking**：我們用了兩種表現不同 embedding model 進行檢索，並對結果的不一致之處使用 LLM 進行處理，提高了系統的魯棒性和準確率。

## 程式說明

1. 這個 repo 提供了用於複現我們方法的程式碼，以及完整的預處理後的資料、index 和模型生成的預測結果。
2. 一些較大的檔案，例如 index 檔案的部份，我們將其放在了 Google Drive 上，請前往[這裡](https://drive.google.com/drive/folders/1UVUw9jKRE-5HdA23BNMaP2AWYPvj9UOZ?usp=sharing)下載。（對於主辦單位提供的參考資料，請在競賽官網下載，並放在`Model`資料夾中）
3. 若要完整的從0開始複現我們的方法，需要準備以下的API key：
    - [LlamaParse](https://cloud.llamaindex.ai/login)   `os.environ["llamaparse"]`
    - [VoyageAI](https://www.voyageai.com/)   `os.environ["voyage"]`
    - [Cohere](https://cohere.com/)   `os.environ["cohere"]`
    - [Claude](https://www.anthropic.com/api)   `os.environ["claude"]`
4. 在`Preprocess`資料夾中，有將PDF轉換成Markdown的程式，以及將參考資料進行 chunking 和 embedding 的程式。
5. 在`Model`資料夾中，有完整的 RAG 系統的程式。
6. 在`main.py`中，可以對主辦單位提供的正式測試資料進行預測，並生成提交檔案。生成的檔案與我們實際提交在 Tbrain 平台上的檔案一致。

### 環境

- Python 3.11
- Ubuntu 20.04
- Intel I7-12700K
- 32GB RAM
- 512GB SSD

## 方法概述

### 1. Preprocess
對於 PDF 格式的 Finance 和 Insurance 資料，我們使用 Llamaindex 開發的服務 `LlamaParse`將文件統一轉換到 Markdown 格式。
（每個人都可以註冊並使用該API，每週有 7000 頁的免費額度）

### 2. Chunking
#### 針對全文的 chunking
為了最大限度地保留語意的完整性，我們以sentence為單位，在給定`max_chunk_size`的情況下，合併相鄰的句子，使chunk的大小不超過`max_chunk_size`。
同時，我們設置了`overlap`的參數，在每個chunk都納入之前的`overlap`個句子，確保上下文的完整性。
經過反覆的消融實驗，我們發現`max_chunk_size=256`和`overlap=2`在 embedding 的 cosine similarity 搜尋的效果最好。
#### 針對表格的 chunking
因為 finance 資料中，存在大量的財報判讀的表格，使用一般的 chunking 方法會導致表格的內容被拆散。
因此我們使用正規表達式，抽取出每個表格之後，將所有表格的純數字的值刪除（使用者的Query中不包含數值），同時根據`max_chunk_size`對表格進行拆分。
最終我們得到了8841個`tabular_chunk`，他們將和普通的chunk一起進行embedding和檢索。

### 3. Embedding
為了追求最好的 embedding 效果，我們分別使用了cohere的`embed-multilingual-v3.0`以及voyageai的`voyage-3` embedding model。
所有chunk都被embedding成向量，並預先儲存。
<br>兩種embedding model都會在稍後被我們使用到。

### 4. Retrieval
對於每個 query，我們將其 embedding 成向量，並計算其和所有給定的source list 中的文件的chunk的 cosine similarity，保留top-100進行備用。

### 5. Ranking
因為發現關鍵的資訊常常分散在文本中不同的地方，但使用過大的chunk_size又會造成過多的噪聲，我們採取了"小chunk去retrival，大chunk去rerank"的策略，先找出文檔中關鍵的線索，再將這些線索合併成更大的文檔進行rerank。<br>
我們使用`voyageai`的`rerank-2` 作為 ranking model。
對於每個query，在它對應的source_list中的f個文件，每個文件的相似度最高的4個chunk會被選出並合併成doc_chunk，得到最多f個doc_chunk。
接著我們用rerank model，將這些doc_chunk進行排序，並選出分數最高的doc_chunk作為最終的答案。

### 6. Stacking

對於cohere和voyageai的方法，都在第3~5步驟中進行了實驗。在主辦單位提供的150筆資料中，我們發現兩者的效果都不錯，但在insurance的表現更好，而後者在finance的表現更好。
於是我們將兩個方法的結果進行stacking，先比對兩個方法的結果，如果兩個方法的結果不同，則提取出兩個方法檢索出的文檔的全文，利用 Claude-3.5 Sonnet 模型進行比對，確定最終的答案。<br>
在最終的實驗中，我們發現這樣的方法不僅保持了兩個方法的優勢，提高最終的準確率，也讓我們的系統具有很強的魯棒性和可解釋性。
