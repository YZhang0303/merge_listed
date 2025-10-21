# 岗位描述本地Embedding处理（分块 + 落盘 memmap 方案 A）
# 使用Sentence Transformers本地模型进行文本嵌入处理

import pandas as pd
import numpy as np
import json
import torch
import os
import time
from tqdm import tqdm
import re
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer

# -------- 配置 --------
MODEL_NAME = "autodl-tmp/Qwen/Qwen3-Embedding-4B"   # 你的模型
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16                   # A40 支持 BF16；也可改 torch.float16
ATTN_IMPL = "sdpa"                       # 不装 flash-attn 也能吃到高性能内核
BATCH_SIZE = 32
CHUNK_SIZE = 50_000                     # 每批写盘的样本数；按内存情况可调大/小
EMB_MEMMAP_PATH = "autodl-tmp/embeddings.f16.memmap"
META_JSON_PATH  = "autodl-tmp/embeddings.meta.json"
INPUT_PARQUET   = "autodl-fs/list_par2.parquet"
INPUT_COLUMN    = "responsibilities"     # 你的文本列名
# 带回写的新文件路径
OUTPUT_PARQUET_MATCHED = "autodl-fs/list_par2_with_matches.parquet"
# ----------------------

# 新增路径：映射与形状数据（压缩存储）
MAPPING_NPZ_PATH = "autodl-tmp/emb_mapping.npz"     # 存放 exploded -> unique 的映射
ROW_COUNTS_NPZ_PATH = "autodl-tmp/emb_row_counts.npz"  # 存放每行的元素数量（清洗后）
CKPT_JSON_PATH = META_JSON_PATH + ".ckpt.json"  # 断点续跑元数据

# 检查必要的库
# -------- 工具函数：清洗与签名 --------
def clean_text(x: Any) -> Optional[str]:
    """将任意对象转换为干净的字符串；返回 None 表示应丢弃。
    规则：去首尾空白；过滤空串与常见空值标记；合并多余空白。
    """
    if x is None:
        return None
    s = str(x)
    s = s.strip()
    if not s:
        return None
    lower = s.lower()
    if lower in {"nan", "none", "null", "na", "n/a"}:
        return None
    s = re.sub(r"\s+", " ", s)
    return s if s else None


def normalize_to_list(x: Any) -> List[str]:
    """将单元格值归一化为清洗后的字符串列表。
    - 列表/元组/ndarray：逐项清洗
    - 其他标量：单元素列表
    - 缺失值：空列表
    """
    if isinstance(x, (list, tuple, np.ndarray)):
        items = x
    else:
        # 注意：pd.isna 对列表会报错，因此仅在非序列时判断
        items = [] if (x is None or (not isinstance(x, (list, tuple, np.ndarray)) and pd.isna(x))) else [x]
    cleaned: List[str] = []
    for it in items:
        t = clean_text(it)
        if t is not None:
            cleaned.append(t)
    return cleaned


def compute_input_signature(input_file: str, column_name: str, num_rows: int, total_texts: int, unique_texts: int) -> str:
    """用于断点续跑的一致性校验，尽量不引入昂贵哈希。
    由文件绝对路径、mtime、size 与统计量构成。
    """
    try:
        st = os.stat(input_file)
        return f"{os.path.abspath(input_file)}|{column_name}|{st.st_size}|{int(st.st_mtime)}|rows={num_rows}|M={total_texts}|U={unique_texts}"
    except Exception:
        return f"{os.path.abspath(input_file)}|{column_name}|rows={num_rows}|M={total_texts}|U={unique_texts}"


# 初始化本地模型
print("🔄 正在加载本地 embedding 模型...")
try:
    model = SentenceTransformer(
        MODEL_NAME,
        device=DEVICE,
        model_kwargs={
            "attn_implementation": ATTN_IMPL,
            "dtype": DTYPE,       
        },
        tokenizer_kwargs={
            "padding_side": "left"      
        },
    )
    print("✅ 模型加载成功")
    print(f"📊 模型信息: max_seq_len={model.max_seq_length}, dim={model.get_sentence_embedding_dimension()}")
except Exception as e:
    raise RuntimeError(f"❌ 模型加载失败: {e}\n💡 首次使用需要下载模型，请确保网络连接正常")

# 读取 + 清洗 + 形状保留
print("📥 读取并清洗数据...")
df = pd.read_parquet(INPUT_PARQUET)
col = df[INPUT_COLUMN]

# 按行清洗，保留列表形状
row_lists: List[List[str]] = [normalize_to_list(v) for v in col.tolist()]
row_counts = np.array([len(lst) for lst in row_lists], dtype=np.int32)
N = len(row_lists)
M = int(row_counts.sum())
print(f"✅ 原始行数: {N:,}，清洗后总文本数: {M:,}")

# 展平并全局去重，构建 exploded -> unique 的映射
print("🔁 去重并构建映射...")
texts: List[str] = [t for lst in row_lists for t in lst]
text_to_uid: Dict[str, int] = {}
unique_texts: List[str] = []
mapping = np.empty(M, dtype=np.uint32 if M < (1 << 32) else np.int64)
write_index = 0
for t in texts:
    uid = text_to_uid.get(t)
    if uid is None:
        uid = len(unique_texts)
        text_to_uid[t] = uid
        unique_texts.append(t)
    mapping[write_index] = uid
    write_index += 1
U = len(unique_texts)
print(f"✅ 唯一文本数: {U:,}  （去重率 {(1 - (U / M) if M else 0) * 100:.2f}%）")

# 写入形状与映射（压缩 npz）
np.savez_compressed(ROW_COUNTS_NPZ_PATH, counts=row_counts)
np.savez_compressed(MAPPING_NPZ_PATH, mapping=mapping)
print(f"🧾 已写出形状与映射: {ROW_COUNTS_NPZ_PATH}, {MAPPING_NPZ_PATH}")

# 预分配磁盘映射矩阵（float16 存储，空间更省）
d = model.get_sentence_embedding_dimension()
if U == 0:
    # 无可编码文本，仅输出元数据并退出
    meta = {
        "version": 2,
        "model": MODEL_NAME,
        "count_total": int(M),
        "unique_count": int(U),
        "num_rows": int(N),
        "embedding_dim": int(d),
        "stored_dtype": "float16",
        "normalize_embeddings": True,
        "attn_implementation": ATTN_IMPL,
        "torch_dtype": str(DTYPE).replace("torch.", ""),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": {"file": INPUT_PARQUET, "column": INPUT_COLUMN},
        "row_counts_path": ROW_COUNTS_NPZ_PATH,
        "mapping_path": MAPPING_NPZ_PATH,
    }
    with open(META_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("✅ 无可编码文本，已仅写出元数据与形状/映射文件。")
    raise SystemExit(0)

print("📝 预分配/打开 memmap 文件...")
input_sig = compute_input_signature(INPUT_PARQUET, INPUT_COLUMN, N, M, U)

write_ptr = 0
mode = "w+"
if os.path.exists(EMB_MEMMAP_PATH) and os.path.exists(CKPT_JSON_PATH):
    try:
        with open(CKPT_JSON_PATH, "r", encoding="utf-8") as f:
            ckpt = json.load(f)
        if (
            ckpt.get("input_signature") == input_sig
            and ckpt.get("model") == MODEL_NAME
            and int(ckpt.get("embedding_dim", -1)) == int(d)
            and int(ckpt.get("unique_count", -1)) == int(U)
        ):
            write_ptr = int(ckpt.get("write_ptr", 0))
            mode = "r+"
            print(f"🔁 检测到断点续跑，已完成 {write_ptr:,}/{U:,}")
        else:
            print("⚠️ 断点文件不匹配，将重新开始计算。")
    except Exception:
        pass

emb_mm = np.memmap(EMB_MEMMAP_PATH, dtype="float16", mode=mode, shape=(U, d))

# 分块编码并写入磁盘
print("🚀 开始分块编码并落盘（去重后写入，支持断点续跑）...")
for start in tqdm(range(write_ptr, U, CHUNK_SIZE), desc="Encoding (chunked)"):
    end = min(start + CHUNK_SIZE, U)
    batch_texts = unique_texts[start:end]

    vec = model.encode(
        batch_texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,     # 单位化，便于后续用点乘/内积
        convert_to_numpy=True,         # 直接回到 CPU/numpy，省 VRAM
        show_progress_bar=False,
    )

    # 降为 float16 再写盘（单位化后精度影响很小）
    emb_mm[start:end] = vec.astype("float16", copy=False)

    # 及时刷盘 + 释放显存
    emb_mm.flush()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 更新断点信息
    ckpt = {
        "model": MODEL_NAME,
        "embedding_dim": int(d),
        "unique_count": int(U),
        "write_ptr": int(end),
        "input_signature": input_sig,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "row_counts_path": ROW_COUNTS_NPZ_PATH,
        "mapping_path": MAPPING_NPZ_PATH,
        "memmap_path": EMB_MEMMAP_PATH,
        "finished": False,
    }
    with open(CKPT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(ckpt, f, ensure_ascii=False, indent=2)

# 关闭 memmap（删除对象触发刷盘）
del emb_mm

# 写入元数据，便于后续读取
meta = {
    "version": 2,
    "model": MODEL_NAME,
    "count_total": int(M),
    "unique_count": int(U),
    "num_rows": int(N),
    "embedding_dim": int(d),
    "stored_dtype": "float16",
    "normalize_embeddings": True,
    "attn_implementation": ATTN_IMPL,
    "torch_dtype": str(DTYPE).replace("torch.", ""),
    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    "source": {"file": INPUT_PARQUET, "column": INPUT_COLUMN},
    "row_counts_path": ROW_COUNTS_NPZ_PATH,
    "mapping_path": MAPPING_NPZ_PATH,
}
with open(META_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

# 最终更新断点信息
final_ckpt = {
    "model": MODEL_NAME,
    "embedding_dim": int(d),
    "unique_count": int(U),
    "write_ptr": int(U),
    "input_signature": input_sig,
    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    "row_counts_path": ROW_COUNTS_NPZ_PATH,
    "mapping_path": MAPPING_NPZ_PATH,
    "memmap_path": EMB_MEMMAP_PATH,
    "finished": True,
}
with open(CKPT_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(final_ckpt, f, ensure_ascii=False, indent=2)

print("✅ 全部完成")
print(f"📦 向量已写入: {EMB_MEMMAP_PATH}  （约 {U*d*2/1e9:.2f} GB）")
print(f"🧾 元数据: {META_JSON_PATH}")
print(f"🔗 形状与映射: {ROW_COUNTS_NPZ_PATH}, {MAPPING_NPZ_PATH}")

# 使用示例（以后读取时）：
# emb_unique = np.memmap(EMB_MEMMAP_PATH, dtype="float16", mode="r", shape=(U, d))
# counts = np.load(ROW_COUNTS_NPZ_PATH)["counts"]
# mapping = np.load(MAPPING_NPZ_PATH)["mapping"]
# emb_exploded = emb_unique[mapping]  # 形状：(M, d)
# # 恢复为列表形状：
# offsets = np.cumsum(counts)[:-1]
# emb_per_row = np.split(emb_exploded, offsets)
# 直接把 emb_unique 交给 Faiss/HNSWlib/Annoy 建索引即可；或分块读取以节省内存。

############################################
# 第二向量库：Embedding + FAISS 最近邻检索  #
############################################

# -------- 配置（请按需修改） --------
SECOND_PARQUET = "autodl-fs/second_lib.parquet"   # 第二库数据文件（约 2k 条）
SECOND_COLUMN  = "text"                             # 第二库文本列名

# 第二库向量落盘（与上方风格保持一致）
SECOND_EMB_MEMMAP_PATH = "autodl-tmp/second_lib.emb.f16.memmap"
SECOND_META_JSON_PATH  = "autodl-tmp/second_lib.emb.meta.json"

# 最近邻搜索输出（针对本脚本刚写出的 emb_unique 与第二库）
NN_TOPK = 1  # 需要的 TopK（题意为“最相似元素”，默认 1，可改）
NN_INDICES_MEMMAP_PATH = "autodl-tmp/nn_topk.indices.i32.memmap"
NN_SCORES_MEMMAP_PATH  = "autodl-tmp/nn_topk.scores.f32.memmap"
NN_META_JSON_PATH      = "autodl-tmp/nn_search.meta.json"
Q_CHUNK_SIZE = 200_000  # 查询分块大小（根据内存可调整）
# GPU FAISS 相关
USE_FAISS_GPU = True
GPU_DEVICE_ID = 0
FAISS_GPU_USE_FP16 = True  # 将索引向量以 FP16 存储，降低 VRAM 占用
# ----------------------------------

print("🔎 准备对第二库进行 embedding，并用 FAISS 检索...")

# 独立重跑自检：如需要，补齐关键常量与模型
if "META_JSON_PATH" not in globals():
    META_JSON_PATH = "autodl-tmp/embeddings.meta.json"
if "EMB_MEMMAP_PATH" not in globals():
    EMB_MEMMAP_PATH = "autodl-tmp/embeddings.f16.memmap"
if "MAPPING_NPZ_PATH" not in globals():
    MAPPING_NPZ_PATH = "autodl-tmp/emb_mapping.npz"
if "ROW_COUNTS_NPZ_PATH" not in globals():
    ROW_COUNTS_NPZ_PATH = "autodl-tmp/emb_row_counts.npz"
if "INPUT_PARQUET" not in globals():
    INPUT_PARQUET = "autodl-fs/list_par2.parquet"
if "INPUT_COLUMN" not in globals():
    INPUT_COLUMN = "responsibilities"
if "OUTPUT_PARQUET_MATCHED" not in globals():
    OUTPUT_PARQUET_MATCHED = "autodl-fs/list_par2_with_matches.parquet"

# 基础库导入（独立重跑时补齐）
try:
    json  # type: ignore[name-defined]
except NameError:
    import json
try:
    np  # type: ignore[name-defined]
except NameError:
    import numpy as np
try:
    pd  # type: ignore[name-defined]
except NameError:
    import pandas as pd
try:
    os  # type: ignore[name-defined]
except NameError:
    import os
try:
    time  # type: ignore[name-defined]
except NameError:
    import time
try:
    List  # type: ignore[name-defined]
except NameError:
    from typing import List

# 若模型未初始化，则按默认配置加载
try:
    model  # type: ignore[name-defined]
except NameError:
    try:
        from sentence_transformers import SentenceTransformer  # noqa: F401
        import torch  # noqa: F401
    except Exception:
        pass
    if "MODEL_NAME" not in globals():
        MODEL_NAME = "autodl-tmp/Qwen/Qwen3-Embedding-4B"
    if "DEVICE" not in globals():
        DEVICE = "cuda" if ("torch" in globals() and getattr(torch, "cuda", None) and torch.cuda.is_available()) else "cpu"
    if "DTYPE" not in globals():
        DTYPE = torch.bfloat16 if "torch" in globals() else None
    if "ATTN_IMPL" not in globals():
        ATTN_IMPL = "sdpa"
    if "BATCH_SIZE" not in globals():
        BATCH_SIZE = 32
    print("🔄 正在加载本地 embedding 模型(独立重跑)...")
    model = SentenceTransformer(
        MODEL_NAME,
        device=DEVICE,
        model_kwargs={
            "attn_implementation": ATTN_IMPL,
            "dtype": DTYPE,
        },
        tokenizer_kwargs={
            "padding_side": "left",
        },
    )
    print("✅ 模型加载成功(独立重跑)")

# 支持独立重跑：加载第一步产物的元数据、形状与映射
try:
    with open(META_JSON_PATH, "r", encoding="utf-8") as f:
        _meta0 = json.load(f)
    d_meta = int(_meta0.get("embedding_dim", 0))
    U_meta = int(_meta0.get("unique_count", 0))
    row_counts_path = _meta0.get("row_counts_path", ROW_COUNTS_NPZ_PATH)
    mapping_path = _meta0.get("mapping_path", MAPPING_NPZ_PATH)
    # 校验维度与当前模型一致
    d_model = model.get_sentence_embedding_dimension()
    if d_meta != d_model:
        raise RuntimeError(
            f"❌ 第一库向量维度({d_meta})与当前模型维度({d_model})不一致，请确认使用同一模型。"
        )
    d = d_model
    U = U_meta
    row_counts = np.load(row_counts_path)["counts"]
    mapping = np.load(mapping_path)["mapping"]
except Exception as _err:
    raise RuntimeError("❌ 重跑第二步失败：无法加载第一步的元数据/形状/映射。") from _err

# 1) 加载已存在的第二库向量（memmap 或通过 meta 推断）
print("📥 加载第二库向量...")
_second_emb_path = globals().get("SECOND_EMB_MEMMAP_PATH", None)
if not _second_emb_path or not os.path.exists(_second_emb_path):
    # 尝试通过 meta 读取
    try:
        with open(SECOND_META_JSON_PATH, "r", encoding="utf-8") as f:
            _meta2 = json.load(f)
        _second_emb_path = _meta2.get("memmap_path")
        d_from_meta = int(_meta2.get("embedding_dim", 0))
        if d_from_meta:
            d = d_from_meta
    except Exception:
        pass
if not _second_emb_path or not os.path.exists(_second_emb_path):
    raise FileNotFoundError("❌ 未找到第二库向量文件，请检查 SECOND_EMB_MEMMAP_PATH 或 SECOND_META_JSON_PATH")

# 从元数据或文件大小推断 S
try:
    if "_meta2" not in locals():
        with open(SECOND_META_JSON_PATH, "r", encoding="utf-8") as f:
            _meta2 = json.load(f)
    S = int(_meta2.get("count"))
    d_meta2 = int(_meta2.get("embedding_dim", d))
    d = d_meta2 or d
except Exception:
    # 兜底：根据文件大小推断
    st = os.stat(_second_emb_path)
    elem_bytes = 2  # float16
    d = int(d)
    S = st.st_size // (elem_bytes * max(d, 1))

vec2_mm = np.memmap(_second_emb_path, dtype="float16", mode="r", shape=(S, d))
vec2 = np.asarray(vec2_mm, dtype="float32")  # faiss 要 float32

# 3) 构建 FAISS 索引（内积）
print("🏗️  构建 FAISS IndexFlatIP 索引（GPU 优先）...")
try:
    import faiss  # pip install faiss-gpu
except Exception as e:
    raise RuntimeError("❌ 导入 faiss 失败。请先安装：pip install faiss-gpu") from e

if USE_FAISS_GPU:
    if not hasattr(faiss, "StandardGpuResources"):
        raise RuntimeError("❌ 当前 faiss 未启用 GPU 模块，请安装 GPU 版：pip install faiss-gpu")
    gpu_id = int(GPU_DEVICE_ID)
    if hasattr(faiss, "get_num_gpus"):
        num_gpus = faiss.get_num_gpus()
        if gpu_id < 0 or gpu_id >= num_gpus:
            raise RuntimeError(f"❌ 可用 GPU 数量为 {num_gpus}，配置的 GPU_DEVICE_ID={gpu_id} 不可用")
    res = faiss.StandardGpuResources()
    index_cpu = faiss.IndexFlatIP(d)
    # 采用 FP16 存储以降低 VRAM；如需禁用，设 FAISS_GPU_USE_FP16=False
    co = faiss.GpuClonerOptions()
    co.useFloat16 = bool(FAISS_GPU_USE_FP16)
    index = faiss.index_cpu_to_gpu(res, gpu_id, index_cpu, co)
else:
    index = faiss.IndexFlatIP(d)

index.add(vec2.astype("float32", copy=False))

# 4) 加载本脚本写出的 unique 向量作为查询，进行分块检索
print("🔍 开始对已保存的向量进行最近邻检索（按第二库）...")
emb_unique_q = np.memmap(EMB_MEMMAP_PATH, dtype="float16", mode="r", shape=(U, d))

# 结果 memmap：索引与相似度
I_mm = np.memmap(NN_INDICES_MEMMAP_PATH, dtype="int32", mode="w+", shape=(U, NN_TOPK))
D_mm = np.memmap(NN_SCORES_MEMMAP_PATH, dtype="float32", mode="w+", shape=(U, NN_TOPK))

# 进度条按查询向量数量推进
with tqdm(total=U, desc="Searching (vectors)", unit="vec") as pbar:
    for start in range(0, U, Q_CHUNK_SIZE):
        end = min(start + Q_CHUNK_SIZE, U)
        q = emb_unique_q[start:end].astype("float32", copy=False)
        D, I = index.search(q, NN_TOPK)
        I_mm[start:end] = I
        D_mm[start:end] = D
        pbar.update(end - start)

I_mm.flush()
D_mm.flush()
del I_mm, D_mm, emb_unique_q

# 5) 检索元数据
nn_meta = {
    "topk": int(NN_TOPK),
    "query_count": int(U),
    "index_count": int(S),
    "similarity": "cosine via inner product (normalized)",
    "query_memmap": EMB_MEMMAP_PATH,
    "index_memmap": SECOND_EMB_MEMMAP_PATH,
    "index_source": {"file": SECOND_PARQUET, "column": SECOND_COLUMN},
    "model": MODEL_NAME,
    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    "indices_memmap": NN_INDICES_MEMMAP_PATH,
    "scores_memmap": NN_SCORES_MEMMAP_PATH,
}
with open(NN_META_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(nn_meta, f, ensure_ascii=False, indent=2)

print("✅ 最近邻检索完成")
print(f"📦 第二库向量: {SECOND_EMB_MEMMAP_PATH}  （约 {S*d*2/1e6:.2f} MB）")
print(f"📦 Top{NN_TOPK} 索引: {NN_INDICES_MEMMAP_PATH}")
print(f"📦 Top{NN_TOPK} 分数: {NN_SCORES_MEMMAP_PATH}")
print(f"🧾 检索元数据: {NN_META_JSON_PATH}")

# 小示例：读取 Top1 结果并查看对应文本（如需要，可取消注释）
# I = np.memmap(NN_INDICES_MEMMAP_PATH, dtype="int32", mode="r", shape=(U, NN_TOPK))
# D = np.memmap(NN_SCORES_MEMMAP_PATH, dtype="float32", mode="r", shape=(U, NN_TOPK))
# with open(SECOND_META_JSON_PATH, "r", encoding="utf-8") as f:
#     _meta2 = json.load(f)
# top1_indices = I[:10, 0]
# for qi, idx in enumerate(top1_indices):
#     print(qi, idx, second_texts[idx] if 0 <= idx < len(second_texts) else None)

############################################
# 回写匹配文本到原始 DataFrame 并落盘            #
############################################

print("📎 正在将 Top1 匹配文本回写到 DataFrame 并保存...")

# 独立重跑所需：补齐依赖、路径与统计
if "NN_META_JSON_PATH" not in globals():
    NN_META_JSON_PATH = "autodl-tmp/nn_search.meta.json"
if "SECOND_META_JSON_PATH" not in globals():
    SECOND_META_JSON_PATH = "autodl-tmp/second_lib.emb.meta.json"
if "INPUT_PARQUET" not in globals():
    INPUT_PARQUET = "autodl-fs/list_par2.parquet"
if "INPUT_COLUMN" not in globals():
    INPUT_COLUMN = "responsibilities"
if "OUTPUT_PARQUET_MATCHED" not in globals():
    OUTPUT_PARQUET_MATCHED = "autodl-fs/list_par2_with_matches.parquet"

try:
    json  # type: ignore[name-defined]
except NameError:
    import json
try:
    np  # type: ignore[name-defined]
except NameError:
    import numpy as np
try:
    pd  # type: ignore[name-defined]
except NameError:
    import pandas as pd
try:
    os  # type: ignore[name-defined]
except NameError:
    import os
try:
    time  # type: ignore[name-defined]
except NameError:
    import time
try:
    re  # type: ignore[name-defined]
except NameError:
    import re
try:
    List  # type: ignore[name-defined]
except NameError:
    from typing import List

# 若必要的清洗函数在当前会话未定义，则补充精简版
try:
    normalize_to_list  # type: ignore[name-defined]
except NameError:
    def clean_text(x):
        if x is None:
            return None
        s = str(x).strip()
        if not s:
            return None
        if s.lower() in {"nan", "none", "null", "na", "n/a"}:
            return None
        s = re.sub(r"\s+", " ", s)
        return s if s else None
    def normalize_to_list(x):
        if isinstance(x, (list, tuple, np.ndarray)):
            items = x
        else:
            items = [] if (x is None or (not isinstance(x, (list, tuple, np.ndarray)) and pd.isna(x))) else [x]
        out = []
        for it in items:
            t = clean_text(it)
            if t is not None:
                out.append(t)
        return out

# 载入第一步 meta、形状与映射（若当前会话尚未有这些变量）
try:
    U  # type: ignore[name-defined]
    mapping  # type: ignore[name-defined]
    row_counts  # type: ignore[name-defined]
except NameError:
    if "META_JSON_PATH" not in globals():
        META_JSON_PATH = "autodl-tmp/embeddings.meta.json"
    with open(META_JSON_PATH, "r", encoding="utf-8") as f:
        _meta0 = json.load(f)
    U = int(_meta0.get("unique_count", 0))
    _row_counts_path = _meta0.get("row_counts_path", "autodl-tmp/emb_row_counts.npz")
    _mapping_path = _meta0.get("mapping_path", "autodl-tmp/emb_mapping.npz")
    row_counts = np.load(_row_counts_path)["counts"]
    mapping = np.load(_mapping_path)["mapping"]

# 载入检索 meta，获取 indices 路径与 topk
try:
    with open(NN_META_JSON_PATH, "r", encoding="utf-8") as f:
        _nnm = json.load(f)
    _indices_path = _nnm.get("indices_memmap", "autodl-tmp/nn_topk.indices.i32.memmap")
    _topk_read = int(_nnm.get("topk", 1))
    _second_src = _nnm.get("index_source", {})
except Exception:
    _indices_path = "autodl-tmp/nn_topk.indices.i32.memmap"
    _topk_read = 1
    _second_src = {}

# 构造第二库文本列表（用于将索引还原为文本）
try:
    _second_file = _second_src.get("file") if isinstance(_second_src, dict) else None
    _second_col = _second_src.get("column") if isinstance(_second_src, dict) else None
except Exception:
    _second_file, _second_col = None, None

if not _second_file:
    _second_file = globals().get("SECOND_PARQUET", "autodl-fs/second_lib.parquet")
if not _second_col:
    _second_col = globals().get("SECOND_COLUMN", "text")

def _safe_read_parquet_second(path: str) -> pd.DataFrame:
    try:
        return pd.read_parquet(path, use_legacy_dataset=True)
    except Exception:
        pass
    try:
        return pd.read_parquet(path, engine="fastparquet")
    except Exception:
        pass
    try:
        import pyarrow.parquet as pq  # type: ignore
        _table = pq.read_table(path)
        return _table.to_pandas()
    except Exception as e:
        raise e

df2_back = _safe_read_parquet_second(_second_file)
if _second_col not in df2_back.columns:
    raise KeyError(f"❌ 第二库文件中不存在列: {_second_col}")
second_texts: List[str] = []
for v in df2_back[_second_col].tolist():
    second_texts.extend(normalize_to_list(v))
S = len(second_texts)

# 读取 Top1 索引（U,)
I_ro = np.memmap(_indices_path, dtype="int32", mode="r", shape=(U, _topk_read))
uid_to_top1 = np.asarray(I_ro[:, 0], dtype=np.int64)
del I_ro

# 将 unique 级别的匹配结果映射回 exploded（长度 M）
exploded_top1 = uid_to_top1[mapping]

# 逐行组装成列表，长度与清洗后 responsibilities 对齐
matches_per_row: List[List[str]] = []
cursor = 0
for cnt in row_counts.tolist():
    if cnt == 0:
        matches_per_row.append([])
        continue
    idxs = exploded_top1[cursor:cursor + cnt]
    row_matches = [second_texts[int(j)] if 0 <= int(j) < S else None for j in idxs]
    matches_per_row.append(row_matches)
    cursor += cnt

# 添加新列并写出 Parquet（支持独立重跑时重新读取原文件）

# 确定原始数据源路径：优先用第一步元数据中的 source.file
try:
    with open(META_JSON_PATH, "r", encoding="utf-8") as _fsrc:
        _meta_for_source = json.load(_fsrc)
    _src_file = _meta_for_source.get("source", {}).get("file", INPUT_PARQUET)
except Exception:
    _src_file = INPUT_PARQUET

def _safe_read_parquet(path: str) -> pd.DataFrame:
    # 1) pandas + pyarrow legacy 路径
    try:
        return pd.read_parquet(path, use_legacy_dataset=True)
    except Exception:
        pass
    # 2) pandas + fastparquet（可选安装）
    try:
        return pd.read_parquet(path, engine="fastparquet")
    except Exception:
        pass
    # 3) 直接用 pyarrow 读取
    try:
        import pyarrow.parquet as pq  # type: ignore
        _table = pq.read_table(path)
        return _table.to_pandas()
    except Exception as e:
        raise e

# 若 df 已在内存且形状匹配就用之，否则稳妥地重读
try:
    _df_tmp = df  # type: ignore[name-defined]
    if INPUT_COLUMN not in _df_tmp.columns or len(_df_tmp) != len(matches_per_row):
        raise RuntimeError("reload")
except Exception:
    _df_tmp = _safe_read_parquet(_src_file)

_df_tmp[INPUT_COLUMN + "_match"] = matches_per_row
_df_tmp.to_parquet(OUTPUT_PARQUET_MATCHED, index=False)
print(f"💾 已保存回写结果: {OUTPUT_PARQUET_MATCHED}")