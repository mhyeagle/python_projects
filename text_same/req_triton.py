import tritonclient.http as httpclient
from transformers import AutoTokenizer
import numpy as np

# 初始化客户端
client = httpclient.InferenceServerClient(url="localhost:8000")

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
texts = ["明月几时有", "把酒问青天"]
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="np")

# 准备 Triton 输入
input_ids = httpclient.InferInput("input_ids", inputs["input_ids"].shape, "INT64")
attention_mask = httpclient.InferInput("attention_mask", inputs["attention_mask"].shape, "INT64")
token_type_ids = httpclient.InferInput("token_type_ids", inputs["token_type_ids"].shape, "INT64")

input_ids.set_data_from_numpy(inputs["input_ids"].astype(np.int64))
attention_mask.set_data_from_numpy(inputs["attention_mask"].astype(np.int64))
token_type_ids.set_data_from_numpy(inputs["token_type_ids"].astype(np.int64))

# 发送请求
response = client.infer(
    model_name="paraphrase",
    inputs=[input_ids, attention_mask, token_type_ids]
)

# 获取输出（使用实际名称 'last_hidden_state'）
output = response.as_numpy("last_hidden_state")  # 形状: (2, 8, 384)
print("Output shape:", output.shape)  # 应输出: (2, 8, 384)

# 加权平均池化（考虑注意力掩码）
mask = inputs["attention_mask"]  # 形状: (2, 8)
mask_expanded = np.expand_dims(mask, axis=-1).astype(np.float32)  # 形状: (2, 8, 1)

# 计算加权平均值
sum_embeddings = np.sum(output * mask_expanded, axis=1)  # 形状: (2, 384)
sum_mask = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)  # 形状: (2, 1)
sentence_embeddings = sum_embeddings / sum_mask  # 形状: (2, 384)

print("Sentence embeddings shape:", sentence_embeddings.shape)  # 应输出: (2, 384)
print(sentence_embeddings)  # 打印嵌入向量