import numpy as np
import tritonclient.grpc as grpcclient

# Triton server 地址
TRITON_GRPC = "localhost:8001"
MODEL_NAME = "gn_sjtjs_xgb_json"

client = grpcclient.InferenceServerClient(url=TRITON_GRPC)

# 构造输入数据 (batch=2, feature=514)
# 这里用随机数据，你可以换成自己的特征
batch_size = 10
feature_size = 455
input_data = np.random.rand(batch_size, feature_size).astype(np.float32)
# print("输入数据:", input_data)

# grpc 创建输入对象
infer_input = grpcclient.InferInput("input__0", input_data.shape, "FP32")
infer_input.set_data_from_numpy(input_data)

# grpc 创建输出请求
outputs = [grpcclient.InferRequestedOutput("output__0")]

# grpc 创建推理请求
response = client.infer(model_name=MODEL_NAME, inputs=[infer_input], outputs=outputs)
print("推理结果:", response.get_response())

# 获取并打印结果
probs = response.as_numpy("output__0")  # shape: [batch, 2]
labels = response.as_numpy("output__1")

print("输入数据 shape:", input_data.shape)
print("预测概率:\n", probs)
print("预测标签:\n", labels)
