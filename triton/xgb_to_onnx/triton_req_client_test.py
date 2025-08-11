import numpy as np
# import tritonclient.http as httpclient
import tritonclient.grpc as grpcclient

# Triton server 地址
TRITON_URL = "localhost:8000"
TRITON_GRPC = "localhost:8001"
# MODEL_NAME = "gr_youmightlike_xgb"
MODEL_NAME = "gn_sjtjs_xgb"

# 创建 HTTP 客户端
# client = httpclient.InferenceServerClient(url=TRITON_URL)
client = grpcclient.InferenceServerClient(url=TRITON_GRPC)

# 构造输入数据 (batch=2, feature=514)
# 这里用随机数据，你可以换成自己的特征
batch_size = 2
feature_size = 455
input_data = np.random.rand(batch_size, feature_size).astype(np.float32)

# http 创建输入对象
# inputs = []
# inputs.append(httpclient.InferInput("input", input_data.shape, "FP32"))
# inputs[0].set_data_from_numpy(input_data)

# grpc 创建输入对象
infer_input = grpcclient.InferInput("input", input_data.shape, "FP32")
infer_input.set_data_from_numpy(input_data)

# http 创建输出请求
# outputs = []
# outputs.append(httpclient.InferRequestedOutput("probabilities"))
# outputs.append(httpclient.InferRequestedOutput("label"))

# grpc 创建输出请求
outputs = [grpcclient.InferRequestedOutput("probabilities"), grpcclient.InferRequestedOutput("label")]

# http 发送推理请求
# response = client.infer(model_name=MODEL_NAME, inputs=inputs, outputs=outputs)

# grpc 创建推理请求
response = client.infer(model_name=MODEL_NAME, inputs=[infer_input], outputs=outputs)

# 获取并打印结果
probs = response.as_numpy("probabilities")  # shape: [batch, 2]
labels = response.as_numpy("label")         # shape: [batch, 1]

print("输入数据 shape:", input_data.shape)
print("预测概率:\n", probs)
print("预测标签:\n", labels)
