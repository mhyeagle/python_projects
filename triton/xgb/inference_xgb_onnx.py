# infer.py
import requests
import json

# 准备输入数据（Iris-setosa）
data = [5.1, 3.5, 1.4, 0.2]

# Triton 推理请求
request_body = {
    "inputs": [
        {
            "name": "float_input",
            "shape": [1, 4],
            "datatype": "FP32",
            "data": data
        }
    ]
}

# 发送请求
response = requests.post("http://localhost:8000/v2/models/iris_xgb/infer", json=request_body)

# 打印结果
print(json.dumps(response.json(), indent=2))