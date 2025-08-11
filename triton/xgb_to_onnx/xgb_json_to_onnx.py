# 这个成功了
import xgboost as xgb
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType
import sys
import os
import onnxruntime as ort
import numpy as np


def serialize_model_to_json(json_path):
    booster = xgb.Booster()
    booster.load_model(json_path)  # 支持 JSON 格式
    booster.save_model("model.json")
    booster.save_model("model.ubj")
    print(f"已成功保存 JSON 模型到：model.json")

def json_to_onnx(json_path, onnx_path, n_features):
    booster = xgb.Booster()
    booster.load_model(json_path)  # 支持 JSON 格式

    initial_type = [('input', FloatTensorType([None, n_features]))]

    onnx_model = onnxmltools.convert_xgboost(booster, initial_types=initial_type)

    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    print(f"已成功转换并保存 ONNX 模型到：{onnx_path}")

def test_run(onnx_path, n_features):
    session = ort.InferenceSession(onnx_path)
    inputs = np.random.rand(1, n_features).astype(np.float32)
    output = session.run(None, {'input': inputs})
    print("推理输出:", output)

if __name__ == "__main__":
    if len(sys.argv) == 2:
        json_file = sys.argv[1]
        serialize_model_to_json(json_file)
        sys.exit(0)

    if len(sys.argv) < 4 or len(sys.argv) > 5:
        print("用法: python json_to_onnx.py 模型.json 输出.onnx 特征数 isTest")
        sys.exit(1)

    json_file = sys.argv[1]
    onnx_file = sys.argv[2]
    features = int(sys.argv[3])
    isTest = sys.argv[4].lower() == 'true' if len(sys.argv) == 5 else False

    if not os.path.exists(json_file):
        print(f"模型文件不存在: {json_file}")
        sys.exit(1)

    if not isTest:
        json_to_onnx(json_file, onnx_file, features)

    test_run(onnx_file, features)
