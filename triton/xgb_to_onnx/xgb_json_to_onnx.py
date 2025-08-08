# 这个成功了
import xgboost as xgb
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType
import sys
import os

def json_to_onnx(json_path, onnx_path, n_features):
    booster = xgb.Booster()
    booster.load_model(json_path)  # 支持 JSON 格式

    initial_type = [('input', FloatTensorType([None, n_features]))]

    onnx_model = onnxmltools.convert_xgboost(booster, initial_types=initial_type)

    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    print(f"已成功转换并保存 ONNX 模型到：{onnx_path}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("用法: python json_to_onnx.py 模型.json 输出.onnx 特征数")
        sys.exit(1)

    json_file = sys.argv[1]
    onnx_file = sys.argv[2]
    features = int(sys.argv[3])

    if not os.path.exists(json_file):
        print(f"模型文件不存在: {json_file}")
        sys.exit(1)

    json_to_onnx(json_file, onnx_file, features)
