import xgboost as xgb
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import json
import onnxruntime as ort
import numpy as np
import sys
import os

def gbdt_to_onnx(gbdt_path, onnx_path, model_type="auto", test_run=True):
    # 1. 加载 Booster
    booster = xgb.Booster()
    booster.load_model(gbdt_path)

    booster.save_model("model.json")
    booster.save_model("model.bin")


    n_features = 514

    print(f"检测到特征数: {n_features}")


    # 8. 可选：测试推理
    if test_run:
        session = ort.InferenceSession(onnx_path)
        inputs = np.random.rand(1, n_features).astype(np.float32)
        output = session.run(None, {'input': inputs})
        print("推理输出:", output)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python gbdt_to_onnx.py model.gbdt model.onnx [classifier|regressor|auto]")
        sys.exit(1)

    gbdt_file = sys.argv[1]
    onnx_file = sys.argv[2]
    model_type = sys.argv[3] if len(sys.argv) > 3 else "auto"

    if not os.path.exists(gbdt_file):
        print(f"模型文件不存在: {gbdt_file}")
        sys.exit(1)

    gbdt_to_onnx(gbdt_file, onnx_file, model_type)
