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
    booster.save_model("model.onnx")

    # with open("model.onnx", "wb") as f:
    #     f.write(onnx_model)

    # 2. 自动检测特征数（通过 dump_model 的 JSON）
    booster_json = json.loads(booster.save_raw("json").decode("utf-8"))
    # print(json.dumps(booster_json, indent=2, ensure_ascii=False))
    # first_tree = booster_json[0]
    # feature_ids = set()
    #
    # def collect_features(node):
    #     if "split" in node:
    #         feature_ids.add(node["split"])
    #         collect_features(node["children"][0])
    #         collect_features(node["children"][1])
    #
    # collect_features(first_tree)
    n_features = 514

    print(f"检测到特征数: {n_features}")

    # 3. 自动选择分类或回归
    # if model_type == "auto":
    #     if booster.attr("objective") and "reg" in booster.attr("objective"):
    #         model_type = "regressor"
    #     else:
    #         model_type = "classifier"
    #
    # model_type = "regressor"
    #
    # print(f"模型类型: {model_type}")
    #
    # # 4. 转成 sklearn API 模型
    # if model_type == "classifier":
    #     model = xgb.XGBClassifier()
    # else:
    #     model = xgb.XGBRegressor()
    # model._Booster = booster
    #
    # # 5. 定义输入
    # initial_type = [('input', FloatTensorType([None, n_features]))]
    #
    # # 6. 转换成 ONNX
    # onnx_model = convert_sklearn(model, initial_types=initial_type)
    #
    # # 7. 保存
    # with open(onnx_path, "wb") as f:
    #     f.write(onnx_model.SerializeToString())
    #
    # print(f"已保存 ONNX 模型到: {onnx_path}")

    # 8. 可选：测试推理
    # if test_run:
    #     session = ort.InferenceSession(onnx_path)
    #     inputs = np.random.rand(1, n_features).astype(np.float32)
    #     output = session.run(None, {'input': inputs})
    #     print("推理输出:", output)


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
