# train_model.py

from sklearn.datasets import load_iris
from xgboost import XGBClassifier
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType
import numpy as np

# 1. 加载数据
iris = load_iris()
X, y = iris.data, iris.target

# 2. 训练 XGBoost 模型
model = XGBClassifier(
    n_estimators=10,
    max_depth=3,
    random_state=42,
    objective='multi:softprob'
)
model.fit(X, y)

# 3. 定义输入类型 —— 使用 onnxmltools 的 FloatTensorType
# ✅ 正确：是一个包含元组的列表
initial_types = [('float_input', FloatTensorType([None, 4]))]  # 注意：是 ( ... ) 元组！

# 4. 转换为 ONNX
onnx_model = onnxmltools.convert_xgboost(
    model,
    initial_types=initial_types,
    target_opset=12
)

# 5. 保存模型
onnxmltools.utils.save_model(onnx_model, 'xgb_iris.onnx')

print("✅ ONNX 模型已成功保存：xgb_iris.onnx")

# 6. 可选：测试 ONNX 模型
import onnxruntime as rt

sess = rt.InferenceSession("xgb_iris.onnx")
input_name = sess.get_inputs()[0].name
pred = sess.run(None, {input_name: X[:1].astype(np.float32)})

print("📌 ONNX 推理结果 - 类别:", pred[0])
print("📌 ONNX 推理结果 - 概率:", pred[1])