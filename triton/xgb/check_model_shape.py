# check_model.py
import onnx

model = onnx.load("xgb_iris.onnx")

for out in model.graph.output:
    print(f"Output: {out.name}")
    shape = [dim.dim_value if dim.HasField('dim_value') else -1 for dim in out.type.tensor_type.shape.dim]
    print(f"Shape: {shape}")