import numpy as np
from PIL import Image
import tritonclient.http as httpclient


def preprocess(img_path):
    # 读取并调整图像大小
    img = Image.open(img_path).convert("RGB").resize((224, 224))

    # 转换为numpy数组并确保为float32
    img = np.array(img, dtype=np.float32)  # 先转换为float32

    # 执行所有数值运算
    img = img / 255.0  # 归一化
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std  # 标准化

    # 调整维度顺序并添加batch维度
    img = img.transpose(2, 0, 1)  # HWC → CHW
    img = np.expand_dims(img, axis=0)  # 添加batch维度 [1,3,224,224]

    # 最终确认数据类型
    assert img.dtype == np.float32, f"Expected float32 but got {img.dtype}"
    return img


def main():
    # 连接Triton服务器
    client = httpclient.InferenceServerClient(url="localhost:8000")

    try:
        # 预处理图像
        input_data = preprocess("img2.jpg")
        print(f"Input shape: {input_data.shape}, dtype: {input_data.dtype}")  # 调试信息

        # 构造请求
        inputs = [httpclient.InferInput("image_tensor", input_data.shape, "FP32")]
        inputs[0].set_data_from_numpy(input_data, binary_data=True)  # 显式设置binary_data

        outputs = [httpclient.InferRequestedOutput("class_logits")]

        # 发送推理请求
        response = client.infer(model_name="densenet_onnx", inputs=inputs, outputs=outputs)
        logits = response.as_numpy("class_logits")

        # 后处理
        probs = np.exp(logits - np.max(logits)) / np.sum(np.exp(logits - np.max(logits)), axis=1, keepdims=True)
        top5 = np.argsort(probs[0])[-5:][::-1]

        print("Top5 class indices:", top5)
        print("Probabilities:", probs[0][top5])

    except Exception as e:
        print(f"Error during inference: {str(e)}")
        raise


if __name__ == "__main__":
    main()