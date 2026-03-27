import os
import json
import ctypes
import cv2
import numpy as np

# ===================== 配置 =====================
RKLLM_MODEL = "./models/qwen3-vl-2b-instruct_w8a8_rk3588.rkllm"
VISION_MODEL = "./models/qwen3-vl-2b_vision_rk3588.rknn"
IMAGE_PATH = "3fe6db1a8d291be39192f9a06c74ce99.png"
PROMPT = "这张图片里有什么？详细描述一下"

# ===================== 加载 librkllmrt.so =====================
lib = ctypes.CDLL("./lib/librkllmrt.so")

# ===================== 图像预处理 =====================
def get_image_embedding(image_path):
    print("提取图像特征...")
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (448, 448))
    img = img.astype(np.float32) / 255.0
    return img.tobytes()

# ===================== 运行推理 =====================
print("加载模型...")

# 构造输入
img_data = get_image_embedding(IMAGE_PATH)

# 调用 RKLLM 推理
lib.rkllm_create.argtypes = [ctypes.c_char_p]
lib.rkllm_create.restype = ctypes.c_void_p

llm = lib.rkllm_create(RKLLM_MODEL.encode())

# 输入图文
prompt_bytes = (f"<img>{IMAGE_PATH}</img>{PROMPT}").encode()
lib.rkllm_process_input(llm, prompt_bytes, img_data, len(img_data))

# 输出回答
print("\n模型回答：")
result = ctypes.c_char_p()
while lib.rkllm_stream_output(llm, ctypes.byref(result)) == 0:
    if result.value:
        print(result.value.decode(), end="", flush=True)

print("\n\n完成！")
lib.rkllm_destroy(llm)
