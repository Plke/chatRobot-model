import torch
import torchvision
from flask import Flask, jsonify, request
from PIL import Image

# 加载模型
model = torch.nn.Sequential(torch.nn.Linear(784, 128), torch.nn.ReLU(), torch.nn.Linear(128, 10))
model.load_state_dict(torch.load('mnist_model.pth'))
model.eval()

# 创建Flask应用程序
app = Flask(__name__)


# 定义API路由
@app.route('/predict', methods=['POST'])
def predict():
    # 从请求中获取图像数据
    image = request.files['image'].read()
    image = Image.open(io.BytesIO(image))

    # 预处理图像数据
    image = torchvision.transforms.ToTensor()(image)
    image = image.view(-1, 28 * 28)

    # 运行模型进行预测
    output = model(image)
    prediction = torch.argmax(output, dim=1)

    # 返回预测结果
    return jsonify({'prediction': prediction.item()})


# 运行应用程序
if __name__ == '__main__':
    app.run()