import torch
import torchvision
from flask import Flask, jsonify, request
import pickle
import config

# 加载模型
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
word_map = pickle.load(open(config.word_sequence_dict, "rb"))  # 词典
checkpoint = torch.load('model.pth.rar')
model = checkpoint['gpt']

model.eval()
# 初始输入是空，每次加上后面的对话信息
sentence = []
# 创建Flask应用程序
app = Flask(__name__)


# 定义API路由
@app.route('/predict', methods=['GET'])
def predict():
    """
    根据输入的文本，预测并返回模型的回答
    :param input: 用户输入的文本字符串
    :param model: 已训练好的模型
    :param word_map: 词汇表映射对象
    :param device: PyTorch设备（例如：'cpu'或'torch.device('cuda')')
    :return: 预测的回答字符串
    """
    input_text = request.args.get('input', '').strip()
    sentence = list(input_text) + ['<EOS>']
    sentence_vec = word_map.transform(sentence, max_len=None, add_eos=False)
    dec_input = torch.LongTensor(sentence_vec).to(device).unsqueeze(0)

    terminal = False
    start_dec_len = len(dec_input[0])

    while not terminal:
        if len(dec_input[0]) - start_dec_len > 100:
            next_symbol = word_map.dict['<EOS>']
            dec_input = torch.cat(
                [dec_input.detach(), torch.tensor([[next_symbol]], dtype=dec_input.dtype, device=device)], -1)
            break

        projected = model(dec_input)
        prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]
        next_word = prob.data[-1]
        next_symbol = next_word
        if next_symbol == word_map.dict['<EOS>']:
            terminal = True

        dec_input = torch.cat(
            [dec_input.detach(), torch.tensor([[next_symbol]], dtype=dec_input.dtype, device=device)], -1)

    out = dec_input.squeeze(0)
    out = word_map.inverse_transform(out.tolist())

    eos_indexs = [i for i in range(len(out)) if out[i] == '<EOS>']

    if len(eos_indexs) < 2:
        return "预测过程中出现问题，无法生成回答"

    answer = out[eos_indexs[-2] + 1:-1]
    answer = "".join(answer)

    return answer


# 运行应用程序
if __name__ == '__main__':
    app.run()