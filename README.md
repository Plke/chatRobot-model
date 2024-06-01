# 聊天机器人模型接口

使用 pytorch 实现的中文聊天机器人模型接口，基于 GPT 模型，支持中文输入，支持中文输出

模型经过 70 次训练，效果一般，
想要模型文件的可以联系我

[前端 vue3 页面](https://github.com/Plke/chatRobot-front)
</br>
[后端 java 实现](https://github.com/Plke/chatRobot-backend)

数据集和模型代码来源于：
</br>
https://github.com/tt-s-t/Small-chinese-Chatbot-based-on-gpt
</br>
https://blog.csdn.net/weixin_55073640/article/details/131135003?spm=1001.2014.3001.5501

使用 flask 新增加了模型预测接口文档

## 使用方法

（注：origin_train.txt 是原始数据集，dataset.txt 是预处理后的数据文件，ws.pkl 是生成的词表，你想直接用或者自己重新生成都 ok）。<br />
在 config.py 里把参数设置好，然后先运行"sol_data.py"文件生成预处理后的数据文件。<br />
然后将 word2vec.py 文件中最后注释掉的部分取消注释，运行一遍得到词表，再将这部分内容进行注释。<br />
接着运行 train.py 文件进行模型训练，ok 之后运行 chat.py 文件测试你的模型结果啦~

运行 predict.py 文件开启接口
接口地址为

```
http://127.0.0.1:5000/predict?input=你好
```
