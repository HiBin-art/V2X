使用 Python 3.6 + TensorFlow 1.12.0 进行训练和测试。

如何使用代码：
要训练多智能体 RL 模型：main_marl_train.py + Environment_marl.py + replay_memory.py
要训练基准单智能体 RL 模型：main_sarl_train.py + Environment_marl.py + replay_memory.py
要在同一环境中测试所有模型：main_test.py + Environment_marl_test.py + replay_memory.py + '/model'。。
不建议在 “main_marl_train.py” 中使用 “Test” 模式。
