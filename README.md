# 强化学习系统使用指南

## 环境要求
- **Python**: 3.6
- **TensorFlow**: 1.12.0

## 代码使用说明

### 训练多智能体RL模型
需要同时运行以下三个文件：
```bash
python main_marl_train.py
python Environment_marl.py 
python replay_memory.py
训练基准单智能体RL模型
需要同时运行以下三个文件：

bash
python main_sarl_train.py
python Environment_marl.py 
python replay_memory.py
测试所有模型
在同一环境中测试需要运行：

bash
python main_test.py
python Environment_marl_test.py
python replay_memory.py
并确保模型文件存放在/model目录下
