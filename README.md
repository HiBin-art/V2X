# V2X强化学习使用指南

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

### 训练单智能体RL模型
需要同时运行以下三个文件：
```bash
python main_sarl_train.py
python Environment_marl.py 
python replay_memory.py

### 测试所有智能体RL模型
需要同时运行以下三个文件：
```bash
python main_test.py
python Environment_marl_test.py
python replay_memory.py
