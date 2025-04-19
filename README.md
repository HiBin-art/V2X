# V2X 强化学习使用指南

## 🛠 环境要求
- Python: 3.6
- TensorFlow: 1.12.0

---

## 🚀 使用说明

### 🧠 训练多智能体强化学习模型（MARL）

```bash
# 需同时运行以下三个文件：
python main_marl_train.py          # 多智能体训练主程序
python Environment_marl.py         # 多智能体环境配置
python replay_memory.py            # 经验回放模块


### 🤖 训练单智能体强化学习模型（SARL）

```bash
# 需同时运行以下三个文件：
python main_sarl_train.py          # 单智能体训练主程序
python Environment_marl.py         # 共享环境配置
python replay_memory.py            # 经验回放模块
