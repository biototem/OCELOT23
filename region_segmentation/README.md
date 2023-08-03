# 说明文档

## 内容列表

- [项目介绍](#项目介绍)
- [环境要求](#环境要求)
- [流程简介](#流程简介)

## 项目介绍

本项目是**分割-细胞检测**任务的一个子项目，来自 ocelot 比赛，官网连接是：[Home - Grand Challenge (grand-challenge.org)](https://ocelot2023.grand-challenge.org/)。

本项目架构开发时要求将 ```./main``` 设为 ```source_path``` ，运行时要求按如下格式调用：

```sh
# 指定工作目录与配置文件（仅共不同配置文件对比实验使用）
python ./main/start_train.py workspace=$pwd config=CONFIG.yaml

# 指定工作目录，默认配置文件（默认为工作目录下的 ./CONFIG.yaml）
python ./main/start_train.py workspace=$pwd

# 指定配置文件，默认工作目录（默认为当前命令行所在路径）
python ./main/start_train.py config=CONFIG.yaml

# 默认工作目录，默认配置文件（通常用法）
python ./main/start_train.py config=CONFIG.yaml
```

## 环境要求

本项目的开发环境是 ```python 3.9 + pytorch 1.10.1+cu113```。其他依赖酌情参考`requirements.txt`


## 流程简介

流程的启动代码是 ```./main/start_{process_name}.py``` ，而其对应的实现代码一律放在```./main/process/{name}``` 中。

本项目包含一套预研流程和三套主流程，分述如下：

### 预研流程
用于探索数据结构，或实现一些非标准化的手工作业，此类代码均在 ```./dev``` 下，每个文件均可独立运行

### 预构建流程
用于加载数据源信息，启动名 ```process_name = 'build'```，执行代码是：```python ./main/start_build.py```

该流程将在 ```./cache``` 下生成训练流程中会用到的重要数据，由于本项目是比赛项目，预处理流程较为简单，该数据仅包括 ```source.lib```

### 训练流程
用于获得有效权重，启动名 ```process_name = 'train'```，执行代码是：```python ./main/start_train.py```

该流程将在 ```./output/{output.target}``` 下生成每代的训练权重，其中 ```output.target``` 是  ```CONFIG.yaml ``` 中的配置项

### 评估流程
用于评选模型，启动名 ```process_name = 'evalute’```，执行代码是 ```python ./main/start_evaluate.py``` 和 ```python ./main/start_metric_visual.py``` 

```start_evaluate``` 将在目标路径下生成评估文档，以 Table 格式存储（详见 ```./resource/document/modules``` 关于 evaluate 的说明）

该流程涵盖了对 ROI 的预测和可视化（只需令 ```visual=True``` ），还有一个影子流程 ```process_name = 'predict'```，执行代码  ```python ./main/start_predict.py```  专用于以某一确定权重进行预测和可视化。
