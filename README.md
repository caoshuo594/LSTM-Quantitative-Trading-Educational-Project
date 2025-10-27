# LSTM量化交易教学项目 - EURUSD H1策略

## 项目简介

基于深度学习LSTM网络的外汇量化交易完整解决方案,使用真实EURUSD历史数据训练模型,通过ONNX格式在MT5平台实现实盘交易。本项目涵盖从数据获取、模型训练、导出到回测的完整工作流,是学习AI量化交易的理想教学案例。

## 技术栈

- **深度学习框架**: PyTorch 2.5+ (支持CUDA GPU加速)
- **模型架构**: LSTM (Long Short-Term Memory)
- **模型格式**: ONNX (跨平台部署)
- **交易平台**: MetaTrader 5
- **编程语言**: Python 3.11 + MQL5
- **数据源**: MT5实时行情数据

## LSTM在量化交易中的优势

**LSTM (长短期记忆网络)** 是循环神经网络(RNN)的变体,特别适合时间序列预测:

1. **长期依赖捕捉**: 通过门控机制(遗忘门、输入门、输出门)记住长期价格趋势
2. **自动特征提取**: 无需手动构造技术指标,直接从原始OHLCV数据学习
3. **非线性建模**: 可拟合复杂的市场动态,优于传统线性模型
4. **梯度稳定性**: 解决传统RNN的梯度消失问题,支持深层网络训练

## 项目文件结构

```
LSTM/
├── requirements.txt      # Python依赖包清单
├── train_lstm.py         # 训练脚本(支持GPU加速)
├── lstm_model.onnx       # 训练生成的ONNX模型(~11KB)
├── scaler_params.npy     # MinMax归一化参数
├── training_loss.png     # 训练曲线图(自动生成)
├── LSTM_EA.mq5           # MQL5交易EA代码
├── prompt_simple.txt     # 简单版提示词模板
├── prompt_professional.txt # 专业版提示词模板
└── README.md             # 项目文档
```

## 快速开始

### 环境准备

**硬件要求**:
- CPU: 任意现代处理器
- GPU: NVIDIA显卡(可选,推荐RTX系列)
- 内存: 8GB+
- 硬盘: 1GB可用空间

**软件要求**:
- Python 3.11+
- MetaTrader 5 (已登录账户)
- CUDA 12.1+ (GPU训练需要)

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

**依赖包说明**:
- `torch`: PyTorch深度学习框架
- `onnx`: ONNX模型格式支持
- `MetaTrader5`: MT5 Python API
- `pandas`: 数据处理
- `scikit-learn`: 数据归一化
- `numpy`: 数值计算

**安装GPU版PyTorch** (如有NVIDIA显卡):
```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 2. 训练模型

```bash
python train_lstm.py
```

**训练流程**:
1. 自动检测CUDA并使用GPU加速(如可用)
2. 连接MT5终端获取EURUSD历史数据
3. 使用MinMaxScaler归一化OHLCV数据
4. **数据集划分**: 80%训练集 + 20%验证集
5. **智能训练**: 最多500个epoch，支持早停机制
6. **验证监控**: 实时追踪训练Loss和验证Loss
7. **最佳模型保存**: 自动保存验证Loss最低的模型
8. **可视化**: 生成训练曲线图(training_loss.png)
9. 导出ONNX格式模型
10. 保存归一化参数供推理使用

**训练输出示例**:
```
使用设备: cuda
GPU型号: NVIDIA GeForce RTX 4090
获取到 31,053 条EURUSD H1数据

训练集形状: torch.Size([24833, 10, 5])
验证集形状: torch.Size([6209, 10, 5])

开始训练...
Epoch  10 | 训练Loss: 0.307397 | 验证Loss: 0.301387
Epoch  20 | 训练Loss: 0.198500 | 验证Loss: 0.189706
...
Epoch 164 | 训练Loss: 0.000424 | 验证Loss: 0.000152  ← 最佳模型

早停触发! 验证损失已经20个epoch没有改善
已恢复最佳模型 (Epoch 164)

训练总结:
总训练轮数: 184 epochs
最佳模型: Epoch 164
最佳验证Loss: 0.000152
✓ 模型状态: 良好，没有明显的过拟合

训练曲线已保存: training_loss.png
ONNX模型已保存: lstm_model.onnx (10.27 KB)
```

### 3. 部署到MT5

#### 步骤A: 复制ONNX模型
```
将 lstm_model.onnx 复制到:
MT5安装目录/MQL5/Files/lstm_model.onnx
```

#### 步骤B: 验证归一化参数
训练脚本会输出归一化参数,确保与 `LSTM_EA.mq5` 第20-21行一致:
```cpp
double data_min[5] = {0.95383, 0.95586, 0.95351, 0.95383, 9.00000};
double data_max[5] = {1.23394, 1.23485, 1.23327, 1.23396, 24004.00000};
```

#### 步骤C: 编译EA
1. 将 `LSTM_EA.mq5` 复制到 `MT5/MQL5/Experts/`
2. 在MetaEditor中打开并编译(F7)

#### 步骤D: 策略回测
1. 打开策略测试器(Ctrl+R)
2. 配置参数:
   - EA: LSTM_EA
   - 品种: EURUSD
   - 周期: H1
   - 模式: 每笔成交
   - 日期: 选择样本外数据(如最近3个月)
   - 初始资金: 10,000 USD
3. 点击"开始"运行回测

### 4. EA参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| LotSize | 0.01 | 每笔交易手数 |
| StopLoss | 100 | 止损点数 |
| TakeProfit | 150 | 止盈点数 |

## 模型架构详解

### 网络结构

```
输入层: [batch_size, 10, 5]
   ├─ 10: 时间步长(10根历史K线)
   └─ 5: 特征维度(OHLCV)
    ↓
LSTM层: input_size=5, hidden_size=20, num_layers=1
   ├─ 遗忘门: 决定丢弃哪些历史信息
   ├─ 输入门: 决定存储哪些新信息
   └─ 输出门: 决定输出哪些信息
    ↓
全连接层: Linear(20 → 1)
    ↓
输出层: [batch_size, 1]
   └─ 预测下一根K线收盘价(归一化值)
```

**模型参数**:
- 总参数量: ~2,040个
- ONNX文件大小: 10.27 KB
- 推理速度: <1ms/次 (GPU)

### 数据预处理

**归一化公式**:
```python
normalized = (value - min) / (max - min)  # 映射到[0,1]
```

**反归一化公式**:
```python
real_value = normalized * (max - min) + min
```

### 训练参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 数据划分 | 80%/20% | 训练集/验证集 |
| 优化器 | Adam | 自适应学习率 |
| 学习率 | 0.001 | 初始学习率 |
| 损失函数 | MSE | 均方误差 |
| 批大小 | 全量 | 使用所有训练数据 |
| 最大轮数 | 500 | 最多训练500个epoch |
| 早停耐心 | 20 | 验证Loss 20轮不降则停止 |
| 最佳模型 | 自动保存 | 保存验证Loss最低的模型 |

## 交易逻辑

```python
每小时新K线形成时:
1. 获取最近10根H1 K线 [Open, High, Low, Close, Volume]
2. 使用训练时的参数归一化数据
3. 输入LSTM模型进行推理
4. 反归一化得到预测收盘价
5. 交易决策:
   - 预测价 > 当前价 * 1.0001 → 买入
   - 预测价 < 当前价 * 0.9999 → 卖出
   - 其他情况 → 不交易
6. 设置止损止盈并提交订单
```

## 性能优化

### 提升模型效果

1. **增加模型容量**:
   ```python
   self.lstm = nn.LSTM(input_size=5, hidden_size=50, num_layers=2)
   ```

2. **扩展时间窗口**:
   ```python
   lookback = 20  # 使用20根K线预测
   ```

3. **调整训练参数**:
   ```python
   patience = 30  # 增加早停耐心值
   optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)  # 降低学习率
   ```

4. **添加技术指标**:
   ```python
   features = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'macd']
   ```

5. **数据增强**:
   - 添加噪声
   - 时间窗口滑动
   - 多时间周期融合

### 防止过拟合

✅ **已实现的功能**:

1. **训练集/验证集分割**:
   ```python
   train_size = int(len(X) * 0.8)  # 80%训练，20%验证
   X_train = X[:train_size]
   X_val = X[train_size:]
   ```

2. **早停法(Early Stopping)**:
   ```python
   patience = 20  # 验证Loss 20轮不降则停止
   if val_loss < best_val_loss:
       best_model_state = model.state_dict()  # 保存最佳模型
   ```

3. **验证Loss监控**:
   ```python
   # 每10轮显示训练和验证Loss对比
   print(f"训练Loss: {train_loss:.6f} | 验证Loss: {val_loss:.6f}")
   ```

4. **训练曲线可视化**:
   - 自动生成 `training_loss.png`
   - 双曲线对比(训练vs验证)
   - 标记最佳模型位置

⚙️ **可选的进阶方法**:

5. **添加Dropout**:
   ```python
   self.dropout = nn.Dropout(0.2)
   ```

6. **L2正则化**:
   ```python
   optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
   ```

7. **学习率衰减**:
   ```python
   scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
   ```

## 常见问题排查

### 训练相关

**Q: 训练时CUDA不可用?**
```bash
# 检查PyTorch版本
python -c "import torch; print(torch.__version__)"

# 应显示 2.5.1+cu121 (带+cu121后缀)
# 如果是 2.5.1+cpu,需重装GPU版本:
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

**Q: MT5连接失败?**
```
解决方案:
1. 确保MT5已启动并登录
2. 检查工具→选项→EA交易→允许DLL导入
3. 尝试手动在MT5图表上加载EURUSD H1
```

**Q: 训练Loss不下降?**
```
可能原因:
1. 学习率过大 → 降低到0.0001
2. 数据未归一化 → 检查scaler
3. 数据质量差 → 检查MT5历史数据完整性
4. 模型过于简单 → 增加hidden_size或num_layers
```

**Q: 验证Loss远高于训练Loss?**
```
原因: 模型过拟合训练数据
解决方案:
1. 查看训练曲线图(training_loss.png)确认
2. 减少训练轮数或降低模型复杂度
3. 增加训练数据量
4. 添加Dropout层或L2正则化
5. 使用早停机制(已自动启用)
```

**Q: 如何判断模型训练效果好坏?**
```
关键指标:
1. ✅ 验证Loss越小越好 (如<0.001)
2. ✅ 训练Loss和验证Loss接近 (差异<10%)
3. ✅ Loss曲线平稳收敛不震荡
4. ✅ 早停在合理位置触发

查看方法:
- 打开 training_loss.png 查看曲线
- 阅读终端输出的"训练总结"
- 关注"模型状态"的诊断结果
```

### 回测相关

**Q: ONNX推理失败 "wrong dimension"?**
```cpp
// 确保在OnInit()中设置输入输出形状
ulong input_shape[] = {1, 10, 5};
OnnxSetInputShape(model_handle, 0, input_shape);

ulong output_shape[] = {1, 1};
OnnxSetOutputShape(model_handle, 0, output_shape);
```

**Q: 回测没有交易信号?**
```
检查清单:
1. 归一化参数是否一致
2. ONNX模型是否正确加载
3. 时间周期是否为H1
4. 查看Expert日志的预测值
```

**Q: 回测收益为负?**
```
这是正常的,原因:
1. LSTM难以预测随机市场
2. 模型过拟合训练数据
3. 交易成本(点差/滑点)未考虑
4. 需要优化止损止盈参数

建议:
- 仅用于学习,不要实盘
- 尝试不同市场和时间周期
- 结合其他技术指标
```

## 进阶扩展方向

### 1. 模型改进
- **双向LSTM**: 同时考虑过去和未来信息
- **GRU**: 更简单的门控单元,训练更快
- **Transformer**: 注意力机制,捕捉长期依赖
- **CNN-LSTM**: 卷积提取局部特征+LSTM捕捉时序

### 2. 特征工程
- **技术指标**: RSI, MACD, 布林带, KDJ
- **市场情绪**: VIX指数, 恐慌指数
- **基本面**: 经济数据, 新闻情绪分析
- **多周期**: M5/M15/H4数据融合

### 3. 策略优化
- **强化学习**: PPO/DQN优化交易决策
- **仓位管理**: Kelly公式动态调整手数
- **风控系统**: 最大回撤限制, 连续亏损止损
- **集成学习**: 多模型投票/加权平均

### 4. 工程化部署
- **实时推理服务**: FastAPI + Docker
- **模型监控**: MLflow跟踪模型性能
- **在线学习**: 定期用新数据微调模型
- **多品种交易**: 扩展到黄金/原油/股指

## 性能基准

### 训练性能 (RTX 4090)

| 指标 | 数值 |
|------|------|
| 数据量 | 31,053条 |
| 训练集 | 24,833条 (80%) |
| 验证集 | 6,209条 (20%) |
| 训练时间 | ~30秒 (含早停) |
| 实际轮数 | 164-184 (早停触发) |
| 最佳验证Loss | ~0.0001-0.0002 |
| 显存占用 | ~500MB |

### 推理性能 (MT5)

| 指标 | 数值 |
|------|------|
| 单次推理 | <1ms |
| H1周期延迟 | 可忽略 |
| CPU占用 | <1% |

## 免责声明

**重要提示**:

1. 本项目仅用于教学和研究目的,不构成任何投资建议
2. 金融市场存在不可预测性,过去表现不代表未来收益
3. LSTM模型可能过拟合历史数据,实盘表现可能与回测不符
4. 实盘交易前务必在模拟账户充分测试
5. 交易有风险,投资需谨慎,请勿使用无法承受损失的资金

## 参考资料

- [PyTorch官方文档](https://pytorch.org/docs/stable/index.html)
- [ONNX格式规范](https://onnx.ai/)
- [MQL5语言参考](https://www.mql5.com/zh/docs)
- [LSTM原理论文](https://www.bioinf.jku.at/publications/older/2604.pdf)
- [量化交易策略](https://www.quantstart.com/)

## 许可证

本项目采用 MIT 许可证开源,允许自由使用、修改和分发。

## 作者与贡献

本项目由AI辅助生成,用于教学演示。欢迎提出改进建议和Bug反馈。

---

**最后更新**: 2025-10-27
**版本**: 2.0.0
**更新内容**: 
- ✅ 新增训练集/验证集划分 (80%/20%)
- ✅ 新增早停机制 (Early Stopping)
- ✅ 新增最佳模型自动保存
- ✅ 新增训练曲线可视化
- ✅ 新增训练状态智能诊断
- ✅ 优化训练流程和文档说明

**适用人群**: 量化交易初学者、深度学习实践者、MQL5开发者
