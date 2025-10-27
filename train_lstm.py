# 导入PyTorch核心库，用于深度学习模型构建和训练
import torch
# 导入PyTorch神经网络模块，包含各种网络层的定义
import torch.nn as nn
# 导入NumPy，用于数值计算和数组操作
import numpy as np
# 导入Pandas，用于数据处理和分析
import pandas as pd
# 导入MetaTrader5接口，用于获取外汇交易数据
import MetaTrader5 as mt5
# 从scikit-learn导入数据归一化工具
# 从scikit-learn导入数据归一化工具
from sklearn.preprocessing import MinMaxScaler

# ==================== CUDA设备检查 ====================
# 强制检查CUDA是否可用，如果不可用则给出警告和安装指引
if not torch.cuda.is_available():
    # 打印分隔线，使警告信息更醒目
    print("=" * 60)
    # 提示用户CUDA不可用
    print("警告: CUDA不可用!")
    # 显示当前安装的PyTorch版本号
    print(f"当前PyTorch版本: {torch.__version__}")
    # 说明安装的是CPU版本
    print("您安装的是CPU版本的PyTorch")
    # 提供换行，使信息更清晰
    print("\n请卸载当前PyTorch并安装CUDA版本:")
    # 提供卸载当前版本的命令
    print("pip uninstall torch torchvision torchaudio")
    # 提供安装CUDA版本的命令（针对CUDA 12.1）
    print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    # 打印分隔线
    print("=" * 60)
    # 询问用户是否继续使用CPU进行训练
    response = input("\n是否继续使用CPU训练? (y/n): ")
    # 如果用户不输入'y'，则退出程序
    if response.lower() != 'y':
        quit()

# 根据CUDA是否可用来选择计算设备（GPU或CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 打印分隔线
print(f"\n{'='*60}")
# 显示当前使用的计算设备
print(f"使用设备: {device}")
# 如果CUDA可用，显示GPU详细信息
if torch.cuda.is_available():
    # 显示GPU型号名称
    print(f"GPU型号: {torch.cuda.get_device_name(0)}")
    # 显示CUDA版本
    print(f"CUDA版本: {torch.version.cuda}")
    # 显示GPU可用显存大小（转换为GB单位）
    print(f"可用显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
# 打印分隔线
# 打印分隔线
print(f"{'='*60}\n")

# ==================== 连接MT5并获取历史数据 ====================
# 连接MT5获取数据
print("正在连接MT5...")
# 初始化MT5连接，如果失败则退出程序
if not mt5.initialize():
    # 提示用户MT5初始化失败的原因
    print("MT5初始化失败,请确保MT5已启动并登录")
    # 退出程序
    quit()

# 导入datetime模块用于处理日期和时间
from datetime import datetime, timedelta
# 获取当前日期时间作为数据结束时间
end_date = datetime.now()
# 计算开始时间为当前时间往前推5年（5*365天）
start_date = end_date - timedelta(days=5*365)

# 提示用户正在获取数据，显示数据范围
print(f"正在获取EURUSD H1数据 ({start_date.date()} 至 {end_date.date()})...")
# 从MT5获取EURUSD货币对的H1（1小时）时间框架的历史数据
rates = mt5.copy_rates_range("EURUSD", mt5.TIMEFRAME_H1, start_date, end_date)
# 关闭MT5连接，释放资源
mt5.shutdown()

# 检查是否成功获取数据，如果没有数据则退出
if rates is None or len(rates) == 0:
    # 提示用户获取数据失败的可能原因
    print("获取数据失败,请检查MT5中是否有EURUSD历史数据")
    # 退出程序
    quit()

# 将获取的数据转换为Pandas DataFrame格式，便于后续处理
df = pd.DataFrame(rates)
# 显示获取到的数据条数（使用千位分隔符格式化）
# 显示获取到的数据条数（使用千位分隔符格式化）
print(f"获取到 {len(df):,} 条EURUSD H1数据\n")

# ==================== 数据预处理 ====================
# 数据归一化 - 创建MinMaxScaler对象，用于将数据缩放到0-1范围
scaler = MinMaxScaler()
# 对开盘价、最高价、最低价、收盘价和成交量进行归一化处理
# fit_transform会先学习数据的最小值和最大值，然后进行归一化转换
df[['open', 'high', 'low', 'close', 'tick_volume']] = scaler.fit_transform(
    df[['open', 'high', 'low', 'close', 'tick_volume']])

# ==================== 构造训练数据集 ====================
# 构造训练数据 - 定义函数将时间序列数据转换为监督学习格式
def create_dataset(data, lookback=10):
    """
    将时间序列数据转换为监督学习数据集
    参数:
        data: 输入的时间序列数据
        lookback: 回溯窗口大小，表示用过去多少个时间步预测下一个时间步
    返回:
        X: 特征数据（过去lookback个时间步的数据）
        y: 标签数据（下一个时间步的收盘价）
    """
    # 初始化特征列表和标签列表
    X, y = [], []
    # 遍历数据，构造训练样本
    # 减去lookback+1是为了确保每个样本都有对应的标签
    for i in range(len(data) - lookback - 1):
        # 提取从i到i+lookback的数据作为特征（输入）
        X.append(data[i:i+lookback])
        # 提取i+lookback位置的收盘价（索引3）作为标签（输出）
        y.append(data[i+lookback, 3])  # 3代表收盘价列
    # 将列表转换为NumPy数组，并指定数据类型为float32（节省内存）
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32).reshape(-1, 1)

# 从DataFrame中提取需要的特征列（开、高、低、收、成交量）
features = df[['open', 'high', 'low', 'close', 'tick_volume']].values
# 调用create_dataset函数创建训练数据集，默认使用10个时间步作为回溯窗口
X, y = create_dataset(features)

# ==================== 划分训练集和验证集 ====================
# 计算训练集大小（80%用于训练）
train_size = int(len(X) * 0.8)
# 划分训练集特征数据（前80%）
X_train_np = X[:train_size]
# 划分训练集标签数据（前80%）
y_train_np = y[:train_size]
# 划分验证集特征数据（后20%）
X_val_np = X[train_size:]
# 划分验证集标签数据（后20%）
y_val_np = y[train_size:]

# 将训练集NumPy数组转换为PyTorch张量，并移动到指定的计算设备（GPU或CPU）
X_train = torch.from_numpy(X_train_np).to(device)
# 将训练集标签数据也转换为PyTorch张量并移动到设备
y_train = torch.from_numpy(y_train_np).to(device)
# 将验证集特征数据转换为PyTorch张量并移动到设备
X_val = torch.from_numpy(X_val_np).to(device)
# 将验证集标签数据转换为PyTorch张量并移动到设备
y_val = torch.from_numpy(y_val_np).to(device)

# 打印训练集和验证集的形状，便于确认数据维度
print(f"训练集形状: {X_train.shape}")  # 应该是 (训练样本数, 10, 5)
print(f"训练标签形状: {y_train.shape}")  # 应该是 (训练样本数, 1)
print(f"验证集形状: {X_val.shape}")  # 应该是 (验证样本数, 10, 5)
print(f"验证标签形状: {y_val.shape}\n")  # 应该是 (验证样本数, 1)

# ==================== 定义LSTM神经网络模型 ====================
# LSTM模型 - 定义一个简单的LSTM神经网络类
class SimpleLSTM(nn.Module):
    """
    简单的LSTM神经网络模型
    用于预测外汇价格的下一个收盘价
    """
    def __init__(self):
        """初始化模型结构"""
        # 调用父类的初始化方法
        super().__init__()
        # 定义LSTM层：
        # input_size=5: 输入特征数（开、高、低、收、成交量）
        # hidden_size=20: LSTM隐藏层的神经元数量
        # num_layers=1: LSTM层数
        # batch_first=True: 输入数据的第一个维度是batch_size
        self.lstm = nn.LSTM(input_size=5, hidden_size=20, num_layers=1, batch_first=True)
        # 定义全连接层，将LSTM的输出（20维）映射到1维（预测的收盘价）
        self.fc = nn.Linear(20, 1)

    def forward(self, x):
        """
        前向传播函数
        参数:
            x: 输入数据，形状为 (batch_size, sequence_length, input_size)
        返回:
            预测值，形状为 (batch_size, 1)
        """
        # 将输入数据传入LSTM层，获取输出
        # lstm_out形状: (batch_size, sequence_length, hidden_size)
        # _表示隐藏状态和细胞状态，这里不需要使用
        lstm_out, _ = self.lstm(x)
        # 取LSTM输出的最后一个时间步（[:, -1, :]），传入全连接层得到最终预测
        return self.fc(lstm_out[:, -1, :])

# 创建模型实例并移动到指定设备（GPU或CPU）
model = SimpleLSTM().to(device)
# 定义优化器，使用Adam算法，学习率设为0.001
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# 定义损失函数，使用均方误差（MSE）损失
# 定义损失函数，使用均方误差（MSE）损失
criterion = nn.MSELoss()

# ==================== 训练模型 ====================
# 训练
print("开始训练...")
# 打印分隔线，使训练过程更清晰
print(f"{'='*60}")

# 初始化记录训练过程的列表
train_losses = []  # 记录每个epoch的训练损失
val_losses = []  # 记录每个epoch的验证损失
best_val_loss = float('inf')  # 初始化最佳验证损失为无穷大
best_epoch = 0  # 记录最佳模型对应的epoch
patience = 20  # 早停的耐心值（验证损失20个epoch不下降就停止）
patience_counter = 0  # 早停计数器
best_model_state = None  # 保存最佳模型的状态

# 训练最多500个epoch（但可能会提前停止）
for epoch in range(500):
    # ========== 训练阶段 ==========
    model.train()  # 设置模型为训练模式
    # 清空之前的梯度，防止梯度累积
    optimizer.zero_grad()
    # 前向传播：将训练数据输入模型，得到预测输出
    output = model(X_train)
    # 计算训练损失：比较预测值和真实值之间的差异
    train_loss = criterion(output, y_train)
    # 反向传播：计算损失函数对模型参数的梯度
    train_loss.backward()
    # 更新参数：根据梯度和学习率更新模型参数
    optimizer.step()
    
    # ========== 验证阶段 ==========
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():  # 验证时不计算梯度，节省内存
        # 在验证集上进行前向传播
        val_output = model(X_val)
        # 计算验证损失
        val_loss = criterion(val_output, y_val)
    
    # 记录本epoch的损失值
    train_losses.append(train_loss.item())
    val_losses.append(val_loss.item())
    
    # 每10个epoch打印一次训练进度
    if (epoch + 1) % 10 == 0:
        # 显示当前epoch、训练损失和验证损失
        print(f"Epoch {epoch+1:3d} | 训练Loss: {train_loss.item():.6f} | 验证Loss: {val_loss.item():.6f}")
    
    # ========== 早停机制 ==========
    # 如果当前验证损失是最好的
    if val_loss.item() < best_val_loss:
        best_val_loss = val_loss.item()  # 更新最佳验证损失
        best_epoch = epoch + 1  # 记录最佳epoch
        patience_counter = 0  # 重置耐心计数器
        # 保存当前最佳模型的状态（深拷贝）
        best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    else:
        # 如果验证损失没有改善，增加计数器
        patience_counter += 1
    
    # 如果连续patience个epoch验证损失都没有改善，提前停止训练
    if patience_counter >= patience:
        print(f"\n早停触发! 验证损失已经{patience}个epoch没有改善")
        print(f"最佳模型出现在Epoch {best_epoch}，验证Loss: {best_val_loss:.6f}")
        break

# 恢复最佳模型的参数
if best_model_state is not None:
    model.load_state_dict(best_model_state)
    model = model.to(device)  # 确保模型在正确的设备上
    print(f"\n已恢复最佳模型 (Epoch {best_epoch})")

# 打印分隔线，表示训练结束
# 打印分隔线，表示训练结束
print(f"{'='*60}\n")

# ==================== 绘制训练曲线 ====================
# 绘制Loss曲线，可视化训练过程
print("正在生成训练曲线图...")
try:
    # 导入matplotlib绘图库
    import matplotlib.pyplot as plt
    # 导入os模块（如果之前没有导入）
    import os
    # 设置中文字体支持
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    
    # 创建图形和坐标轴
    plt.figure(figsize=(12, 5))
    
    # 第一个子图：训练损失和验证损失
    plt.subplot(1, 2, 1)
    # 绘制训练损失曲线（蓝色）
    plt.plot(train_losses, label='训练Loss', color='blue', linewidth=2)
    # 绘制验证损失曲线（橙色）
    plt.plot(val_losses, label='验证Loss', color='orange', linewidth=2)
    # 标记最佳模型的位置（红色虚线）
    plt.axvline(x=best_epoch-1, color='red', linestyle='--', label=f'最佳模型 (Epoch {best_epoch})')
    # 设置x轴标签
    plt.xlabel('Epoch')
    # 设置y轴标签
    plt.ylabel('Loss')
    # 设置图表标题
    plt.title('训练和验证损失曲线')
    # 显示图例
    plt.legend()
    # 显示网格
    plt.grid(True, alpha=0.3)
    
    # 第二个子图：放大后期的损失曲线（更清楚地看收敛情况）
    plt.subplot(1, 2, 2)
    # 只显示后80%的epoch，方便观察收敛细节
    start_idx = int(len(train_losses) * 0.2)
    plt.plot(range(start_idx, len(train_losses)), train_losses[start_idx:], 
             label='训练Loss', color='blue', linewidth=2)
    plt.plot(range(start_idx, len(val_losses)), val_losses[start_idx:], 
             label='验证Loss', color='orange', linewidth=2)
    plt.axvline(x=best_epoch-1, color='red', linestyle='--', label=f'最佳模型 (Epoch {best_epoch})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('训练和验证损失曲线（后80%）')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 调整子图之间的间距
    plt.tight_layout()
    # 保存图表到文件（获取当前脚本所在目录）
    current_dir = os.path.dirname(os.path.abspath(__file__))
    loss_plot_path = os.path.join(current_dir, "training_loss.png")
    plt.savefig(loss_plot_path, dpi=150, bbox_inches='tight')
    print(f"训练曲线已保存: {loss_plot_path}")
    # 显示图表
    plt.show()
except ImportError:
    # 如果没有安装matplotlib，跳过绘图
    print("未安装matplotlib库，跳过绘图。可运行: pip install matplotlib")
except Exception as e:
    # 捕获其他绘图错误
    print(f"绘图时出错: {e}")

print(f"{'='*60}\n")

# ==================== 导出ONNX模型 ====================
# 导出ONNX (先移到CPU) - ONNX是一种跨平台的模型格式
print("正在导出ONNX模型...")
# 将模型设置为评估模式（关闭dropout等训练特有的层）
model.eval()
# 将模型从GPU移动到CPU，因为ONNX导出在CPU上更稳定
model_cpu = model.cpu()
# 创建一个虚拟输入张量，用于ONNX导出时追踪模型结构
# 形状为 (1, 10, 5)：batch_size=1, sequence_length=10, input_size=5
dummy_input = torch.randn(1, 10, 5)

# 导入os模块用于文件路径操作
import os
# 获取脚本所在目录的绝对路径
script_dir = os.path.dirname(os.path.abspath(__file__))
# 构造ONNX模型的完整保存路径
onnx_path = os.path.join(script_dir, "lstm_model.onnx")
# 构造归一化参数的完整保存路径
scaler_path = os.path.join(script_dir, "scaler_params.npy")

# 显示脚本所在目录
print(f"脚本所在目录: {script_dir}")
# 显示模型将要保存的路径
print(f"模型将保存到: {onnx_path}\n")

# 使用try-except捕获可能的导出错误
try:
    # 导出模型为ONNX格式
    torch.onnx.export(
        model_cpu,  # 要导出的模型
        dummy_input,  # 模型的示例输入
        onnx_path,  # 保存路径
        export_params=True,  # 导出模型参数
        opset_version=11,  # ONNX算子集版本（11是较为通用的版本）
        do_constant_folding=True,  # 是否执行常量折叠优化
        input_names=['input'],  # 输入节点的名称
        output_names=['output'],  # 输出节点的名称
        dynamic_axes={'input': {0: 'batch_size'}}  # 动态维度设置（batch_size可变）
    )

    # 验证文件是否真的生成
    if os.path.exists(onnx_path):
        # 获取生成的ONNX文件大小（转换为KB）
        file_size = os.path.getsize(onnx_path) / 1024
        # 显示保存成功信息
        print(f"ONNX模型已保存: {onnx_path}")
        # 显示文件大小
        print(f"文件大小: {file_size:.2f} KB")

        # 验证ONNX模型的正确性
        import onnx
        # 加载刚才保存的ONNX模型
        onnx_model = onnx.load(onnx_path)
        # 检查模型格式是否正确
        onnx.checker.check_model(onnx_model)
        # 显示验证通过信息
        print("ONNX模型验证通过!")
    else:
        # 如果文件未生成，显示错误信息
        print(f"错误: ONNX文件未生成在 {onnx_path}")

except Exception as e:
    # 捕获并显示导出过程中的任何异常
    print(f"导出ONNX失败: {e}")
    # 导入traceback模块
    import traceback
    # 导入traceback模块
    import traceback
    # 打印详细的错误堆栈信息，便于调试
    traceback.print_exc()

# ==================== 保存归一化参数 ====================
# 保存归一化参数 - 将归一化时使用的最小值和最大值保存下来
# 保存为字典格式，包含'min'和'max'两个键
np.save(scaler_path, {'min': scaler.data_min_, 'max': scaler.data_max_})
# 显示归一化参数保存成功信息
print(f"归一化参数已保存: {scaler_path}")

# ==================== 输出归一化参数供MQL5使用 ====================
# 输出归一化参数供MQL5使用 - 生成可以直接复制到MQL5代码中的数组声明
# 打印分隔线
print(f"\n{'='*60}")
# 提示这些参数需要复制到EA代码中
print("归一化参数 (复制到LSTM_EA.mq5):")
# 打印分隔线
print(f"{'='*60}")
# 生成data_min数组的声明语句，包含5个特征的最小值（保留5位小数）
print(f"double data_min[5] = {{{', '.join([f'{x:.5f}' for x in scaler.data_min_])}}};")
# 生成data_max数组的声明语句，包含5个特征的最大值（保留5位小数）
print(f"double data_max[5] = {{{', '.join([f'{x:.5f}' for x in scaler.data_max_])}}};")
# 打印分隔线
print(f"{'='*60}\n")

# ==================== 输出训练总结 ====================
print(f"{'='*60}")
print("训练总结:")
print(f"{'='*60}")
print(f"总训练轮数: {len(train_losses)} epochs")
print(f"最佳模型: Epoch {best_epoch}")
print(f"最佳验证Loss: {best_val_loss:.6f}")
print(f"最终训练Loss: {train_losses[-1]:.6f}")
print(f"最终验证Loss: {val_losses[-1]:.6f}")
# 计算训练集和验证集的Loss差异，用于判断是否过拟合
loss_diff = abs(train_losses[-1] - val_losses[-1])
print(f"训练/验证Loss差异: {loss_diff:.6f}")
if loss_diff < 0.0001:
    print("✓ 模型状态: 良好，没有明显的过拟合")
elif loss_diff < 0.001:
    print("⚠ 模型状态: 尚可，有轻微的过拟合倾向")
else:
    print("✗ 模型状态: 可能存在过拟合，建议调整模型复杂度或增加数据")
print(f"{'='*60}\n")

# ==================== 输出完成信息和后续步骤 ====================
# 显示训练完成信息
print("训练完成! 下一步:")
# 提示步骤1：将ONNX模型文件复制到MT5的指定目录
print("1. 将 lstm_model.onnx 复制到 MT5/MQL5/Files/")
# 提示步骤2：将归一化参数复制到EA代码中
print("2. 将上面的归一化参数复制到 LSTM_EA.mq5")
# 提示步骤3：编译EA并运行回测
print("3. 编译并运行EA进行回测")
