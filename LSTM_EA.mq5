//+------------------------------------------------------------------+
//|                                                      LSTM_EA.mq5 |
//|                                      LSTM量化交易教学示例          |
//+------------------------------------------------------------------+
#property copyright "LSTM Trading Example"
#property version   "1.00"

#resource "\\Files\\lstm_model.onnx" as uchar lstm_model[]

#include <Trade\Trade.mqh>

input double LotSize = 0.01;           // 手数
input int StopLoss = 100;              // 止损点数
input int TakeProfit = 150;            // 止盈点数

long model_handle;
CTrade trade;

// 归一化参数(从训练脚本自动生成)
double data_min[5] = {0.95383, 0.95586, 0.95351, 0.95383, 9.0};
double data_max[5] = {1.23394, 1.23485, 1.23327, 1.23396, 24004.0};

//+------------------------------------------------------------------+
int OnInit()
{
   model_handle = OnnxCreateFromBuffer(lstm_model, ArraySize(lstm_model));
   if(model_handle == INVALID_HANDLE)
   {
      Print("ONNX模型加载失败");
      return INIT_FAILED;
   }

   // 设置输入形状: [batch_size=1, seq_len=10, features=5]
   ulong input_shape[] = {1, 10, 5};
   if(!OnnxSetInputShape(model_handle, 0, input_shape))
   {
      Print("设置ONNX输入形状失败");
      return INIT_FAILED;
   }

   // 设置输出形状: [batch_size=1, output_size=1]
   ulong output_shape[] = {1, 1};
   if(!OnnxSetOutputShape(model_handle, 0, output_shape))
   {
      Print("设置ONNX输出形状失败");
      return INIT_FAILED;
   }

   Print("ONNX模型加载成功,准备交易EURUSD H1");
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   OnnxRelease(model_handle);
}

//+------------------------------------------------------------------+
void OnTick()
{
   // 仅在新K线时执行
   static datetime last_bar = 0;
   datetime current_bar = iTime(_Symbol, PERIOD_H1, 0);
   if(current_bar == last_bar) return;
   last_bar = current_bar;

   // 准备输入数据: [1, 10, 5]
   // 使用double避免在传参或计算时发生不必要的类型转换警告
   double inputs[50];
   for(int i = 0; i < 10; i++)
   {
      double bar_open = iOpen(_Symbol, PERIOD_H1, i);
      double bar_high = iHigh(_Symbol, PERIOD_H1, i);
      double bar_low = iLow(_Symbol, PERIOD_H1, i);
      double bar_close = iClose(_Symbol, PERIOD_H1, i);
      // 使用double存储成交量，避免精度或类型转换警告
      double bar_volume = (double)iVolume(_Symbol, PERIOD_H1, i);

      // 归一化（结果仍为double）
      inputs[i*5 + 0] = (bar_open - data_min[0]) / (data_max[0] - data_min[0]);
      inputs[i*5 + 1] = (bar_high - data_min[1]) / (data_max[1] - data_min[1]);
      inputs[i*5 + 2] = (bar_low - data_min[2]) / (data_max[2] - data_min[2]);
      inputs[i*5 + 3] = (bar_close - data_min[3]) / (data_max[3] - data_min[3]);
      inputs[i*5 + 4] = (bar_volume - data_min[4]) / (data_max[4] - data_min[4]);
   }

   // 运行推理
   // 输出也使用double，确保与输入数组类型一致
   double output[1];
   if(!OnnxRun(model_handle, ONNX_DEFAULT, inputs, output))
   {
      Print("ONNX推理失败");
      return;
   }

   double current_price = iClose(_Symbol, PERIOD_H1, 0);
   double predicted_price = output[0] * (data_max[3] - data_min[3]) + data_min[3];

   // 交易逻辑: 预测价格上涨买入,下跌卖出
   if(PositionsTotal() == 0)
   {
      if(predicted_price > current_price * 1.0001)
      {
         double sl = current_price - StopLoss * _Point;
         double tp = current_price + TakeProfit * _Point;
         trade.Buy(LotSize, _Symbol, 0, sl, tp);
         Print("买入 | 当前:", current_price, " 预测:", predicted_price);
      }
      else if(predicted_price < current_price * 0.9999)
      {
         double sl = current_price + StopLoss * _Point;
         double tp = current_price - TakeProfit * _Point;
         trade.Sell(LotSize, _Symbol, 0, sl, tp);
         Print("卖出 | 当前:", current_price, " 预测:", predicted_price);
      }
   }
}
//+------------------------------------------------------------------+
