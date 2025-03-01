import torch
import torch.nn as nn

class LinearTrend(nn.Module):
    def __init__(self):
        super(LinearTrend, self).__init__()

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        batch_size, seq_len, num_channels = x.size()
        # 生成时间索引
        time_index = torch.arange(seq_len, dtype=torch.float32, device=x.device).unsqueeze(0).unsqueeze(-1)
        time_index = time_index.repeat(batch_size, 1, num_channels)

        # 计算时间序列数据与时间索引的平均值
        mean_x = x.mean(dim=1, keepdim=True)
        mean_t = time_index.mean(dim=1, keepdim=True)

        # 计算时间序列数据与时间索引的协方差
        cov_xt = ((x - mean_x) * (time_index - mean_t)).mean(dim=1, keepdim=True)

        # 计算时间索引的方差
        var_t = ((time_index - mean_t) ** 2).mean(dim=1, keepdim=True)

        # 计算线性回归的斜率和截距
        slope = cov_xt / var_t
        intercept = mean_x - slope * mean_t

        # 计算趋势成分
        trend = slope * time_index + intercept
        return trend
class FFT_Residual_decomp(nn.Module):
    """
    结合线性回归和傅里叶变换的时间序列分解块
    """

    def __init__(self, top_k=5):
        super(FFT_Residual_decomp, self).__init__()
        self.top_k = top_k
        self.linear_trend = LinearTrend()

    def forward(self, x):
        # 1. 使用线性回归法提取趋势成分
        trend_component = self.linear_trend(x)
        trend_residual = x - trend_component

        # 2. 对趋势残差部分进行傅里叶变换
        xf = torch.fft.rfft(trend_residual)
        freq = abs(xf)
        freq[0] = 0

        # 3. 获取最大的 top_k 个频率，确保 top_k 不超过 freq 的大小
        k = min(self.top_k, freq.shape[-1])
        top_k_freq, top_list = torch.topk(freq, k)
        xf[freq <= top_k_freq.min()] = 0

        # 4. 进行逆傅里叶变换，得到季节性成分
        if xf.numel() == 0:  # 确保 xf 不为空
            seasonal_component = torch.zeros_like(trend_residual)
        else:
            seasonal_component = torch.fft.irfft(xf, n=trend_residual.size(-1))

        # 5. 原始信号减去季节性成分，得到新的趋势成分
        trend_component_adjusted = x - seasonal_component

        return seasonal_component, trend_component_adjusted

class Model(nn.Module):
    """
    Decomposition-Linear
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # Decomposition parameters
        top_k = 5
        self.decomposition = FFT_Residual_decomp(top_k=top_k)
        self.individual = configs.individual
        self.channels = configs.enc_in

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()

            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len, self.pred_len))

                # Use these two lines if you want to visualize the weights
                # self.Linear_Seasonal[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
                # self.Linear_Trend[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

            # Use these two lines if you want to visualize the weights
            # self.Linear_Seasonal.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
            # self.Linear_Trend.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decomposition(x)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                                          dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.pred_len],
                                       dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output
        return x.permute(0, 2, 1)  # to [Batch, Output length, Channel]


