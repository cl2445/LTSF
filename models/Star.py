import torch
import torch.nn as nn
import torch.nn.functional as F

class STAR(nn.Module):
    def __init__(self, d_series, d_core):
        super(STAR, self).__init__()
        """
        STar Aggregate-Redistribute Module
        """

        self.gen1 = nn.Linear(d_series, d_series)
        self.gen2 = nn.Linear(d_series, d_core)
        self.gen3 = nn.Linear(d_series + d_core, d_series)
        self.gen4 = nn.Linear(d_series, d_series)

    def forward(self, input, *args, **kwargs):
        batch_size, channels, d_series = input.shape

        # set FFN
        combined_mean = F.gelu(self.gen1(input))
        combined_mean = self.gen2(combined_mean)

        # stochastic pooling
        if self.training:
            ratio = F.softmax(combined_mean, dim=1)
            ratio = ratio.permute(0, 2, 1)
            ratio = ratio.reshape(-1, channels)
            indices = torch.multinomial(ratio, 1)
            indices = indices.view(batch_size, -1, 1).permute(0, 2, 1)
            combined_mean = torch.gather(combined_mean, 1, indices)
            combined_mean = combined_mean.repeat(1, channels, 1)
        else:
            weight = F.softmax(combined_mean, dim=1)
            combined_mean = torch.sum(combined_mean * weight, dim=1, keepdim=True).repeat(1, channels, 1)

        # mlp fusion
        combined_mean_cat = torch.cat([input, combined_mean], -1)
        combined_mean_cat = F.gelu(self.gen3(combined_mean_cat))
        combined_mean_cat = self.gen4(combined_mean_cat)
        output = combined_mean_cat

        return output, None

class Model(nn.Module):
    """
    Decomposition-Linear with STAR module
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

        # 是否使用 STAR 模块
        self.use_star = getattr(configs, 'use_star', True)  # 默认使用 STAR

        if self.use_star:
            # 初始化 STAR 模块
            self.star_trend = STAR(configs.d_model, configs.d_core)  # 适用于趋势分量
            self.star_seasonal = STAR(configs.d_model, configs.d_core)  # 适用于季节分量

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()

            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len, self.pred_len))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decomposition(x)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)

        if self.use_star:
            # 使用 STAR 模块对趋势和季节性分量进行增强
            trend_init, _ = self.star_trend(trend_init)  # 对趋势分量应用 STAR
            seasonal_init, _ = self.star_seasonal(seasonal_init)  # 对季节性分量应用 STAR

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