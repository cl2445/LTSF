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

        # RevIN 集成
        self.revin = RevIN(num_features=self.channels)

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
        x = self.revin(x, mode='norm')  # RevIN 归一化
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
        x = x.permute(0, 2, 1)  # to [Batch, Output length, Channel]
        x = self.revin(x, mode='denorm')  # RevIN 反归一化
        return x


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        self.mask = None
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str, mask=None):
        # x [b,l,n]
        if mode == 'norm':
            self._get_statistics(x, mask)
            x = self._normalize(x, mask)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x, mask=None):
        self.mask = mask
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            if mask is None:
                self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
            else:
                assert isinstance(mask, torch.Tensor)
                # print(type(mask))
                x = x.masked_fill(mask, 0)  # in case other values are filled
                self.mean = (torch.sum(x, dim=1) / torch.sum(~mask, dim=1)).unsqueeze(1).detach()
                # self.mean could be nan or inf
                self.mean = torch.nan_to_num(self.mean, nan=0.0, posinf=0.0, neginf=0.0)

        if mask is None:
            self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()
        else:
            self.stdev = (torch.sqrt(torch.sum((x - self.mean) ** 2, dim=1) / torch.sum(~mask, dim=1) + self.eps)
                          .unsqueeze(1).detach())
            self.stdev = torch.nan_to_num(self.stdev, nan=0.0, posinf=None, neginf=None)

    def _normalize(self, x, mask=None):
        self.mask = mask
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean

        x = x / self.stdev

        # x should be zero, if the values are masked
        if mask is not None:
            # forward fill
            # x, mask2 = forward_fill(x, mask)
            # x = x.masked_fill(mask2, 0)

            # mean imputation
            x = x.masked_fill(mask, 0)

        if self.affine:
            # Safety check: ensure affine_weight has the correct size
            if self.affine_weight.size(0) != x.size(-1):
                print(f"Warning: affine_weight size ({self.affine_weight.size(0)}) doesn't match input channels ({x.size(-1)}). Reinitializing...")
                self.affine_weight = nn.Parameter(torch.ones(x.size(-1), device=self.affine_weight.device))
                self.affine_bias = nn.Parameter(torch.zeros(x.size(-1), device=self.affine_bias.device))
                self.num_features = x.size(-1)
            
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            # Safety check: ensure affine_weight has the correct size
            if self.affine_weight.size(0) != x.size(-1):
                print(f"Warning: affine_weight size ({self.affine_weight.size(0)}) doesn't match input channels ({x.size(-1)}). Reinitializing...")
                self.affine_weight = nn.Parameter(torch.ones(x.size(-1), device=self.affine_weight.device))
                self.affine_bias = nn.Parameter(torch.zeros(x.size(-1), device=self.affine_bias.device))
                self.num_features = x.size(-1)
            
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x