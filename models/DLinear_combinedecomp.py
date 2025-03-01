import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block using moving average
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class DFT_series_decomp(nn.Module):
    """
    Series decomposition block using DFT
    """

    def __init__(self, top_k=5):
        super(DFT_series_decomp, self).__init__()
        self.top_k = top_k

    def forward(self, x):
        # Perform FFT
        xf = torch.fft.rfft(x)
        # Get frequency magnitudes
        freq = abs(xf)
        # Set DC component (frequency 0) to 0
        freq[0] = 0
        # Extract top_k frequencies
        top_k_freq, top_list = torch.topk(freq, self.top_k)
        # Set frequencies outside the top_k to 0
        xf[freq <= top_k_freq.min()] = 0
        # Perform inverse FFT to get seasonal component
        x_season = torch.fft.irfft(xf)
        # Subtract seasonal component from original signal to get trend component
        x_trend = x - x_season
        return x_season, x_trend


class Model(nn.Module):
    """
    Combined decomposition model using both moving average and DFT
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # Decomposition methods
        kernel_size = 25
        self.moving_avg_decomp = series_decomp(kernel_size)
        self.dft_decomp = DFT_series_decomp(top_k=3)

        self.individual = configs.individual
        self.channels = configs.enc_in

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

        # Decompose using moving average
        seasonal_ma, trend_ma = self.moving_avg_decomp(x)
        # Decompose using DFT
        seasonal_dft, trend_dft = self.dft_decomp(x)

        # Combine seasonal and trend components
        seasonal_combined = (seasonal_ma + seasonal_dft) / 2
        trend_combined = (trend_ma + trend_dft) / 2

        seasonal_combined, trend_combined = seasonal_combined.permute(0, 2, 1), trend_combined.permute(0, 2, 1)

        if self.individual:
            seasonal_output = torch.zeros([seasonal_combined.size(0), seasonal_combined.size(1), self.pred_len],
                                          dtype=seasonal_combined.dtype).to(seasonal_combined.device)
            trend_output = torch.zeros([trend_combined.size(0), trend_combined.size(1), self.pred_len],
                                       dtype=trend_combined.dtype).to(trend_combined.device)

            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_combined[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_combined[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_combined)
            trend_output = self.Linear_Trend(trend_combined)

        x = seasonal_output + trend_output
        return x.permute(0, 2, 1)  # to [Batch, Output length, Channel]
