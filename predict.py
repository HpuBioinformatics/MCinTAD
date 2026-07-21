from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import find_peaks

class TestDataset(Dataset):
    def __init__(self, feature_file):
        data = np.load(feature_file)
        self.features = data[:, :121]
        self.index = data[:, -1]

        self.features = self.features.reshape(-1, 11 , 11)

        self.features = torch.tensor(self.features, dtype=torch.float32)

        if len(self.features.shape) == 3:
            self.features = self.features.unsqueeze(1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], int(self.index[idx])


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=1, padding=0, bias=False):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride,padding=padding, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class EfficientCA(nn.Module):
    def __init__(self, channel, gamma=2, b=1):
        super(EfficientCA, self).__init__()
        t = int(abs((math.log(channel,2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=int(k/2)) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.avg_pool(x) 
        y = self.conv(y.squeeze(-1).transpose(-1, -2)) 
        y = y.transpose(-1, -2).unsqueeze(-1) 
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class CoordAtt(nn.Module):
    def __init__(self, channels, reduction=8):
        super(CoordAtt, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Conv2d(channels, channels//reduction, 1),
            nn.BatchNorm2d(channels//reduction),
            nn.ReLU(inplace=True)
        )

        self.xfc = nn.Conv2d(channels//reduction, channels, 1)
        self.yfc = nn.Conv2d(channels//reduction, channels, 1)

    def forward(self, x):
        B, C, H, W = x.size()

        xap = F.adaptive_avg_pool2d(x, (H, 1))
        yap = F.adaptive_avg_pool2d(x, (1, W))

        mer = torch.cat([xap.transpose_(2, 3), yap], dim=3)
        fc1 = self.fc1(mer)
        
        xat, yat = torch.split(fc1, (H, W), dim=3)

        xat = torch.sigmoid(self.xfc(xat))
        yat = torch.sigmoid(self.yfc(yat))

        out = x * xat * yat
        return out


class Attention(nn.Module):
    def __init__(self, d_model, num_heads=4, dropout_rate=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

        self.norm1 = nn.LayerNorm(d_model)

        self.att = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout_rate, batch_first=True
        )

        self.norm2 = nn.LayerNorm(d_model)
        self.query = nn.Parameter(torch.randn(1, 1, d_model))

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):

        x_norm = self.norm1(x)
        att_out, _ = self.att(x_norm, x_norm, x_norm)   

        x = x + att_out
        x = self.norm2(x) 
        
        query = self.query.expand(x.size(0), -1, -1)
        weights = torch.matmul(query, x.transpose(1, 2))   
        weights = weights / (self.d_model ** 0.5)
        weights = F.softmax(weights, dim=-1)              
        context = torch.matmul(weights, x)             
        context = context.squeeze(1)                 
        
        context_in = context
        context = self.ffn(context)
        context = context_in + context
        return context

class DistanceFactor(nn.Module):
    def __init__(self):
        super().__init__()

        self.raw_beta = nn.Parameter(torch.tensor(0.0))  

        self.bias = nn.Parameter(torch.tensor(0.0))

        coords = torch.arange(11)
        dist_matrix = torch.abs(coords[:, None] - coords[None, :])
        self.register_buffer("dist_matrix", dist_matrix)

    def forward(self, x, resolution):
        device = x.device

        s = self.dist_matrix.to(device).float() * resolution
        s = torch.clamp(s, min=1.0)

        beta = 2.0 * torch.sigmoid(self.raw_beta)
        D = torch.exp(beta * torch.log(s) + self.bias)

        mask = (self.dist_matrix > 0).float().to(device)

        D = D * mask
        D = D / resolution
        D = D.unsqueeze(0).unsqueeze(0)


        return x * D

class MCinTAD(nn.Module):
    def __init__(self):
        super().__init__()

        self.dist_factor = DistanceFactor()
        self.conv1 = nn.Sequential(
            DepthwiseSeparableConv(1,128, kernel_size=3, padding=1),
            nn.ReLU(),
            DepthwiseSeparableConv(128,64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            EfficientCA(channel=64),
            CoordAtt(channels=64, reduction=8),
            nn.Flatten(), 
            nn.Linear(64*3*3,256),
            nn.ReLU()
        )


        self.conv2 = nn.Sequential(
            nn.Conv1d(1, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten()
        )

        self.LSTM = nn.LSTM(
            input_size=256,
            hidden_size=128,
            num_layers=2,
            dropout=0.4,
            batch_first=True,
            bidirectional=True
        )

        self.att = Attention(256, 4, 0.1)

        self.fusion_weight = nn.Parameter(torch.tensor(0.5))

        self.fc = nn.Sequential( 
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64,1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, resolution):
        
        x = F.normalize(x, p=1, dim=-1)
        x = self.dist_factor(x, resolution)

        x1 = self.conv1(x)

        B = x.size(0)
        x2 = x.squeeze(1)
        x2 = x2.reshape(B * 11, 1, 11)
        x2 = self.conv2(x2)
        x2 = x2.view(B, 11, 256)
        lstm_out, _ = self.LSTM(x2)
        x2 = self.att(lstm_out)

        fx = self.fusion_weight * x1 + (1 - self.fusion_weight) * x2
        out = self.fc(fx)
        out = self.sigmoid(out)
        
        return out

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
model = MCinTAD().to(device)
model.load_state_dict(torch.load("best_model.pt", map_location=device))
model.eval()


def generate_sample_features(resolution, chrom_list):
    res_str = f"{int(resolution/1000)}KB"
    for chrom in chrom_list:
        all_data = []

        probs_matrix = np.loadtxt(f'probs_mat/{chrom}_{res_str}_probs_mat.txt')

        pad_probs_matrix = np.pad(probs_matrix, 5,'constant')
        for l in range(len(probs_matrix)):
            sub_probs_mat = pad_probs_matrix[l:l+11, l:l+11]

            all_data.append(np.append(sub_probs_mat.flatten(), [-1, l+1]))

        all_data = np.array(all_data)
        all_data = all_data.astype('float32')
        np.save(f'samples_fea/{res_str}/{chrom}_{res_str}_samples_fea.npy', all_data) 



def find_peak(data):
    arr = np.array(data)
    max_indices, _ = find_peaks(arr)
    max_points = [(i+1) for i in max_indices]
    if data[0] > data[1]:
        max_points.insert(0,1)
    if data[-1] > data[-2]:
        max_points.append(len(data))
    return max_points


def predict_boundaries(resolution, chrom_list, bg_percent=None):
    res_str = f"{int(resolution/1000)}KB"
    for chrom in chrom_list:
        test_dataset = TestDataset(f'samples_fea/{res_str}/{chrom}_{res_str}_samples_fea.npy')
        test_loader = DataLoader(test_dataset, batch_size=32)
        all_probs_map = {}

        with torch.no_grad():
            for x,y in test_loader:
                x = x.to(device)
                probs = model(x, resolution)
                batch_probs = probs.squeeze(1)
                batch_pos = torch.where(batch_probs)[0]
                for i in batch_pos:
                    all_probs_map[y[i].item()] = probs[i].item()
        
        all_probs = []
        sorted_bins = sorted(all_probs_map.keys())
        for bin in sorted_bins:
            all_probs.append(all_probs_map[bin])
               
        peaks= find_peak(all_probs)
        threshold = 0.5
        if bg_percent:
            threshold = np.quantile(all_probs, bg_percent)

        final_boundaries = []
        for boundary in peaks:
            if all_probs_map[boundary] > threshold:
                final_boundaries.append(boundary)


        np.savetxt(f'results/{res_str}/{chrom}_{res_str}_predicted_boundaries.txt', final_boundaries, fmt='%d', delimiter='\t')


if __name__ == '__main__':

    resolution_list = [10000, 25000, 50000]
    chrom_list = [
        "chr20", "chr21", "chr22"
    ]

    bg_percent = None
    for resolution in resolution_list:
        res_str = f"{int(resolution/1000)}KB"
        print(f"predict the results for resolution: {res_str}")
        if not os.path.exists(f"samples_fea/{res_str}"):
            os.makedirs(f"samples_fea/{res_str}")
        if not os.path.exists(f"results/{res_str}"):
            os.makedirs(f"results/{res_str}")
        
        generate_sample_features(resolution, chrom_list)
        predict_boundaries(resolution, chrom_list, bg_percent=bg_percent)