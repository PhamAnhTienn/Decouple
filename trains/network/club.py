import torch
import torch.nn as nn

class CLUBLoss(nn.Module):
    """
    计算两个序列张量间的CLUB互信息上界损失
    """
    def __init__(self, d, hidden_size=64):
        """
        Args:
            d: 输入张量的特征维度
            hidden_size: 条件网络隐藏层大小
        """
        super(CLUBLoss, self).__init__()
        
        # 条件网络：使用LSTM处理序列
        self.lstm = nn.LSTM(d, hidden_size, batch_first=True, bidirectional=True)
        self.net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),  # *2是因为双向LSTM
            nn.ReLU(),
            nn.Linear(hidden_size, d)
        )
        
        # 可学习的log方差
        self.log_var = nn.Parameter(torch.zeros(d))
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
    
    def forward(self, x1, x2, mode='mean'):
        """
        计算CLUB互信息上界
        
        Args:
            x1: 第一个张量 [l, b, d] 或 [b, l, d]
            x2: 第二个张量 [l, b, d] 或 [b, l, d]
            mode: 返回模式，'mean'返回平均损失，'sample'返回每个样本的损失
        """
        # 确保输入是 [b, l, d] 格式
        x1 = x1.transpose(0, 1)
        x2 = x2.transpose(0, 1)
        
        batch_size, seq_len, feat_dim = x1.size()
        
        # 通过LSTM处理序列
        lstm_out, _ = self.lstm(x1)  # [b, l, hidden*2]
        
        # 对每个时间步应用全连接层
        mu = self.net(lstm_out)  # [b, l, d]
        var = torch.exp(self.log_var)  # [d]
        
        # 计算正样本的对数似然
        pos_loss = -0.5 * torch.sum(
            (torch.square(x2 - mu) / var + self.log_var + 1.8379),
            dim=(1, 2)  # 对序列长度和特征维度求和
        )
        
        # 计算负样本的对数似然（使用batch内打乱）
        x2_shuffle = x2[torch.randperm(batch_size)]
        neg_loss = -0.5 * torch.sum(
            (torch.square(x2_shuffle - mu) / var + self.log_var + 1.8379),
            dim=(1, 2)
        )
        
        # 计算互信息上界
        club = pos_loss - neg_loss
        
        if mode == 'mean':
            return club.mean()
        elif mode == 'sample':
            return club
        else:
            raise ValueError(f'Unknown mode: {mode}')
    
    def learning_loss(self, x1, x2):
        """
        计算用于优化参数的学习损失
        """
        # 确保输入是 [b, l, d] 格式
        if x1.size(1) != x2.size(1):
            x1 = x1.transpose(0, 1)
            x2 = x2.transpose(0, 1)
            
        lstm_out, _ = self.lstm(x1)
        mu = self.net(lstm_out)
        var = torch.exp(self.log_var)
        
        loss = 0.5 * torch.mean(
            torch.sum(
                torch.square(x2 - mu) / var + self.log_var,
                dim=(1, 2)  # 对序列长度和特征维度求和
            )
        )
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

def example_usage():
    # 设置随机种子
    torch.manual_seed(0)
    
    # 生成示例序列数据
    batch_size = 32
    seq_len = 10
    dim = 8
    
    # 生成 [b, l, d] 格式的数据
    x1 = torch.randn(batch_size, seq_len, dim)
    x2 = 0.8 * x1 + 0.2 * torch.randn(batch_size, seq_len, dim)
    
    # 初始化CLUB损失
    club_loss = CLUBLoss(d=dim)
    optimizer = torch.optim.Adam(club_loss.parameters(), lr=1e-3)
    
    # 训练CLUB估计器
    print("Training CLUB estimator...")
    for step in range(100):
        optimizer.zero_grad()
        loss = club_loss.learning_loss(x1, x2)
        loss.backward()
        optimizer.step()
        
        if (step + 1) % 20 == 0:
            mi_estimate = club_loss(x1, x2)
            print(f'Step [{step+1}/100], MI Estimate: {mi_estimate.item():.4f}')
    
    # 使用训练好的估计器计算互信息上界
    with torch.no_grad():
        mi_bound = club_loss(x1, x2)
        print(f'\nFinal MI Upper Bound: {mi_bound.item():.4f}')
        
        # 计算每个样本的MI估计
        sample_mi = club_loss(x1, x2, mode='sample')
        print(f'Sample MI shape: {sample_mi.shape}')
        print(f'Sample MI mean: {sample_mi.mean().item():.4f}')
        print(f'Sample MI std: {sample_mi.std().item():.4f}')

if __name__ == "__main__":
    example_usage()