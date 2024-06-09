import torch
import torch.nn as nn
import torch.nn.functional as F

class DATdetector(nn.Module):
    def __init__(self, mode, n_class):
        super(DATdetector, self).__init__()
        
        self.mode = mode
        self.n_class = n_class
        
        self.stage1 = nn.Sequential(
            nn.Conv2d(2048, 256, kernel_size=3),
            nn.ReLU(),
            nn.AvgPool2d(5)
        )
        self.stage1_fc = nn.Linear(256, 1, bias=False)
        
        self.stage2 = nn.Sequential(
            nn.Conv2d(2048, 256, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        self.pool = nn.AvgPool2d(7)
        
        self.stage2_fc = nn.Linear(256, self.n_class, bias=False)
        
        self.reg_net = nn.Sequential(
            nn.Linear(256*7*7, 64),
            nn.ReLU(),
            nn.Linear(64, 6)
        )
        
        self.att_net = nn.Linear(256, 1)
        

        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # initialize to identity transformation
        self.reg_net[2].weight.data.zero_()
        self.reg_net[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))


    def forward(self, x, train=True, b=8):
        # train is False for evaluation only, not validation.
        xsize = x.size()
        
        if self.mode == 1:
            x = self.stage1(x)
            x = x.squeeze((-2, -1))
            
            if train:
                x = x.reshape(b, -1, 256)
                x = x.mean(dim=1)
            
            out = self.stage1_fc(x)
            out = F.sigmoid(out)
                
        elif self.mode == 2:
            x = self.stage2(x)
            
            theta = self.reg_net(x.view(-1, 256*7*7))
            theta = theta.view(-1, 2, 3)
            theta[:, 0, 1] = theta[:, 0, 1] * 0.0
            theta[:, 0, 2] = theta[:, 0, 2] * 0.0
            theta[:, 1, 0] = theta[:, 1, 0] * 0.0
            theta[:, 1, 2] = theta[:, 1, 2] * 0.0
            

            grid = F.affine_grid(theta, x.size())
            x = F.grid_sample(x, grid)
            
            x = self.pool(x)
            x = x.squeeze((-2, -1))
            att = self.att_net(x)
            x = x * att
            
            if train:
                x = x.reshape(b, -1, 256)
                x = x.mean(dim=1)
             
            out = self.stage2_fc(x)
            out = F.softmax(out)
             
        return out
            