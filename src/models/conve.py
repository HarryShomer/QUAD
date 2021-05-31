import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvE(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.dim = config['EMBEDDING_DIM']

        self.n_filters = config['MODEL']['N_FILTERS']
        self.kernel_sz = config['MODEL']['KERNEL_SZ']
        self.k_w = config['MODEL']['K_W']
        self.k_h = config['MODEL']['K_H']

        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(self.n_filters)
        self.bn2 = torch.nn.BatchNorm1d(self.dim)

        self.drop1 = torch.nn.Dropout(config['MODEL']['CONVE_DROP_1'])
        self.drop2 = torch.nn.Dropout(config['MODEL']['CONVE_DROP_2'])
        self.m_conv1 = torch.nn.Conv2d(1, out_channels=self.n_filters, kernel_size=(self.kernel_sz, self.kernel_sz), stride=1, padding=0)

        # 14
        # 14


        flat_sz_h = int(2 * self.k_w) - self.kernel_sz + 1
        flat_sz_w = self.k_h - self.kernel_sz + 1
        self.flat_sz = flat_sz_h * flat_sz_w * self.n_filters
        self.fc = torch.nn.Linear(self.flat_sz, self.dim)


    def concat(self, head_ent, rel, tail_ent):
        """
        Combine triplet info into an "image"
        """
        head_ent  = head_ent.view(-1, 1, self.dim)
        rel       = rel.view(-1, 1, self.dim)
        tail_ent  = head_ent.view(-1, 1, self.dim)
        stack_inp = torch.cat([head_ent, rel, tail_ent], 1)

        return torch.transpose(stack_inp, 2, 1).reshape((-1, 1, 2*self.k_w, self.k_h))


    def forward(self, head_ent, rel, tail_ent):
        """
        Return encoded representation
        """
        
        stk_inp = self.concat(head_ent, rel, tail_ent)
        x       = self.bn0(stk_inp)
        x       = self.m_conv1(x)
        x       = self.bn1(x)
        x       = F.relu(x)
        x       = self.drop1(x)
        #x       = x.view(-1, self.flat_sz)
        x       = self.fc(x)

        print("------>", x.shape)

        ### ???
        # x       = self.drop2(x)
        # x       = self.bn2(x)
        # x       = F.relu(x)

        return x

