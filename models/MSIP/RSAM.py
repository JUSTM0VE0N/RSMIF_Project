import torch
import torch.nn as nn
import torch.nn.functional as F

# Proposed RSAM m=4
class RSAM(nn.Module): 
    def __init__(self, in_planes, out_planes, kernel_att=7, head=4, kernel_conv=3, stride=1, padding=0, dilation=1,
                 groups=1, use_bias=True):
        super(RSAM, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_attn = kernel_att
        self.head = head
        self.kernel_conv = kernel_conv
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = use_bias

        # Generating local adaptive weights
        self.attention1 = nn.Sequential(
            SA(in_planes, kernel_conv * kernel_conv, kernel_att, head=3, kernel_conv=3, stride=stride, dilation=dilation),
            nn.ReLU(inplace=True),
            SA(kernel_conv ** 2, kernel_conv ** 2, kernel_att, head=3, kernel_conv=3, stride=stride, dilation=dilation),
            nn.ReLU(inplace=True),
            SA(kernel_conv ** 2, kernel_conv ** 2, kernel_att, head=3, kernel_conv=3, stride=stride, dilation=dilation),
            nn.ReLU(inplace=True),
            SA(kernel_conv ** 2, kernel_conv ** 2, kernel_att, head=3, kernel_conv=3, stride=stride, dilation=dilation),
            nn.Sigmoid()
        )

        conv1 = nn.Conv2d(in_planes, out_planes, kernel_conv, stride, padding, dilation, groups)
        self.weight = conv1.weight

    def forward(self, x):
        (b, n, H, W) = x.shape
        m = self.out_planes
        k = self.kernel_conv
        n_H = 1 + int((H + 2 * self.padding - k) / self.stride)
        n_W = 1 + int((W + 2 * self.padding - k) / self.stride)
        atw1 = self.attention1(x)

        atw1 = atw1.permute([0, 2, 3, 1])
        atw1 = atw1.unsqueeze(3).repeat([1, 1, 1, n, 1])
        atw1 = atw1.view(b, n_H, n_W, n * k * k)

        atw = atw1
        atw = atw.view(b, n_H * n_W, n * k * k)
        atw = atw.permute([0, 2, 1])

        kx = F.unfold(x, kernel_size=k, stride=self.stride, padding=self.padding)
        atx = atw * kx

        atx = atx.permute([0, 2, 1])
        atx = atx.view(1, b * n_H * n_W, n * k * k)

        w = self.weight.view(m, n * k * k)
        w = w.permute([1, 0])
        y = torch.matmul(atx, w)
        y = y.view(b, n_H * n_W, m)

        y = y.permute([0, 2, 1])  # b,m,n_H*n_W
        y = F.fold(y, output_size=(n_H, n_W), kernel_size=1)
        return y

def position(H, W, is_cuda=True):
    if is_cuda:
        loc_w = torch.linspace(-1.0, 1.0, W).cuda().unsqueeze(0).repeat(H, 1)  # (H, W)
        loc_h = torch.linspace(-1.0, 1.0, H).cuda().unsqueeze(1).repeat(1, W)  # (H, W)
        # loc_w = torch.linspace(-1.0, 1.0, W).to(device).unsqueeze(0).repeat(H, 1)  # (H, W)
        # loc_h = torch.linspace(-1.0, 1.0, H).to(device).unsqueeze(1).repeat(1, W)  # (H, W)
    else:
        loc_w = torch.linspace(-1.0, 1.0, W).unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).unsqueeze(1).repeat(1, W)
    loc = torch.cat([loc_w.unsqueeze(0), loc_h.unsqueeze(0)], 0).unsqueeze(0)  # (1, 2, H, W)
    return loc


def stride(x, stride):
    b, c, h, w = x.shape
    return x[:, :, ::stride, ::stride]

def init_rate_half(tensor):
    if tensor is not None:
        tensor.data.fill_(0.5)

def init_rate_0(tensor):
    if tensor is not None:
        tensor.data.fill_(0.)

class SA(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_att=7, head=4, kernel_conv=3, stride=1, dilation=1):
        super(SA, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.head = head
        self.kernel_att = kernel_att
        self.kernel_conv = kernel_conv
        self.stride = stride
        self.dilation = dilation
        self.rate1 = torch.nn.Parameter(torch.Tensor(1))
        self.head_dim = self.out_planes // self.head

        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv3 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv_p = nn.Conv2d(2, self.head_dim, kernel_size=1)

        self.padding_att = (self.dilation * (self.kernel_att - 1) + 1) // 2
        self.pad_att = torch.nn.ReflectionPad2d(self.padding_att)
        self.unfold = nn.Unfold(kernel_size=self.kernel_att, padding=0, stride=self.stride)
        self.softmax = torch.nn.Softmax(dim=1)

        self.dep_conv = nn.Conv2d(self.kernel_conv * self.kernel_conv * self.head_dim, out_planes,
                                  kernel_size=self.kernel_conv, bias=True, groups=self.head_dim, padding=1,
                                  stride=stride)

        self.reset_parameters()

    def reset_parameters(self):
        init_rate_half(self.rate1)  # 0.5
        kernel = torch.zeros(self.kernel_conv * self.kernel_conv, self.kernel_conv, self.kernel_conv)  # (k*k, k, k)
        for i in range(self.kernel_conv * self.kernel_conv):
            kernel[i, i // self.kernel_conv, i % self.kernel_conv] = 1
        kernel = kernel.squeeze(0).repeat(self.out_planes, 1, 1, 1)  # (out_planes, k*k, k, k)
        self.dep_conv.weight = nn.Parameter(data=kernel, requires_grad=True)
        self.dep_conv.bias = init_rate_0(self.dep_conv.bias)

    def forward(self, x):
        q, k, v = self.conv1(x), self.conv2(x), self.conv3(x)  # （b, 9, 64, 64) --> (b, 32, 64, 64)
        scaling = float(self.head_dim) ** -0.5  # 1/2
        b, c, h, w = q.shape  # b, 32, 64, 64
        h_out, w_out = h // self.stride, w // self.stride  # 64, 64

        # positional encoding
        pe = self.conv_p(position(h, w, x.is_cuda))  # conv_p(1, 2, 64, 64) -> (1, self.head_dim(8), 64, 64)

        q_att = q.view(b * self.head, self.head_dim, h, w) * scaling
        k_att = k.view(b * self.head, self.head_dim, h, w)
        v_att = v.view(b * self.head, self.head_dim, h, w)

        if self.stride > 1:
            q_att = stride(q_att, self.stride)
            q_pe = stride(pe, self.stride)
        else:
            q_pe = pe  # (1, 8, 64, 64)

        # (4b, 8, 49, 64, 64)
        unfold_k = self.unfold(self.pad_att(k_att)).view(b * self.head, self.head_dim, self.kernel_att * self.kernel_att, h_out, w_out)
        # (1, 8, 49, 64, 64)
        unfold_rpe = self.unfold(self.pad_att(pe)).view(1, self.head_dim, self.kernel_att * self.kernel_att, h_out, w_out)

        att = (q_att.unsqueeze(2) * (unfold_k + q_pe.unsqueeze(2) - unfold_rpe)).sum(1)
        att = self.softmax(att)  # (b*head, k_att^2, h_out, w_out)
        out_att = self.unfold(self.pad_att(v_att)).view(b * self.head, self.head_dim, self.kernel_att * self.kernel_att, h_out, w_out)
        out_att = (att.unsqueeze(1) * out_att).sum(2).view(b, self.out_planes, h_out, w_out)

        return self.rate1 * out_att
    
class RSARB(nn.Module):
    def __init__(self, in_planes):
        super(RSARB, self).__init__()
        self.conv1 = RSAM(in_planes, in_planes, kernel_att=7, kernel_conv=3, stride=1, padding=1, use_bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = RSAM(in_planes, in_planes, kernel_att=7, kernel_conv=3, stride=1, padding=1, use_bias=False)

    def forward(self, x):
        res = self.conv1(x)
        res = self.relu1(res)
        res = self.conv2(res)
        x = x + res
        return x

# Proposed Network n=8
class RSANet(nn.Module):
    def __init__(self):
        super(RSANet, self).__init__()
        self.head_conv = nn.Sequential(
            RSAM(9, 32, kernel_att=7, kernel_conv=3, stride=1, padding=1, use_bias=False),
            nn.ReLU(inplace=True)
        )

        self.RB1 = RSARB(32)
        self.RB2 = RSARB(32)
        self.RB3 = RSARB(32)

        self.tail_conv = RSAM(32, 8, kernel_att=7, kernel_conv=3, stride=1, padding=1, use_bias=False)

    def forward(self, pan, lms):
        x = torch.cat([pan, lms], 1)
        x = self.head_conv(x)
        x = self.RB1(x)
        x = self.RB2(x)
        x = self.RB3(x)
        x = self.tail_conv(x)
        sr = lms + x
        return sr