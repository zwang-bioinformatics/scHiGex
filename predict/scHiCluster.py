import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix


def convolution_p(rawHiC, pad=1, device=torch.device('cpu')):
	if pad == 0:
		return torch.from_numpy(rawHiC).float().to(device)
	kernel_size = 2*pad + 1
	conv_filter = torch.ones(1, 1, kernel_size, kernel_size).to(device)
	convHiC = F.conv2d(torch.from_numpy(rawHiC[None,None,:,:]).float().to(device), conv_filter, padding=2*pad)
	return (convHiC[0,0,pad:-pad,pad:-pad] / float(kernel_size*kernel_size))

def random_walk(convHiC, rp=0.5, device=torch.device('cpu')):
	nbins = convHiC.shape[0]
	convHiC.fill_diagonal_(0.0)
	#convHiC = convHiC + torch.diag(torch.sum(convHiC,0)==0).float()
	convHiC[range(nbins), range(nbins)] = (torch.sum(convHiC,0) == 0).type(torch.float)
	
	P = torch.div(convHiC, torch.sum(convHiC, 0))
	#rsum = torch.diag(torch.pow(torch.sum(convHiC,0), -0.5))
	#P = torch.mm(torch.mm(rsum,convHiC),rsum)

	Q = torch.eye(nbins).to(device)
	I = torch.eye(nbins).to(device)
	for i in range(30):
		Q_new = (1-rp)*I + rp*torch.mm(Q, P)
		delta = torch.norm(Q-Q_new, 2)
		Q = Q_new
		if delta < 1e-6:
			break
	return Q

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def scHiCluster(a):
	#A = np.log2(A + A.T + 1)
	#a = np.ones((10,10))
	n = a.shape[0]
	b = np.log2(a)
	b = convolution_p(b, 1)
	c = random_walk(b)
	#d = c.reshape(10*10).cpu().numpy()
	d = c.cpu().numpy()
	e = np.percentile(d, 100 - 20)
	#e = np.percentile(d, 100 - 15)
	f = d > e
	return f.reshape(n*n)

