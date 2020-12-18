import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models
import numpy as np
from transformer_model_2.transformer2 import TransformerModel

class G_LSTM(nn.Module):
	"""
	LSTM implementation proposed by A. Graves (2013),
	it has more parameters compared to original LSTM
	"""
	def __init__(self,input_size=2048,hidden_size=512):
		super(G_LSTM,self).__init__()
		# without batch_norm
		self.input_x = nn.Linear(input_size,hidden_size,bias=True)
		self.forget_x = nn.Linear(input_size,hidden_size,bias=True)
		self.output_x = nn.Linear(input_size,hidden_size,bias=True)
		self.memory_x = nn.Linear(input_size,hidden_size,bias=True)

		self.input_h = nn.Linear(hidden_size,hidden_size,bias=True)
		self.forget_h = nn.Linear(hidden_size,hidden_size,bias=True)
		self.output_h = nn.Linear(hidden_size,hidden_size,bias=True)
		self.memory_h = nn.Linear(hidden_size,hidden_size,bias=True)

		self.input_c = nn.Linear(hidden_size,hidden_size,bias=True)
		self.forget_c = nn.Linear(hidden_size,hidden_size,bias=True)
		self.output_c = nn.Linear(hidden_size,hidden_size,bias=True)

	def forward(self,x,state):
		h, c = state
		i = torch.sigmoid(self.input_x(x) + self.input_h(h) + self.input_c(c))
		f = torch.sigmoid(self.forget_x(x) + self.forget_h(h) + self.forget_c(c))
		g = torch.tanh(self.memory_x(x) + self.memory_h(h))

		next_c = torch.mul(f,c) + torch.mul(i,g)
		o = torch.sigmoid(self.output_x(x) + self.output_h(h) + self.output_c(next_c))
		h = torch.mul(o,next_c)
		state = (h,next_c)

		return state

class Sal_seq(nn.Module):
	def __init__(self,backend,seq_len,hidden_size=512):
		super(Sal_seq,self).__init__()
		self.seq_len = seq_len
		# defining backend
		if backend == 'resnet':
			resnet = models.resnet50(pretrained=True)
			self.init_resnet(resnet)
			input_size = 2048
		elif backend == 'vgg':
			vgg = models.vgg19(pretrained=True)
			self.init_vgg(vgg)
			input_size = 512
		else:
			assert 0, 'Backend not implemented'

		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		# self.rnn = G_LSTM(input_size,hidden_size)
		self.decoder = nn.Linear(hidden_size,1,bias=True) # comment for multi-modal distillation
		self.hidden_size = hidden_size
		self.encoder = TransformerModel(ntoken=14, ninp=2048, nhead=2, nhid=200, nlayers=2, device=device, dropout=0.2)

	def init_resnet(self,resnet):
		self.backend = nn.Sequential(*list(resnet.children())[:-2])

	def init_vgg(self,vgg):
		# self.backend = vgg.features
		self.backend = nn.Sequential(*list(vgg.features.children())[:-2]) # omitting the last Max Pooling

	def init_hidden(self,batch): #initializing hidden state as all zero
		h = torch.zeros(batch,self.hidden_size)
		c = torch.zeros(batch,self.hidden_size)
		# h = torch.zeros(batch,self.hidden_size).cuda()
		# c = torch.zeros(batch,self.hidden_size).cuda()
		return (Variable(h),Variable(c))

	def process_lengths(self,input):
		"""
		Computing the lengths of sentences in current batchs
		"""
		max_length = input.size(1)
		lengths = list(max_length - input.data.eq(0).sum(1).squeeze())
		return lengths

	def crop_seq(self,x,lengths): # x torch.Size([12, 14, 512])
		"""
		Adaptively select the hidden state at the end of sentences
		"""
		batch_size = x.size(0)
		seq_length = x.size(1)
		mask = x.data.new().resize_as_(x.data).fill_(0)
		for i in range(batch_size):
			mask[i][lengths[i]-1].fill_(1)
		mask = Variable(mask)
		x = x.mul(mask)
		x = x.sum(1).view(batch_size, x.size(2))
		return x

	# def forward(self,x,fixation):
	# 	#x: torch.Size([12, 3, 600, 800])
	# 	# fixation: torch.Size([12, 14])
	# 	# fixation = fixation.int()
	# 	valid_len = self.process_lengths(fixation) # computing valid fixation lengths
	# 	x = self.backend(x)
	# 	batch, feat, h, w = x.size() # 12, 2048, 19, 25
	# 	x = x.view(batch,feat,-1) # 12 x 2048 x 475
	#
	# 	# recurrent loop
	# 	state = self.init_hidden(batch) # initialize hidden state
	# 	fixation = fixation.view(fixation.size(0),1,fixation.size(1))
	# 	fixation = fixation.expand(fixation.size(0),feat,fixation.size(2)) # fixation torch.Size([12, 2048, 14])
	# 	x = x.gather(2,fixation) # 12 x 2048 x 14
	# 	output = [] # list of 14 each torch.Size([12, 1, 512])
	# 	for i in range(self.seq_len):
	# 		# extract features corresponding to current fixation
	# 		cur_x = x[:,:,i].contiguous() # torch.Size([12, 2048])
	# 		#LSTM forward
	# 		state = self.rnn(cur_x,state)
	# 		output.append(state[0].view(batch,1,self.hidden_size))
	#
	# 	# selecting hidden states from the valid fixations without padding
	# 	output = torch.cat(output, 1) # torch.Size([12, 14, 512])
	# 	output = self.crop_seq(output,valid_len)
	# 	output = torch.sigmoid(self.decoder(output)) # torch.Size([12, 1])
	# 	return output

	def forward(self,x,fixation, src_mask):
		#x: torch.Size([12, 3, 600, 800])
		# fixation: torch.Size([12, 14])
		# fixation = fixation.int()
		valid_len = self.process_lengths(fixation) # computing valid fixation lengths
		x = self.backend(x)
		batch, feat, h, w = x.size() # 12, 2048, 19, 25
		x = x.view(batch,feat,-1) # 12 x 2048 x 475

		fixation = fixation.view(fixation.size(0),1,fixation.size(1))
		fixation = fixation.expand(fixation.size(0),feat,fixation.size(2)) # fixation torch.Size([12, 2048, 14])
		x = x.gather(2,fixation) # 12 x 2048 x 14
		trans_ip = torch.zeros((x.size(0), self.seq_len, x.size(1)))
		for i in range(self.seq_len):
			# extract features corresponding to current fixation
			cur_x = x[:,:,i].contiguous() # torch.Size([12, 2048])
			for j in range(cur_x.size(0)):
				trans_ip[j,i] = cur_x[j]

		# for transformer pass all 14 fixations at one go
		# collect 12
		op = self.encoder(trans_ip, src_mask)

		# temp = torch.zeros(op.size(0), op.size(1)*op.size(2))
		# for k in range(op.size(0)):
		# 	list = []
		# 	for l in range(op.size(1)):
		# 		list.append(op[k][l])
		# 	t_cat = torch.cat(list, 0)
		# 	temp[k] = t_cat

		# temp torch.Size([12, 28672])
		# selecting hidden states from the valid fixations without padding
		# output = torch.cat(output, 1)
		output = self.crop_seq(op,valid_len)
		output = torch.sigmoid(self.decoder(output)) # torch.Size([12, 1])
		return output
