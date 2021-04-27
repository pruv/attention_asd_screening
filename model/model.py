import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models
import math

class TransformerModel(nn.Module):

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, device, dropout=0.5):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.ninp = ninp
        self.device = device

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, src_mask): # src torch.Size([35, 20])
        src = self.pos_encoder(src) # src torch.Size([35, 20, 200])
        output = self.transformer_encoder(src, src_mask) # output torch.Size([35, 20, 200])
        return output


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class Sal_seq(nn.Module):
	def __init__(self,backend,seq_len,hidden_size=512):
		super(Sal_seq,self).__init__()
		self.seq_len = seq_len
		# defining backend
		if backend == 'resnet':
			resnet = models.resnet50(pretrained=True)
			self.init_resnet(resnet)
		elif backend == 'vgg':
			vgg = models.vgg19(pretrained=True)
			self.init_vgg(vgg)
		else:
			assert 0, 'Backend not implemented'

		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		self.decoder = nn.Linear(hidden_size,1,bias=True) # comment for multi-modal distillation
		self.hidden_size = hidden_size
		self.encoder = TransformerModel(ntoken=14, ninp=2048, nhead=2, nhid=200, nlayers=2, device=device, dropout=0.2)

	def init_resnet(self,resnet):
		self.backend = nn.Sequential(*list(resnet.children())[:-2])

	def init_vgg(self,vgg):
		# self.backend = vgg.features
		self.backend = nn.Sequential(*list(vgg.features.children())[:-2]) # omitting the last Max Pooling

	def process_lengths(self,input):
		"""
		Computing the lengths of sentences in current batchs
		"""
		max_length = input.size(1)
		# lengths = list(max_length - input.data.eq(0).sum(1).squeeze())
		lengths = list(max_length - input.data.eq(0).sum(1))
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

	def forward(self,x,fixation, src_mask):
		valid_len = self.process_lengths(fixation) # computing valid fixation lengths torch.Size([12, 14])
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

		op = self.encoder(trans_ip, src_mask)
		output = self.crop_seq(op,valid_len)
		output = torch.sigmoid(self.decoder(output)) # torch.Size([12, 1])
		return output
