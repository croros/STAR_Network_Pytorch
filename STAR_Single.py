import torch
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class STAR_Single(nn.Module):
	# A single layer STAR Network
	def __init__(self,input_dim,hidden_dim):
		super(STAR_Single,self).__init__()
		self.hidden_dim = hidden_dim
		self.project = nn.Sequential(nn.Linear(input_dim,hidden_dim), nn.Tanh())
		self.gate = nn.Sequential(nn.Linear(input_dim+hidden_dim,hidden_dim),nn.Sigmoid())
		self.tanh_out = nn.Tanh()
	
	def forward(self,x,h=None):
		seq_size, batch_size,_ = x.size()
		hidden_out = torch.zeros(seq_size,batch_size,self.hidden_dim).to(device)
		if h == None:
			h_t = torch.zeros(batch_size,self.hidden_dim).to(device)
		else:
			h_t = torch.squeeze(h)
		for t in range(seq_size):
			x_t = x[t,:,:]
			#Project input into the hidden vector space
			z_t = self.project(x_t)
			
			#Combine input and hidden into gating variable k
			combined = torch.cat((h_t,x_t),-1)
			k_t = self.gate(combined)
			#Control which information is preserved in the current step t
			forget_x = k_t * z_t
			forget_h = (1-k_t) * h_t
			
			#Combine gated information into new hidden
			h_t = self.tanh_out(forget_x+forget_h)
			
			#Save h_t for the current step
			hidden_out[t,:] = h_t
		
		return hidden_out, h_t
			
			
			
		
		
