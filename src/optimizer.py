import matplotlib.pyplot as plt

import torch
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, LambdaLR, LRScheduler, ExponentialLR, SequentialLR
from ignite.handlers.param_scheduler import create_lr_scheduler_with_warmup
import pytorch_warmup as warmup

import params

class Combi_scheduler():
	def __init__(self):
		self.warmup_steps = params.warmup_steps
		self.base_lr = params.learning_rate
		self.epoch = 0
		self.min_loss = None

	def __call__(self, loss):
		self.epoch += 1

		# Distinguish Warmup increase with plateau part
		if self.epoch < self.warmup_steps :
			alpha = (self.epoch / self.warmup_steps)
			return self.base_lr * alpha + (self.base_lr/10)*(1-alpha)		
		else :
			# Track the minimum of the loss
			if self.min_loss is None or loss < self.min_loss :
				self.min_loss = loss
				self.delta_min = 0
			else :
				self.delta_min += 1
				if self.delta_min > params.patience :
					self.base_lr = self.base_lr/10
					self.delta_min = 0
			return self.base_lr


class Optimizer():
	def __init__(self, model):
		self.epoch = 0
		lr = params.learning_rate
		if params.sheduler=="combi": lr = 1
		self.optim = eval(params.optimizer)(model.parameters(), lr=lr)

		self.scheduler = None
		if params.sheduler == "warmup":
			self.lr_scheduler = ExponentialLR(self.optim, gamma=0.999)
			self.warmup_scheduler = warmup.ExponentialWarmup(self.optim, warmup_period=int(params.warmup_steps/2))
		elif params.sheduler == "plateau":
			self.scheduler = ReduceLROnPlateau(self.optim, 'min', patience=params.patience)
		elif params.sheduler == "combi":
			self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optim, Combi_scheduler())


	def step(self):
		self.optim.step()

	def zero_grad(self):
		self.optim.zero_grad()

	def get_lr(self):
		return self.optim.param_groups[0]['lr']

	def epoch_routine(self, loss):
		if params.sheduler == "plateau":
			self.scheduler.step(loss)
		elif params.sheduler == "combi":
			self.scheduler.step(loss)
		else :
			with self.warmup_scheduler.dampening():
				self.lr_scheduler.step()
		self.epoch += 1


	def early_stopping(self):
		if params.sheduler == "plateau":
			return self.get_lr() < params.lr_limit
		else:
			if self.epoch > params.warmup_steps:
				return self.get_lr() < params.lr_limit



class AdvOptimizer(Optimizer):
	def __init__(self, model):
		dev_param = model.devCls.parameters()
		pos_param = model.posCls.parameters()
		encoder_param = model.encoder.parameters()
		self.epoch = 0
		self.dev_optim = eval(params.optimizer)(dev_param, lr=params.learning_rate)
		self.pos_optim = eval(params.optimizer)(pos_param, lr=params.learning_rate)
		self.encoder_optim = eval(params.optimizer)(encoder_param, lr=params.learning_rate)
		self.scheduler = None

	def step(self):
		self.dev_optim.step()
		self.pos_optim.step()
		self.encoder_optim.step()

	def zero_grad(self):
		self.dev_optim.zero_grad()
		self.pos_optim.zero_grad()
		self.encoder_optim.zero_grad()

	def get_lr(self):
		# Arbitrary taking one Optim only
		return self.dev_optim.param_groups[0]['lr']
	

	def epoch_routine(self, loss):
		self.epoch += 1

	def early_stopping(self):
		return False

			


if __name__ == "__main__":
	x=Optimizer()
	for i in range(100):
		x.append(warmup(i))
	plt.plot(x)
	plt.show()
