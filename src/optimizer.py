import matplotlib.pyplot as plt

import torch
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR, LRScheduler, ExponentialLR
from ignite.handlers.param_scheduler import create_lr_scheduler_with_warmup
import pytorch_warmup as warmup

import params


class MyScheduler(LRScheduler):
	def __init__(self, optimizer):
		print(optimizer)
		self.optimizer = optimizer
		super().__init__(optimizer)

	def get_lr(self):
		print("current_step", self.last_epoch)
		training_steps = 1000
		return 1
		if self.last_epoch < params.warmup_steps:  # current_step / warmup_steps * base_lr
			return float(self.last_epoch / params.warmup_steps)
		else:                                 # (num_training_steps - current_step) / (num_training_steps - warmup_steps) * base_lr
			return max(0.0, float(training_steps - self.last_epoch) / float(max(1, training_steps - params.warmup_steps)))




def mywarmup(current_step):
	print("current_step", current_step)
	training_steps = 1000
	return 1
	if current_step < params.warmup_steps:  # current_step / warmup_steps * base_lr
		return float(current_step / params.warmup_steps)
	else:                                 # (num_training_steps - current_step) / (num_training_steps - warmup_steps) * base_lr
		return max(0.0, float(training_steps - current_step) / float(max(1, training_steps - params.warmup_steps)))



class Optimizer():
	def __init__(self, parameters):
		self.epoch = 0
		self.optim = eval(params.optimizer)(parameters, lr=params.learning_rate)
		self.scheduler = None
		if params.sheduler == "warmup":
			self.lr_scheduler = ExponentialLR(self.optim, gamma=0.999)			
			# self.scheduler = create_lr_scheduler_with_warmup(lr_scheduler,
			# 											warmup_start_value=0.0,
			# 											warmup_end_value=params.learning_rate,
			# 											warmup_duration=50)
			self.warmup_scheduler = warmup.ExponentialWarmup(self.optim, warmup_period=int(params.warmup_steps/2))

			# self.scheduler = MyScheduler(self.optim)
			# LambdaLR(self.optim, mywarmup, last_epoch=1, verbose=True)
		elif params.sheduler == "plateau":
			self.scheduler = ReduceLROnPlateau(self.optim, 'min', patience=params.patience)


	def step(self):
		self.optim.step()

	def zero_grad(self):
		self.optim.zero_grad()

	def get_lr(self):
		return self.optim.param_groups[0]['lr']

	def epoch_routine(self, loss):
		if params.sheduler == "plateau":
			self.scheduler.step(loss)
		else :
			with self.warmup_scheduler.dampening():
				self.lr_scheduler.step()
		self.epoch += 1


	def early_stopping(self):
		if params.sheduler == "plateau":
			return self.get_lr() < params.lr_limit
		elif params.sheduler == "warmup":	
			if self.epoch > params.warmup_steps:
				return self.get_lr() < params.lr_limit
		return False
			


if __name__ == "__main__":
	x=[]
	for i in range(100):
		x.append(warmup(i))
	plt.plot(x)
	plt.show()
