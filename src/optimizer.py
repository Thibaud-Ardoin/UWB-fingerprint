import torch
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR

import params


def warmup(current_step: int):
    training_steps = 1000
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
			self.scheduler = LambdaLR(self.optim, warmup)
		elif params.sheduler == "plateau":
			self.scheduler = ReduceLROnPlateau(self.optim, 'min', patience=params.patience)

	def step(self):
		self.optim.step()
		self.epoch += 1

	def zero_grad(self):
		self.optim.zero_grad()

	def get_lr(self):
		return self.scheduler.optimizer.param_groups[0]['lr']

	def early_stopping(self):
		if params.sheduler == "plateau":
			return optimizer.get_lr() < params.lr_limit
		elif params.sheduler == "warmup":	
			if self.epoch > params.warmup_steps:
				return optimizer.get_lr() < params.lr_limit
		return False
			
