##totally from scratch in numpy

import numpy as np

weight_size = 1000 #amount of values


value_range = 4 #value range
all_samples = 1000 * weight_size
batch_size = 4 #batch size
step_size = 100.0 #step size
value_center = value_range*(value_range+1)/3 #square center of value_range
accuracy_size = 10 
rng = np.random.default_rng(5) #seed

#create grid with random integers
def random_integers(count):
  return rng.integers(-value_range, value_range+1, size=(count, weight_size))

#compute with sigmoid
def sig(z):
  return 1 / (1 + np.exp(-np.clip(z, -30, 30)))

#get the neuron score
def get_score(W,Y,x,sq):
  return x @ W + sq @ Y

#compute the target value of list
def make_target(raw_math):
  return (raw_math>0).astype(float)
    
#create random batch
def create_input():
  x = random_integers(batch_size)
  return x, x * x - value_center

#train student on random batches
def train():
  global teacher_x_weights,teacher_sq_weights,student_x_weights,student_sq_weights
  #teacher weights 
  teacher_x_weights,teacher_sq_weights = random_integers(1)[0], random_integers(1)[0]
  #student weights 
  student_x_weights,student_sq_weights = random_integers(1)[0].astype(float),random_integers(1)[0].astype(float)
  for _ in range(all_samples//batch_size):
    x,sq = create_input()
    error = make_target(get_score(teacher_x_weights,teacher_sq_weights,x,sq)) - sig(get_score(student_x_weights,student_sq_weights,x,sq))
    sumed = 1 + (x * x).sum(axis=1) + (sq * sq).sum(axis=1)
    step = step_size / sumed
    student_x_weights += (error * step) @ x /batch_size
    student_sq_weights += (error * step) @ sq /batch_size
  return teacher_x_weights,teacher_sq_weights,student_x_weights,student_sq_weights

#compute accuracy on fresh random batches
def evaluate():
  correct = 0
  for _ in range(accuracy_size * 100):
    x,sq = create_input()
    correct += make_target(get_score(teacher_x_weights,teacher_sq_weights,x,sq)) ==make_target(get_score(student_x_weights,student_sq_weights,x,sq))
  print(correct.sum(axis=0)/(accuracy_size*batch_size))
##written in percent

#run code if this is the main file
if __name__ == "__main__":
  train()
  evaluate()