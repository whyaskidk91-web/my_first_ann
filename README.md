# my_first_ann

## Creation Process:
Created late September early October.
Created only with numpy and only by hand, and a lot of trial and error to learn how numpy works.
Was my first project with numpy.

## The model:

A simple ANN with sigmoid neuron layer that tries to learn the pattern of binary decisions a teacher model(based on the same function with random values) makes.

## Test run:

In seed 5 it returns 98.9%, that means the student predicts the teachers choice in 98.9% cases correctly.
This shows it learned the teacher pattern pretty well.

## Limits and next steps:

The biggest problem is that it uses no real data but only a teacher model. And its missing useful features(eg.: loss,He init) and its very small. Thats what I changed with my following two mlp projects.

## How to run:
```
pip install numpy
python FIRST_ANN.py
```
