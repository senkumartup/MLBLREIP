
# FORWARD PROPAGATION, BACK PROPAGATION AND EPOCHS


```python
import numpy as np
```

## Hyper parameters


```python
learning_rate = 0.01
print learning_rate
```

    0.01
    

## Step 0: Read input and output


```python
input_x1 = [1, 0, 1, 1]
input_x2 = [1, 1, 0, 1]
input_x3 = [1, 0, 1, 0]
input_x = np.array([input_x1, input_x2, input_x3])
print input_x
```

    [[1 0 1 1]
     [1 1 0 1]
     [1 0 1 0]]
    


```python
output_y1 = 1
output_y2 = 1
output_y3 = 0
output_y = np.array([output_y1, output_y2, output_y3]).reshape(input_x.shape[0], 1)
print output_y
```

    [[1]
     [1]
     [0]]
    

## Step 1: Initialize weights and biases with random values


```python
n, input_channels = input_x.shape
print input_channels
```

    4
    


```python
in_kernals =3
print in_kernals
```

    3
    


```python
wh = np.random.random_sample((input_channels * in_kernals,)).reshape(input_channels, in_kernals)
wh = np.round(wh, decimals=2)
print wh
```

    [[ 0.52  0.43  0.1 ]
     [ 0.34  0.7   0.3 ]
     [ 0.54  0.52  0.91]
     [ 0.43  0.62  0.48]]
    


```python
bh = np.random.random_sample((in_kernals,)).reshape(1, in_kernals)
bh = np.round(bh, decimals=2)
print bh
```

    [[ 0.78  0.2   0.53]]
    

## Step 2: Calculate hidden layer input

hidden_layer_input = matrix_dot_product(X,wh) + bh


```python
hidden_layer_input = np.dot(input_x, wh) + bh
print hidden_layer_input
```

    [[ 2.27  1.77  2.02]
     [ 2.07  1.95  1.41]
     [ 1.84  1.15  1.54]]
    

## Step 3: Perform non-linear transformation on hidden linear input

hiddenlayer_activations = sigmoid(hidden_layer_input)


```python
import math

def sigmoid(x):
  return 1 / (1 + math.exp(-x))
```


```python
hidden_layer_activations = [[sigmoid(hidden_layer_input[i,j]) for j in range(hidden_layer_input.shape[1])] for i in range(hidden_layer_input.shape[0])]
hidden_layer_activations = np.round(hidden_layer_activations, decimals=2)
print hidden_layer_activations
```

    [[ 0.91  0.85  0.88]
     [ 0.89  0.88  0.8 ]
     [ 0.86  0.76  0.82]]
    

### Step 4: Perform linear and non-linear transformation of hidden layer activation at output layer


```python
out_kernals = 1
print out_kernals
```

    1
    


```python
out_channels = 3
print out_channels
```

    3
    


```python
wout = np.random.random_sample(out_kernals * out_channels).reshape(out_channels, out_kernals)
wout = np.round(wout, decimals=2)
print wout
```

    [[ 0.37]
     [ 0.27]
     [ 0.62]]
    


```python
bout = np.random.random_sample(out_kernals)
bout = np.round(bout, decimals=2)
print bout
```

    [ 0.33]
    

output_layer_input = matrix_dot_product (hidden_layer_activations * wout ) + bout


```python
output_layer_input = np.dot(hidden_layer_activations, wout) + bout
output_layer_input = output_layer_input.round(decimals=2)
print output_layer_input
```

    [[ 1.44]
     [ 1.39]
     [ 1.36]]
    

output = sigmoid(output_layer_input)


```python
output = [[sigmoid(output_layer_input[i,j]) for j in range(output_layer_input.shape[1])] for i in range(output_layer_input.shape[0])]
output = np.round(output_layer_input, decimals=2)
print output
```

    [[ 1.44]
     [ 1.39]
     [ 1.36]]
    

## Step 5: Calculate gradient of Error(E) at output layer

E = y - output


```python
E = output_y - output
print E
```

    [[-0.44]
     [-0.39]
     [-1.36]]
    

## Step 6: Compute slope at output and hidden layer

f = 1/(1+exp(-x))
df = f * (1 - f)


```python
def derivatives_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))
```

Slope_output_layer = derivatives_sigmoid(output)


```python
slope_output_layer = np.array([[derivatives_sigmoid(output[i,j]) for j in range(output.shape[1])] for i in range(output.shape[0])])
slope_output_layer = slope_output_layer.round(decimals=2)
print slope_output_layer
```

    [[ 0.15]
     [ 0.16]
     [ 0.16]]
    

Slope_hidden_layer = derivatives_sigmoid(hiddenlayer_activations)


```python
slope_hidden_layer = np.array([[derivatives_sigmoid(hidden_layer_activations[i,j]) for j in range(hidden_layer_activations.shape[1])] for i in range(hidden_layer_activations.shape[0])])
slope_hidden_layer = slope_hidden_layer.round(decimals=2)
print slope_hidden_layer
```

    [[ 0.2   0.21  0.21]
     [ 0.21  0.21  0.21]
     [ 0.21  0.22  0.21]]
    

## Step 7: Compute delta at output layer

d_output = E \* slope_output_layer \* lr


```python
d_output = E * slope_output_layer * learning_rate
print d_output
```

    [[-0.00066 ]
     [-0.000624]
     [-0.002176]]
    

## Step 8: Calculate Error at hidden layer

Error_at_hidden_layer = matrix_dot_product(d_output, wout.Transpose)


```python
error_at_hidden_layer = np.dot(d_output, wout.T)
print error_at_hidden_layer
```

    [[-0.0002442  -0.0001782  -0.0004092 ]
     [-0.00023088 -0.00016848 -0.00038688]
     [-0.00080512 -0.00058752 -0.00134912]]
    

## Step 9: Compute delta at hidden layer

d_hiddenlayer = Error_at_hidden_layer * slope_hidden_layer


```python
d_hidden_layer = error_at_hidden_layer * slope_hidden_layer
print d_hidden_layer
```

    [[ -4.88400000e-05  -3.74220000e-05  -8.59320000e-05]
     [ -4.84848000e-05  -3.53808000e-05  -8.12448000e-05]
     [ -1.69075200e-04  -1.29254400e-04  -2.83315200e-04]]
    

## Step 10: Update weight at both output and hidden layer

wout = wout + matrix_dot_product (hiddenlayer_activations.Transpose, d_output) * learning_rate


```python
print wout
wout = wout + np.dot(hidden_layer_activations, d_output) * learning_rate
print wout
```

    [[ 0.37]
     [ 0.27]
     [ 0.62]]
    [[ 0.36996954]
     [ 0.26997123]
     [ 0.61997174]]
    

wh = wh+ matrix_dot_product (X.Transpose,d_hiddenlayer) * learning_rate


```python
print wh
wh = wh + np.dot(input_x.T, d_hidden_layer) * learning_rate
print wh
```

    [[ 0.52  0.43  0.1 ]
     [ 0.34  0.7   0.3 ]
     [ 0.54  0.52  0.91]
     [ 0.43  0.62  0.48]]
    [[ 0.51999734  0.42999798  0.0999955 ]
     [ 0.33999952  0.69999965  0.29999919]
     [ 0.53999782  0.51999833  0.90999631]
     [ 0.42999903  0.61999927  0.47999833]]
    

## Step 11: Update biases at both output and hidden layer

bh = bh + sum(d_hiddenlayer, axis=0) * learning_rate


```python
print bh
bh = bh + np.sum(d_hidden_layer, axis=0) * learning_rate
print bh
```

    [[ 0.78  0.2   0.53]]
    [[ 0.77999734  0.19999798  0.5299955 ]]
    

bout = bout + sum(d_output, axis=0)*learning_rate


```python
print bout
bout = bout + np.sum(d_output, axis=0) * learning_rate
print bout
```

    [ 0.33]
    [ 0.3299654]
    
