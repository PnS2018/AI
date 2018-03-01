
#ex 1

input_a=K.placeholder((5,))
input_b=K.placeholder((5,))
input_c=K.placeholder((5,))
inputs_squared=input_a**2+input_b**2+input_c**2+2*input_b*input_c
squared_function=K.function(inputs=(input_a,input_b,input_c),outputs=(inputs_squared,))

#ex 2

input_x=K.variable((1,))
tanh=1-(2/(K.exp(2*input_x)+1))
tanhyp=K.function((input_x,),(tanh,))
grad_tensor=K.gradients(loss=tanh,variables=[input_x])

tanhyp((—100,),(tanh,))
grad_tensor(loss=tanh,variables=[-100])

tanhyp((-1,),(tanh,))
grad_tensor(loss=tanh,variables=[-1])

tanhyp((0,),(tanh,))
grad_tensor(loss=tanh,variables=[0])

tanhyp((1,),(tanh,))
grad_tensor(loss=tanh,variables=[1])

tanhyp((100,),(tanh,))
grad_tensor(loss=tanh,variables=[100])

#ex 3
w=K.variable((2,))
b=K.placeholder((1,))
x=K.placeholder((2,))
buffer=w*x+b
buffer_add=K.function((w,b,x),(buffer,))
func=1/(1+K.exp(-buffer))
func_exp=K.function((buffer,),(func,))
grad_2_tensor=K.gradients(loss=func,variables=[w])

#ex 4

n=2
x=K.placeholder(())
y = K.variable((n+1,))

output = None
for idx in range(n+1):
	if output is None:
		output = y[idx] * x ** idx
	else:
		output += y[idx] * x ** idx
grad_3_tensor=K.gradient(loss=output,variables=[y])






