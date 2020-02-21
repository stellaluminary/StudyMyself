# IPython log file

a=5
a
import numpy as np
data = {i:np.random.randn() for i in range(7)}
data
from numpy.random import randn
data = {i:randn() for i in range(7)}
data
an_apple = 27
an_example = 42
an_apple, an_example
b = [1,2,3]
import datetime
datetime.date
get_ipython().run_line_magic('pinfo', 'b')
def add_numbers(a,b):
    return a+b
get_ipython().run_line_magic('pinfo', 'add_numbers')
get_ipython().run_line_magic('pinfo2', 'add_numbers')
get_ipython().run_line_magic('psearch', 'np.*load*')
get_ipython().run_line_magic('run', 'ipython_script_test.py')
c
result
x=5
y=7
if x >= 5:
    x += 1
    
    y=8
x,y
a = np.random.randn(100,100)
get_ipython().run_line_magic('timeit', 'np.dot(a,a)')
get_ipython().run_line_magic('pinfo', '%reset')
get_ipython().run_line_magic('pinfo', '%magic')
2**27
foo = 'bar'
foo
_i22
_22
# exec _i40
get_ipython().run_line_magic('logstart', '')
import time
start = time.time()
iterations = 1
elapsed_per = (time.time() - start) / iterations
strings = ['foo','foobar','baz','qux','python','Guido Van Rossum','scari']*100000

get_ipython().run_line_magic('time', "method1 = [x for x in strings if x.startswith('foo')]")
get_ipython().run_line_magic('time', "method2 = [x for x in strings if x[:3] == 'foo']")
get_ipython().run_line_magic('timeit', "[x for x in strings if x.startswith('foo')]")
get_ipython().run_line_magic('timeit', "[x for x in strings if x[:3] == 'foo']")
x = 'foobar'
y = 'foo'

get_ipython().run_line_magic('timeit', 'x.startswith(y)')
get_ipython().run_line_magic('timeit', 'x[:3] == y')
