import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
a = np.array([1,2,3]) # call for list 
print("list of a ", a) 
print("lenght of a is ", len(a)) # lengh of a 
print("first element of list is a_0=", a[0]) # print value in first position in list . in python we start with 0 
print("second element of list is a_1=", a[1]) # print seconde value in list 
b = range(11)
print("range of b is ", len(b))
c = np.arange(11)
print('the size of c is ',c.size) # size of c 
def new_func(c): # function return size of array
    print(c.itemsize)
new_func(c)
print(c.size*c.itemsize)
a = np.array ([[1,2], [3,4], [5,6]]) 
print(a)
print(a.ndim) # n dimension 
print(a.itemsize) #
print(a.shape) # dimension n,p here (3,2)
a = np.array ([[1,2],[3,4],[5,6]], dtype= np.float64) # converge all values to float 
print(a)
print(a.dtype) # type of data
print(np.zeros((3,4))) # matrix zeros 
print(np.ones((3,4))) #matrix ones 
#char can do same us excel
print(np.char.add(['hello ', 'hi '] , ['abc','xyz'])) # add x+y in matrix char for just 2
print(np.char.multiply(['Hello my name is hatim '], 3)) # print multiple time the char
print(np.char.center(' hello ', 20, fillchar= '-'))# print char in centre and repeat fillchar must be one caracter  
print(np.char.split('hello my name is hatim')) # spilt a char to multiple chars
print(np.char.upper(['python', 'data'])) # .lower or title or capitalize 
print(np.char.replace('he is a good person','is', 'was')) # .replace a first char by seconde one 
a= np.arange(9)
print(a)
b=a.reshape(3,3)
print("the modified array :")
print(b)
print(b.flatten()) # get together all array in one 
print(b.flatten(order ="F")) # order by column not row and False
a = np.arange(12).reshape(4,3)
print(a) # matrix or array range from 0 to 12
print(np.transpose(a)) # transpose of matrix 
#other operation np.add(a,b) or np.subtract(a,b) or np.multiply(a,b) or np.divide(a,b)
a = np.arange(9)
print(a)
print(a[4:])
s = slice(2,9,2) #start from 2 and end in 9 with step 2
print(s)
a = a.reshape(3,3)
print(a)
for x in np.nditer(a): #matrix vector linear
    print(x)
for x in np.nditer(a, order ="F"): #matrix vector column
    print(x)
for x in np.nditer(a, order ="C"): #matrix vector column
    print(x)
a = np.linspace(1,3,9) # start from 1 to 3 to 10 number
print(a)
