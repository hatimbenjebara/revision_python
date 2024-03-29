import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
a = np.array([(1,2,3),(4,5,6)]) 
print(a.sum(axis = 0)) #sum of columns 
print(a.sum(axis =1)) # sum of rows
print(np.sqrt(a)) #square root 
print(np.std(a)) #standard deviation 
print( a.ravel()) # ravel function 
print(np.log10(a)) # log of a 
x = np.arange(0,3*np.pi, 0.1)
y = np.sin(x)
plt.plot(x,y) #show variable of sin in function of x
plt.title("variable of x acording to y")
fig,axes=plt.subplots()
axes.legend(["label1","label2"])
axes.set_title("title : variable of x acording to y ") # title of plot
axes.set_xlabel("x label") # title of x axe
axes.set_ylabel("y label") # title of axe y
axes.plot(x,x**2) #other function
axes.plot(x,x**3) # other function y=x^3
axes.legend(["y=x^2","y=x^3"], loc=2) # add name
axes.plot(x,y,"r") # color red
fig,axes=plt.subplots(dpi=150) # add new figure and dpi for lengh
axes.plot(x,x+1, color='red')
axes.plot(x,x+5, color='pink')
axes.plot(x,10*x*np.sin(x), color='black') # choose color by color=
axes.set_xlabel("x")
axes.set_ylabel("y")
axes.legend(["y=x+1","y=x+2","y=x"])
plt.show()
#create a 6*6 two a dimension array 
Z = np.zeros((6,6), dtype=int)
Z[1::2, ::2]=1
print(Z)
Z[::2,1::2]
print(Z)
#this i so important find the total of number of missing values
z= np.random.rand(10,10)
print(z)
z[np.random.randint(10,size = 5), np.random.randint(10,size =5)] = np.nan # let give some value nan value
print(z)
print("total number of missing values :\n", np.isnan(z).sum()) # total number of missing 
print("indexes of missing values : \n", np.argwhere(np.isnan(z))) # where this missing values
index = np.where(np.isnan(z)) # np.where is where to find a value
print(index)
z[index] = 0# replace missing value by zero. other ways replace it by average or std 
print(z)
#pandas now 
#creating a series from a list
a = [0,1,2,3,4,5]
s = pd.Series(a)
print(s)
n = np.random.randn(5)
index = ['a','b','c','d','e']
s1 = pd.Series(n, index= index)
print(s1)
#create a series from dictionary 
d = { 'a':1, 'b':2, 'c':3, 'd':4, 'e':5} # give index to values in array
s2 = pd.Series(d)
print(s2)
s.index=['A','B','C','D','E','F']
print(s) #np.drop to delete a value or s1.append(s2) to add s2 to s1 or s1.add(s2)
print("median of a : \n", s.median())# there is no median to list ... series 
print("max of a : \n", s.max())
print("min of a : \n", s.min())
dates = pd.date_range('today', periods=6) # define time sequence as index
print( dates) # print a list of date start from today to 6 days 
num_a = np.random.randn(6,4) 
print(num_a)
columns = ['A', 'B', 'C', 'D'] # use the table as the column name 
df= pd.DataFrame(num_a, index=dates, columns=columns) # here we create data 
print(df)
#create a data 
data = { 'animal' : ['cat', 'cat', 'snake', 'dog', 'cat','dog'],
         'age': [ 2, 1, np.NAN , 6 , 7, 2],
         'visits':[1 , 3 ,4, 5,6,1],
         'priority':['yes','yes','no','yes','no','yes']} # declaration d un dictionnaire
labels = ['a', 'b', 'c', 'd', 'e', 'f'] # call a list
data2 = pd.DataFrame(data, index=labels) # make modification in dictionnary is not autorosate 
#for that, we converte a dictionnary to data by pd.DataFrame
print(data2)
print(data2.dtypes)
print(data2.tail(3))
print(data2.index)
print(data2.columns)
print(data2.describe ()) # statitical data of dataframe 
print(data2.T) # transpose the data 
print(data2.sort_values(by='age')) #class by age in descenting 
print(data2.isnull()) 
data2.loc['b','age']=2 # change value in column age and index b 
print(data2.mean()) # mean of data
print(data2['age'].sum()) # sum of column age
print(data2['age'].max()) #max of column age
print(data2.sum()) # sum of data
string = pd.Series(['A','B','C','D',np.nan,'CBA','cow'])
print(string.str.lower()) # make string in data string lower
print(string.str.upper()) #make string in data string upper
#operations for DataFrame missing values 
df=data2.copy() # copy of data cause we are going to work on it and we dont want to lose 
print(df)
print(df.fillna(2)) # replace nan by value = 2
df=data2.copy()
meanage = df['age'].mean()
print(meanage)
print(df.fillna(meanage)) #replace nan by value = mean
df=data2.copy()
print(df.dropna(how='any')) # delete all NAN values
ts= pd.Series(np.random.randn(50), index=pd.date_range('today', periods=50))# data 
ts = ts.cumsum() # return the cumulative sum of the elements along a given axis.
ts.plot()
plt.show()
#multiple functions in one plt
df= pd.DataFrame(np.random.randn(50,4), index=ts.index, columns = ['A', 'B','X','Y']) 
df=df.cumsum()
df.plot()
plt.show()
#remove repeated data
df = pd.DataFrame({'A':[1,2,2,2,4,4,5,5,6,6,7,8,8]})
print(df)
print(df.loc[ df['A'].shift() != df['A']]) # DataFrame.shift(periods=1, freq=None, axis=0,
#fill_value=_NoDefault.no_default)[source] 
#Shift index by desired number of periods with an optional time freq.


