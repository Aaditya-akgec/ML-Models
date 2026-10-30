import numpy as np
arr = np.array([2,3,4,6,7,8,9,10,15,18,20,25,29,33,38,201])
q1 = np.percentile(arr,25)
q3 = np.percentile(arr,75)
iqr = q3 - q1
uf = q3 + 1.5*iqr
lf = q1 - 1.5*iqr

l = []
for i in arr:
    if i<=uf and i>=lf:
        l.append(i)
arr2 = np.array(l)
print(arr2)
print(arr)
import matplotlib.pyplot as plt
import seaborn as sns
sns.boxplot(x=arr)
plt.show()