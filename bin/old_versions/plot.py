import numpy as np
import matplotlib.pyplot as plt

r,pr,dpr = np.genfromtxt('pr.d',skip_header=0,usecols=[0,1,2],unpack=True)
plt.plot(r,pr)

plt.show()
plt.savefig('pr.png')
