import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

def show_anim(L, interval=200):
    fig, ax = plt.subplots()
    
    vmin = min(np.min(m) for m in L)
    vmax = max(np.max(m) for m in L)
    
    im = ax.imshow(L[0], vmin=vmin, vmax=vmax)
    
    def update(frame):
        im.set_array(L[frame])
        return [im]
    
    anim = animation.FuncAnimation(fig, update, frames=len(L), 
                                 interval=interval, blit=True, repeat=False)
    
    plt.show()
    return anim


def show(M):
    fig, ax = plt.subplots()
    img = ax.imshow(M)
    plt.show()

def show2(M,NA):
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    img1 = ax1.imshow(M)
    img2 = ax2.imshow(NA)
    plt.show()
    
def show_n(L,n,m):
    l = len(L)
    fig,axs = plt.subplots(n,m)
    
    k=0
    for i in range(n):
        for j in range(m):
            if k<l:
                axs[i,j].imshow(L[k])
            axs[i,j].set_xticklabels([])
            axs[i,j].set_yticklabels([])
            
                
            k+=1
    plt.show()
