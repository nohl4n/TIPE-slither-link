import matplotlib.pyplot as plt


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
