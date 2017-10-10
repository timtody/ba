from matplotlib import pyplot as plt
from scipy.io import loadmat

images_with_debris = loadmat("DIGIT_data/light_debris_with_debris.mat")
i=1
for image, label in zip(images_with_debris["images"][:10], images_with_debris["targets"][:10]):
    plt.matshow(image[:,:,0], cmap=plt.cm.gray, vmin=0, vmax=255)
    print("Figure{} ".format(i), label)
    i+= 1

plt.show()

