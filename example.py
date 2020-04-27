from mnist import MNIST
import random


# Loads content of data/
mndata = MNIST('data')

# images: list of unsigned bytes
# labels: array of unsigned bytes
images, labels = mndata.load_training()
images, labels = mndata.load_testing()

# Pick a random image
index = random.randrange(0, len(images))
# Display it in the terminal
print(mndata.display(images[index]))