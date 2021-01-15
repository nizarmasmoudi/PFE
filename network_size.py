import os
import matplotlib.pyplot as plt
from utils import env

train_images = os.listdir(env.TRAIN_PATH_ + 'images')
train_images = [env.TRAIN_PATH_ + 'images/' + image for image in train_images]
test_images = os.listdir(env.TEST_PATH_ + 'images')
test_images = [env.TEST_PATH_ + 'images/' + image for image in test_images]
valid_images = os.listdir(env.VALIDATION_PATH_ + 'images')
valid_images = [env.VALIDATION_PATH_ + 'images/' + image for image in valid_images]

train_sizes = []
test_sizes = []
valid_sizes = []
for i, image in enumerate(train_images): 
    train_sizes.append(plt.imread(image).shape[:2])
    print(i, end='\r')
print(' '*4, end='\r')
for i, image in enumerate(test_images): 
    test_sizes.append(plt.imread(image).shape[:2])
    print(i, end='\r')
print(' '*4, end='\r')
for i, image in enumerate(valid_images): 
    valid_sizes.append(plt.imread(image).shape[:2])
    print(i, end='\r')

train_sizes = list(set(train_sizes))
test_sizes = list(set(test_sizes))
valid_sizes = list(set(valid_sizes))

plt.figure()
plt.scatter(
    x = [shape[0] for shape in train_sizes],
    y = [shape[1] for shape in train_sizes],
    alpha = .4,
    c = 'red',
    label = 'Train sizes'
)
plt.scatter(
    x = [shape[0] for shape in test_sizes],
    y = [shape[1] for shape in test_sizes],
    alpha = .4,
    c = 'blue',
    label = 'Test sizes'
)
plt.scatter(
    x = [shape[0] for shape in valid_sizes],
    y = [shape[1] for shape in valid_sizes],
    alpha = .4,
    c = 'green',
    label = 'Validation sizes'
)
plt.legend()
plt.grid(True)
plt.show()