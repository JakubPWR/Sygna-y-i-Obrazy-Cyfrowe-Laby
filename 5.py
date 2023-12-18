import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def max_pooling(image, pool_size=(2, 2)):
    if len(image.shape) == 3:
        channels = []
        for channel in range(image.shape[2]):
            channel_matrix = image[:, :, channel]
            pooled_channel = max_pooling(channel_matrix, pool_size)
            channels.append(pooled_channel)
        return np.stack(channels, axis=2)

    height, width = image.shape
    new_height = height // pool_size[0]
    new_width = width // pool_size[1]
    pooled_image = np.zeros((new_height, new_width))

    for i in range(new_height):
        for j in range(new_width):
            region = image[i * pool_size[0]:(i + 1) * pool_size[0],
                     j * pool_size[1]:(j + 1) * pool_size[1]]
            pooled_image[i, j] = np.max(region)

    return pooled_image


def downsample(image, scale_factor):
    new_height = int(image.shape[0] / scale_factor)
    new_width = int(image.shape[1] / scale_factor)

    return np.array(Image.fromarray(image).resize((new_width, new_height)))


def upsample_nearest_neighbor(image, scale_factor):
    new_height = int(image.shape[0] * scale_factor)
    new_width = int(image.shape[1] * scale_factor)

    return np.array(Image.fromarray(image).resize((new_width, new_height), Image.NEAREST))


# Importowanie obrazu za pomocą PIL
image_path = "/Users/jakubgodlewski/PycharmProjects/KZInterpolationP21/Cat1.jpg"
original_image = Image.open(image_path)

# Konwersja obrazu do macierzy NumPy
image_matrix = np.array(original_image)

# Wywołanie funkcji max_pooling
result_matrix = max_pooling(image_matrix, pool_size=(2, 2))

# Wyświetlenie oryginalnego i przeskalowanego obrazu po max pooling
plt.subplot(1, 3, 1)
plt.imshow(original_image)
plt.title("Original Image")

plt.subplot(1, 3, 2)
plt.imshow(Image.fromarray(result_matrix.astype(np.uint8)))
plt.title("Image after Max Pooling")

# Wywołanie funkcji downsample
downsampled_matrix = downsample(image_matrix, scale_factor=20)

# Wyświetlenie oryginalnego i zeskalowanego obrazu
plt.subplot(1, 3, 3)
plt.imshow(Image.fromarray(downsampled_matrix.astype(np.uint8)))
plt.title("Downsampled Image")

plt.show()

# Wywołanie funkcji upsample
upsampled_matrix = upsample_nearest_neighbor(downsampled_matrix, scale_factor=16)

# Wyświetlenie zeskalowanego i ponownie zeskalowanego obrazu
plt.subplot(1, 2, 1)
plt.imshow(Image.fromarray(downsampled_matrix.astype(np.uint8)))
plt.title("Downsampled Image")

plt.subplot(1, 2, 2)
plt.imshow(Image.fromarray(upsampled_matrix.astype(np.uint8)))
plt.title("Upsampled Image")

plt.show()
