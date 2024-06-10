import imageio
import jax.numpy as jnp
import matplotlib.pyplot as plt
from skimage import img_as_ubyte


def create_radial_pattern(shape):
    """
    Create a basic radial pattern image.

    Args:
        shape (tuple): Shape of the image (height, width).

    Returns:
        jnp.ndarray: Radial pattern image.
    """
    # Create a grid of coordinates
    y, x = jnp.indices(shape)

    # Calculate the center of the image
    center_y, center_x = shape[0] // 2, shape[1] // 2

    # Compute the distances from the center
    distances = jnp.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

    # Normalize distances to range [0, 2*pi] for phase pattern
    max_distance = jnp.sqrt(center_x**2 + center_y**2)
    phase_pattern = (distances / max_distance) * 2 * jnp.pi

    return phase_pattern


def save_phase_pattern():
    # Create the radial pattern
    shape = (512, 512)
    radial_pattern = create_radial_pattern(shape)

    # Save the pattern as a PNG file
    plt.imshow(radial_pattern, cmap="hsv")
    plt.colorbar()
    plt.title("Radial Phase Pattern")
    plt.axis("off")  # Hide the axis
    plt.savefig("data/radial_pattern.png", bbox_inches="tight", pad_inches=0)
    plt.show()


def normalize_grayscale_image(input_path, output_path):
    # Read the image
    img = imageio.imread(input_path)

    # Normalize the grayscale image
    normalized_img = img / img.max()

    # Convert the normalized image to 8-bit unsigned integer format
    normalized_img_ubyte = img_as_ubyte(normalized_img)

    # Save the normalized grayscale image as a PNG
    imageio.imsave(output_path, normalized_img_ubyte)
