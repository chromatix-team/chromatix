import numpy as np
from typing import Tuple

try:
    import cv2

    USE_CV2 = True
except ModuleNotFoundError:
    USE_CV2 = False


def siemens_star(num_pixels: int = 512, num_spokes: int = 32, radius: int = None) -> np.ndarray:
    """
    Generates a 2D Siemens star image of shape ``num_pixels``. A single input is interpreted as a square-shaped array.
    ``radius`` is the radius of the star in pixels. If not provided, it will be half of the image size along each dimension.

    Number of spokes in the star can be controlled with ``num_spokes``. Spokes
    will alternate between black and white (0.0 and 1.0).
    """

    num_pixels = np.atleast_1d(num_pixels)
    if num_pixels.size == 1:
        num_pixels = np.repeat(num_pixels, 2)        

    radius = np.atleast_1d(radius)
    if (radius is None):
        radius = num_pixels / 2

    if radius.size == 1:
        radius = np.repeat(radius, 2)

    ctr = num_pixels // 2
    X, Y = np.mgrid[-ctr[0]:num_pixels[0]-ctr[0], num_pixels[1]-ctr[1]:-ctr[1]:-1]
    R = np.sqrt((X/radius[1])**2 + (Y/radius[0])**2)
    theta = np.arctan2(X, Y) + np.pi
    S = np.zeros_like(R)
    for spoke in range(num_spokes):
        in_spoke = (theta >= ((spoke) * 2 * np.pi / num_spokes)) & (
            theta <= ((spoke + 1) * 2 * np.pi / num_spokes)
        )
        if not spoke % 2:
            S[in_spoke] = 1.0
    S *= R < 1.0
    return S


if USE_CV2:

    def draw_disks(
        shape: Tuple[int, int], coordinates: np.ndarray, radius: int, color: int = 255
    ) -> np.ndarray:
        """
        Create a grayscale image with disks drawn at each provided coordinate.

        Args:
            image_size: The desired image size as (height, width).
            coordinates: A list of (x, y) coordinates where disks should be
                drawn.
            radius: The radius of the disks.
            color: An optional intensity for the disks (0-255).

        Returns:
            numpy.ndarray: The resulting grayscale image with disks drawn at
                the specified coordinates.
        """
        image = np.zeros(shape, dtype=np.uint8)
        for coord in coordinates:
            cv2.circle(image, (coord[0], coord[1]), radius, color, -1)
        return image

else:

    def draw_disks(
        shape: Tuple[int, int], coordinates: np.ndarray, radius: int, color: int = 255
    ) -> np.ndarray:
        """
        Create a grayscale image with disks drawn at each provided coordinate.

        Args:
            image_size: The desired image size as (height, width).
            coordinates: A list of (x, y) coordinates where disks should be
                drawn.
            radius: The radius of the disks.
            color: An optional intensity for the disks (0-255).

        Returns:
            numpy.ndarray: The resulting grayscale image with disks drawn at
                the specified coordinates.
        """
        image = np.zeros([s + radius * 2 for s in shape], dtype=np.uint8)
        _samples = np.linspace(-radius, radius, num=radius * 2, dtype=np.float32)
        circle = color * np.uint8(
            np.sum(np.array(np.meshgrid(_samples, _samples)) ** 2, axis=0)
            <= radius**2
        )
        for c in coordinates:
            slices = (slice(c[0], c[0] + radius * 2), slice(c[1], c[1] + radius * 2))
            image[slices] = image[slices] | circle
        image = image[radius : radius + shape[0], radius : radius + shape[1]]
        return image


class RandDiskGenerator:  # TODO avoid overlapping disks
    def __init__(
        self,
        N: int,
        num_points: int,
        radius: int,
        shape: Tuple[int, int],
        z_range: Tuple[int, int],
    ):
        """
        Create a dataset of random 3D coordinates and their associated image.
        Each generated sample consists of an array of x y z coordinates with
        shape: (n_points 3), accompanied by a 3D image with shape `shape`.
        The last dimension of `shape` represents the z axis, if it exists. The
        number of planes will be inferred from the `shape` argument. This is
        meant for TensorFlow and PyTorch data loaders that support generators.

        Args:
            N: Number of samples in the dataset. Avoid large N as the coordinates
                on each epoch are pre-stored for speed. On each new epoch the
                samples are randomized again (new random coordinates are generated)
                therefore you can easily use small `N` to avoid memory issues.
            num_points: Number of points in each sample. For 3D samples these samples
                will be randomly split between the planes.
            radius: Radius of the disks to be drawn on each plane.
            shape: Shape of the output image. Dimensions are [h w n_z] where n_z is
                the number of planes in 3D. For 2D samples use z=1.
            z_range: list
                Minimum and Maximum values for z values [min, max]. The returned
                coordinates are [x y z] and this parameter determines the range of
                the z coordinates.
        """

        assert (
            len(shape) == 3
        ), "Shape must specify three dimensions, shape parameter is: {}".format(
            len(shape)
        )
        self.N = N
        self.radius = radius
        self.shape = shape
        self.z_range = z_range
        self.num_points = num_points
        self.num_planes = shape[-1]
        self.reset()

    def reset(self):
        """
        Generate all the random coordinates. This is called when generator
        is instantiated or the last sample in the generator is reached.
        """
        self.y = np.random.randint(
            low=self.radius,
            high=self.shape[0] - self.radius,
            size=(self.N, self.num_points),
        )
        self.x = np.random.randint(
            low=self.radius,
            high=self.shape[1] - self.radius,
            size=(self.N, self.num_points),
        )

        if self.num_planes > 1:
            self.z_indices = np.random.randint(
                low=0, high=self.num_planes, size=(self.N, self.num_points)
            )
            self.z_values = np.random.rand(self.N, self.num_planes) * (
                self.z_range[1] - self.z_range[0]
            )
            self.z_values += self.z_range[0]
            self.z_values.sort(axis=1)
            self.z = np.zeros_like(self.x).astype(np.float32)

    def __len__(self) -> int:
        return self.N

    def __getitem__(self, idx: int) -> np.ndarray:
        """
        Get a new sample.

        Args:
            idx: Index of the current sample that needs to be generated.

        Returns:
            numpy.ndarray
                A (num_points 2) or (num_points 3) array containing the
                    coordinates.
            numpy.ndarray
                2D or 3D Image corresponding to the coordinates.
        """
        coords = np.array([self.x[idx], self.y[idx]]).T

        if self.num_planes > 1:
            canvas = np.zeros(self.shape)
            for i in range(self.num_planes):
                canvas[:, :, i] = draw_disks(
                    self.shape[:-1],
                    coords[self.z_indices[idx] == i, :],
                    self.radius,
                    color=255,
                )  # TODO add weight
                print(self.z_values[idx, i])
                self.z[idx, self.z_indices[idx] == i] = self.z_values[idx, i]
            return (
                np.array([self.x[idx], self.y[idx], self.z[idx]]).T,
                canvas,
            )  # TODO add weight

        else:
            image = draw_disks(self.shape[:-1], coords, self.radius, color=255)[
                ..., None
            ]  # TODO add weight
            return coords, image

    def __call__(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get a new sample. Automatically iterates through samples of coordinates
        with every call. Will cause the random coordinates to be regenerated
        when the last sample is reached.

        Returns:
            numpy.ndarray
                A (num_points 2) or (num_points 3) array containing the
                    coordinates.
            numpy.ndarray
                2D or 3D Image corresponding to the coordinates.
        """
        for i in range(self.__len__()):
            yield self.__getitem__(i)
            if i == self.__len__() - 1:
                self.reset()
