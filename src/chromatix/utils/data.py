import numpy as np
from typing import Optional, Tuple, Union

try:
    import cv2

    USE_CV2 = True
except ModuleNotFoundError:
    USE_CV2 = False


def sqr_dist_to_line(
    z: np.ndarray, y: np.ndarray, x: np.ndarray, start: np.ndarray, n: np.ndarray
) -> np.ndarray:
    """
    Returns an array with each pixel being assigned to the square distance to
    that line and an array with the distance along the line.
    """
    dx = x - start[2]
    dy = y - start[1]
    dz = z - start[0]
    dot_dn = dx * n[2] + dy * n[1] + dz * n[0]
    return (dx - dot_dn * n[2]) ** 2 + (dy - dot_dn * n[1]) ** 2 + (
        dz - dot_dn * n[0]
    ) ** 2, dot_dn


def draw_line(
    arr: np.ndarray,
    start: np.ndarray,
    stop: np.ndarray,
    thickness: float = 0.3,
    intensity: float = 1.0,
) -> np.ndarray:
    """
    Draw a line in a 3D object with a given thickness and intensity.

    Args:
        arr: The object to draw the line in.
        start: The start of the line.
        end: The end of the line.
        thickness: The thickness of the line.
        intensity: The intensity of the line.
    """
    direction = np.subtract(stop, start)
    line_length = np.sqrt(np.sum(np.square(direction)))
    n = direction / line_length

    sigma2 = 2 * thickness**2

    z, y, x = np.meshgrid(
        np.arange(arr.shape[0]),
        np.arange(arr.shape[1]),
        np.arange(arr.shape[2]),
        indexing="ij",
    )
    d2, t = sqr_dist_to_line(z, y, x, start, n)

    line_weight = (
        (t > 0) * (t < line_length)
        + (t <= 0) * np.exp(-(t**2) / sigma2)
        + (t >= line_length) * np.exp(-((t - line_length) ** 2) / sigma2)
    )
    return arr + intensity * np.exp(-d2 / sigma2) * line_weight


def filaments_3d(
    sz: Tuple[int, int, int],
    intensity: float = 1.0,
    radius: Union[float, Tuple[float, float, float]] = 0.8,
    rand_offset: float = 0.05,
    rel_theta: float = 1.0,
    num_filaments: int = 50,
    apply_seed: bool = True,
    thickness: float = 0.3,
) -> np.ndarray:
    """
    Create a 3D representation of filaments.

    Args:
        sz: A 3D shape tuple representing the size of the object.
        radius: A tuple of real numbers (or a single real number) representing
            the relative radius of the volume in which the filaments will be
            created. Default is 0.8. If a tuple is used, the filamets will be
            created in a corresponding elliptical volume. Note that the radius
            is only enforced in the version `filaments_3d` which creates the
            array rather than adding.
        rand_offset: A tuple of real numbers representing the random offsets of
            the filaments in relation to the size. Default is 0.05.
        rel_theta: A real number representing the relative theta range of the
            filaments. Default is 1.0.
        num_filaments: An integer representing the number of filaments to be
            created. Default is 50.
        apply_seed: A boolean representing whether to apply a seed to the random
            number generator. Default is ``True``.
        thickness: A real number representing the thickness of the filaments in
            pixels. Default is 0.8.

    This code is based on the SyntheticObjects.jl package by Hossein Zarei
    Oshtolagh and Rainer Heintzmann.
    """

    sz = np.array(sz)
    radius = np.array(radius)

    # Save the state of the rng to reset it after the function is done
    rng_state = np.random.get_state()
    if apply_seed:
        np.random.seed(42)

    # Create the object
    obj = np.zeros(sz, dtype=np.float32)

    mid = sz // 2

    # Draw random lines equally distributed over the 3D sphere
    for n in range(num_filaments):
        phi = 2 * np.pi * np.random.rand()
        # Theta should be scaled such that the distribution over the unit sphere is uniform
        theta = np.arccos(rel_theta * (1 - 2 * np.random.rand()))
        pos = (sz * radius / 2) * np.array(
            [
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta),
            ]
        )
        pos_offset = np.array(rand_offset * sz * (np.random.rand(3) - 0.5))
        # Draw line
        obj = draw_line(
            obj,
            pos + pos_offset + mid,
            mid + pos_offset - pos,
            thickness=thickness,
            intensity=intensity,
        )

    # Reset the rng to the state before this function was called
    np.random.set_state(rng_state)
    return obj


def pollen_3d(
    sz: Tuple[int, int, int],
    intensity: float = 1.0,
    radius: float = 0.8,
    dphi: float = 0.0,
    dtheta: float = 0.0,
    thickness: float = 0.8,
    filled: bool = False,
    filled_rel_intensity=0.1,
) -> np.ndarray:
    """
    Create a 3D representation of a pollen grain.

    Args:
        sz: A tuple of three integers representing the size of the volume in
            which the pollen grain will be created. Default is ``(128, 128,
            128)``.
        radius: Roughly the relative radius of the pollen grain.
        dphi: A float representing the phi angle offset in radians. Default
            is 0.0.
        dtheta: A float representing the theta angle offset in radians. Default
            is 0.0.
        thickness: A float representing the thickness of the pollen grain.
            Default is 0.8.
        filled: A boolean representing whether the pollen grain should be
            filled. Default is ``False``.
        filled_rel_intensity: A float representing the relative intensity of the
            filled part of the pollen grain. Default is 0.1.
    Returns:
        ret: A 3D array representing the pollen grain.

    This code is based on the SyntheticObjects.jl package by Hossein Zarei
    Oshtolagh and Rainer Heintzmann and the original code by Kai Wicker.
    """

    sz = np.array(sz)
    z, y, x = np.meshgrid(
        np.linspace(-radius, radius, sz[0]),
        np.linspace(-radius, radius, sz[1]),
        np.linspace(-radius, radius, sz[2]),
        indexing="ij",
    )
    thickness = thickness / sz[0]

    r = x**2 + y**2 + z**2

    phi = np.atan2(y, x)
    theta = np.asin(z / (np.sqrt(x**2 + y**2 + z**2) + 1e-2)) + dtheta

    a = np.abs(np.cos(theta * 20))
    b = np.abs(
        np.sin(
            (phi + dphi) * np.sqrt(np.maximum(0, np.cos(theta) * (20.0**2)))
            - theta
            + np.pi / 2
        )
    )

    # calculate the relative distance to the surface of the pollen grain
    dc = ((0.4 + 1 / 20.0 * (a * b) ** 5) + np.cos(phi + dphi) * 1 / 20) - r
    # return dc

    sigma2 = 2 * (thickness**2)
    res = (
        intensity * np.exp(-(dc**2) / sigma2)
        + filled * (dc > 0) * intensity * filled_rel_intensity
    )

    return res


def siemens_star(
    num_pixels: int = 512, num_spokes: int = 32, radius: Optional[int] = None
) -> np.ndarray:
    """
    Generates a 2D Siemens star image of shape ``num_pixels``. A single input
    is interpreted as a square-shaped array. ``radius`` is the radius of the
    star in pixels. If not provided, it will be half of the image size along
    each dimension.

    Number of spokes in the star can be controlled with ``num_spokes``. Spokes
    will alternate between black and white (0.0 and 1.0).
    """

    num_pixels = np.atleast_1d(num_pixels)
    if num_pixels.size == 1:
        num_pixels = np.repeat(num_pixels, 2)
    if radius is None:
        radius = num_pixels / 2
    radius = np.atleast_1d(radius)
    if radius.size == 1:
        radius = np.repeat(radius, 2)
    ctr = num_pixels // 2
    X, Y = np.mgrid[
        -ctr[0] : num_pixels[0] - ctr[0], num_pixels[1] - ctr[1] : -ctr[1] : -1
    ]
    R = np.sqrt((X / radius[1]) ** 2 + (Y / radius[0]) ** 2)
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
            np.sum(np.array(np.meshgrid(_samples, _samples)) ** 2, axis=0) <= radius**2
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
