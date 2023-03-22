import numpy as np
import jax.numpy as jnp
import os
import cv2


def siemens_star(num_pixels=512, num_spokes=32):
    """
    Generates a 2D Siemens star image of shape ``[num_pixels num_pixels]``.

    Number of spokes in the star can be controlled with ``num_spokes``. Spokes
    will alternate between black and white (0.0 and 1.0).
    """
    X, Y = np.mgrid[0:num_pixels, num_pixels:0:-1] - (num_pixels / 2.0)
    R = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(X, Y) + np.pi
    S = np.zeros_like(R)
    for spoke in range(num_spokes):
        in_spoke = (theta >= ((spoke) * 2 * np.pi / num_spokes)) & (
            theta <= ((spoke + 1) * 2 * np.pi / num_spokes)
        )
        if not spoke % 2:
            S[in_spoke] = 1.0
    S *= R < (num_pixels / 2.0)
    return S


def draw_disks(image_size, coordinates, radius, color=255):
    """
    Create a grayscale image with disks drawn at each provided coordinate.

    Args:
        image_size (tuple): The desired image size as (height, width).
        coordinates (list or numpy.ndarray): A list of (x, y) coordinates where disks should be drawn.
        radius (int): The radius of the disks.
        color (int): An optional intensity for the disks (0-255).

    Returns:
        numpy.ndarray: The resulting grayscale image with disks drawn at the specified coordinates.
    """
    # Create a blank grayscale image with the desired size
    image = np.zeros(image_size, dtype=np.uint8)

    # Draw a disk at each coordinate
    for coord in coordinates:
        cv2.circle(image, (coord[0], coord[1]), radius, color, -1)

    return image

class RandDiskGenerator: # TODO avoid overlapping disks
    def __init__(self,
                 N,
                 n_points,
                 radius,
                 shape,
                 z_range):
        '''
        Create a dataset of random 3D coordinates and their associated image.
        Each generated sample consists of an array of [n_points x y z] with
        shape: n_points * 3, accompanied by a 3D image with shape `shape`. The
        last dimension of `shape` represents the z axis. The number of planes
        will be infered from the `shape` argument. This is meant for Tensorflow
        and PyTorch data loaders that support generators.
        
        Parameters
        ----------
        N : int
            Number of samples in the dataset. Avoid large N as the coordinates
            on each epoch are pre-stored for speed. On each new epoch the samples
            are randomized again (new random coordinates are generated) therefore
            you can easily use small `N` to avoid memory issues.
        n_points : int
            Number of points in each sample. For 3D samples these samples will
            be randomly split between the planes.
        radius : int
            Radius of the disks to be drawn on each plane.
        shape : tuple or list
            Shape of the output image. Dimensions are [h w n_z] where n_z is the
            number of planes in 3D. For 2D samples use z=1.
        z_range : list
            Minimum and Maximum values for z values [min, max]. The returned
            coordinates are [x y z] and this parameter determines the range of
            the z coordinates.

        Returns
        -------
        None.

        '''
        
        assert len(shape) == 3, 'Shape must specify three dimensions, shape parameter is: {}'.format(len(shape))
        self.N = N
        self.radius = radius
        self.shape = shape
        self.z_range = z_range
        self.n_points = n_points
        self.num_planes = shape[-1]
        
        self.__get_randomized_coords()
        
    
    def __get_randomized_coords(self):
        '''
        Generate all the random coordinates. This is called when generator
        is instanciated or end of epoch is reached.

        Returns
        -------
        None.

        '''
        self.centery = np.random.randint(low = self.radius,
                                         high = self.shape[0]-self.radius,
                                         size = (self.N, self.n_points))
        self.centerx = np.random.randint(low = self.radius,
                                         high = self.shape[1]-self.radius,
                                         size = (self.N, self.n_points))
        
        if self.num_planes > 1:
            self.z_indices = np.random.randint(low = 0,
                                               high = self.num_planes,
                                               size = (self.N, self.n_points))
            self.z_values = np.random.rand(self.N, self.num_planes) * (self.z_range[1] - self.z_range[0])
            self.z_values += self.z_range[0]
            self.z_values.sort(axis=1)
            self.z = np.zeros_like(self.centerx).astype(np.float32)
        

    def __len__(self):
        return self.N
    

    def __getitem__(self, idx):
        '''
        Get a new sample.

        Parameters
        ----------
        idx : int
            Index of the current sample that needs to be generated.
            Automatically provided.

        Returns
        -------
        numpy.ndarray
            A [num_points 2] or [num_points 3] array containing the coordinates.
        numpy.ndarray
            2D or 3D Image corresponding to the coordinates.

        '''
        coords = np.array([self.centerx[idx], self.centery[idx]]).T
        
        if self.num_planes > 1:
            canvas = np.zeros(self.shape)
            for i in range(self.num_planes):
                canvas[:, :, i] = draw_disks(self.shape[:-1],
                                            coords[self.z_indices[idx] == i, :],
                                            self.radius,
                                            color = 255) #TODO add weight
                print(self.z_values[idx, i])
                self.z[idx, self.z_indices[idx] == i] = self.z_values[idx, i]
            return np.array([self.centerx[idx], self.centery[idx], self.z[idx]]).T, canvas #TODO add weight
                
        else:
            image = draw_disks(self.shape[:-1],
                               coords,
                               self.radius,
                               color = 255)[..., None] #TODO add weight
            return coords, image
    
    
    def __call__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)
            
            if i == self.__len__()-1:
                self.on_epoch_end()
        
        
    #shuffles the dataset at the end of each epoch
    def on_epoch_end(self):
        self.__get_randomized_coords()
