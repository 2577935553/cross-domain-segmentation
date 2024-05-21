import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

def elastic_transform_by_scale(image, label, scale, centre, width, smooth):
  '''
  :param image: HW format
  :param label: HW format or None
  :param scale: tuple (x_scale, y_scale). Scale factor of 1 means no deformation
  :param centre: tuple (x_centre, y_centre)
  :param width: tuple (x_width, y_width)
  :param smooth: recommended to be within 10 - 15  分smooth_x和smooth_y
  :return:
  '''
  x_scale, y_scale = scale
  x_centre, y_centre = centre
  x_width, y_width = width

  x_scale_map = np.ones_like(image)
  x_scale_map[x_centre-x_width : x_centre+x_width, y_centre-y_width : y_centre+y_width] = 1.0 / x_scale
  x_scale_map = gaussian_filter(x_scale_map, smooth[0], mode="constant", cval=1)

  y_scale_map = np.ones_like(image)
  y_scale_map[x_centre-x_width : x_centre+x_width, y_centre-y_width : y_centre+y_width] = 1.0 / y_scale
  y_scale_map = gaussian_filter(y_scale_map, smooth[1], mode="constant", cval=1)

  shapes = list(map(lambda x: slice(0, x, None), image.shape))
  grid = np.broadcast_arrays(*np.ogrid[shapes])

  indices = ((grid[0] - x_centre) * x_scale_map + x_centre, (grid[1] - y_centre) * y_scale_map + y_centre)

  transformed_image = map_coordinates(image, indices, order=1, mode='constant', cval=0).reshape(image.shape)

  if label is not None:
    transformed_label = map_coordinates(label, indices, order=0, mode='constant', cval=0).reshape(image.shape)
  else:
    transformed_label = None

  return transformed_image, transformed_label

# Parameters:
# deform_x_list = [1.0, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0]
# deform_y_list = [1.0, 0.8, 0.8, 0.7, 0.7, 0.6, 0.6]
# x_mid, y_mid are midpoints of the heart based on segmentation label
#
# for s in range(image.shape[-1]):
#     deformed_image[:,:,s], deformed_label[:,:,s] = elastic_transform_by_scale(image[:,:,s], label[:,:,s],
#                                                                   scale=(deform_x, deform_y),
#                                                                   centre=(x_mid, y_mid),
#                                                                   width=(30,30),
#                                                                   smooth=14)
if __name__ == '__main__':
    x=np.random.randint(0,1,[10,10])
    label_range = np.argwhere(x)
    print(label_range.size)

