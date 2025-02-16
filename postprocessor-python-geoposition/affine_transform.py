import numpy as np
from scipy.optimize import minimize


def affine_transform(params, pixel_coordinates, lat_lon_coordinates):
  a, b, c, d, e, f = params  # Affine parameters
  pixel_x, pixel_y = pixel_coordinates.T  # Transpose to get x and y
  # Calculate lat/lon from the affine parameters
  lat = a * pixel_x + b * pixel_y + c
  lon = d * pixel_x + e * pixel_y + f
  # Return the residual (difference) between calculated and actual lat/lon
  return np.concatenate([lat - lat_lon_coordinates[:, 0], lon - lat_lon_coordinates[:, 1]])

def transform_point(pixel, params):
  a, b, c, d, e, f = params
  pixel_x, pixel_y = pixel
  lat = a * pixel_x + b * pixel_y + c
  lon = d * pixel_x + e * pixel_y + f
  return lat, lon


def get_pixel_to_coordinates(known_points, pixel) -> tuple[float, float]:
  pixel_coordinates = np.array([p["pixel"] for p in known_points])
  lat_lon_coordinates = np.array([p["lat_lon"] for p in known_points])

  initial_params = [0, 0, 0, 0, 0, 0]  # Initial guess
  result = minimize(
    lambda params: np.sum(affine_transform(params, pixel_coordinates, lat_lon_coordinates) ** 2),
    initial_params,
  )
  # Extract optimized parameters
  optimized_params = result.x

  return transform_point(pixel, optimized_params)