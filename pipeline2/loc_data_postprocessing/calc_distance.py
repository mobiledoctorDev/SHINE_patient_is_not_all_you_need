from math import pi, sin, cos, sqrt, atan2


def distance_with_angle(lat1, lon1, lat2, lon2):
    """generally used geo measurement function"""
    R = 6378.137  # Radius of earth in KM
    dLat = lat2 * pi / 180 - lat1 * pi / 180
    dLon = lon2 * pi / 180 - lon1 * pi / 180
    a = sin(dLat / 2) * sin(dLat / 2) + cos(lat1 * pi / 180) * cos(lat2 * pi / 180) * sin(dLon / 2) * sin(dLon / 2)
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    d = R * c
    return d * 1000  # Km -> meters
