import numpy as np

TOP_X_MAX = 70.3
TOP_X_MIN = 0
TOP_Y_MIN = -40
TOP_Y_MAX = 40
RES = 0.1
LIDAR_HEIGHT = 1.73
CAR_HEIGHT = 1.56


def _lidar_to_bv_coord(x, y):
    X0, Xn = 0, int((TOP_X_MAX - TOP_X_MIN) // RES) + 1
    Y0, Yn = 0, int((TOP_Y_MAX - TOP_Y_MIN) // RES) + 1

    xx = Yn - (y - TOP_Y_MIN) // RES
    yy = Xn - (x - TOP_X_MIN) // RES

    return xx, yy


def lidar_to_bv_single(rois_3d):
    """
    cast lidar 3d points(x, y, z, l, w, h) to bird view (x1, y1, x2, y2)
    """
    assert(rois_3d.shape[0] == 6)
    rois = np.zeros((4))

    rois[0] = rois_3d[0] + rois_3d[3] * 0.5
    rois[1] = rois_3d[1] + rois_3d[4] * 0.5
    rois[2] = rois_3d[0] - rois_3d[3] * 0.5
    rois[3] = rois_3d[1] - rois_3d[4] * 0.5

    rois[0], rois[1] = _lidar_to_bv_coord(rois[0], rois[1])
    rois[2], rois[3] = _lidar_to_bv_coord(rois[2], rois[3])

    return rois


def lidar_to_bv(rois_3d):
    """
    cast lidar 3d points(x, y, z, l, w, h) to bird view (x1, y1, x2, y2)
    """

    rois = np.zeros((rois_3d.shape[0], 5))
    rois[:, 0] = rois_3d[:, 0]

    rois[:, 1] = rois_3d[:, 1] + rois_3d[:, 4] * 0.5
    rois[:, 2] = rois_3d[:, 2] + rois_3d[:, 5] * 0.5
    rois[:, 3] = rois_3d[:, 1] - rois_3d[:, 4] * 0.5
    rois[:, 4] = rois_3d[:, 2] - rois_3d[:, 5] * 0.5

    rois[:, 1], rois[:, 2] = _lidar_to_bv_coord(rois[:, 1], rois[:, 2])
    rois[:, 3], rois[:, 4] = _lidar_to_bv_coord(rois[:, 3], rois[:, 4])

    return rois.astype(np.float32)


def _bv_to_lidar_coord(x, y):
    Y0, Yn = 0, int((TOP_X_MAX - TOP_X_MIN) // RES) + 1
    X0, Xn = 0, int((TOP_Y_MAX - TOP_Y_MIN) // RES) + 1
    yy = (Yn - y) * RES + TOP_Y_MIN
    xx = (Xn - x) * RES + TOP_X_MIN
    return xx, yy


def _camera_to_lidar(x, y, z=None):
    return None


def camera_to_lidar(pts_3D, P):
    """
    convert camera(x, y, z, l, w, h) to lidar (x, y, z, l, w, h)
    """
    points = np.ones((1, 4))
    points[0, :3] = pts_3D[:3]
    points = points.reshape((4, 1))
    # print(points)

    R = np.linalg.inv(P[:, :3])

    # T = -P[:, 3].reshape((3, 1))
    T = np.zeros((3, 1))
    T[0] = -P[1,3] 
    T[1] = -P[2,3]
    T[2] = P[0,3]
    RT = np.hstack((R, T))

    points_lidar = np.dot(RT, points)

    pts_3D_lidar = np.zeros(6)
    pts_3D_lidar[:3] = points_lidar.flatten()

    pts_3D_lidar[3:6] = pts_3D[3:6]

    return pts_3D_lidar

def lidar_to_corners(pts_3D):
    """ 
    convert pts_3D_lidar (x, y, z, l, w, h) to
    8 corners (x0, ... x7, y0, ...y7, z0, ... z7)

    (x0, y0, z0) at left,forward, up.
    clock-wise
    """
    l = pts_3D[3]
    w = pts_3D[4]
    h = pts_3D[5]

    x_corners = np.array([l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2])
    y_corners = np.array([w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2])
    z_corners = np.array([h,h,h,h,0,0,0,0])

    corners = np.vstack((x_corners, y_corners, z_corners))

    corners[0,:] = corners[0,:] + pts_3D[0]
    corners[1,:] = corners[1,:] + pts_3D[1]
    corners[2,:] = corners[2,:] + pts_3D[2]

    return corners.reshape(-1).astype(np.float32)


def _projectToImage(pts_3D, P):
    """
    PROJECTTOIMAGE projects 3D points in given coordinate system in the image
    plane using the given projection matrix P.

    Usage: pts_2D = projectToImage(pts_3D, P)
    input: pts_3D: 3xn matrix
          P:      3x4 projection matrix
    output: pts_2D: 2xn matrix

    last edited on: 2012-02-27
    Philip Lenz - lenz@kit.edu
    """
    # project in image
    mat = np.vstack((pts_3D, np.ones((pts_3D.shape[1]))))

    pts_2D = np.dot(P, mat)
    # print(pts_2D)

    # scale projected points
    pts_2D[0, :] = pts_2D[0, :] / pts_2D[2, :]
    pts_2D[1, :] = pts_2D[1, :] / pts_2D[2, :]
    pts_2D = np.delete(pts_2D, 2, 0)
    # pts_2D[2,:] = np.zeros(())
    return pts_2D

# TODO
def lidar_to_image(pts_3D, P):
    return None

if __name__ == '__main__':
    P = np.array([6.927964000000e-03, -9.999722000000e-01, -2.757829000000e-03,
                  -2.457729000000e-02, -1.162982000000e-03, 2.749836000000e-03,
                  -9.999955000000e-01, -6.127237000000e-02, 9.999753000000e-01,
                  6.931141000000e-03, -1.143899000000e-03, -3.321029000000e-01]).astype(np.float32).reshape((3, 4))
    camera = [1.84, 1., 8.41, 5.78, 1.90, 2.72]
    lidar = camera_to_lidar(camera, P)
    corners = lidar_to_corners(lidar)
    corners = corners.reshape((3, 8))
    # print(lidar)
    # print(corners)
