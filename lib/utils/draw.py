import cv2
import numpy as np
from utils.transform import corners_to_img
from utils.transform import projectToImage
import mayavi.mlab as mlab
def drawBox3D(img, corners):

    # img = np.copy(img)
    corners = corners.astype(np.int32)

    cv2.line(img, (corners[0,0], corners[1,0]), (corners[0,1], corners[1,1]), thickness=2, color=(0, 255, 255))
    cv2.line(img, (corners[0,1], corners[1,1]), (corners[0,2], corners[1,2]), thickness=2, color=(0, 255, 255))
    cv2.line(img, (corners[0,2], corners[1,2]), (corners[0,3], corners[1,3]), thickness=2, color=(0, 255, 255))
    cv2.line(img, (corners[0,3], corners[1,3]), (corners[0,0], corners[1,0]), thickness=2, color=(0, 255, 255))

    cv2.line(img, (corners[0,4], corners[1,4]), (corners[0,5], corners[1,5]), thickness=2, color=(0, 255, 255))
    cv2.line(img, (corners[0,5], corners[1,5]), (corners[0,6], corners[1,6]), thickness=2, color=(0, 255, 255))
    cv2.line(img, (corners[0,6], corners[1,6]), (corners[0,7], corners[1,7]), thickness=2, color=(0, 255, 255))
    cv2.line(img, (corners[0,7], corners[1,7]), (corners[0,4], corners[1,4]), thickness=2, color=(0, 255, 255))

    cv2.line(img, (corners[0,0], corners[1,0]), (corners[0,4], corners[1,4]), thickness=2, color=(0, 255, 255))
    cv2.line(img, (corners[0,1], corners[1,1]), (corners[0,5], corners[1,5]), thickness=2, color=(0, 255, 255))
    cv2.line(img, (corners[0,2], corners[1,2]), (corners[0,6], corners[1,6]), thickness=2, color=(0, 255, 255))
    cv2.line(img, (corners[0,3], corners[1,3]), (corners[0,7], corners[1,7]), thickness=2, color=(0, 255, 255))

    return img


def show_lidar_corners(test_image, lidar_corners, calib):
    test = np.copy(test_image)
    for i in range(lidar_corners.shape[0]):
        img_corners = corners_to_img(lidar_corners[i], calib[3], calib[2], calib[0])
        test = drawBox3D(test, img_corners/img_corners[2,:])
    return test

def show_cam_corners(test_image, cam_corners, calib):
    test = np.copy(test_image)
    for i in range(cam_corners.shape[0]):
        cam_corners_i = cam_corners[i]
        if cam_corners[i].shape[0] == 24:
            cam_corners_i = cam_corners[i].reshape((3, 8))
        img_corners = projectToImage(cam_corners_i, calib[0])
        test = drawBox3D(test, img_corners)
    return test

def show_image_boxes(test_image, img_boxes):
    test = np.copy(test_image)
    num = len(img_boxes)
    for n in range(num):
        x1,y1,x2,y2 = img_boxes[n]
        cv2.rectangle(test,(x1,y1), (x2,y2), (255,255,0), 2)
    return test

def scale_to_255(a, min, max, dtype=np.uint8):
    """ Scales an array of values from specified min, max range to 0-255
        Optionally specify the data type of the output (default is uint8)
    """
    return (((a - min) / float(max - min)) * 255).astype(dtype)

def draw_lidar(lidar, is_grid=False, is_top_region=False, fig=None):
    pxs=lidar[:,0]
    pys=lidar[:,1]
    pzs=lidar[:,2]
    prs=lidar[:,3]

    if fig is None: fig = mlab.figure(figure=None, bgcolor=(0,0,0), fgcolor=None, engine=None, size=(1000, 500))
    mlab.points3d(
        pxs, pys, pzs, prs,
        mode='point',  # 'point'  'sphere'
        colormap='gnuplot',  #'bone',  #'spectral',  #'copper',
        scale_factor=1,
        figure=fig)
    #draw grid
    if is_grid:
        mlab.points3d(0, 0, 0, color=(1,1,1), mode='sphere', scale_factor=0.2)

        for y in np.arange(-50,50,1):
            x1,y1,z1 = -50, y, 0
            x2,y2,z2 =  50, y, 0
            mlab.plot3d([x1, x2], [y1, y2], [z1,z2], color=(0.5,0.5,0.5), tube_radius=None, line_width=1, figure=fig)

        for x in np.arange(-50,50,1):
            x1,y1,z1 = x,-50, 0
            x2,y2,z2 = x, 50, 0
            mlab.plot3d([x1, x2], [y1, y2], [z1,z2], color=(0.5,0.5,0.5), tube_radius=None, line_width=1, figure=fig)

    #draw axis
    if 1:
        mlab.points3d(0, 0, 0, color=(1,1,1), mode='sphere', scale_factor=0.2)

        axes=np.array([
            [2.,0.,0.,0.],
            [0.,2.,0.,0.],
            [0.,0.,2.,0.],
        ],dtype=np.float64)
        fov=np.array([  ##<todo> : now is 45 deg. use actual setting later ...
            [20., 20., 0.,0.],
            [20.,-20., 0.,0.],
        ],dtype=np.float64)


        mlab.plot3d([0, axes[0,0]], [0, axes[0,1]], [0, axes[0,2]], color=(1,0,0), tube_radius=None, figure=fig)
        mlab.plot3d([0, axes[1,0]], [0, axes[1,1]], [0, axes[1,2]], color=(0,1,0), tube_radius=None, figure=fig)
        mlab.plot3d([0, axes[2,0]], [0, axes[2,1]], [0, axes[2,2]], color=(0,0,1), tube_radius=None, figure=fig)
        mlab.plot3d([0, fov[0,0]], [0, fov[0,1]], [0, fov[0,2]], color=(1,1,1), tube_radius=None, line_width=1, figure=fig)
        mlab.plot3d([0, fov[1,0]], [0, fov[1,1]], [0, fov[1,2]], color=(1,1,1), tube_radius=None, line_width=1, figure=fig)

    #draw top_image feature area
    if is_top_region:
        x1 = TOP_X_MIN
        x2 = TOP_X_MAX
        y1 = TOP_Y_MIN
        y2 = TOP_Y_MAX
        mlab.plot3d([x1, x1], [y1, y2], [0,0], color=(0.5,0.5,0.5), tube_radius=None, line_width=1, figure=fig)
        mlab.plot3d([x2, x2], [y1, y2], [0,0], color=(0.5,0.5,0.5), tube_radius=None, line_width=1, figure=fig)
        mlab.plot3d([x1, x2], [y1, y1], [0,0], color=(0.5,0.5,0.5), tube_radius=None, line_width=1, figure=fig)
        mlab.plot3d([x1, x2], [y2, y2], [0,0], color=(0.5,0.5,0.5), tube_radius=None, line_width=1, figure=fig)

    mlab.orientation_axes()
    mlab.view(azimuth=180,elevation=None,distance=50,focalpoint=[ 12.0909996 , -1.04700089, -2.03249991])#2.0909996 , -1.04700089, -2.03249991

def draw_gt_boxes3d(gt_boxes3d, fig, color=(1,1,1), line_width=2):

    num = len(gt_boxes3d)
    for n in range(num):
        b = gt_boxes3d[n]

        mlab.text3d(b[0,0], b[0,1], b[0,2], '%d'%n, scale=(1, 1, 1), color=color, figure=fig)
        for k in range(0,4):

            #http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
            i,j=k,(k+1)%4
            mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)

            i,j=k+4,(k+1)%4 + 4
            mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)

            i,j=k,k+4
            mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)

    mlab.view(azimuth=180,elevation=None,distance=50,focalpoint=[ 12.0909996 , -1.04700089, -2.03249991])#2.0909996 , -1.04700089, -2.03249991
