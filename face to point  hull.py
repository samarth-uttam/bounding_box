import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull
import pandas as pd
import math
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import sympy as sym
from mpl_toolkits.mplot3d import Axes3D
import plotter

import pyenvelope

import homogenous_matrixes


xy_to_uv_file =  "C:/Users/Samarth/OneDrive - HKUST Connect/Thesis Project/IEDA/march/8/coarse_sheet.xls"
# xy_to_uv_file =  "C:/Users/Samarth/OneDrive - HKUST Connect/Thesis Project/IEDA/march/8/bunny.xls"
b = pd.read_excel(xy_to_uv_file)
uv_conv = pd.DataFrame(b)
uv_conv = uv_conv.drop(columns="u")
uv_conv = uv_conv.drop(columns="v")
uv_conv = uv_conv.drop(columns="i")
uv_conv = uv_conv.drop(columns="j")
pts = array_from_dataframe = uv_conv.to_numpy()

hull = ConvexHull(pts)

# plotter.threeD_plot(pts)


filler_points = []
for i in range(len(hull.points)):
    filler_points.append([0])

# print(filler_points)

points_matrix = np.append(hull.points, filler_points, axis=1)
# print(points_matrix)

face_eq = []

for i in range(len(hull.simplices)):
    point_1 = hull.simplices[i][0]
    point_2 = hull.simplices[i][1]
    point_3 = hull.simplices[i][2]
    # print(point_1, point_2, point_3)

    cord_point1 = hull.points[point_1]
    cord_point2 = hull.points[point_2]
    cord_point3 = hull.points[point_3]

    # print(cord_point1, cord_point2, cord_point3)

    v1 = cord_point3 - cord_point1
    v2 = cord_point2 - cord_point1
    cp = np.cross(v1, v2)
    coff_a , coff_b , coff_c = cp
    coff_d = (-1) * (np.dot(cp, cord_point3))

    # face equations in the form of ax +by + cz + d = 0 (abcd and coffecients )

    current_face = [coff_a, coff_b, coff_c, coff_d]
    # print(current_face)
    face_eq.append(current_face)

# print(face_eq)

#  get the distance of all the points from a given face and get the maximum of all

face_point_distance_array = []
max_face_point = []
face = j = 0
for j in range(len(face_eq)):

    current_distance_array = []
    for i in range(len(hull.points)):

        current_point = hull.points[i]
        p_1 = hull.points[i][0]
        p_2 = hull.points[i][1]
        p_3 = hull.points[i][2]

        # print(p_1, p_2, p_3)
        current_distance_num = abs((p_1 * face_eq[j][0]) + (p_2 * face_eq[j][1]) + (p_3 * face_eq[j][2]) + (face_eq[j][3]))
        current_distance_den =math.sqrt((face_eq[j][0] ** 2) + ((face_eq[j][1] ** 2 )+ (face_eq[j][2] ** 2) ))
        current_distance =  current_distance_num / current_distance_den
        # print(current_distance)
        current_distance_array.append(current_distance)
        face_point_distance_array_current = [j, i, current_distance]
        # print(face_point_distance_array_current)
        face_point_distance_array.append(face_point_distance_array_current)
    # print(current_distance_array)
    # print(max(current_distance_array))
    # print('\n')
    max_face_point.append(max(current_distance_array))
# print(max_face_point)
# print(len(max_face_point))

shortest_len = min(max_face_point)
print(shortest_len)
# print(len(face_point_distance_array))

for i in range(len(face_point_distance_array)):
    if face_point_distance_array[i][2] == shortest_len:
        # print(face_point_distance_array[i])
        edge_face = face_point_distance_array[i][0]
        edge_point = face_point_distance_array[i][1]
        edge_distance = face_point_distance_array[i][2]


print('edge face')
print(edge_face)
print('edge face points ')
print(hull.simplices[edge_face])
print('edge point')
print(edge_point)

print('base face coordinates')




edge_face_p1 = hull.simplices[edge_face][0]
edge_face_p2 = hull.simplices[edge_face][1]
edge_face_p3 = hull.simplices[edge_face][2]

print(hull.points[edge_face_p1], hull.points[edge_face_p2], hull.points[edge_face_p3])
print(edge_face_p1, edge_face_p2, edge_face_p3)

print('farthest point is = ')

print(hull.points[edge_point])


# print('edge face equation ')
# print(face_eq[edge_face])


    # getting the projection of the points on this plane

# this plane we know via hull simplices. z is the normal to this plane. x is the line joining the first two points of the hull simplices (hull.simplices[0][0,1])

#  standard point transformation. original coordinates are aligned with 0,0,0 and cartesian planes
# new co-ordinate system has origin at [point 0 of simplices 0],
# z prependicular to face above, x axis [joining simplices[0][point 0 nd 1] and y is normal to given plane and line or ( cross of x and line joining simplices[0][0] and highest point)


#  fist transform to now origin, simplices[0][0] and then rotate about it



new_org  = hull.points[hull.simplices[edge_face][0]]

x_v = hull.points[hull.simplices[edge_face][1]] - hull.points[hull.simplices[edge_face][0]]
x_v_unit =( x_v/np.linalg.norm(x_v)).tolist()
# print(x_v_unit)

z_v = np.cross((hull.points[hull.simplices[edge_face][2]] - hull.points[hull.simplices[edge_face][0]])  , (hull.points[hull.simplices[edge_face][1]] - hull.points[hull.simplices[edge_face][0]]))
# print(z_v)
#
#  making the z normal from plaane to point (so that point is on the top)

v_face_to_p = hull.points[edge_point] - hull.points[hull.simplices[edge_face][0]]

if np.dot(z_v, v_face_to_p) >=0:
    z_v = z_v
    # print('same directions')
else:
    z_v = -1 * (z_v)
    # print('changing directions')

z_v_unit = (z_v/np.linalg.norm(z_v)).tolist()

# print(z_v_unit)
y_v = np.cross(z_v, x_v)
y_v_unit = (np.cross(z_v_unit, x_v_unit)).tolist()

# print(y_v_unit)

new_frame = [x_v_unit, y_v_unit, z_v_unit]

old_x = [1,0,0]
old_y = [0,1,0]
old_z = [0,0,1]
old_frame = [old_x, old_y, old_z]

old_org = [0,0,0]
new_origin = hull.points[hull.simplices[edge_face][0]]

front_rot = homogenous_matrixes.homogenous_transformation_matrix(old_frame, new_frame)


# print('front rot')
# print(front_rot)
# print('front rot transpose')
# print(front_rot.transpose())
# print('front rot inverse')
# print(np.linalg.inv(front_rot))
# print(np.matmul(front_rot, np.linalg.inv(front_rot)))
# print(old_frame)
# print(new_frame)
# #
# print(np.dot(x_v, y_v))
# print(np.dot(y_v, z_v))
# print(np.dot(z_v, x_v))
#

# rotating all the points to new frame


rot_points = []
for i in range(len(hull.points)):
    curr = []
    curr = np.matmul(front_rot, hull.points[i])
    rot_points.append(curr)
#
# print(len(hull.points))
# print(len(rot_points))
# translating  all the points in the new origin [old origin to new origin]



point_rot_trans = []
for i in range(len(rot_points)):
    curr = []
    curr = homogenous_matrixes.translation(old_org, new_org, rot_points[i] )
    # curr = curr.tolist()
    point_rot_trans.append(curr)

# print(len(point_rot_trans))



# # to check the distance of all the points from the plaene if they are alomst 0 (for my given sheet )

# for i in range(len(point_rot_trans)):
#     distaa_n = abs((face_eq[edge_face][0] * point_rot_trans[i][0]) * (face_eq[edge_face][1] * point_rot_trans[i][1]) * (face_eq[edge_face][2] * point_rot_trans[i][2]) + (face_eq[edge_face][3]))
#     dista_d = ((face_eq[edge_face][0]**2) + (face_eq[edge_face][1]**2) + (face_eq[edge_face][2]**2))**0.5
#     d = distaa_n/dista_d
#     print(d)

#
# #  cutting the z of all the points

point_proj_2d = []
for i in range(len(point_rot_trans)):
    curr_xy = []
    x = point_rot_trans[i][0]
    y = point_rot_trans[i][1]
    curr_xy = [x, y]
    point_proj_2d.append(curr_xy)
# print(point_proj_2d)

xy_proj_1 = np.array(point_proj_2d)
xy_proj_1_tuple = tuple([tuple(row) for row in xy_proj_1])
# print(xy_proj_1)

#
a  = pyenvelope.get_minimum_bounding_rectangle(xy_proj_1)
#
print(a)

hull_2d = ConvexHull(xy_proj_1)
#
# # a = pyenvelope.get_minimum_bounding_rectangle(xy_proj_1)
#
rect_points  = a

#
rect_points_3d = []
for i in range(len(rect_points)):
    curr = []
    curr = [rect_points[i][0],rect_points[i][1], 0]
    rect_points_3d.append(curr)

rect_points_3d = list(rect_points_3d)
#
print(rect_points_3d)

print(len(rect_points_3d))
# tranlate back to origin

rect_points_3d_trans = []
#
for i in range(len(rect_points_3d)):
    curr = []
    curr = homogenous_matrixes.translation(new_org, old_org, rect_points_3d[i])

    rect_points_3d_trans.append(curr)
# print(len(rect_points_3d_trans))
print(rect_points_3d_trans)

# rotate all these points back to original frame


# # rotate back there 3d rectangle points
back_rot = front_rot.transpose()

final_rect_points= []
for i in range(len(rect_points_3d_trans)):
    curr = []
    curr = np.matmul(back_rot, rect_points_3d_trans[i])

    final_rect_points.append(curr)


print(final_rect_points)



original_poiints = pts
rotated_points = np.array(rot_points)
translated_points = np.array(point_rot_trans)
proj_rectangle = np.array(rect_points_3d)
proj_rectangle_trans = np.array(rect_points_3d_trans)
proj_rect_final = np.array(final_rect_points)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# origins and xyz vectors

o_1 = np.array(new_frame)
o_2 = np.array(old_frame)

#
# plotting origins
#
# ax.plot(o_1.T[0], o_1.T[1], o_1.T[2], "m")
# ax.plot(o_1.T[0], o_1.T[1], o_1.T[2], "m")


# Plot defining model points
ax.plot(original_poiints.T[0], original_poiints.T[1], original_poiints.T[2], "r")
ax.plot(rotated_points.T[0], rotated_points.T[1], rotated_points.T[2], "b")
ax.plot(translated_points.T[0], translated_points.T[1], translated_points.T[2], "y")

ax.plot(proj_rectangle.T[0], proj_rectangle.T[1], proj_rectangle.T[2], "y")
ax.plot(proj_rectangle_trans.T[0], proj_rectangle_trans.T[1], proj_rectangle_trans.T[2], "y")
ax.plot(proj_rect_final.T[0], proj_rect_final.T[1], proj_rect_final.T[2], "r")

# 12 = 2 * 6 faces are the simplices (2 simplices per square face)
# for s in hull.simplices:
#     s = np.append(s, s[0])  # Here we cycle back to the first coordinate
#     ax.plot(pts[s, 0], pts[s, 1], pts[s, 2], "r-")

plt.plot(xy_proj_1[:,0], xy_proj_1[:,1], 'o')
for simplex in hull.simplices:
    plt.plot(xy_proj_1[simplex,0], xy_proj_1[simplex,1], 'k-')

# plt.plot(xy_proj_1[hull_2d.vertices,0], xy_proj_1[hull_2d.vertices,1], 'r--', lw=2)
# plt.plot(xy_proj_1[hull_2d.vertices[0],0], xy_proj_1[hull_2d.vertices[0],1], 'ro')

# Make axis label
for i in ["x", "y", "z"]:
    eval("ax.set_{:s}label('{:s}')".format(i, i))


# plots making the rectangles

# ax.plot(final_box.T[0], final_box.T[1], final_box.T[2], "y")

plt.show()