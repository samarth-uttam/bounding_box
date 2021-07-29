
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull
import pandas as pd
import math
import sympy as sym
import delete_duplicate_edges


def homogenous_rotation_matrix(old_fr, new_fr):
    # old_unit = [list((old_frame[0] / np.linalg.norm(old_frame[0]))), list(old_frame[1] / np.linalg.norm(old_frame[1])),
    #             list(old_frame[2] / np.linalg.norm(old_frame[2]))]
    # new_unit = [list((new_frame[0] / np.linalg.norm(new_frame[0]))), list(new_frame[1] / np.linalg.norm(new_frame[1])),
    #             list(new_frame[2] / np.linalg.norm(new_frame[2]))]

    old_x, old_y, old_z = old_fr
    new_x, new_y, new_z = new_fr

    r11 = np.dot(new_x, old_x)
    r21 = np.dot(new_x, old_y)
    r31 = np.dot(new_x, old_z)
    r12 = np.dot(new_y, old_x)
    r22 = np.dot(new_y, old_y)
    r32 = np.dot(new_y, old_z)
    r13 = np.dot(new_z, old_x)
    r23 = np.dot(new_z, old_y)
    r33 = np.dot(new_z, old_z)

    rot = np.array([[r11, r12, r13],
                    [r21, r22, r23],
                    [r31, r32, r33],
                    ])
    # a = abs(np.dot(old_frame[0], old_frame[1]))
    # b = abs(np.dot(old_frame[1], old_frame[2]))
    # c = abs(np.dot(old_frame[2], old_frame[0]))
    #
    # d = abs(np.dot(new_frame[0], new_frame[1]))
    # e = abs(np.dot(new_frame[1], new_frame[2]))
    # f = abs(np.dot(new_frame[2], new_frame[1]))
    # print(a, b, c, d, e, f)
    # diff = 0.00001
    # if a < diff or b < diff or c < diff or d < diff or e < diff or f < diff:
    #     return 'vectors not orthogonal'
    # else:
    return rot


def homogenous_translation(old_org, new_org, p):
    final_point = []
    del_x = new_org[0] - old_org[0]
    del_y = new_org[1] - old_org[1]
    del_z = new_org[2] - old_org[2]

    new_x = p[0] + del_x
    new_y = p[1] + del_y
    new_z = p[2] + del_z

    final_point = [new_x, new_y, new_z]

    return final_point



xy_to_uv_file =  "C:/Users/Samarth/OneDrive - HKUST Connect/Thesis Project/IEDA/march/8/bunny.xls"
b = pd.read_excel(xy_to_uv_file)
uv_conv = pd.DataFrame(b)
uv_conv = uv_conv.drop(columns="u")
uv_conv = uv_conv.drop(columns="v")
uv_conv = uv_conv.drop(columns="i")
uv_conv = uv_conv.drop(columns="j")
pts = array_from_dataframe = uv_conv.to_numpy()


hull = ConvexHull(pts)
t = 0

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Plot defining corner points # t[0] is x of all points , t[1] is y of all points , t[3] is z of all pioints
ax.plot(pts.T[0], pts.T[1], pts.T[2], "ko")
# Make axis label
for i in ["x", "y", "z"]:
    eval("ax.set_{:s}label('{:s}')".format(i, i))

for s in hull.simplices:
    s = np.append(s, s[0])  # Here we cycle back to the first coordinate
    ax.plot(pts[s, 0], pts[s, 1], pts[s, 2], "r-")

# plt.show()

# print(len(hull.simplices))
orig_edge_pair = []

#  getting the edge pairs (taking 3 points on simplices and making 3 edges for each)

for i in range(len(hull.simplices)):
    p1 = hull.simplices[i][0]
    p2 = hull.simplices[i][1]
    p3 = hull.simplices[i][2]
    # print(p1, p2, p3)
    edge_pair_1 = [p1, p2]
    edge_pair_2 = [p2, p3]
    edge_pair_3 = [p3, p1]
    orig_edge_pair.append(edge_pair_1)
    orig_edge_pair.append(edge_pair_2)
    orig_edge_pair.append(edge_pair_3)

# print('edge_pair')
# print(edge_pair)
# print(len(edge_pair))

print(len(orig_edge_pair))
edge_pair = delete_duplicate_edges.duplicate_edge_remover(orig_edge_pair)

print(len(edge_pair))
# getting the circle points on xy plane at increment of 1 degree each at a unit distance from origin
print(edge_pair)

circle_points = []
theta = math.pi / 180
# print(theta)
for i in range(180):
    current_xy = []

    x = math.cos(math.radians(i))
    y = math.sin(math.radians(i))
    z = 0
    current_xy = [x, y, z]
    # print(math.radians(i))
    # print(current_xy)
    circle_points.append(current_xy)
# print(circle_points)


pair = []
for edge_1 in range(len(edge_pair)):
    # print(edge_pair[edge_1])
    e1_p1 = hull.points[edge_pair[edge_1][0]]
    e1_p2 = hull.points[edge_pair[edge_1][1]]
    e1p1 = e1_p1
    e1p2 = e1_p2
    # print(e1_p1, e1_p2)
    v1 = e1_p2 - e1_p1
    v1_unit = v1 /np.linalg.norm(v1)
    # print('vectors ')
    # print(v1)
    # print(v1_unit)

    # plane is of the form ax + by + cz + d  = 0
    coff_a , coff_b, coff_c = v1
    coff_d = -1 * (coff_a * e1_p1[0] + coff_b * e1_p1[1] + coff_c * e1_p1[2])
    plane_eq = [coff_a, coff_b, coff_c, coff_d]
    # print('plane')
    # print(plane_eq)
    if coff_c ==0:

        phi = -1 * (coff_b/coff_a)
        p_0 = [e1_p1[0] + phi, e1_p1[1] + 1, e1_p1[2]]
    else:
        delta = -1 * (coff_a/coff_c)
        p_0 = [ e1_p1[0] +1,  e1_p1[1] ,  e1_p1[2] + delta ]

    original_point = e1_p1
    a = original_point[0]*coff_a + original_point[1]*coff_b + original_point[2]*coff_c + coff_d
    # print(original_point)
    # print(a)

    new_point = p_0
    b = new_point[0]*coff_a + new_point[1]*coff_b + new_point[2]*coff_c + coff_d
    # print(new_point)
    # print(b)

    x_new = p_0 - original_point
    x_new_unit = x_new / np.linalg.norm(x_new)
    z_new = [coff_a, coff_b, coff_c]
    z_new_unit = z_new / np.linalg.norm(z_new)
    y_new = np.cross(z_new_unit,x_new_unit)
    y_new_unit = y_new / np.linalg.norm(y_new)

    x_old_unit = [1,0,0]
    y_old_unit = [0,1,0]
    z_old_unit = [0,0,1]

    old_frame = [x_old_unit, y_old_unit, z_old_unit]
    new_frame = [x_new_unit, y_new_unit, z_new_unit]
    old_origin = [0,0,0]
    new_origin = original_point

    # print(old_frame, new_frame, old_origin, new_origin)

    #  converting xy plane 3d points to current plane points with new origin and new frame

    curr_rotation_matrix = homogenous_rotation_matrix(old_frame, new_frame)

    transformed_circle_points = []
    rot_c_points = []
    rot_trans_c_points = []
    for i in range(len(circle_points)):
        curr  = []
        curr = np.matmul(curr_rotation_matrix,circle_points[i])
        rot_c_points.append(curr)

    # print(rot_c_points)

    for i in range(len(rot_c_points)):
        curr = []
        curr  = homogenous_translation(old_origin, new_origin, rot_c_points[i])
        rot_trans_c_points.append(curr)



    # checking the distance of the points form the given plane, they must all be zero
    #
    # for i in range(len(rot_trans_c_points)):
    #     d = coff_a*rot_trans_c_points[i][0] +  coff_b*rot_trans_c_points[i][1] +coff_c*rot_trans_c_points[i][2] + coff_d
    #     print(d)

    # make a plane passing through point from circle projection and 2 other points of the original edge


    for edge_2 in range(len(edge_pair)):
        if edge_pair[edge_1][0] == edge_pair[edge_2][0] or edge_pair[edge_1][0] == edge_pair[edge_2][1] or edge_pair[edge_1][1] == edge_pair[edge_2][0] or edge_pair[edge_1][1] == edge_pair[edge_2][1]:
            some_value = 0
        else:
            e2p1 = hull.points[edge_pair[edge_2][0]]
            e2p2 = hull.points[edge_pair[edge_2][1]]
            p2_count = edge_pair[edge_2][1]
            #
            plane_1 = []
            plane_2 = []
            distance_p2 = []
            dist_e_p = []
            for i3 in range(len(rot_trans_c_points)):
                e1p3 = rot_trans_c_points[i3]

                # print(e1p1, e1p2, e1p3)

                v1 = e1p3 - e1p1
                v2 = e1p2 - e1p1
                cp = np.cross(v1, v2)
                c_a , c_b, c_c = cp
                c_d = (-1) * np.dot(cp, e1p3)

                # print(p1, p2, p3)
                # print(c_a, c_b, c_c, c_d)
                # face equations in the form of ax +by + cz + d = 0 (abcd and coffecients )

                curr_plane = [c_a, c_b, c_c, c_d]
                # print(curr_plane)
                plane_1.append(curr_plane)

                new_c_d =((-1) * np.dot(cp, e2p1))

                new_plane = [c_a, c_b, c_c, new_c_d]
                # print(new_plane)
                plane_2.append(new_plane)

                point_2 = e2p2
                distance = abs((new_plane[0] * point_2[0]) + (new_plane[1] * point_2[1])  + (new_plane[2] * point_2[2]) + new_plane[3] ) / ( (new_plane[0])**2 + (new_plane[1])**2 + (new_plane[2])**2)**0.5
                # print(distance)
                # getting the distance of the point e2p2 from all the planes
                distance_p2.append(distance)

            # print(distance_p2)
            min_d = min(distance_p2)
            # print(min_d)
            angle = distance_p2.index(min(distance_p2))

            front_plane = plane_1[angle]
            back_plane = plane_2[angle]

            p1 = []
            p2 = []
            for n in range(len(hull.points)):

                if n == p2_count:

                    some_value = 0
                else:

                    p_1 = (front_plane[0] * hull.points[n][0]) + (front_plane[1] * hull.points[n][1]) + (front_plane[2] * hull.points[n][2]) + (front_plane[3])
                    if p_1 >= 0:
                        p1_sign = True
                    else:
                        p1_sign = False
                    p1.append(p1_sign)

                    p_2 = (back_plane[0] * hull.points[n][0]) + (back_plane[1] * hull.points[n][1]) + (back_plane[2] * hull.points[n][2]) + (back_plane[3])
                    if p_2 >= 0:
                        p2_sign = True
                    else:
                        p2_sign = False
                    p2.append(p2_sign)

            result1 = all(element == p1[0] for element in p1)
            result2 = all(element == p2[0] for element in p2)

            # print(result1, result2)

            if result2 == True and result1 == True:
                curr = []
                plane_dist = abs(plane_1[angle][3] - plane_2[angle][3])/ (plane_1[angle][0]**2 + plane_1[angle][1]**2 + plane_1[angle][2]**2)**0.5
                curr = [edge_1, edge_2 ,rot_trans_c_points[angle], distance]
                print(curr)
                pair.append(curr)
                print('pair found')
            # print('one second edged one ')
            # print(edge1, edge2 )
    print('one edge donee.........................................................................................................................................')
    print(edge_1)
print(pair)

plane_dist  = []
for i in range(len(curr)):
    dist = curr[i][3]
    plane_dist.append(dist)

plane_min = min(plane_dist)
final_pair = plane_dist.index(plane_min)

print(final_pair)