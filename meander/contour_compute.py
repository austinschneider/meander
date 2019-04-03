import numpy as np
import scipy.spatial

def connect_line_segments(lines):
    """ Create polygons from unsorted line segments """
    points = np.unique([p for pp in lines for p in pp])

    # map from points to lines segments
    l_dict = dict([(p,list()) for p in points])
    for i, (p0, p1) in enumerate(lines):
        l_dict[p0].append(i)
        l_dict[p1].append(i)

    polygons = []
    polygon = []
    while True:
        if len(l_dict) == 0:
            break

        # pop a point, and a line segment that contains the point
        p0 = l_dict.keys()[0]
        l0 = l_dict[p0][0]

        # find the other point in that line segment
        p1 = [p for p in lines[l0] if p != p0][0]

        # start the polygon
        polygon.append(p0)
        polygon.append(p1)

        # store the first line segment
        l_dict_entry = l_dict[p0]

        # delete the point that we've already handled
        del l_dict[p0]

        # next
        p0 = p1

        while True:
            # check if we're finished
            if p0 not in l_dict:
                polygons.append(polygon)
                polygon = []
                break

            # check if we've hit the end of a chain
            if len(l_dict[p0]) < 2:
                # check if we've already done both ends
                if l_dict_entry is None:
                    polygons.append(polygon)
                    polygon = []
                    break

                # try the other end of the chain
                l_dict[polygon[-1]] = l_dict_entry
                p0 = polygon[0]
                polygon = list(reversed(polygon))
                l_dict_entry = None
                continue

            # get both lines
            l0 = l_dict[p0][0]
            l1 = l_dict[p0][1]

            # get the point we don't already have in the chain
            p1 = [p for l in [lines[l0], lines[l1]] for p in l if (p != p0) and (p != polygon[-2])][0]

            # next
            polygon.append(p1)
            del l_dict[p0]
            p0 = p1

    return polygons

def _simplex_intersections(samples, levels, simplices):
    """ Compute the simplex intersections of a iso-lines for a scalar field in a 2d space """

    # to simplify the problem each point has an index
    # and each line has an index

    simplex_lines = np.zeros((len(levels),0)).tolist() # lines between the simplex edges that cross the levels
    lines = [] # the lines that compose the simplex edges
    li_to_si = dict() # maps line index -> simplex indices
    l_to_li = dict() # maps line -> line index
    for si, points in enumerate(simplices):
        points = sorted(points)
        # points and lines are ordered so lx does not contain px (px is opposite lx)
        p0, p1, p2 = points
        these_lines = [(p1,p2), (p0,p2), (p0,p1)]

        # fill the maps
        for l in these_lines:
            if l not in l_to_li:
                li_to_si[len(lines)] = []
                l_to_li[l] = len(lines)
                lines.append(l)
            li = l_to_li[l]
            li_to_si[li].append(si)

        for level_i, level in enumerate(levels):
            # which points are above / below the level
            b = np.array([samples[p] >= level for p in points]).astype(bool)
            b_sum = np.sum(b)

            # skip triangles thar are all above or all below the level
            # for triangles spanning the level, get the lines that cross the level
            if b_sum == 1:
                line_indices = np.array([l_to_li[l] for l in these_lines])[np.array([i for i in xrange(3) if i != np.argmax(b)])]
            elif b_sum == 2:
                line_indices = np.array([l_to_li[l] for l in these_lines])[np.array([i for i in xrange(3) if i != np.argmin(b)])]
            else:
                continue
            simplex_lines[level_i].append(tuple(sorted(line_indices)))

    simplex_intersections_by_level = []
    for level_i, level in enumerate(levels):
        connected_line_segments = connect_line_segments(simplex_lines[level_i])
        simplex_intersections_by_level.append([np.array([lines[li] for li in index_contour]) for index_contour in connected_line_segments])

    return simplex_intersections_by_level

def _cartesian_geodesic(p0, p1, x):
    # This function describes the geodesic between p0 and p1 in a 2d cartesian space
    # p0 and p1 are assumed to be of the form (x0, y0) and (x1, y1)
    assert(x >= 0.0 and x <= 1.0)
    p = np.asarray(p0) * (1.0-x) + np.asarray(p1) * x
    return p

def _spherical_geodesic(p0, p1, x):
    # This function describes the geodesic between p0 and p1 on a unit sphere
    # p0 and p1 are assumed to be of the form (theta0, phi0) and (theta1, phi1)
    # where theta and phi are the polar and azimuthal angles respectively measured in radians
    assert(x >= 0.0 and x <= 1.0)
    v0, v0 = [np.array((np.cos(phi)*np.sin(theta), np.sin(phi)*np.sin(theta), np.cos(theta))) for theta, phi in [p0, p1]]
    c = np.sqrt(np.sum((v1-v0)**2))
    theta = 2.0*np.arcsin(c/2.0)
    theta0 = x*theta
    theta1 = theta * (1.0 - x)
    theta2 = (np.pi - theta) / 2.0
    theta4 = np.pi - (theta1 + theta2)
    c0 = np.sin(theta0) / np.sin(theta4)

    v2 = v1 - v0
    v2 = v2 / np.sqrt(np.sum(v2**2.0))
    v3 = v0 + v2*c0
    v3 = v3 / np.sqrt(np.sum(v3**2.0))

    p = (np.arccos(v3[2]), np.arctan2(v3[1], v3[0]))
    return p

def _interpolate_simplex_intersections(sample_points, samples, levels, simplex_intersections_by_level, geodesic=_cartesian_geodesic):
    contours_by_level = []
    for level_i, level in enumerate(levels):
        connected_line_segments = simplex_intersections_by_level[level_i]
        lines_in_contours = np.unique([line for index_contour in connected_line_segments for line in index_contour], axis=0)

        line_ps = dict()

        for line in lines_in_contours:
            p0, p1 = line
            (p0, y0), (p1, y1) = sorted([(p0, samples[p0]), (p1, samples[p1])], key=lambda x: x[1])
            x = (level-y0) / (y1-y0)
            p = geodesic(sample_points[p0], sample_points[p1], x)
            line_ps[li] = p
        line_ps = np.array(line_ps)

        contours_by_level.append([np.array([line_ps[li] for li in index_contour]) for index_contour in connect_line_segments(simplex_lines)])
    return contours_by_level

def compute_contours(sample_points, samples, level):
    """ Compute contours (iso-lines of a scalar field) in a 2d cartesian space """
    # compute the delaunay triangulation so we have a set of
    # triangles to run the meandering triangles algorithm on
    tri = scipy.spatial.Delaunay(sample_points)
    simplices = tri.simplices

    # to simplify the problem each point has an index
    # and each line has an index

    simplex_lines = [] # lines between the simplex edges that cross the level
    lines = [] # the lines that compose the simplex edges
    li_to_si = dict() # maps line index -> simplex indices
    l_to_li = dict() # maps line -> line index
    for si, points in enumerate(simplices):
        points = sorted(points)
        # points and lines are ordered so lx does not contain px (px is opposite lx)
        p0, p1, p2 = points
        these_lines = [(p1,p2), (p0,p2), (p0,p1)]

        # fill the maps
        for l in these_lines:
            if l not in l_to_li:
                li_to_si[len(lines)] = []
                l_to_li[l] = len(lines)
                lines.append(l)
            li = l_to_li[l]
            li_to_si[li].append(si)

        # which points are above / below the level
        b = np.array([samples[p] >= level for p in points]).astype(bool)
        b_sum = np.sum(b)

        # skip triangles thar are all above or all below the level
        # for triangles spanning the level, get the lines that cross the level
        if b_sum == 1:
            line_indices = np.array([l_to_li[l] for l in these_lines])[np.array([i for i in xrange(3) if i != np.argmax(b)])]
        elif b_sum == 2:
            line_indices = np.array([l_to_li[l] for l in these_lines])[np.array([i for i in xrange(3) if i != np.argmin(b)])]
        else:
            continue
        simplex_lines.append(tuple(sorted(line_indices)))

    # simplex_lines specifies which simplex lines to draw the contour lines between
    # but we still need to know where on those lines to put the points
    # we can estimate where the level is by linearly interpolating between the sample points

    line_ps = np.zeros(len(lines)).tolist()
    #line_xs = np.zeros(len(lines)).tolist()

    for li in xrange(len(lines)):
        p0, p1 = lines[li]
        (p0, y0), (p1, y1) = sorted([(p0, samples[p0]), (p1, samples[p1])], key=lambda x: x[1])
        x = (level-y0) / (y1-y0)
        p = np.array(sample_points[p0]) * (1.0-x) + np.array(sample_points[p1]) * x
        #line_xs[li] = x
        line_ps[li] = p
    line_ps = np.array(line_ps)
    #line_xs = np.array(line_xs)
    #mask = np.logical_and(line_xs < 1.0, line_xs > 0.0)
    #simplex_lines = [(l0, l1) for l0, l1 in simplex_lines if mask[l0] and mask[l1]]

    return [np.array([line_ps[li] for li in index_contour]) for index_contour in connect_line_segments(simplex_lines)]
