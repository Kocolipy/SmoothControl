import numpy as np
import Constants

# Defines a centripetal Catmull-Rom spline.
class Spline(object):
    def __init__(self, P0, P1, P2, P3):
        # Convert the points to numpy so that we can do array multiplication
        P0, P1, P2, P3 = map(np.array, [P0, P1, P2, P3])

        # Calculate t0 to t4
        def tj(ti, Pi, Pj):
            alpha = 0.5
            xi, yi = Pi
            xj, yj = Pj
            return (((xj - xi) ** 2 + (yj - yi) ** 2) ** 0.5) ** alpha + ti

        # Only calculate points between P1 and P2
        t1 = tj(0, P0, P1)
        t2 = tj(t1, P1, P2)
        t3 = tj(t2, P2, P3)
        t21 = t2 - t1

        m1 = t21 * ((P1 - P0) / t1 - (P2 - P0) / t2 + (P2 - P1) / t21)
        m2 = t21 * ((P2 - P1) / t21 - (P3 - P1) / (t3 - t1) + (P3 - P2) / (t3 - t2))
        a = 2 * P1 - 2 * P2 + m1 + m2
        b = -3 * P1 + 3 * P2 - 2 * m1 - m2
        c = m1
        d = P1

        self._m1 = m1
        self._m2 = m2
        self._P0 = P0
        self._P1 = P1
        self._P2 = P2
        self._P3 = P3
        self._coefs = np.array([a, b, c, d])

    @property
    def control_nodes(self):
        return np.array([self._P0, self._P1, self._P2, self._P3])

    @property
    def coefs(self):
        return self._coefs

    @coefs.setter
    def coefs(self, array):
        self._coefs = array

    def poly(self, t):
        return self.coefs[0] * t ** 3 + self.coefs[1] * t ** 2 + self.coefs[2] * t + self.coefs[3]

    def discretise(self, points=Constants.NUM_POINTS):
        t = np.linspace(0, 1, points)
        points = [self.poly(x) for x in t]
        return points

    def pointOnSpline(self, point):
        # Return t such that spline(t) is the point closest to the input point
        A, B = point
        (ax, ay), (bx, by), (cx, cy), (dx, dy) = self._coefs

        t5 = 3 * ax * ax + 3 * ay * ay
        t4 = 5 * ax * bx + 5 * ay * by
        t3 = 4 * ax * cx + 2 * bx * bx + 4 * ay * cy + 2 * by * by
        t2 = 3 * ax * dx - 3 * ax * A + 3 * bx * cx + 3 * ay * dy - 3 * ay * B + 3 * by * cy
        t1 = 2 * bx * dx - 2 * bx * A + cx * cx + 2 * by * dy - 2 * by * B + cy * cy
        t0 = cx * dx - cx * A + cy * dy - cy * B
        roots = np.roots([t5, t4, t3, t2, t1, t0])

        # Only consider real roots
        rroots = roots[np.logical_not(np.iscomplex(roots))].real
        boundary_check = np.logical_and(rroots <= 1, rroots >= 0)
        if np.any(boundary_check):
            # If root is between 0 and 1
            return rroots[boundary_check][0]
        else:
            # root lie outside of spline, take either start or end of spline
            end = sum(self._coefs)
            start = self._coefs[3]
            if np.linalg.norm(point - end) > np.linalg.norm(point - start):
                return 0
            else:
                return 1

    def VecToSpline(self, point):
        # Return AB where A is the point and B is the point on the spline with the shortest distance to A
        t = self.pointOnSpline(point)
        return self.poly(t) - point

    def gradient(self, t):
        return 3*self.coefs[0] * t ** 2 + 2*self.coefs[1] * t + self.coefs[2]

    def secondderiv(self, t):
        return 6*self.coefs[0] * t + 2*self.coefs[1]

    def __repr__(self):
        return str(self._coefs)

    def __str__(self):
        return str(self._coefs)


