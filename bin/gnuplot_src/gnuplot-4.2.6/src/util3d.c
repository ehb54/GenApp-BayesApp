#ifndef lint
static char *RCSid() { return RCSid("$Id: util3d.c,v 1.27.2.4 2009/01/07 22:58:27 sfeam Exp $"); }
#endif

/* GNUPLOT - util3d.c */

/*[
 * Copyright 1986 - 1993, 1998, 2004   Thomas Williams, Colin Kelley
 *
 * Permission to use, copy, and distribute this software and its
 * documentation for any purpose with or without fee is hereby granted,
 * provided that the above copyright notice appear in all copies and
 * that both that copyright notice and this permission notice appear
 * in supporting documentation.
 *
 * Permission to modify the software is granted, but not the right to
 * distribute the complete modified source code.  Modifications are to
 * be distributed as patches to the released version.  Permission to
 * distribute binaries produced by compiling modified sources is granted,
 * provided you
 *   1. distribute the corresponding source modifications from the
 *    released version in the form of a patch file along with the binaries,
 *   2. add special version identification to distinguish your version
 *    in addition to the base release version number,
 *   3. provide your name and address as the primary contact for the
 *    support of your modified version, and
 *   4. retain our contact information in regard to use of the base
 *    software.
 * Permission to distribute the released version of the source code along
 * with corresponding source modifications in the form of a patch file is
 * granted with same provisions 2 through 4 for binary distributions.
 *
 * This software is provided "as is" without express or implied warranty
 * to the extent permitted by applicable law.
]*/


/*
 * 19 September 1992  Lawrence Crowl  (crowl@cs.orst.edu)
 * Added user-specified bases for log scaling.
 *
 * 3.6 - split graph3d.c into graph3d.c (graph),
 *                            util3d.c (intersections, etc)
 *                            hidden3d.c (hidden-line removal code)
 *
 */

#include "util3d.h"

#include "axis.h"
#include "hidden3d.h"
#include "pm3d.h"
#include "term_api.h"

/* HBB 990826: all that stuff referenced from other modules is now
 * exported in graph3d.h, instead of being listed here */

/* Prototypes for local functions */
static void mat_unit __PROTO((transform_matrix mat));
static GP_INLINE void draw3d_point_unconditional __PROTO((p_vertex, struct lp_style_type *));

static void
mat_unit(transform_matrix mat)
{
    int i, j;

    for (i = 0; i < 4; i++)
	for (j = 0; j < 4; j++)
	    if (i == j)
		mat[i][j] = 1.0;
	    else
		mat[i][j] = 0.0;
}

#if 0 /* HBB 990829: unused --> commented out */
void
mat_trans(double tx, double ty, double tz, transform_matrix mat)
{
    mat_unit(mat);		/* Make it unit matrix. */
    mat[3][0] = tx;
    mat[3][1] = ty;
    mat[3][2] = tz;
}
#endif /* commented out */

void
mat_scale(double sx, double sy, double sz, transform_matrix mat)
{
    mat_unit(mat);		/* Make it unit matrix. */
    mat[0][0] = sx;
    mat[1][1] = sy;
    mat[2][2] = sz;
}

void
mat_rot_x(double teta, transform_matrix mat)
{
    double cos_teta, sin_teta;

    teta *= DEG2RAD;
    cos_teta = cos(teta);
    sin_teta = sin(teta);

    mat_unit(mat);		/* Make it unit matrix. */
    mat[1][1] = cos_teta;
    mat[1][2] = -sin_teta;
    mat[2][1] = sin_teta;
    mat[2][2] = cos_teta;
}

#if 0 /* HBB 990829: unused --> commented out */
void
mat_rot_y(double teta, transform_matrix mat)
{
    double cos_teta, sin_teta;

    teta *= DEG2RAD;
    cos_teta = cos(teta);
    sin_teta = sin(teta);

    mat_unit(mat);		/* Make it unit matrix. */
    mat[0][0] = cos_teta;
    mat[0][2] = -sin_teta;
    mat[2][0] = sin_teta;
    mat[2][2] = cos_teta;
}
#endif /* commented out */

void
mat_rot_z(double teta, transform_matrix mat)
{
    double cos_teta, sin_teta;

    teta *= DEG2RAD;
    cos_teta = cos(teta);
    sin_teta = sin(teta);

    mat_unit(mat);		/* Make it unit matrix. */
    mat[0][0] = cos_teta;
    mat[0][1] = -sin_teta;
    mat[1][0] = sin_teta;
    mat[1][1] = cos_teta;
}

/* Multiply two transform_matrix. Result can be one of two operands. */
void
mat_mult(
    transform_matrix mat_res,
    transform_matrix mat1, transform_matrix mat2)
{
    int i, j, k;
    transform_matrix mat_res_temp;

    for (i = 0; i < 4; i++)
	for (j = 0; j < 4; j++) {
	    mat_res_temp[i][j] = 0;
	    for (k = 0; k < 4; k++)
		mat_res_temp[i][j] += mat1[i][k] * mat2[k][j];
	}
    for (i = 0; i < 4; i++)
	for (j = 0; j < 4; j++)
	    mat_res[i][j] = mat_res_temp[i][j];
}

#define IN_AXIS_RANGE(val, axis)					\
	inrange((val), axis_array[axis].min, axis_array[axis].max)


/* single edge intersection algorithm */
/* Given two points, one inside and one outside the plot, return
 * the point where an edge of the plot intersects the line segment defined
 * by the two points.
 */
void
edge3d_intersect(
    struct coordinate GPHUGE *points,	/* the points array */
    int i,				/* line segment from point i-1 to point i */
    double *ex, double *ey, double *ez)	/* the point where it crosses an edge */
{
    int count;
    double ix = points[i - 1].x;
    double iy = points[i - 1].y;
    double iz = points[i - 1].z;
    double ox = points[i].x;
    double oy = points[i].y;
    double oz = points[i].z;
    double x, y, z;		/* possible intersection point */

    if (points[i].type == INRANGE) {
	/* swap points around so that ix/ix/iz are INRANGE and ox/oy/oz are OUTRANGE */
	x = ix;
	ix = ox;
	ox = x;
	y = iy;
	iy = oy;
	oy = y;
	z = iz;
	iz = oz;
	oz = z;
    }

    /* nasty degenerate cases, effectively drawing to an infinity point (?)
       cope with them here, so don't process them as a "real" OUTRANGE point

       If more than one coord is -VERYLARGE, then can't ratio the "infinities"
       so drop out by returning FALSE */

    count = 0;
    if (ox == -VERYLARGE)
	count++;
    if (oy == -VERYLARGE)
	count++;
    if (oz == -VERYLARGE)
	count++;

    /* either doesn't pass through 3D volume *or*
       can't ratio infinities to get a direction to draw line, so return the INRANGE point */
    if (count > 1) {
	*ex = ix;
	*ey = iy;
	*ez = iz;

	return;
    }
    if (count == 1) {
	*ex = ix;
	*ey = iy;
	*ez = iz;

	if (ox == -VERYLARGE) {
	    *ex = AXIS_ACTUAL_MIN(FIRST_X_AXIS);
	    return;
	}
	if (oy == -VERYLARGE) {
	    *ey = AXIS_ACTUAL_MIN(FIRST_Y_AXIS);
	    return;
	}
	/* obviously oz is -VERYLARGE and (ox != -VERYLARGE && oy != -VERYLARGE) */
	*ez = AXIS_ACTUAL_MIN(FIRST_Z_AXIS);
	return;
    }
    /*
     * Can't have case (ix == ox && iy == oy && iz == oz) as one point
     * is INRANGE and one point is OUTRANGE.
     */
    if (ix == ox) {
	if (iy == oy) {
	    /* line parallel to z axis */

	    /* assume iy in yrange, && ix in xrange */
	    *ex = ix;		/* == ox */
	    *ey = iy;		/* == oy */

	    if (inrange(AXIS_ACTUAL_MAX(FIRST_Z_AXIS), iz, oz))
		*ez = AXIS_ACTUAL_MAX(FIRST_Z_AXIS);
	    else if (inrange(AXIS_ACTUAL_MIN(FIRST_Z_AXIS), iz, oz))
		*ez = AXIS_ACTUAL_MIN(FIRST_Z_AXIS);
	    else {
		graph_error("error in edge3d_intersect");
	    }

	    return;
	}
	if (iz == oz) {
	    /* line parallel to y axis */

	    /* assume iz in zrange && ix in xrange */
	    *ex = ix;		/* == ox */
	    *ez = iz;		/* == oz */

	    if (inrange(AXIS_ACTUAL_MAX(FIRST_Y_AXIS), iy, oy))
		*ey = AXIS_ACTUAL_MAX(FIRST_Y_AXIS);
	    else if (inrange(AXIS_ACTUAL_MIN(FIRST_Y_AXIS), iy, oy))
		*ey = AXIS_ACTUAL_MIN(FIRST_Y_AXIS);
	    else {
		graph_error("error in edge3d_intersect");
	    }

	    return;
	}

	/* nasty 2D slanted line in a yz plane */


#define INTERSECT_PLANE(cut, axis, eff, eff_axis, res_x, res_y, res_z)	\
	do {								\
	    if (inrange(cut, i##axis, o##axis)				\
		&& cut != i##axis					\
		&& cut != o##axis) {					\
		eff = (cut - i##axis)					\
		    * ((o##eff - i##eff) / (o##axis - i##axis))		\
		    + i##eff;						\
		if (IN_AXIS_RANGE(eff, eff_axis)) {			\
		    *ex = res_x;					\
		    *ey = res_y;					\
		    *ez = res_z;					\
		    return;						\
		}							\
	    }								\
	} while (0)

	INTERSECT_PLANE(AXIS_ACTUAL_MIN(FIRST_Y_AXIS), y, z, FIRST_Z_AXIS,
			ix, AXIS_ACTUAL_MIN(FIRST_Y_AXIS), z);
	INTERSECT_PLANE(AXIS_ACTUAL_MAX(FIRST_Y_AXIS), y, z, FIRST_Z_AXIS,
			ix, AXIS_ACTUAL_MAX(FIRST_Y_AXIS), z);
	INTERSECT_PLANE(AXIS_ACTUAL_MIN(FIRST_Z_AXIS), z, y, FIRST_Y_AXIS,
			ix, y, AXIS_ACTUAL_MIN(FIRST_Z_AXIS));
	INTERSECT_PLANE(AXIS_ACTUAL_MAX(FIRST_Z_AXIS), z, y, FIRST_Y_AXIS,
			ix, y, AXIS_ACTUAL_MAX(FIRST_Z_AXIS));
    } /* if (ix == ox) */

    if (iy == oy) {
	/* already checked case (ix == ox && iy == oy) */
	if (oz == iz) {
	    /* line parallel to x axis */

	    /* assume inrange(iz) && inrange(iy) */
	    *ey = iy;		/* == oy */
	    *ez = iz;		/* == oz */

	    if (inrange(AXIS_ACTUAL_MAX(FIRST_X_AXIS), ix, ox))
		*ex = AXIS_ACTUAL_MAX(FIRST_X_AXIS);
	    else if (inrange(AXIS_ACTUAL_MIN(FIRST_X_AXIS), ix, ox))
		*ex = AXIS_ACTUAL_MIN(FIRST_X_AXIS);
	    else {
		graph_error("error in edge3d_intersect");
	    }

	    return;
	}
	/* nasty 2D slanted line in an xz plane */

	INTERSECT_PLANE(AXIS_ACTUAL_MIN(FIRST_X_AXIS), x, z, FIRST_Z_AXIS,
			AXIS_ACTUAL_MIN(FIRST_X_AXIS), iy, z);
	INTERSECT_PLANE(AXIS_ACTUAL_MAX(FIRST_X_AXIS), x, z, FIRST_Z_AXIS,
			AXIS_ACTUAL_MAX(FIRST_X_AXIS), iy, z);
	INTERSECT_PLANE(AXIS_ACTUAL_MIN(FIRST_Z_AXIS), z, x, FIRST_X_AXIS,
			x, iy, AXIS_ACTUAL_MIN(FIRST_Z_AXIS));
	INTERSECT_PLANE(AXIS_ACTUAL_MAX(FIRST_Z_AXIS), z, x, FIRST_X_AXIS,
			x, iy, AXIS_ACTUAL_MAX(FIRST_Z_AXIS));
    } /* if(iy==oy) */

    if (iz == oz) {
	/* already checked cases (ix == ox && iz == oz) and (iy == oy
	   && iz == oz) */

	/* 2D slanted line in an xy plane */

	/* assume inrange(oz) */

	INTERSECT_PLANE(AXIS_ACTUAL_MIN(FIRST_X_AXIS), x, y, FIRST_Y_AXIS,
			AXIS_ACTUAL_MIN(FIRST_X_AXIS), y, iz);
	INTERSECT_PLANE(AXIS_ACTUAL_MAX(FIRST_X_AXIS), x, y, FIRST_Y_AXIS,
			AXIS_ACTUAL_MAX(FIRST_X_AXIS), y, iz);
	INTERSECT_PLANE(AXIS_ACTUAL_MIN(FIRST_Y_AXIS), y, x, FIRST_X_AXIS,
			x, AXIS_ACTUAL_MIN(FIRST_Y_AXIS), iz);
	INTERSECT_PLANE(AXIS_ACTUAL_MAX(FIRST_Y_AXIS), y, x, FIRST_X_AXIS,
			x, AXIS_ACTUAL_MAX(FIRST_Y_AXIS), iz);
    } /* if(iz==oz) */
#undef INTERSECT_PLANE

    /* really nasty general slanted 3D case */

#define INTERSECT_DIAG(cut, axis, eff, eff_axis, eff2, eff2_axis,	\
		       res_x, res_y, res_z)				\
	do {								\
	    if (inrange(cut, i##axis, o##axis)				\
		&& cut != i##axis					\
		&& cut != o##axis) {					\
		eff = (cut - i##axis)					\
		    * ((o##eff - i##eff) / (o##axis - i##axis))		\
		    + i##eff;						\
		eff2 = (cut - i##axis)					\
		    * ((o##eff2 - i##eff2) / (o##axis - i##axis))	\
		    + i##eff2;						\
		if (IN_AXIS_RANGE(eff, eff_axis)			\
		    && IN_AXIS_RANGE(eff2, eff2_axis)) {		\
		    *ex = res_x;					\
		    *ey = res_y;					\
		    *ez = res_z;					\
		    return;						\
		}							\
	    }								\
	} while (0)

    INTERSECT_DIAG(AXIS_ACTUAL_MIN(FIRST_X_AXIS), x,
		   y, FIRST_Y_AXIS, z, FIRST_Z_AXIS,
		   AXIS_ACTUAL_MIN(FIRST_X_AXIS), y, z);
    INTERSECT_DIAG(AXIS_ACTUAL_MAX(FIRST_X_AXIS), x,
		   y, FIRST_Y_AXIS, z, FIRST_Z_AXIS,
		   AXIS_ACTUAL_MAX(FIRST_X_AXIS), y, z);

    INTERSECT_DIAG(AXIS_ACTUAL_MIN(FIRST_Y_AXIS), y,
		   x, FIRST_X_AXIS, z, FIRST_Z_AXIS,
		   x, AXIS_ACTUAL_MIN(FIRST_Y_AXIS), z);
    INTERSECT_DIAG(AXIS_ACTUAL_MAX(FIRST_Y_AXIS), y,
		   x, FIRST_X_AXIS, z, FIRST_Z_AXIS,
		   x, AXIS_ACTUAL_MAX(FIRST_Y_AXIS), z);

    INTERSECT_DIAG(AXIS_ACTUAL_MIN(FIRST_Z_AXIS), z,
		   x, FIRST_X_AXIS, y, FIRST_Y_AXIS,
		   x, y, AXIS_ACTUAL_MIN(FIRST_Z_AXIS));
    INTERSECT_DIAG(AXIS_ACTUAL_MAX(FIRST_Z_AXIS), z,
		   x, FIRST_X_AXIS, y, FIRST_Y_AXIS,
		   x, y, AXIS_ACTUAL_MAX(FIRST_Z_AXIS));

#undef INTERSECT_DIAG

    /* If we reach here, the inrange point is on the edge, and
     * the line segment from the outrange point does not cross any
     * other edges to get there. In this case, we return the inrange
     * point as the 'edge' intersection point. This will basically draw
     * line.
     */
    *ex = ix;
    *ey = iy;
    *ez = iz;
    return;
}

/* double edge intersection algorithm */
/* Given two points, both outside the plot, return
 * the points where an edge of the plot intersects the line segment defined
 * by the two points. There may be zero, one, two, or an infinite number
 * of intersection points. (One means an intersection at a corner, infinite
 * means overlaying the edge itself). We return FALSE when there is nothing
 * to draw (zero intersections), and TRUE when there is something to
 * draw (the one-point case is a degenerate of the two-point case and we do
 * not distinguish it - we draw it anyway).
 */
TBOOLEAN			/* any intersection? */
two_edge3d_intersect(
    struct coordinate GPHUGE *points,	/* the points array */
    int i,				/* line segment from point i-1 to point i */
    double *lx, double *ly, double *lz)	/* lx[2], ly[2], lz[2]: points where it crosses edges */
{
    int count;
    /* global axis_array[FIRST_{X,Y,Z}_AXIS].{min,max} */
    double ix = points[i - 1].x;
    double iy = points[i - 1].y;
    double iz = points[i - 1].z;
    double ox = points[i].x;
    double oy = points[i].y;
    double oz = points[i].z;
    double t[6];
    double swap;
    double x, y, z;		/* possible intersection point */
    double t_min, t_max;

    /* nasty degenerate cases, effectively drawing to an infinity point (?)
       cope with them here, so don't process them as a "real" OUTRANGE point

       If more than one coord is -VERYLARGE, then can't ratio the "infinities"
       so drop out by returning FALSE */

    count = 0;
    if (ix == -VERYLARGE)
	count++;
    if (ox == -VERYLARGE)
	count++;
    if (iy == -VERYLARGE)
	count++;
    if (oy == -VERYLARGE)
	count++;
    if (iz == -VERYLARGE)
	count++;
    if (oz == -VERYLARGE)
	count++;

    /* either doesn't pass through 3D volume *or*
       can't ratio infinities to get a direction to draw line, so simply return(FALSE) */
    if (count > 1) {
	return (FALSE);
    }

    if (ox == -VERYLARGE || ix == -VERYLARGE) {
	if (ix == -VERYLARGE) {
	    /* swap points so ix/iy/iz don't have a -VERYLARGE component */
	    x = ix;
	    ix = ox;
	    ox = x;
	    y = iy;
	    iy = oy;
	    oy = y;
	    z = iz;
	    iz = oz;
	    oz = z;
	}
	/* check actually passes through the 3D graph volume */

	if (ix > axis_array[FIRST_X_AXIS].max
	    && IN_AXIS_RANGE(iy, FIRST_Y_AXIS)
	    && IN_AXIS_RANGE(iz, FIRST_Z_AXIS)) {
	    lx[0] = axis_array[FIRST_X_AXIS].min;
	    ly[0] = iy;
	    lz[0] = iz;

	    lx[1] = axis_array[FIRST_X_AXIS].max;
	    ly[1] = iy;
	    lz[1] = iz;

	    return (TRUE);
	} else {
	    return (FALSE);
	}
    }
    if (oy == -VERYLARGE || iy == -VERYLARGE) {
	if (iy == -VERYLARGE) {
	    /* swap points so ix/iy/iz don't have a -VERYLARGE component */
	    x = ix;
	    ix = ox;
	    ox = x;
	    y = iy;
	    iy = oy;
	    oy = y;
	    z = iz;
	    iz = oz;
	    oz = z;
	}
	/* check actually passes through the 3D graph volume */
	if (iy > axis_array[FIRST_Y_AXIS].max
	    && IN_AXIS_RANGE(ix, FIRST_X_AXIS)
	    && IN_AXIS_RANGE(iz, FIRST_Z_AXIS)) {
	    lx[0] = ix;
	    ly[0] = axis_array[FIRST_Y_AXIS].min;
	    lz[0] = iz;

	    lx[1] = ix;
	    ly[1] = axis_array[FIRST_Y_AXIS].max;
	    lz[1] = iz;

	    return (TRUE);
	} else {
	    return (FALSE);
	}
    }
    if (oz == -VERYLARGE || iz == -VERYLARGE) {
	if (iz == -VERYLARGE) {
	    /* swap points so ix/iy/iz don't have a -VERYLARGE component */
	    x = ix;
	    ix = ox;
	    ox = x;
	    y = iy;
	    iy = oy;
	    oy = y;
	    z = iz;
	    iz = oz;
	    oz = z;
	}
	/* check actually passes through the 3D graph volume */
	if (iz > axis_array[FIRST_Z_AXIS].max
	    && IN_AXIS_RANGE(ix, FIRST_X_AXIS)
	    && IN_AXIS_RANGE(iy, FIRST_Y_AXIS)) {
	    lx[0] = ix;
	    ly[0] = iy;
	    lz[0] = axis_array[FIRST_Z_AXIS].min;

	    lx[1] = ix;
	    ly[1] = iy;
	    lz[1] = axis_array[FIRST_Z_AXIS].max;

	    return (TRUE);
	} else {
	    return (FALSE);
	}
    }
    /*
     * Quick outcode tests on the 3d graph volume
     */

    /* test z coord first --- most surface OUTRANGE points generated
     * between axis_array[FIRST_Z_AXIS].min and baseplane (i.e. when
     * ticslevel is non-zero)
     */
    if (GPMAX(iz, oz) < axis_array[FIRST_Z_AXIS].min
	|| GPMIN(iz, oz) > axis_array[FIRST_Z_AXIS].max)
	return (FALSE);

    if (GPMAX(ix, ox) < axis_array[FIRST_X_AXIS].min
	|| GPMIN(ix, ox) > axis_array[FIRST_X_AXIS].max)
	return (FALSE);

    if (GPMAX(iy, oy) < axis_array[FIRST_Y_AXIS].min
	|| GPMIN(iy, oy) > axis_array[FIRST_Y_AXIS].max)
	return (FALSE);

    /* Special horizontal/vertical, etc. cases are checked and
     * remaining slant lines are checked separately.
     *
     * The slant line intersections are solved using the parametric
     * form of the equation for a line, since if we test x/y/z min/max
     * planes explicitly then e.g. a line passing through a corner
     * point (x_min,y_min,z_min) actually intersects all 3 planes and
     * hence further tests would be required to anticipate this and
     * similar situations. */

    /* Can have case (ix == ox && iy == oy && iz == oz) as both points
     * OUTRANGE */
    if (ix == ox && iy == oy && iz == oz) {
	/* but as only define single outrange point, can't intersect
	 * 3D graph volume */
	return (FALSE);
    }

    if (ix == ox) {
	if (iy == oy) {
	    /* line parallel to z axis */

	    /* x and y coords must be in range, and line must span
	     * both FIRST_Z_AXIS->min and ->max.
	     * 
	     * note that spanning FIRST_Z_AXIS->min implies spanning
	     * ->max as both points OUTRANGE */

	    if (!IN_AXIS_RANGE(ix, FIRST_X_AXIS)
		|| !IN_AXIS_RANGE(iy, FIRST_Y_AXIS)) {
		return (FALSE);
	    }
	    if (inrange(axis_array[FIRST_Z_AXIS].min, iz, oz)) {
		lx[0] = ix;
		ly[0] = iy;
		lz[0] = axis_array[FIRST_Z_AXIS].min;

		lx[1] = ix;
		ly[1] = iy;
		lz[1] = axis_array[FIRST_Z_AXIS].max;

		return (TRUE);
	    } else
		return (FALSE);
	}
	if (iz == oz) {
	    /* line parallel to y axis */
	    if (!IN_AXIS_RANGE(ix, FIRST_X_AXIS)
		|| !IN_AXIS_RANGE(iz, FIRST_Z_AXIS)) {
		return (FALSE);
	    }
	    if (inrange(axis_array[FIRST_Y_AXIS].min, iy, oy)) {
		lx[0] = ix;
		ly[0] = axis_array[FIRST_Y_AXIS].min;
		lz[0] = iz;

		lx[1] = ix;
		ly[1] = axis_array[FIRST_Y_AXIS].max;
		lz[1] = iz;

		return (TRUE);
	    } else
		return (FALSE);
	}


	/* nasty 2D slanted line in a yz plane */
	if (!IN_AXIS_RANGE(ox, FIRST_X_AXIS))
	    return (FALSE);

	t[0] = (axis_array[FIRST_Y_AXIS].min - iy) / (oy - iy);
	t[1] = (axis_array[FIRST_Y_AXIS].max - iy) / (oy - iy);

	if (t[0] > t[1]) {
	    swap = t[0];
	    t[0] = t[1];
	    t[1] = swap;
	}
	t[2] = (axis_array[FIRST_Z_AXIS].min - iz) / (oz - iz);
	t[3] = (axis_array[FIRST_Z_AXIS].max - iz) / (oz - iz);

	if (t[2] > t[3]) {
	    swap = t[2];
	    t[2] = t[3];
	    t[3] = swap;
	}
	t_min = GPMAX(GPMAX(t[0], t[2]), 0.0);
	t_max = GPMIN(GPMIN(t[1], t[3]), 1.0);

	if (t_min > t_max)
	    return (FALSE);

	lx[0] = ix;
	ly[0] = iy + t_min * (oy - iy);
	lz[0] = iz + t_min * (oz - iz);

	lx[1] = ix;
	ly[1] = iy + t_max * (oy - iy);
	lz[1] = iz + t_max * (oz - iz);

	/* Can only have 0 or 2 intersection points -- only need test
	 * one coord */
	if (IN_AXIS_RANGE(ly[0], FIRST_Y_AXIS)
	    && IN_AXIS_RANGE(lz[0], FIRST_Z_AXIS)) {
	    return (TRUE);
	}
	return (FALSE);
    }

    if (iy == oy) {
	/* already checked case (ix == ox && iy == oy) */
	if (oz == iz) {
	    /* line parallel to x axis */
	    if (!IN_AXIS_RANGE(iy, FIRST_Y_AXIS)
		|| !IN_AXIS_RANGE(iz, FIRST_Z_AXIS)) {
		return (FALSE);
	    }
	    if (inrange(axis_array[FIRST_X_AXIS].min, ix, ox)) {
		lx[0] = axis_array[FIRST_X_AXIS].min;
		ly[0] = iy;
		lz[0] = iz;

		lx[1] = axis_array[FIRST_X_AXIS].max;
		ly[1] = iy;
		lz[1] = iz;

		return (TRUE);
	    } else
		return (FALSE);
	}
	/* nasty 2D slanted line in an xz plane */

	if (!IN_AXIS_RANGE(oy, FIRST_Y_AXIS))
	    return (FALSE);

	t[0] = (axis_array[FIRST_X_AXIS].min - ix) / (ox - ix);
	t[1] = (axis_array[FIRST_X_AXIS].max - ix) / (ox - ix);

	if (t[0] > t[1]) {
	    swap = t[0];
	    t[0] = t[1];
	    t[1] = swap;
	}
	t[2] = (axis_array[FIRST_Z_AXIS].min - iz) / (oz - iz);
	t[3] = (axis_array[FIRST_Z_AXIS].max - iz) / (oz - iz);

	if (t[2] > t[3]) {
	    swap = t[2];
	    t[2] = t[3];
	    t[3] = swap;
	}
	t_min = GPMAX(GPMAX(t[0], t[2]), 0.0);
	t_max = GPMIN(GPMIN(t[1], t[3]), 1.0);

	if (t_min > t_max)
	    return (FALSE);

	lx[0] = ix + t_min * (ox - ix);
	ly[0] = iy;
	lz[0] = iz + t_min * (oz - iz);

	lx[1] = ix + t_max * (ox - ix);
	ly[1] = iy;
	lz[1] = iz + t_max * (oz - iz);

	/*
	 * Can only have 0 or 2 intersection points -- only need test one coord
	 */
	if (IN_AXIS_RANGE(lx[0], FIRST_X_AXIS)
	    && IN_AXIS_RANGE(lz[0], FIRST_Z_AXIS)) {
	    return (TRUE);
	}
	return (FALSE);
    }
    if (iz == oz) {
	/* already checked cases (ix == ox && iz == oz) and (iy == oy
	   && iz == oz) */

	/* nasty 2D slanted line in an xy plane */

	if (!IN_AXIS_RANGE(oz, FIRST_Z_AXIS))
	    return (FALSE);

	t[0] = (axis_array[FIRST_X_AXIS].min - ix) / (ox - ix);
	t[1] = (axis_array[FIRST_X_AXIS].max - ix) / (ox - ix);

	if (t[0] > t[1]) {
	    swap = t[0];
	    t[0] = t[1];
	    t[1] = swap;
	}
	t[2] = (axis_array[FIRST_Y_AXIS].min - iy) / (oy - iy);
	t[3] = (axis_array[FIRST_Y_AXIS].max - iy) / (oy - iy);

	if (t[2] > t[3]) {
	    swap = t[2];
	    t[2] = t[3];
	    t[3] = swap;
	}
	t_min = GPMAX(GPMAX(t[0], t[2]), 0.0);
	t_max = GPMIN(GPMIN(t[1], t[3]), 1.0);

	if (t_min > t_max)
	    return (FALSE);

	lx[0] = ix + t_min * (ox - ix);
	ly[0] = iy + t_min * (oy - iy);
	lz[0] = iz;

	lx[1] = ix + t_max * (ox - ix);
	ly[1] = iy + t_max * (oy - iy);
	lz[1] = iz;

	/*
	 * Can only have 0 or 2 intersection points -- only need test one coord
	 */
	if (IN_AXIS_RANGE(lx[0], FIRST_X_AXIS) 
	    && IN_AXIS_RANGE(ly[0], FIRST_Y_AXIS)) {
	    return (TRUE);
	}
	return (FALSE);
    }
    /* really nasty general slanted 3D case */

    /*
       Solve parametric equation

       (ix, iy, iz) + t (diff_x, diff_y, diff_z)

       where 0.0 <= t <= 1.0 and

       diff_x = (ox - ix);
       diff_y = (oy - iy);
       diff_z = (oz - iz);
     */

    t[0] = (axis_array[FIRST_X_AXIS].min - ix) / (ox - ix);
    t[1] = (axis_array[FIRST_X_AXIS].max - ix) / (ox - ix);

    if (t[0] > t[1]) {
	swap = t[0];
	t[0] = t[1];
	t[1] = swap;
    }
    t[2] = (axis_array[FIRST_Y_AXIS].min - iy) / (oy - iy);
    t[3] = (axis_array[FIRST_Y_AXIS].max - iy) / (oy - iy);

    if (t[2] > t[3]) {
	swap = t[2];
	t[2] = t[3];
	t[3] = swap;
    }
    t[4] = (iz == oz) ? 0.0 : (axis_array[FIRST_Z_AXIS].min - iz) / (oz - iz);
    t[5] = (iz == oz) ? 1.0 : (axis_array[FIRST_Z_AXIS].max - iz) / (oz - iz);

    if (t[4] > t[5]) {
	swap = t[4];
	t[4] = t[5];
	t[5] = swap;
    }
    t_min = GPMAX(GPMAX(t[0], t[2]), GPMAX(t[4], 0.0));
    t_max = GPMIN(GPMIN(t[1], t[3]), GPMIN(t[5], 1.0));

    if (t_min > t_max)
	return (FALSE);

    lx[0] = ix + t_min * (ox - ix);
    ly[0] = iy + t_min * (oy - iy);
    lz[0] = iz + t_min * (oz - iz);

    lx[1] = ix + t_max * (ox - ix);
    ly[1] = iy + t_max * (oy - iy);
    lz[1] = iz + t_max * (oz - iz);

    /*
     * Can only have 0 or 2 intersection points -- only need test one coord
     */
    if (IN_AXIS_RANGE(lx[0], FIRST_X_AXIS) 
	&& IN_AXIS_RANGE(ly[0], FIRST_Y_AXIS)
	&& IN_AXIS_RANGE(lz[0], FIRST_Z_AXIS)) {
	return (TRUE);
    }
    return (FALSE);
}

/* Performs transformation from 'user coordinates' to a normalized
 * vector in 'graph coordinates' (-1..1 in all three directions).  */
void
map3d_xyz(
    double x, double y, double z,		/* user coordinates */
    p_vertex out)
{
    int i, j;
    double V[4], Res[4];	/* Homogeneous coords. vectors. */

    /* Normalize object space to -1..1 */
    V[0] = map_x3d(x);
    V[1] = map_y3d(y);
    V[2] = map_z3d(z);
    V[3] = 1.0;

    /* Res[] = V[] * trans_mat[][] (uses row-vectors) */
    for (i = 0; i < 4; i++) {
	Res[i] = trans_mat[3][i];		/* V[3] is 1. anyway */
	for (j = 0; j < 3; j++)
	    Res[i] += V[j] * trans_mat[j][i];
    }

    if (Res[3] == 0)
	Res[3] = 1.0e-5;

    out->x = Res[0] / Res[3];
    out->y = Res[1] / Res[3];
    out->z = Res[2] / Res[3];
    /* store z for later color calculation */
    out->real_z = z;
#ifdef EAM_DATASTRINGS
    out->label = NULL;
#endif
}


/* DJS (20 Aug 2004):  A more precise double version of map3d_xy() is
 * is required for the image routine.  The original intention was to
 * reuse the double version of the routine to generate the unsigned
 * int versin of the routine.  However, that caused rounding problems
 * such that PostScript versions of all the demos didn't come out
 * quite exactly the same.
 *
 * The define switch below will allow either code reuse or code
 * replication.  My advice is to study the rounding problem and
 * decide if the code reuse rounding is just as well as the code
 * replication approach.  If so, go the code reuse route and toss
 * the replicated code.
 */
#define REPLICATE_CODE_FOR_BACKWARD_COMPATIBLE_ROUNDING 1

#if REPLICATE_CODE_FOR_BACKWARD_COMPATIBLE_ROUNDING

/* Function to map from user 3D space to normalized 'camera' view
 * space, and from there directly to terminal coordinates */
void
map3d_xy(
    double x, double y, double z,
    unsigned int *xt, unsigned int *yt)
{
    int i, j;
    double v[4], res[4],	/* Homogeneous coords. vectors. */
     w = trans_mat[3][3];

    v[0] = map_x3d(x);		/* Normalize object space to -1..1 */
    v[1] = map_y3d(y);
    v[2] = map_z3d(z);
    v[3] = 1.0;

    for (i = 0; i < 2; i++) {	/* Dont use the third axes (z). */
	res[i] = trans_mat[3][i];	/* Initiate it with the weight factor */
	for (j = 0; j < 3; j++)
	    res[i] += v[j] * trans_mat[j][i];
    }

    for (i = 0; i < 3; i++)
	w += v[i] * trans_mat[i][3];
    if (w == 0)
	w = 1e-5;

    if (lmargin.scalex == screen || rmargin.scalex == screen)
	*xt = res[0] * xscaler/w + xmiddle;
    else
	*xt = (unsigned int) ((res[0] * xscaler / w) + xmiddle);

    if (tmargin.scalex == screen || bmargin.scalex == screen)
	*yt = res[1] * yscaler/w + ymiddle;
    else
	*yt = (unsigned int) ((res[1] * yscaler / w) + ymiddle);
}

/* Function to map from user 3D space to normalized 'camera' view
 * space, and from there directly to terminal coordinates */
void
map3d_xy_double(
    double x, double y, double z,
    double *xt, double *yt)
{
    int i, j;
    double v[4], res[4],	/* Homogeneous coords. vectors. */
     w = trans_mat[3][3];

    v[0] = map_x3d(x);		/* Normalize object space to -1..1 */
    v[1] = map_y3d(y);
    v[2] = map_z3d(z);
    v[3] = 1.0;

    for (i = 0; i < 2; i++) {	/* Dont use the third axes (z). */
	res[i] = trans_mat[3][i];	/* Initiate it with the weight factor */
	for (j = 0; j < 3; j++)
	    res[i] += v[j] * trans_mat[j][i];
    }

    for (i = 0; i < 3; i++)
	w += v[i] * trans_mat[i][3];
    if (w == 0)
	w = 1e-5;

    if (lmargin.scalex == screen || rmargin.scalex == screen)
	*xt = res[0] * xscaler + xmiddle;
    else
	*xt = (res[0] * xscaler / w) + xmiddle;

    if (tmargin.scalex == screen || bmargin.scalex == screen)
	*yt = res[1] * yscaler + ymiddle;
    else
	*yt = (res[1] * yscaler / w) + ymiddle;
}

#else /* REPLICATE_CODE_FOR_BACKWARD_COMPATIBLE_ROUNDING */

/* Function to map from user 3D space to normalized 'camera' view
 * space, and from there directly to terminal coordinates */
void
map3d_xy(
    double x, double y, double z,
    unsigned int *xt, unsigned int *yt)
{
#ifdef WITH_IMAGE
    double xtd, ytd;
    map3d_xy_double(x, y, z, &xtd, &ytd);
    *xt = (unsigned int) xtd;
    *yt = (unsigned int) ytd;
}

void
map3d_xy_double(
    double x, double y, double z,
    double *xt, double *yt)
{
#endif
    int i, j;
    double v[4], res[4],	/* Homogeneous coords. vectors. */
     w = trans_mat[3][3];

    v[0] = map_x3d(x);		/* Normalize object space to -1..1 */
    v[1] = map_y3d(y);
    v[2] = map_z3d(z);
    v[3] = 1.0;

    for (i = 0; i < 2; i++) {	/* Dont use the third axes (z). */
	res[i] = trans_mat[3][i];	/* Initiate it with the weight factor */
	for (j = 0; j < 3; j++)
	    res[i] += v[j] * trans_mat[j][i];
    }

    for (i = 0; i < 3; i++)
	w += v[i] * trans_mat[i][3];
    if (w == 0)
	w = 1e-5;

#ifdef WITH_IMAGE
    *xt = ((res[0] * xscaler / w) + xmiddle);
    *yt = ((res[1] * yscaler / w) + ymiddle);
#else
    *xt = (unsigned int) ((res[0] * xscaler / w) + xmiddle);
    *yt = (unsigned int) ((res[1] * yscaler / w) + ymiddle);
#endif
}

#endif /* REPLICATE_CODE_FOR_BACKWARD_COMPATIBLE_ROUNDING */


/* HBB 20020313: New routine, broken out of draw3d_point, to be used
 * to output a single point without any checks for hidden3d */
static GP_INLINE void
draw3d_point_unconditional(p_vertex v, struct lp_style_type *lp)
{
    unsigned int x, y;

    TERMCOORD(v, x, y);
    term_apply_lp_properties(lp);
    /* HBB 20010822: implemented "linetype palette" for points, too */
    if (lp->use_palette) {
	set_color(cb2gray( z2cb(v->real_z) ));
    }
    if (!clip_point(x, y))
	(term->point) (x, y, lp->p_type);
}

/* Moved this upward, to make optional inlining in draw3d_line easier
 * for compilers */
/* HBB 20021128: removed GP_INLINE qualifier to avoid MSVC++ silliness */
void
draw3d_line_unconditional(
    p_vertex v1, p_vertex v2,
    struct lp_style_type *lp,
    int linetype)
{
    unsigned int x1, y1, x2, y2;
    struct lp_style_type ls = *lp;

    /* HBB 20020312: v2 can be NULL, if this call is coming from
    draw_line_hidden. --> redirect to point drawing routine */
    if (! v2) {
	draw3d_point_unconditional(v1, lp);
	return;
    }

    TERMCOORD(v1, x1, y1);
    TERMCOORD(v2, x2, y2);

    /* User-specified line styles */
    if (prefer_line_styles && linetype >= 0)
	lp_use_properties(&ls, linetype+1, FALSE);

    /* The usual case of auto-generated line types */
    else
	ls.l_type = linetype;

    /* Color by Z value */
    if (ls.pm3d_color.type == TC_Z)
	    ls.pm3d_color.value = (v1->real_z + v2->real_z) * 0.5;

    term_apply_lp_properties(&ls);
    draw_clip_line(x1,y1,x2,y2);
}

void
draw3d_line (p_vertex v1, p_vertex v2, struct lp_style_type *lp)
{
#ifndef LITE
    /* hidden3d routine can't work if no surface was drawn at all */
    if (hidden3d && draw_surface) {
	draw_line_hidden(v1, v2, lp);
	return;
    }
#endif

    draw3d_line_unconditional(v1, v2, lp, lp->l_type);

}

/* HBB 20000621: new routine, to allow for hiding point symbols behind
 * the surface */
void
draw3d_point(p_vertex v, struct lp_style_type *lp)
{
#ifndef LITE
    /* hidden3d routine can't work if no surface was drawn at all */
    if (hidden3d && draw_surface) {
	/* Draw vertex as a zero-length edge */
	draw_line_hidden(v, NULL, lp);
	return;
    }
#endif

    draw3d_point_unconditional(v, lp);
}

/* HBB NEW 20031218: tools for drawing polylines in 3D with a semantic
 * like term->move() and term->vector() */

/* Previous points 3D position */
static vertex polyline3d_previous_vertex;

void
polyline3d_start(p_vertex v1)
{
    unsigned int x1, y1;

    polyline3d_previous_vertex = *v1;
#ifndef LITE
    if (hidden3d && draw_surface)
	return;
#endif /* LITE */

    TERMCOORD(v1, x1, y1);
    /* HBB FIXME 20031219: no clipping!? */
    term->move(x1, y1);
}

void
polyline3d_next(p_vertex v2, struct lp_style_type *lp)
{
    unsigned int x2, y2;

    /* Copied from draw3d_line(): */
#ifndef LITE
    /* FIXME HBB 20031218: hidden3d mode will still create isolated
     * edges! */
    if (hidden3d && draw_surface) {
	draw_line_hidden(&polyline3d_previous_vertex, v2, lp);
	polyline3d_previous_vertex = *v2;
	return;
    }
#endif

    /* Copied from draw3d_line_unconditional: */
    /* If use_palette is active, polylines can't be used -->
     * revert back to old method */
    if (lp->use_palette) {
	draw3d_line_unconditional(&polyline3d_previous_vertex, v2,
				  lp, lp->l_type);
	polyline3d_previous_vertex = *v2;
	return;

    }

    TERMCOORD(v2, x2, y2);
    /* FIXME HBB 20031219: no clipping?! */
    term->vector(x2, y2);

    polyline3d_previous_vertex = *v2;
}
