#ifndef lint
static char *RCSid() { return RCSid("$Id: hidden3d.c,v 1.56.2.3 2008/06/02 19:40:10 sfeam Exp $"); }
#endif

/* GNUPLOT - hidden3d.c */

/*[
 * Copyright 1986 - 1993, 1998, 1999, 2004   Thomas Williams, Colin Kelley
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
 * 1999 Hans-Bernhard Broeker (Broeker@physik.rwth-aachen.de)
 *
 * Major rewrite, affecting just about everything
 *
 */

#include "hidden3d.h"
#include "color.h"
#include "pm3d.h"
#include "alloc.h"
#include "axis.h"
#include "command.h"
#include "dynarray.h"
#include "graph3d.h"
#include "tables.h"
#include "term_api.h"
#include "util.h"
#include "util3d.h"


/*************************/
/* Configuration section */
/*************************/

/* If this module is compiled with HIDDEN3D_GRIDBOX = 1 defined, it
 * will store the information about {x|y}{min|max} in an other
 * (additional!) form: a bit mask, with each bit representing one
 * horizontal or vertical strip of the screen. The bits for strips a
 * polygon spans are set to one. This allows to test for xy overlap
 * of an edge with a polygon simply by comparing bit patterns.  */
#ifndef HIDDEN3D_GRIDBOX
#define HIDDEN3D_GRIDBOX 0
#endif

/* HBB 19991204: new code started to finally implement a spatially
 * ordered data structure to store the polygons in. This is meant to
 * speed up the HLR process. Before, the hot spot of hidden3d was the
 * loop in in_front, where by far most of the polygons are rejected by
 * the first test, already. The idea is to _not_ to loop over all
 * those polygons far away from the edge under consideration, in the
 * first place. Instead, store the polygons in an xy grid of lists,
 * so we can select a sample of these lists to test a given edge
 * against. */
#ifndef HIDDEN3D_QUADTREE
#define HIDDEN3D_QUADTREE 0
#endif
#if HIDDEN3D_QUADTREE && HIDDEN3D_GRIDBOX
# warning HIDDEN3D_QUADTREE & HIDDEN3D_GRIDBOX do not work together, sensibly!
#endif

/* If you don't want the color-distinction between the
 * 'top' and 'bottom' sides of the surface, like I do, then just compile
 * with -DBACKSIDE_LINETYPE_OFFSET = 0. */
#ifndef BACKSIDE_LINETYPE_OFFSET
# define BACKSIDE_LINETYPE_OFFSET 1
#endif

/* This #define lets you choose if the diagonals that
 * divide each original quadrangle in two triangles will be drawn
 * visible or not: do draw them, define it to be 7L, otherwise let be
 * 3L */
#ifndef TRIANGLE_LINESDRAWN_PATTERN
# define TRIANGLE_LINESDRAWN_PATTERN 3L
#endif

/* Handle out-of-range or undefined points. Compares the maximum
 * marking (0=inrange, 1=outrange, 2=undefined) of the coordinates of
 * a vertex to the value #defined here. If not less, the vertex is
 * rejected, and all edges that hit it, as well. NB: if this is set to
 * anything above 1, gnuplot may crash with a floating point exception
 * in hidden3d. You get what you asked for ... */
#ifndef HANDLE_UNDEFINED_POINTS
# define HANDLE_UNDEFINED_POINTS 1
#endif
/* Symbolic value for 'do not handle Undefined Points specially' */
#define UNHANDLED (UNDEFINED+1)

/* If both subtriangles of a quad were cancelled, try if using the
 * other diagonal is better. This only makes a difference if exactly
 * one vertex of the quad is unusable, and that one is on the 'usual'
 * tried diagonal. In such a case, the border of the hole in the
 * surface will be less rough than with the previous method, as the
 * border follows the undefined region as close as it can. */
#ifndef SHOW_ALTERNATIVE_DIAGONAL
# define SHOW_ALTERNATIVE_DIAGONAL 1
#endif

/* If the two triangles in a quad are both drawn, and they show
 * different sides to the user (the quad is 'bent over'), then it's
 * preferrable to force the diagonal being visible to avoid other
 * parts of the scene being obscured by a line the user can't
 * see. This avoids unnecessary user surprises. */
#ifndef HANDLE_BENTOVER_QUADRANGLES
# define HANDLE_BENTOVER_QUADRANGLES 1
#endif

/* The actual configuration is stored in these variables, modifiable
 * at runtime through 'set hidden3d' options */
static int hiddenBacksideLinetypeOffset = BACKSIDE_LINETYPE_OFFSET;
static long hiddenTriangleLinesdrawnPattern = TRIANGLE_LINESDRAWN_PATTERN;
static int hiddenHandleUndefinedPoints = HANDLE_UNDEFINED_POINTS;
static int hiddenShowAlternativeDiagonal = SHOW_ALTERNATIVE_DIAGONAL;
static int hiddenHandleBentoverQuadrangles = HANDLE_BENTOVER_QUADRANGLES;


/**************************************************************/
/**************************************************************
 * The 'real' code begins, here.                              *
 *                                                            *
 * first: types and global variables                          *
 **************************************************************/
/**************************************************************/

/* precision of calculations in normalized space. Coordinates closer to
 * each other than an absolute difference of EPSILON are considered
 * equal, by some of the routines in this module. */
#define EPSILON 1e-5

/* The code used to die messily if the scale parameters got over-large.
 * Prevent this from happening due to mousing by locking out the mouse
 * response. */
TBOOLEAN disable_mouse_z = FALSE;

/* Some inexact operations: == , > , >=, sign() */
#define EQ(X,Y)  (fabs( (X) - (Y) ) < EPSILON)	/* X == Y */
#define GR(X,Y)  ((X) >  (Y) + EPSILON)		/* X >  Y */
#define GE(X,Y)  ((X) >= (Y) - EPSILON)		/* X >= Y */
#define SIGN(X)  ( ((X)<-EPSILON) ? -1: ((X)>EPSILON) )

/* A plane equation, stored as a four-element vector. The equation
 * itself is: x*p[0]+y*p[1]+z*p[2]+1*p[3]=0 */
typedef coordval t_plane[4];

/* One edge of the mesh. The edges are (currently) organized into a
 * linked list as a method of traversing them back-to-front. */
typedef struct edge {
    long v1, v2;		/* the vertices at either end */
    int style;			/* linetype index */
    struct lp_style_type *lp;	/* line/point style attributes */
    long next;			/* index of next edge in z-sorted list */
} edge;
typedef edge GPHUGE *p_edge;


/* One polygon (actually: always a triangle) of the surface
 * mesh(es). */
#define POLY_NVERT 3
typedef struct polygon {
    long vertex[POLY_NVERT];    /* The vertices (indices on vlist) */
    /* min/max in all three directions */
    coordval xmin, xmax, ymin, ymax, zmin, zmax;
    t_plane plane;		/* the plane coefficients */
    TBOOLEAN frontfacing;	/* is polygon facing front- or backwards? */
#if ! HIDDEN3D_QUADTREE
    long next;			/* index of next polygon in z-sorted list */
#endif
#if HIDDEN3D_GRIDBOX
    unsigned long xbits;	/* x coverage mask of bounding box */
    unsigned long ybits;	/* y coverage mask of bounding box */
#endif
} polygon;
typedef polygon GPHUGE *p_polygon;

#if HIDDEN3D_GRIDBOX
# define UINT_BITS (CHAR_BIT * sizeof(unsigned int))
# define COORD_TO_BITMASK(x,shift)							\
  (~0U << (unsigned int) ((((x) / surface_scale) + 1.0) / 2.0 * UINT_BITS + (shift)))
# define CALC_BITRANGE(range_min, range_max)				 \
  ((~COORD_TO_BITMASK((range_max), 1)) & COORD_TO_BITMASK(range_min, 0))
#endif

/* Enumeration of possible types of line, for use with the
 * store_edge() function. Influences the position in the grid the
 * second vertex will be put to, relative to the one that is passed
 * in, as another argument to that function. edir_none is for
 * single-pixel 'blind' edges, which exist only to facilitate output
 * of 'points' style splots.
 *
 * Directions are interpreted in a pseudo-geographical coordinate
 * system of the data grid: within the isoline, we count from left to
 * right (west to east), and the isolines themselves are counted from
 * top to bottom, described as north and south. */
typedef enum edge_direction {
    edir_west, edir_north,
    edir_NW, edir_NE,
    edir_impulse, edir_point,
    edir_vector
} edge_direction;

/* direction into which the polygon is facing (the corner with the
 * right angle, inside the mesh, that is). The reference identifiying
 * the whole cell is always the lower right, i.e. southeast one. */
typedef enum polygon_direction {
    pdir_NE, pdir_SE, pdir_SW, pdir_NW
} polygon_direction;

/* Three dynamical arrays that describe what we have to plot: */
static dynarray vertices, edges, polygons;

/* convenience #defines to make the generic vector useable as typed arrays */
#define vlist ((p_vertex) vertices.v)
#define plist ((p_polygon) polygons.v)
#define elist ((p_edge) edges.v)

static long pfirst;		/* first polygon in zsorted chain*/
static long efirst;		/* first edges in zsorted chain */

#if HIDDEN3D_QUADTREE
/* HBB 20000716: spatially oriented hierarchical data structure to
 * store polygons in. For now, it's a simple xy grid of z-sorted
 * lists. A single polygon can appear in several lists, if it spans
 * cell borders */
typedef struct qtreelist {
    long p;			/* the polygon */
    long next;			/* next element in this chain */
} qtreelist;
typedef qtreelist GPHUGE *p_qtreelist;

/* the number of cells in x and y direction: */
# ifndef QUADTREE_GRANULARITY
#  define QUADTREE_GRANULARITY 10
# endif
/* indices of the heads of all the cells' chains: */
static long quadtree[QUADTREE_GRANULARITY][QUADTREE_GRANULARITY];

/* and a routine to calculate the cells' position in that array: */
static int
coord_to_treecell(coordval x)
{
    int index;
    index = ((((x) / surface_scale) + 1.0) / 2.0) * QUADTREE_GRANULARITY;
    if (index >= QUADTREE_GRANULARITY)
	index = QUADTREE_GRANULARITY - 1;
    else if (index < 0)
	index = 0;

    return index;
}

/* the dynarray to actually store all that stuff in: */
static dynarray qtree;
#define qlist ((p_qtreelist) qtree.v)
#endif /* HIDDEN3D_QUADTREE*/

/* Prototypes for internal functions of this module. */
static long int store_vertex __PROTO((struct coordinate GPHUGE *point,
				      lp_style_type *lp_style, TBOOLEAN color_from_column));
static long int make_edge __PROTO((long int vnum1, long int vnum2,
				   struct lp_style_type *lp,
				   int style, int next));
static long int store_edge __PROTO((long int vnum1, edge_direction direction,
				    long int crvlen, struct lp_style_type *lp,
				    int style));
static GP_INLINE double eval_plane_equation __PROTO((t_plane p, p_vertex v));
static long int store_polygon __PROTO((long int vnum1,
				       polygon_direction direction,
				       long int crvlen));
static void color_edges __PROTO((long int new_edge, long int old_edge,
				 long int new_poly, long int old_poly,
				 int style_above, int style_below));
static void build_networks __PROTO((struct surface_points * plots,
				    int pcount));
int compare_edges_by_zmin __PROTO((SORTFUNC_ARGS p1, SORTFUNC_ARGS p2));
int compare_polys_by_zmax __PROTO((SORTFUNC_ARGS p1, SORTFUNC_ARGS p2));
static void sort_edges_by_z __PROTO((void));
static void sort_polys_by_z __PROTO((void));
static TBOOLEAN get_plane __PROTO((p_polygon p, t_plane plane));
static long split_line_at_ratio __PROTO((long int vnum1, long int vnum2, double w));
static GP_INLINE double area2D __PROTO((p_vertex v1, p_vertex v2,
					p_vertex v3));
static void draw_vertex __PROTO((p_vertex v));
static GP_INLINE void draw_edge __PROTO((p_edge e, p_vertex v1, p_vertex v2));
static GP_INLINE void handle_edge_fragment __PROTO((long int edgenum,
						    long int vnum1,
						    long int vnum2,
						    long int firstpoly));
static int in_front __PROTO((long int edgenum,
			     long int vnum1, long int vnum2,
			     long int *firstpoly));


/* Set the options for hidden3d. To be called from set.c, when the
 * user has begun a command with 'set hidden3d', to parse the rest of
 * that command */
void
set_hidden3doptions()
{
    struct value t;
    int tmp;

    while (!END_OF_COMMAND) {
	switch (lookup_table(&set_hidden3d_tbl[0], c_token)) {
	case S_HI_DEFAULTS:
	    /* reset all parameters to defaults */
	    reset_hidden3doptions();
	    c_token++;
	    if (!END_OF_COMMAND)
		int_error(c_token,
			  "No further options allowed after 'defaults'");
	    return;
	    break;
	case S_HI_OFFSET:
	    c_token++;
	    hiddenBacksideLinetypeOffset = (int) real(const_express(&t));
	    c_token--;
	    break;
	case S_HI_NOOFFSET:
	    hiddenBacksideLinetypeOffset = 0;
	    break;
	case S_HI_TRIANGLEPATTERN:
	    c_token++;
	    hiddenTriangleLinesdrawnPattern = (int) real(const_express(&t));
	    c_token--;
	    break;
	case S_HI_UNDEFINED:
	    c_token++;
	    tmp = (int) real(const_express(&t));
	    if (tmp <= 0 || tmp > UNHANDLED)
		tmp = UNHANDLED;
	    hiddenHandleUndefinedPoints = tmp;
	    c_token--;
	    break;
	case S_HI_NOUNDEFINED:
	    hiddenHandleUndefinedPoints = UNHANDLED;
	    break;
	case S_HI_ALTDIAGONAL:
	    hiddenShowAlternativeDiagonal = 1;
	    break;
	case S_HI_NOALTDIAGONAL:
	    hiddenShowAlternativeDiagonal = 0;
	    break;
	case S_HI_BENTOVER:
	    hiddenHandleBentoverQuadrangles = 1;
	    break;
	case S_HI_NOBENTOVER:
	    hiddenHandleBentoverQuadrangles = 0;
	    break;
	case S_HI_INVALID:
	    int_error(c_token, "No such option to hidden3d (or wrong order)");
	default:
	    break;
	}
	c_token++;
    }
}

void
show_hidden3doptions()
{
    fprintf(stderr,"\
\t  Back side of surfaces has linestyle offset of %d\n\
\t  Bit-Mask of Lines to draw in each triangle is %ld\n\
\t  %d: ",
	    hiddenBacksideLinetypeOffset, hiddenTriangleLinesdrawnPattern,
	    hiddenHandleUndefinedPoints);

    switch (hiddenHandleUndefinedPoints) {
    case OUTRANGE:
	fputs("Outranged and undefined datapoints are omitted from the surface.\n",
	      stderr);
	break;
    case UNDEFINED:
	fputs("Only undefined datapoints are omitted from the surface.\n",
	      stderr);
	break;
    case UNHANDLED:
	fputs("Will not check for undefined datapoints (may cause crashes).\n",
	      stderr);
	break;
    default:
	fputs("Value stored for undefined datapoint handling is illegal!!!\n",
	      stderr);
	break;
    }

    fprintf(stderr,"\
\t  Will %suse other diagonal if it gives a less jaggy outline\n\
\t  Will %sdraw diagonal visibly if quadrangle is 'bent over'\n",
	    hiddenShowAlternativeDiagonal ? "" : "not ",
	    hiddenHandleBentoverQuadrangles ? "" : "not ");
}

/* Implements proper 'save'ing of the new hidden3d options... */
void
save_hidden3doptions(FILE *fp)
{
    if (!hidden3d) {
	fputs("unset hidden3d\n", fp);
	return;
    }
    fprintf(fp, "set hidden3d offset %d trianglepattern %ld undefined %d %saltdiagonal %sbentover\n",
	    hiddenBacksideLinetypeOffset,
	    hiddenTriangleLinesdrawnPattern,
	    hiddenHandleUndefinedPoints,
	    hiddenShowAlternativeDiagonal ? "" : "no",
	    hiddenHandleBentoverQuadrangles ? "" : "no");
}

/* Initialize the necessary steps for hidden line removal and
   initialize global variables. */
void
init_hidden_line_removal()
{
    /* Check for some necessary conditions to be set elsewhere: */
    /* HandleUndefinedPoints mechanism depends on these: */
    assert(OUTRANGE == 1);
    assert(UNDEFINED == 2);

    /* Re-mapping of this value makes the test easier in the critical
     * section */
    if (hiddenHandleUndefinedPoints < OUTRANGE)
	hiddenHandleUndefinedPoints = UNHANDLED;

    init_dynarray(&vertices, sizeof(vertex), 100, 100);
    init_dynarray(&edges, sizeof(edge), 100, 100);
    init_dynarray(&polygons, sizeof(polygon), 100, 100);
#if HIDDEN3D_QUADTREE
    init_dynarray(&qtree, sizeof(qtreelist), 100, 100);
#endif

}

/* Reset the hidden line data to a fresh start. */
void
reset_hidden_line_removal()
{
    vertices.end = 0;
    edges.end = 0;
    polygons.end = 0;
#if HIDDEN3D_QUADTREE
    qtree.end = 0;
#endif
}


/* Terminates the hidden line removal process.                  */
/* Free any memory allocated by init_hidden_line_removal above. */
void
term_hidden_line_removal()
{
    free_dynarray(&polygons);
    free_dynarray(&edges);
    free_dynarray(&vertices);
#if HIDDEN3D_QUADTREE
    free_dynarray(&qtree);
#endif
}


#if 0 /* UNUSED ! */
/* Do we see the top or bottom of the polygon, or is it 'on edge'? */
#define GET_SIDE(vlst,csign)						\
do {									\
    double ctmp =							\
	vlist[vlst[0]].x * (vlist[vlst[1]].y - vlist[vlst[2]].y) +	\
	vlist[vlst[1]].x * (vlist[vlst[2]].y - vlist[vlst[0]].y) +	\
	vlist[vlst[2]].x * (vlist[vlst[0]].y - vlist[vlst[1]].y);	\
    csign = SIGN (ctmp);						\
} while (0)
#endif /* UNUSED */

static long int
store_vertex (
    struct coordinate GPHUGE * point,
    lp_style_type *lp_style,
    TBOOLEAN color_from_column)
{
    p_vertex thisvert = nextfrom_dynarray(&vertices);

    thisvert->lp_style = lp_style;
    if ((int) point->type >= hiddenHandleUndefinedPoints) {
	FLAG_VERTEX_AS_UNDEFINED(*thisvert);
	return (-1);
    }
    map3d_xyz(point->x, point->y, point->z, thisvert);
    if (color_from_column) {
	thisvert->real_z = point->CRD_COLOR;
	thisvert->lp_style->pm3d_color.lt = LT_COLORFROMCOLUMN;
    } else
	thisvert->real_z = point->z;

#ifdef HIDDEN3D_VAR_PTSIZE
    /* Store pointer back to original point */
    /* Needed to support variable pointsize */
    thisvert->original = point;
#endif
	
    return (thisvert - vlist);
}

/* A part of store_edge that does the actual storing. Used by
 * in_front(), as well, so I separated it out. */
static long int
make_edge(
    long vnum1, long vnum2,
    struct lp_style_type *lp,
    int style,
    int next)
{
    p_edge thisedge = nextfrom_dynarray(&edges);
    p_vertex v1 = vlist + vnum1;
    p_vertex v2 = vlist + vnum2;

    /* ensure z ordering inside each edge */
    if (v1->z >= v2->z) {
	thisedge->v1 = vnum1;
	thisedge->v2 = vnum2;
    } else {
	thisedge->v1 = vnum2;
	thisedge->v2 = vnum1;
    }

    thisedge->style = style;
    thisedge->lp = lp;
    thisedge->next = next;

    return thisedge - elist;
}

/* store the edge from vnum1 to vnum2 into the edge list. Ensure that
 * the vertex with higher z is stored in v1, to ease sorting by zmax */
static long int
store_edge(
    long int vnum1,
    edge_direction direction,
    long int crvlen,
    struct lp_style_type *lp,
    int style)
{
    p_vertex v1 = vlist + vnum1;
    p_vertex v2 = NULL;		/* just in case: initialize... */
    long int vnum2;
    unsigned int drawbits = (0x1 << direction);

    switch (direction) {
    case edir_vector:
	v2 = v1 + 1;
	drawbits = 0;
	break;
    case edir_west:
	v2 = v1 - 1;
	break;
    case edir_north:
	v2 = v1 - crvlen;
	break;
    case edir_NW:
	v2 = v1 - crvlen - 1;
	break;
    case edir_NE:
	v2 = v1 - crvlen;
	v1 -= 1;
	drawbits >>= 1;		/* altDiag is handled like normal NW one */
	break;
    case edir_impulse:
	v2 = v1 - 1;
	drawbits = 0;		/* don't care about the triangle pattern */
	break;
    case edir_point:
	v2 = v1;
	drawbits = 0;		/* nothing to draw, but disable check */
	break;
    }

    vnum2 = v2 - vlist;

    if (VERTEX_IS_UNDEFINED(*v1)
	|| VERTEX_IS_UNDEFINED(*v2)) {
	return -2;
    }

    if (drawbits &&		/* no bits set: 'blind' edge --> no test! */
	! (hiddenTriangleLinesdrawnPattern & drawbits)
	)
	style = -3;

    return make_edge(vnum1, vnum2, lp, style, -1);
}


/* Calculate the normal equation coefficients of the plane of polygon
 * 'p'. Uses is the 'signed projected area' method. Its benefit is
 * that it doesn't rely on only three of the vertices of 'p', as the
 * naive cross product method does. */
static TBOOLEAN
get_plane(p_polygon poly, t_plane plane)
{
    int i;
    p_vertex v1, v2;
    double x, y, z, s;
    TBOOLEAN frontfacing=TRUE;

    /* calculate the signed areas of the polygon projected onto the
     * planes x=0, y=0 and z=0, respectively. The three areas form
     * the components of the plane's normal vector: */
    v1 = vlist + poly->vertex[POLY_NVERT - 1];
    v2 = vlist + poly->vertex[0];
    plane[0] = (v1->y - v2->y) * (v1->z + v2->z);
    plane[1] = (v1->z - v2->z) * (v1->x + v2->x);
    plane[2] = (v1->x - v2->x) * (v1->y + v2->y);
    for (i = 1; i < POLY_NVERT; i++) {
	v1 = v2;
	v2 = vlist + poly->vertex[i];
	plane[0] += (v1->y - v2->y) * (v1->z + v2->z);
	plane[1] += (v1->z - v2->z) * (v1->x + v2->x);
	plane[2] += (v1->x - v2->x) * (v1->y + v2->y);
    }

    /* Normalize the resulting normal vector */
    s = sqrt(plane[0] * plane[0] + plane[1] * plane[1] + plane[2] * plane[2]);

    if (GE(0.0, s)) {
	/* The normal vanishes, i.e. the polygon is degenerate. We build
	 * another vector that is orthogonal to the line of the polygon */
	v1 = vlist + poly->vertex[0];
	for (i = 1; i < POLY_NVERT; i++) {
	    v2 = vlist + poly->vertex[i];
	    if (!V_EQUAL(v1, v2))
		break;
	}

	/* build (x,y,z) that should be linear-independant from <v1, v2> */
	x = v1->x;
	y = v1->y;
	z = v1->z;
	if (EQ(y, v2->y))
	    y += 1.0;
	else
	    x += 1.0;

	/* Re-do the signed area computations */
	plane[0] = v1->y * (v2->z - z) + v2->y * (z - v1->z) + y * (v1->z - v2->z);
	plane[1] = v1->z * (v2->x - x) + v2->z * (x - v1->x) + z * (v1->x - v2->x);
	plane[2] = v1->x * (v2->y - y) + v2->x * (y - v1->y) + x * (v1->y - v2->y);
	s = sqrt(plane[0] * plane[0] + plane[1] * plane[1] + plane[2] * plane[2]);
    }

    /* ensure that normalized c is > 0 */
    if (plane[2] < 0.0) {
	s *= -1.0;
	frontfacing = FALSE;
    }

    plane[0] /= s;
    plane[1] /= s;
    plane[2] /= s;

    /* Now we have the normalized normal vector, insert one of the
     * vertices into the equation to get 'd'. For an even better result,
     * an average over all the vertices might be used */
    plane[3] = -plane[0] * v1->x - plane[1] * v1->y - plane[2] * v1->z;

    return frontfacing;
}


/* Evaluate the plane equation represented a four-vector for the given
 * vector. For points in the plane, this should result in values ==0.
 * < 0 is 'away' from the polygon, > 0 is infront of it */
static GP_INLINE double
eval_plane_equation(t_plane p, p_vertex v)
{
    return (p[0]*v->x + p[1]*v->y + p[2]*v->z + p[3]);
}


/* Build the data structure for this polygon. The return value is the
 * index of the newly generated polygon. This is memorized for access
 * to polygons in the previous isoline, from the next-following
 * one. */
static long int
store_polygon(long vnum1, polygon_direction direction, long crvlen)
{
    long int v[POLY_NVERT];
    p_vertex v1, v2, v3;
    p_polygon p;

    switch (direction) {
    case pdir_NE:
	v[0] = vnum1;
	v[2] = vnum1 - crvlen;
	v[1] = v[2] - 1;
	break;
    case pdir_SW:
	/* triangle points southwest, here */
	v[0] = vnum1;
	v[1] = vnum1 - 1;
	v[2] = v[1] - crvlen;
	break;
    case pdir_SE:
	/* alt-diagonal, case 1: southeast triangle: */
	v[0] = vnum1;
	v[2] = vnum1 - crvlen;
	v[1] = vnum1 - 1;
	break;
    case pdir_NW:
	v[2] = vnum1 - crvlen;
	v[0] = vnum1 - 1;
	v[1] = v[0] - crvlen;
	break;
    }

    v1 = vlist + v[0];
    v2 = vlist + v[1];
    v3 = vlist + v[2];

    if (VERTEX_IS_UNDEFINED(*v1)
	|| VERTEX_IS_UNDEFINED(*v2)
	|| VERTEX_IS_UNDEFINED(*v3)
	)
	return (-2);

    /* Check if polygon is degenerate */
    if (V_EQUAL(v1,v2) || V_EQUAL(v2,v3) || V_EQUAL(v3,v1))
	return (-2);

    /* All else OK, fill in the polygon: */

    p = nextfrom_dynarray(&polygons);

    memcpy (p->vertex, v, sizeof(v));
#if ! HIDDEN3D_QUADTREE
    p->next = -1;
#endif

    /* Some helper macros for repeated code blocks: */

    /* Gets Minimum 'var' value of polygon 'poly' into variable
     * 'min. C is one of x, y, or z: */
#define GET_MIN(poly, var, min)			\
    do {					\
	int i;					\
	long *v = poly->vertex;			\
						\
	min = vlist[*v++].var;			\
	for (i = 1; i< POLY_NVERT; i++, v++)	\
	    if (vlist[*v].var < min)		\
		min = vlist[*v].var;		\
        if (min < -surface_scale) disable_mouse_z = TRUE;	\
    } while (0)

    /* Gets Maximum 'var' value of polygon 'poly', as with GET_MIN */
#define GET_MAX(poly, var, max)			\
    do {					\
	int i;					\
	long *v = poly->vertex;			\
						\
	max = vlist[*v++].var;			\
	for (i = 1; i< POLY_NVERT; i++, v++)	\
	    if (vlist[*v].var > max)		\
		max = vlist[*v].var;		\
        if (max > surface_scale) disable_mouse_z = TRUE;	\
    } while (0)

    GET_MIN(p, x, p->xmin);
    GET_MIN(p, y, p->ymin);
    GET_MIN(p, z, p->zmin);
    GET_MAX(p, x, p->xmax);
    GET_MAX(p, y, p->ymax);
    GET_MAX(p, z, p->zmax);
#undef GET_MIN
#undef GET_MAX

#if HIDDEN3D_GRIDBOX
    p->xbits = CALC_BITRANGE(p->xmin, p->xmax);
    p->ybits = CALC_BITRANGE(p->ymin, p->ymax);
#endif

    p->frontfacing = get_plane(p, p->plane);

    return (p - plist);
}

/* color edges, based on the orientation of polygon(s). One of the two
 * edges passed in is a new one, meaning there is no other polygon
 * sharing it, yet. The other, 'old' edge is common to the new polygon
 * and another one, which was created earlier on. If these two polygon
 * differ in their orientation (one front-, the other backsided to the
 * viewer), this routine has to resolve that conflict.  Edge colours
 * are changed only if the edge wasn't invisible, before */
static void
color_edges(
    long int new_edge,		/* index of 'new', conflictless edge */
    long int old_edge,		/* index of 'old' edge, may conflict */
    long int new_poly,		/* index of current polygon */
    long int old_poly,		/* index of poly sharing old_edge */
    int above,			/* style number for front of polygons */
    int below)			/* style number for backside of polys */
{
    int casenumber;

    if (new_poly > -2) {
	/* new polygon was built successfully */
	if (old_poly <= -2)
	    /* old polygon doesn't exist. Use new_polygon for both: */
	    old_poly = new_poly;

	casenumber =
	    (plist[new_poly].frontfacing ? 1 : 0)
	    + 2 * (plist[old_poly].frontfacing ? 1 : 0);
	switch (casenumber) {
	case 0:
	    /* both backfacing */
	    if (elist[new_edge].style >= -2)
		elist[new_edge].style	= below;
	    if (elist[old_edge].style >= -2)
		elist[old_edge].style = below;
	    break;
	case 2:
	    if (elist[new_edge].style >= -2)
		elist[new_edge].style = below;
	    /* FALLTHROUGH */
	case 1:
	    /* new front-, old one backfacing, or */
	    /* new back-, old one frontfacing */
	    if (((new_edge == old_edge)
		 && hiddenHandleBentoverQuadrangles) /* a diagonal edge! */
		|| (elist[old_edge].style >= -2)) {
		/* conflict has occured: two polygons meet here, with opposige
		 * sides being shown. What's to do?
		 * 1) find a vertex of one polygon outside this common
		 * edge
		 * 2) check wether it's in front of or behind the
		 * other polygon's plane
		 * 3) if in front, color the edge accoring to the
		 * vertex' polygon, otherwise, color like the other
		 * polygon */
		long int vnum1 = elist[old_edge].v1;
		long int vnum2 = elist[old_edge].v2;
		p_polygon p = plist + new_poly;
		long int pvert = -1;
		double point_to_plane;

		if (p->vertex[0] == vnum1) {
		    if (p->vertex[1] == vnum2) {
			pvert = p->vertex[2];
		    } else if (p->vertex[2] == vnum2) {
			pvert = p->vertex[1];
		    }
		} else if (p->vertex[1] == vnum1) {
		    if (p->vertex[0] == vnum2) {
			pvert = p->vertex[2];
		    } else if (p->vertex[2] == vnum2) {
			pvert = p->vertex[0];
		    }
		} else if (p->vertex[2] == vnum1) {
		    if (p->vertex[0] == vnum2) {
			pvert = p->vertex[1];
		    } else if (p->vertex[1] == vnum2) {
			pvert = p->vertex[0];
		    }
		}
		assert (pvert >= 0);

		point_to_plane =
		    eval_plane_equation(plist[old_poly].plane, vlist + pvert);

		if (point_to_plane > 0) {
		    /* point in new_poly is in front of old_poly plane */
		    elist[old_edge].style = p->frontfacing ? above : below;
		}	else {
		    elist[old_edge].style =
			plist[old_poly].frontfacing ? above : below;
		}
	    }
	    break;
	case 3:
	    /* both frontfacing: nothing to do */
	    break;
	} /* switch */
    } else {
	/* Ooops? build_networks() must have guessed incorrectly that
	 * this polygon should exist. */
	return;
    }
}


/* This somewhat monstrous routine fills the vlist, elist and plist
 * dynamic arrays with values from all those plots. It strives to
 * respect all the topological linkage between vertices, edges and
 * polygons. E.g., it has to find the correct color for each edge,
 * based on the orientation of the two polygons sharing it, WRT both
 * the observer and each other. */
/* NEW FEATURE HBB 20000715: allow non-grid datasets too, by storing
 * only vertices and 'direct' edges, but no polygons or 'cross' edges
 * */
static void
build_networks(struct surface_points *plots, int pcount)
{
    long int i;
    struct surface_points *this_plot;
    int surface;		/* count the surfaces (i.e. sub-plots) */
    long int crv, ncrvs;	/* count isolines */
    long int nverts;		/* count vertices */
    long int max_crvlen;	/* maximal length of isoline in any plot */
    long int nv, ne, np;	/* local poly/edge/vertex counts */
    long int *north_polygons;	/* stores polygons of isoline above */
    long int *these_polygons;	/* same, being built for use by next turn */
    long int *north_edges;	/* stores edges of polyline above */
    long int *these_edges;	/* same, being built for use by next turn */
    struct iso_curve *icrvs;
    int above = -3, below;	/* line type for edges of front/back side*/
    struct lp_style_type *lp;	/* pointer to line and point properties */

    /* Count out the initial sizes needed for the polygon and vertex
     * lists. */
    nv = ne = np = 0;
    max_crvlen = -1;

    for (this_plot = plots, surface = 0;
	surface < pcount;
	this_plot = this_plot->next_sp, surface++) {
	long int crvlen;
	
	/* Quietly skip empty plots */
	if (this_plot->plot_type == NODATA)
	    continue;

	crvlen = this_plot->iso_crvs->p_count;

	/* Allow individual plots to opt out of hidden3d calculations */
	if (this_plot->opt_out_of_hidden3d)
	    continue;

	/* register maximal isocurve length. Only necessary for
	 * grid-topology plots that will create polygons, so I can do
	 * it here, already. */
	if (crvlen > max_crvlen)
	    max_crvlen = crvlen;

	/* count 'curves' (i.e. isolines) and vertices in this plot */
	nverts = 0;
	if(this_plot->plot_type == FUNC3D) {
	    ncrvs = 0;
	    for(icrvs = this_plot->iso_crvs;
		icrvs; icrvs = icrvs->next) {
		ncrvs++;
	    }
	    nverts += ncrvs * crvlen;
	} else if(this_plot->plot_type == DATA3D) {
	    ncrvs = this_plot->num_iso_read;
	    if (this_plot->has_grid_topology)
		nverts += ncrvs * crvlen;
	    else if (this_plot->plot_style == VECTOR)
		nverts += this_plot->iso_crvs->p_count;
	    else {
		/* have to check each isoline separately: */
		for (icrvs = this_plot->iso_crvs; icrvs; icrvs = icrvs->next)
		    nverts += icrvs->p_count;
	    }
	} else {
	    graph_error("Plot type is neither function nor data");
	    return;
	}

	/* To avoid possibly suprising error messages, several 2d-only
	 * plot styles are mapped to others, that are genuinely
	 * available in 3d. */
	switch (this_plot->plot_style) {
	case LINESPOINTS:
	case STEPS:
	case FSTEPS:
	case HISTEPS:
	case LINES:
	    nv += nverts;
	    ne += nverts - ncrvs;
	    if (this_plot->has_grid_topology) {
		ne += 2 * nverts - ncrvs - 2 * crvlen + 1;
		np += 2 * (ncrvs - 1) * (crvlen - 1);
	    }
	    break;
	case BOXES:
	case FILLEDCURVES:
	case IMPULSES:
	case VECTOR:
	    nv += 2 * nverts;
	    ne += nverts;
	    break;
	case POINTSTYLE:
	default:
	    /* treat all remaining ones like 'points' */
	    nv += nverts;
	    ne += nverts; /* a 'phantom edge' per isolated point */
	    break;
	} /* switch */
    } /* for (plots) */

    /* Check for no data at all */
    if (max_crvlen <= 0)
	return;

    /* allocate all the lists to the size we need: */
    resize_dynarray(&vertices, nv);
    resize_dynarray(&edges, ne);
    resize_dynarray(&polygons, np);

    /* allocate the storage for polygons and edges of the isoline just
     * above the current one, to allow easy access to them from the
     * current isoline */
    north_polygons = gp_alloc(2 * max_crvlen * sizeof (long int),
			      "hidden north_polys");
    these_polygons = gp_alloc(2 * max_crvlen * sizeof (long int),
			      "hidden these_polys");
    north_edges = gp_alloc(3 * max_crvlen * sizeof (long int),
			   "hidden north_edges");
    these_edges = gp_alloc(3 * max_crvlen * sizeof (long int),
			   "hidden these_edges");

    /* initialize the lists, all in one large loop. This is different
     * from the previous approach, which went over the vertices,
     * first, and only then, in new loop, built polygons */
    for (this_plot = plots, surface = 0;
	 surface < pcount;
	 this_plot = this_plot->next_sp, surface++) {
	lp_style_type *lp_style = NULL;
	TBOOLEAN color_from_column = this_plot->pm3d_color_from_column;
	long int crvlen;

	/* Quietly skip empty plots */
	if (this_plot->plot_type == NODATA)
	    continue;

	crvlen = this_plot->iso_crvs->p_count;

	/* Allow individual plots to opt out of hidden3d calculations */
	if (this_plot->opt_out_of_hidden3d)
	    continue;

	lp = &(this_plot->lp_properties);
	above = this_plot->lp_properties.l_type;
	below = this_plot->lp_properties.l_type + hiddenBacksideLinetypeOffset;

	/* calculate the point symbol type: */
	/* Assumest hat upstream functions have made sure this is
	 * initialized sensibly --- thou hast been warned */
	lp_style = &(this_plot->lp_properties);

	/* HBB 20000715: new initialization code block for non-grid
	 * structured datasets. Sufficiently different from the rest
	 * to warrant separate code, I think. */
	if (! this_plot->has_grid_topology) {
	    for (crv = 0, icrvs = this_plot->iso_crvs;
		 icrvs;
		 crv++, icrvs = icrvs->next) {
		struct coordinate GPHUGE *points = icrvs->points;
		long int previousvertex = -1;

#ifdef EAM_DATASTRINGS
		/* To handle labels we must look inside a separate list */
		/* rather than just walking through the points arrays.  */
		if (this_plot->plot_style == LABELPOINTS) {
		    struct text_label *label;
		    long int thisvertex;
		    struct coordinate labelpoint = {0};

		    lp->pointflag = 1; /* Labels can use the code for hidden points */
		    for (label = this_plot->labels->next; label != NULL; label = label->next) {
			labelpoint.x = label->place.x;
			labelpoint.y = label->place.y;
			labelpoint.z = label->place.z;
			thisvertex = store_vertex(&labelpoint, 
				&(this_plot->lp_properties), color_from_column);
			if (thisvertex < 0)
			    continue;
			(vlist+thisvertex)->label = label;
			store_edge(thisvertex, edir_point, crvlen, lp, above);
		    }
		} else
#endif

		for (i = 0; i < icrvs->p_count; i++) {
		    long int thisvertex, basevertex;

		    thisvertex = store_vertex(points + i, lp_style,
					      color_from_column);

		    if (this_plot->plot_style == VECTOR) {
			store_vertex(icrvs->next->points+i, 0, 0);
		    }

		    if (thisvertex < 0) {
			previousvertex = thisvertex;
			continue;
		    }

		    switch (this_plot->plot_style) {
		    case LINESPOINTS:
		    case STEPS:
		    case FSTEPS:
		    case HISTEPS:
		    case LINES:
			if (previousvertex >= 0)
			    store_edge(thisvertex, edir_west, 0, lp, above);
			break;
		    case VECTOR:
			store_edge(thisvertex, edir_vector, 0, lp, above);
			break;
		    case BOXES:
		    case FILLEDCURVES:
		    case IMPULSES:
			/* set second vertex to the low end of zrange */
			{
			    coordval remember_z = points[i].z;

			    points[i].z = axis_array[FIRST_Z_AXIS].min;
			    basevertex = store_vertex(points + i, lp_style,
						      color_from_column);
			    points[i].z = remember_z;
			}
			if (basevertex > 0)
			    store_edge(basevertex, edir_impulse, 0, lp, above);
			break;

		    case POINTSTYLE:
		    default:	/* treat all the others like 'points' */
			store_edge(thisvertex, edir_point, crvlen, lp, above);
			break;
		    } /* switch(plot_style) */

		    previousvertex = thisvertex;
		} /* for(vertex) */
	    } /* for(crv) */

	    continue;		/* done with this plot! */
	}

	/* initialize stored indices of north-of-this-isoline polygons and
	 * edges properly */
	for (i=0; i < this_plot->iso_crvs->p_count; i++) {
	    north_polygons[2 * i]
		= north_polygons[2 * i + 1]
		= north_edges[3 * i]
		= north_edges[3 * i + 1]
		= north_edges[3 * i + 2]
		= -3;
	}

	for (crv = 0, icrvs = this_plot->iso_crvs;
	     icrvs;
	     crv++, icrvs = icrvs->next) {
	    struct coordinate GPHUGE *points = icrvs->points;

	    for (i = 0; i < icrvs->p_count; i++) {
		long int thisvertex, basevertex;
		long int e1, e2, e3;
		long int pnum;

		thisvertex = store_vertex(points + i, lp_style,
					  color_from_column);

		/* Preset the pointers to the polygons and edges
		 * belonging to this isoline */
		these_polygons[2 * i] = these_polygons[2 * i + 1]
		    = these_edges[3 * i] = these_edges[3 * i + 1]
		    = these_edges[3 * i + 2]
		    = -3;

		switch (this_plot->plot_style) {
		case LINESPOINTS:
		case STEPS:
		case FSTEPS:
		case HISTEPS:
		case LINES:
		    if (i > 0) {
			/* not first point, so we might want to set up
			 * the edge(s) to the left of this vertex */
			if (thisvertex < 0) {
			    if ((crv > 0)
				&& (hiddenShowAlternativeDiagonal)
				) {
				/* this vertex is invalid, but the
				 * other three might still form a
				 * valid triangle, facing northwest to
				 * do that, we'll need the 'wrong'
				 * diagonal, which goes from SW to NE:
				 * */
				these_edges[i*3+2] = e3
				    = store_edge(vertices.end - 1, edir_NE, crvlen,
						 lp, above);
				if (e3 > -2) {
				    /* don't store this polygon for
				     * later: it doesn't share edges
				     * with any others to the south or
				     * east, so there's need to */
				    pnum
					= store_polygon(vertices.end - 1, pdir_NW, crvlen);
				    /* The other two edges of this
				     * polygon need to be checked
				     * against the neighboring
				     * polygons' orientations, before
				     * being coloured */
				    color_edges(e3, these_edges[3*(i-1) +1],
						pnum, these_polygons[2*(i-1) + 1],
						above, below);
				    color_edges(e3, north_edges[3*i],
						pnum, north_polygons[2*i], above, below);
				}
			    }
			    break; /* nothing else to do for invalid vertex */
			}

			/* Coming here means that the current vertex
			 * is valid: check the other three of this
			 * cell, by trying to set up the edges from
			 * this one to there */
			these_edges[i*3] = e1
			    = store_edge(thisvertex, edir_west, crvlen, lp, above);

			if (crv > 0) { /* vertices to the north exist */
			    these_edges[i*3 + 1] = e2
				= store_edge(thisvertex, edir_north, crvlen, lp, above);
			    these_edges[i*3 + 2] = e3
				= store_edge(thisvertex, edir_NW, crvlen, lp, above);
			    if (e3 > -2) {
				/* diagonal edge of this cell is OK,
				 * so try to build both the polygons:
				 * */
				if (e1 > -2) {
				    /* one pair of edges is valid: put
				     * first polygon, which points
				     * towards the southwest */
				    these_polygons[2*i] = pnum
					= store_polygon(thisvertex, pdir_SW, crvlen);
				    color_edges(e1, these_edges[3*(i-1)+1],
						pnum, these_polygons[2*(i-1)+ 1], above, below );
				}
				if (e2 > -2) {
				    /* other pair of two is fine, put
				     * the northeast polygon: */
				    these_polygons[2*i + 1] = pnum
					= store_polygon(thisvertex, pdir_NE, crvlen);
				    color_edges(e2, north_edges[3*i],
						pnum, north_polygons[2*i], above, below);
				}
				/* In case these two new polygons
				 * differ in orientation, find good
				 * coloring of the diagonal */
				color_edges(e3, e3, these_polygons[2*i],
					    these_polygons[2*i+1], above, below);
			    } /* if e3 valid */
			    else if ((e1 > -2) && (e2 > -2)
				     && hiddenShowAlternativeDiagonal) {
				/* looks like all but the north-west
				 * vertex are usable, so we set up the
				 * southeast-pointing triangle, using
				 * the 'wrong' diagonal: */
				these_edges[3*i + 2] = e3
				    = store_edge(thisvertex, edir_NE, crvlen, lp, above);
				if (e3 > -2) {
				    /* fill this polygon into *both*
				     * polygon places for this
				     * quadrangle, as this triangle
				     * coincides with both edges that
				     * will be used by later polygons
				     * */
				    these_polygons[2*i] = these_polygons[2*i+1] = pnum
					= store_polygon(thisvertex, pdir_SE, crvlen);
				    /* This case is somewhat special:
				     * all edges are new, so there is
				     * no other polygon orientation to
				     * consider */
				    if (!plist[pnum].frontfacing)
					elist[e1].style = elist[e2].style = elist[e3].style
					    = below;
				}
			    }
			}
		    } else if ((crv > 0)
			       && (thisvertex >= 0)) {
			/* We're at the west border of the grid, but
			 * not on the north one: put vertical end-wall
			 * edge:*/
			these_edges[3*i + 1] =
			    store_edge(thisvertex, edir_north, crvlen, lp, above);
		    }
		    break;

		case BOXES:
		case FILLEDCURVES:
		case IMPULSES:
		    if (thisvertex < 0)
			break;

		    /* set second vertex to the low end of zrange */
		    {
			coordval remember_z = points[i].z;

			points[i].z = axis_array[FIRST_Z_AXIS].min;
			basevertex = store_vertex(points + i, lp_style,
						  color_from_column);
			points[i].z = remember_z;
		    }
		    if (basevertex > 0)
			store_edge(basevertex, edir_impulse, 0, lp, above);
		    break;

		case POINTSTYLE:
		default:	/* treat all the others like 'points' */
		    if (thisvertex < 0) /* Ignore invalid vertex */
			break;
		    store_edge(thisvertex, edir_point, crvlen, lp, above);
		    break;
		} /* switch */
	    } /* for(i) */

	    /* Swap the 'north' lists of polygons and edges with
	     * 'these' ones, which have been filled in the pass
	     * through this isocurve */
	    {
		long int *temp = north_polygons;
		north_polygons = these_polygons;
		these_polygons = temp;

		temp = north_edges;
		north_edges = these_edges;
		these_edges = temp;
	    }
	} /* for(isocrv) */
    } /* for(plot) */

    free (these_polygons);
    free (north_polygons);
    free (these_edges);
    free (north_edges);
}

/* Sort the elist in order of growing zmax. Uses qsort on an array of
 * plist indices, and then fills in the 'next' fields in struct
 * polygon to store the resulting order inside the plist */
/* HBB 20010720: removed 'static' to avoid HP-sUX gcc bug */
int
compare_edges_by_zmin(SORTFUNC_ARGS p1, SORTFUNC_ARGS p2)
{
    return SIGN(vlist[elist[*(const long *) p1].v2].z
		- vlist[elist[*(const long *) p2].v2].z);
}

static void
sort_edges_by_z()
{
    long *sortarray, i;
    p_edge this;

    if (!edges.end)
	return;

    sortarray = gp_alloc((unsigned long) sizeof(long) * edges.end,
			 "hidden sort edges");
    /* initialize sortarray with an identity mapping */
    for (i = 0; i < edges.end; i++)
	sortarray[i] = i;
    /* sort it */
    qsort(sortarray, (size_t) edges.end, sizeof(long), compare_edges_by_zmin);

    /* traverse plist in the order given by sortarray, and set the
     * 'next' pointers */
    this = elist + sortarray[0];
    for (i = 1; i < edges.end; i++) {
	this->next = sortarray[i];
	this = elist + sortarray[i];
    }
    this->next = -1L;

    /* 'efirst' is the index of the leading element of plist */
    efirst = sortarray[0];

    free(sortarray);
}

/* HBB 20010720: removed 'static' to avoid HP-sUX gcc bug */
int
compare_polys_by_zmax(SORTFUNC_ARGS p1, SORTFUNC_ARGS p2)
{
    return (SIGN(plist[*(const long *) p1].zmax
		 - plist[*(const long *) p2].zmax));
}

static void
sort_polys_by_z()
{
    long *sortarray, i;
    p_polygon this;

    if (!polygons.end)
	return;

    sortarray = gp_alloc((unsigned long) sizeof(long) * polygons.end,
			 "hidden sortarray");

    /* initialize sortarray with an identity mapping */
    for (i = 0; i < polygons.end; i++)
	sortarray[i] = i;

    /* sort it */
    qsort(sortarray, (size_t) polygons.end, sizeof(long),
	  compare_polys_by_zmax);

    /* traverse plist in the order given by sortarray, and set the
     * 'next' pointers */
#if HIDDEN3D_QUADTREE
    /* HBB 20000716: Loop backwards, to ease construction of
     * linked lists from the head: */
    {
	int grid_x, grid_y;
	int grid_x_low, grid_x_high, grid_y_low, grid_y_high;

	for (grid_x = 0; grid_x < QUADTREE_GRANULARITY; grid_x++)
	    for (grid_y = 0; grid_y < QUADTREE_GRANULARITY; grid_y++)
		quadtree[grid_x][grid_y] = -1;

	for (i=polygons.end - 1; i >= 0; i--) {
	    this = plist + sortarray[i];

	    grid_x_low = coord_to_treecell(this->xmin);
	    grid_x_high = coord_to_treecell(this->xmax);
	    grid_y_low = coord_to_treecell(this->ymin);
	    grid_y_high = coord_to_treecell(this->ymax);

	    for (grid_x = grid_x_low; grid_x <= grid_x_high; grid_x++) {
		for (grid_y = grid_y_low; grid_y <= grid_y_high; grid_y++) {
		    p_qtreelist newhead = nextfrom_dynarray(&qtree);

		    newhead->next = quadtree[grid_x][grid_y];
		    newhead->p = sortarray[i];

		    quadtree[grid_x][grid_y] = newhead - qlist;
		}
	    }
	}
    }

#else /* HIDDEN3D_QUADTREE */
    this = plist + sortarray[0];
    for (i = 1; i < polygons.end; i++) {
	this->next = sortarray[i];
	this = plist + sortarray[i];
    }
    this->next = -1L;
    /* 'pfirst' is the index of the leading element of plist */
#endif /* HIDDEN3D_QUADTREE */
    pfirst = sortarray[0];

    free(sortarray);
}


#define LASTBITS (USHRT_BITS -1)	/* ????? */

/************************************************/
/*******            Drawing the polygons ********/
/************************************************/

/* draw a single vertex as a point symbol, if requested by the chosen
 * plot style (linespoints, points, or dots...) */
static void
draw_vertex(p_vertex v)
{
    unsigned int x, y;

    TERMCOORD(v, x, y);
    if (v->lp_style && v->lp_style->p_type >= 0 && !clip_point(x,y)) {
	int colortype = v->lp_style->pm3d_color.type;

#ifdef EAM_DATASTRINGS
	if (v->label)  {
	    write_label(x,y, v->label);
	    v->lp_style = NULL;
	    return;
	}
#endif

	/* EAM DEBUG - Check for extra point properties */
	if (colortype == TC_LT)
	    /* Should have been set already! */
	    ;
	else if (colortype == TC_RGB && v->lp_style->pm3d_color.lt == LT_COLORFROMCOLUMN)
	    set_rgbcolor(v->real_z);
	else if (colortype == TC_RGB)
	    set_rgbcolor(v->lp_style->pm3d_color.lt);
	else if (colortype == TC_Z)
	    set_color( cb2gray(z2cb(v->real_z)) );

#ifdef HIDDEN3D_VAR_PTSIZE
	if (v->lp_style->p_size == PTSZ_VARIABLE)
	    (term->pointsize)(pointsize * v->original->CRD_PTSIZE);
#endif

	(term->point)(x,y, v->lp_style->p_type);

	/* vertex has been drawn --> flag it as done */
	v->lp_style = NULL;
    }
}


/* The function that actually does the drawing of the visible portions
 * of lines */
/* HBB 20001108: changed to take the pointers to the end vertices as
 * additional arguments. */
static void
draw_edge(p_edge e, p_vertex v1, p_vertex v2)
{
    assert (e >= elist && e < elist + edges.end);

    draw3d_line_unconditional(v1, v2, e->lp, e->style);
    if (e->lp->pointflag) {
	draw_vertex(v1);
	draw_vertex(v2);
    }
}


/*************************************************************/
/*************************************************************/
/*******   The depth sort algorithm (in_front) and its  ******/
/*******   whole lot of helper functions                ******/
/*************************************************************/
/*************************************************************/

/* Split a given line segment into two at an inner point. The inner
 * point is specified as a fraction of the line-length (0 is V1, 1 is
 * V2) */
/* HBB 20001108: changed to now take two vertex pointers as its
 * arguments, rather than an edge pointer. */
/* HBB 20001204: changed interface again. Now use vertex indices,
 * rather than pointers, to avoid problems with dangling pointers
 * after nextfrom_dynarray() call. */
static long
split_line_at_ratio(
    long vnum1, long vnum2, 	/* vertex indices of line to split */
    double w)			/* where to split it */
{
    p_vertex v;

    if (EQ(w, 0.0))
	return vnum1;
    if (EQ(w, 1.0))
	return vnum2;

    /* Create a new vertex */
    v = nextfrom_dynarray(&vertices);

    v->x = (vlist[vnum2].x - vlist[vnum1].x) * w + vlist[vnum1].x;
    v->y = (vlist[vnum2].y - vlist[vnum1].y) * w + vlist[vnum1].y;
    v->z = (vlist[vnum2].z - vlist[vnum1].z) * w + vlist[vnum1].z;
    v->real_z = (vlist[vnum2].real_z - vlist[vnum1].real_z) * w
	+ vlist[vnum1].real_z;

    /* no point symbol for vertices generated by splitting an edge */
    v->lp_style = NULL;

    /* additional checks to prevent adding unnecessary vertices */
    if (V_EQUAL(v, vlist + vnum1)) {
	droplast_dynarray(&vertices);
	return vnum1;
    }
    if (V_EQUAL(v, vlist + vnum2)) {
	droplast_dynarray(&vertices);
	return vnum2;
    }

    return (v - vlist);
}


/* Compute the 'signed area' of 3 points in their 2d projection
 * to the x-y plane. Essentially the z component of the crossproduct.
 * Should come out positive if v1, v2, v3 are ordered counter-clockwise */

static GP_INLINE double
area2D(p_vertex v1, p_vertex v2, p_vertex v3)
{
    double
	dx12 = v2->x - v1->x,	/* x/y components of (v2-v1) and (v3-v1) */
	dx13 = v3->x - v1->x,
	dy12 = v2->y - v1->y,
	dy13 = v3->y - v1->y;
    return (dx12 * dy13 - dy12 * dx13);
}

/* Utility routine: takes an edge and makes a new one, which is a fragment
 * of the old one. The fragment inherits the line style and stuff of the
 * given edge; only the two new vertices are different. The new edge
 * is then passed to in_front, for recursive handling */
/* HBB 20001108: Changed from edge pointer to edge index. Don't
 * allocate a fresh anymore, as this is no longer needed after the
 * change to in_front().  What remains of this function may no longer
 * be worth having. I.e. it can be replaced by a direct recursion call
 * of in_front(), sometime soon. */
static GP_INLINE void
handle_edge_fragment(long edgenum, long vnum1, long vnum2, long firstpoly)
{
#if !HIDDEN3D_QUADTREE
    /* Avoid checking against the same polygon again. */
    firstpoly = plist[firstpoly].next;
#endif
    in_front(edgenum, vnum1, vnum2, &firstpoly);
}

/*********************************************************************/
/* The actual heart of all this: determines if edge at index 'edgenum'
 * of the elist is in_front of all the polygons, or not. If necessary,
 * it will recursively call itself to isolate more than one visible
 * fragment of the input edge. Wherever possible, recursion is
 * avoided, by in-place modification of the edge.
 *
 * The visible fragments are then drawn by a call to 'draw_edge' from
 * inside this routine. */
/*********************************************************************/
/* HBB 20001108: changed to now take the vertex numbers as additional
 * arguments. The idea is to not overwrite the endpoint stored with
 * the edge, so Test 2 will catch on even after the subject edge has
 * been split up before one of its two polygons is tested against
 * it. FIXME: allocates new vertices when splitting, but never frees
 * them, currently. */

static int
in_front(
    long edgenum,		/* number of the edge in elist */
    long vnum1, long vnum2,	/* numbers of its endpoints */
    long *firstpoly)		/* first plist index to consider */
{
    p_polygon p;		/* pointer to current testing polygon */
    long int polynum;		/* ... and its index in the plist */
    p_vertex v1, v2;		/* pointers to vertices of input edge */

    coordval xmin, xmax;	/* all of these are for the edge */
    coordval ymin, ymax;
    coordval zmin, zmax;
#if HIDDEN3D_GRIDBOX
    unsigned int xextent;	/* extent bitmask in x direction */
    unsigned int yextent;	/* same, in y direction */

# define SET_XEXTENT \
  xextent = CALC_BITRANGE(xmin, xmax);
# define SET_YEXTENT \
  yextent = CALC_BITRANGE(ymin, ymax);
#else
# define SET_XEXTENT /* nothing */
# define SET_YEXTENT /* nothing */
#endif
#if HIDDEN3D_QUADTREE
    int grid_x, grid_y;
    int grid_x_low, grid_x_high;
    int grid_y_low, grid_y_high;
    long listhead;
#endif

    /* zmin of the edge, as it started out. This is needed separately to
     * allow modifying '*firstpoly', without moving it too far to the
     * front. */
    coordval first_zmin;

    /* macro for eliminating tail-recursion inside in_front: when the
     * current edge is modified, recompute all function-wide status
     * variables. Note that it guarantees that v1 is always closer to
     * the viewer than v2 (in z direction) */
    /* HBB 20001108: slightly changed so it can be called with vnum1
     * and vnum2 as its arguments, too */
#define setup_edge(vert1, vert2)		\
    do {					\
	if (vlist[vert1].z > vlist[vert2].z) {	\
	    v1 = vlist + (vert1);		\
	    v2 = vlist + (vert2);		\
	} else {				\
	    v1 = vlist + (vert2);		\
	    v2 = vlist + (vert1);		\
	}					\
	vnum1 = v1 - vlist;			\
	vnum2 = v2 - vlist;			\
	zmax = v1->z;	zmin = v2->z;		\
						\
	if (v1->x > v2->x) {			\
	    xmin = v2->x;	xmax = v1->x;	\
	} else {				\
	    xmin = v1->x;	xmax = v2->x;	\
	}					\
	SET_XEXTENT;				\
						\
	if (v1->y > v2->y) {			\
	    ymin = v2->y;	ymax = v1->y;	\
	} else {				\
	    ymin = v1->y;	ymax = v2->y;	\
	}					\
	SET_YEXTENT;				\
    } while (0) /* end macro setup_edge */

    /* use the macro for initial setup, too: */
    setup_edge(vnum1, vnum2);

    first_zmin = zmin;

#if HIDDEN3D_QUADTREE
    grid_x_low = coord_to_treecell(xmin);
    grid_x_high = coord_to_treecell(xmax);
    grid_y_low = coord_to_treecell(ymin);
    grid_y_high = coord_to_treecell(ymax);

    for (grid_x = grid_x_low; grid_x <= grid_x_high; grid_x ++)
	for (grid_y = grid_y_low; grid_y <= grid_y_high; grid_y ++)
	    for (listhead = quadtree[grid_x][grid_y];
		 listhead >= 0;
		 listhead = qlist[listhead].next)
#else /* HIDDEN3D_QUADTREE */
    /* loop over all the polygons in the sorted list, starting at the
     * currently first (i.e. furthest, from the viewer) polgon. */
    for (polynum = *firstpoly; polynum >=0; polynum = p->next)
#endif /* HIDDEN3D_QUADTREE */
	{
	    /* shortcut variables for the three vertices of 'p':*/
	    p_vertex w1, w2, w3;
	    /* signed areas of each of e's vertices wrt. the edges of
	     * p. I store them explicitly, because they are used more
	     * than once, eventually */
	    double e_side[2][POLY_NVERT];
	    /* signed areas of each of p's vertices wrt. to edge e. Stored
	     * for re-use in intersection calculations, as well */
	    double p_side[POLY_NVERT];
	    /* values got from throwing the edge vertices into the plane
	     * equation of the polygon. Used for testing if the vertices are
	     * in front of or behind the plane, and also to determine the
	     * point where the edge goes through the polygon, by
	     * interpolation. */
	    double v1_rel_pplane, v2_rel_pplane;
	    /* Orientation of polygon wrt. to the eye: front or back side
	     * visible? */
	    coordval polyfacing;
	    int whichside;
	    /* stores classification of cases as 4 2-bit patterns, mainly */
	    unsigned int classification[POLY_NVERT + 1];

#if HIDDEN3D_QUADTREE
	    polynum = qlist[listhead].p;
#endif
	    p = plist + polynum;

	    /* OK, off we go with the real work. This algo is mainly
	     * following the one of 'HLines.java', as described in the
	     * book 'Computer Graphics for Java Programmers', by Dutch
	     * professor Leen Ammeraal, published by J.Wiley & Sons,
	     * ISBN 0 471 98142 7. It happens to the only(!) actual
	     * code or algorithm example I got out of a question to
	     * the Newsgroup comp.graphics.algorithms, asking for a
	     * hidden *line* removal algorithm. */

	    /* Test 1 (2D): minimax tests. Do x/y ranges of polygon
	     * and edge have any overlap? */
	    if (0
#if HIDDEN3D_GRIDBOX
		/* First, check by comparing the extent bit patterns: */
		|| (!(xextent & p->xbits))
		|| (!(yextent & p->ybits))
#endif
		|| (p->xmax < xmin)
		|| (p->xmin > xmax)
		|| (p->ymax < ymin)
		|| (p->ymin > ymax)
		)
		continue;

	    /* Tests 2 and 3 switched... */

	    /* Test 3 (3D): Is edge completely in front of polygon? */
	    if (p->zmax < zmin) {
		/* Polygon completely behind this edge. Move start of
		 * relevant plist to this point, to speed up next
		 * run. This makes use of the fact that elist is also
		 * kept in upwardly sorted order of zmin, i.e. the
		 * condition found here will also hold for all coming
		 * edges in the list */
		if (p->zmax < first_zmin)
		    *firstpoly = polynum;
		continue;	/* this polygon is done with */
	    }

	    /* Test 2 (0D): does edge belong to this very polygon? */
	    /* 20001108: to make this rejector more effective, do keep
	     * the original edge vertices unchanged */
	    if (1
		&& (0
		    || (p->vertex[0] == elist[edgenum].v1)
		    || (p->vertex[1] == elist[edgenum].v1)
		    || (p->vertex[2] == elist[edgenum].v1)
		    )
		&& (0
		    || (p->vertex[0] == elist[edgenum].v2)
		    || (p->vertex[1] == elist[edgenum].v2)
		    || (p->vertex[2] == elist[edgenum].v2)
		    )
		)
		continue;

	    w1 = vlist + p->vertex[0];
	    w2 = vlist + p->vertex[1];
	    w3 = vlist + p->vertex[2];


	    /* Test 4 (2D): Is edge fully on the 'wrong side' of one of the
	     * polygon edges?
	     *
	     * Done by comparing signed areas (cross-product z components of
	     * both edge endpoints) to that of the polygon itself (as given by
	     * the plane normal z component). If both v1 and v2 are found on
	     * the 'outside' side of a polygon edge (interpreting it as
	     * infinite line), the polygon can't obscure this edge. All the
	     * signed areas are remembered, for later tests. */
	    polyfacing = p->plane[2];
	    whichside = SIGN(polyfacing);
	    if (!whichside)
		/* We're looking at the border of this polygon. It
		 * looks like an infitisemally thin line, i.e. it'll
		 * be practically invisible*/
		/* FIXME: should such a polygon have been stored in the plist,
		 * in the first place??? */
		continue;

	    /* p->plane[2] is now forced >=0, so it doesn't tell anything
	     * about which side the polygon is facing */
	    whichside = (p->frontfacing) ? -1 : 1;

#if 0
	    /* Just make sure I got this the right way round: */
	    if (p->frontfacing)
		assert (GE(area2D(w1, w2, w3),0));
	    else
		assert (GE(0,area2D(w1, w2, w3)));
#endif

	    /* classify both endpoints of the test edge, by their
	     * position relative to some plane that defines the volume
	     * shadowed by a triangle. Used both by macro 'checkside',
	     * and for individual checking, below --> put this into a
	     * macro */
#define classify(par1, par2, classnum)					\
	    do {							\
		if (GR(par1, 0)) {					\
		    /* v1 strictly outside --> v2 in-plane is enough,	\
                       already : */					\
		    classification[classnum]				\
			= (1 << 0) | ((GE(par2, 0)) << 1);		\
		} else if (GR(par2, 0)) {				\
		    /* v2 strictly outside --> v1 in-plane is enough,	\
                       already : */					\
		    classification[classnum]				\
			= ((GE(par1, 0)) << 0) | (1 << 1);		\
		} else {						\
		    /* as neither one of them is strictly outside,	\
		     * there's no need for any edge-splitting, at all.	\
		     * */						\
		    classification[classnum] = 0;			\
		}							\
	    } while (0)

	    /* code block used thrice, so put into a macro. The
	     * 'area2D' calls return <0, ==0 or >0 if the three
	     * vertices passed to it are in clockwise order, colinear,
	     * or in conter-clockwise order, respectively. So the sign
	     * of this can be used to determine if a point is inside
	     * or outside the shadow volume. */
#define checkside(a,b,s)					\
	    e_side[0][s] = whichside * area2D((a), (b), v1);	\
	    e_side[1][s] = whichside * area2D((a), (b), v2);	\
	    classify(e_side[0][s], e_side[1][s], s);		\
	    if (classification[s] == 3)				\
		continue;

	    checkside(w1, w2, 0);
	    checkside(w2, w3, 1);
	    checkside(w3, w1, 2);
#undef checkside		/* get rid of that macro, again */


	    /* Test 5 (2D): Does the whole polygon lie on one and the same
	     * side of the tested edge? Again, area2D is used to determine on
	     * which side of a given edge a vertex lies. */
	    /* HBB 20000621: this test only makes sense if this is a
	     * 'real' edge, not just a single point */
	    /* HBB 20051206: but p_side[] may still have to be computed, to be used
	     * by Test 9 */
	    p_side[0] = area2D(v1, v2, w1);
	    p_side[1] = area2D(v1, v2, w2);
	    p_side[2] = area2D(v1, v2, w3);
	    if (v1 != v2) {
		/* HBB 20001104: made this more restrictive: only
		 * reject p if areas are greater than 0, i.e. don't
		 * allow EQ(..,0). Otherwise, edges coincident with a
		 * polygon edge in 2D will not be hidden by it. */
		/* FIXME HBB 20001108: some more code will be needed
		 * here or further down below. It will currently fail
		 * if two edges fall exactly on top of each other
		 * (like the crossover line in the Klein-bottle demo):
		 * both edges will be hidden by the other's polygon,
		 * essentially */
#if 0
		if (0
		    || (GR(p_side[0], 0)
			&& GR(p_side[1], 0)
			&& GR(p_side[2], 0)
			)
		    || (GR(0 , p_side[0])
			&& GR(0 , p_side[1])
			&& GR(0 , p_side[2])
			)
		    )
		    continue;
#else
		/* HBB 20020406: try to repair this */
		if (0
		    || (GE(p_side[0], 0)
			&& GE(p_side[1], 0)
			&& GE(p_side[2], 0)
			)
		    || (GE(0 , p_side[0])
			&& GE(0 , p_side[1])
			&& GE(0 , p_side[2])
			)
		    )
		    continue;
#endif
	    }

	    /* Test 6 (3D): does the whole edge lie on the viewer's
	     * side of the polygon's plane? */
	    v1_rel_pplane = eval_plane_equation(p->plane, v1);
	    v2_rel_pplane = eval_plane_equation(p->plane, v2);

	    /* Condense into classification bit pattern for v1 and v2
	     * wrt.  this plane, and store into entry 'POLY_NVERT' of
	     * the classification array, following the three edges of
	     * the polygon. */
	    classify(v1_rel_pplane, v2_rel_pplane, POLY_NVERT);

	    if (classification[POLY_NVERT] == 3)
		continue;									/* edge completely in front of polygon */

	    /* Test 7 (2D): is edge completely inside the triangle?
	     * The stored results from Test 4 make this rather easy to
	     * check: if a point is oriented equally wrt. all three
	     * edges, and also to the triangle itself, the point is
	     * inside the 2D triangle: */

	    /* search for any one bits (--> they mean 'outside') */
	    if (! (classification[0]
		   | classification[1]
		   | classification[2])) {

		/* Edge is completely inside the polygon's 2D
		 * projection. Unlike Ammeraal, I have to check for
		 * edges penetrating polygons, as well. */

		/* Check if edge really doesn't intersect the triangle
		 * (this happens in roughly one third of the
		 * cases). If both vertices have been classified as
		 * 'behind', the result is true even in the case of
		 * possibly intersecting polygons */
		/* HBB 20020408: softened this check to allow edges
		 * not to be hidden by polygons they're fully inside.
		 * I.e. only declare the edge invisible if at least
		 * one of its endpoints is truly behind, not inside
		 * the polygon's plane. ---> FIXME 20001108 above */
		if (!classification[POLY_NVERT]
		    && (GR(0, v1_rel_pplane) || GR(0, v2_rel_pplane)))
		    return 0;
	    }


	    /* Test 8 (3D): If no edge intersects any polygon, the edge must
	     * lie infront of the poly if the 'inside' vertex in front of it.
	     * This test doesn't work if lines might intersect polygons, so I
	     * took it out of my version of Ammeraal's algorithm. */


	    /* Test 9 (3D): The final 'catch-all' handler. We now know
	     * that the edges passes through the walls of the
	     * semifinite polygonal cylinder of invisible space
	     * stamped out by the polygon, somewhere, so we have to
	     * compute the intersection points with the 2D projection
	     * of the polygon and check whether they are infront of or
	     * behind the polygon, to isolate the visible fragments of
	     * the line. Method is to calc. the [0..1] parameter that
	     * describes the way from V1 to V2, for all intersections
	     * of the edge with any of the planes that define the
	     * 'shadow volume' of this polygon, and deciding what to
	     * do, with all this.
	     *
	     * The remaining work can be classified by the values of
	     * the variables calculated in previous tests:
	     * v{1/2}_rel_pplane, e_side[2][POLY_NVERT],
	     * p_side[POLY_NVERT]. Setting aside cases where one of
	     * them is zero, this leaves a total of 2^11 = 2048
	     * separate cases, in principal. Several of those have
	     * been sorted out, already, by the previous tests. That
	     * leaves us with 3*27 cases, essentially. The '3' cases
	     * differ by which of the three vertices of the polygon is
	     * alone on its side of the test edge.  That already fixes
	     * which two edges might intersect the test edge.  The 27
	     * cases differ by which of the edge points lies on which
	     * of the three planes in space (two sides, and the
	     * polygon plane) that remain interesting for the tests.
	     * That would normally make 2^2 = 4 cases per plane. But
	     * the 'both outside' cases for all three planes have
	     * already been taken care of, earlier, leaving 3 of those
	     * cases to be handled. This is the same for all three
	     * relevant planes, so we get 3^3 = 27 cases. */

	    {
		/* the interception point parameters: */
		double front_hit, this_hit, hit_in_poly;
		long newvert[2];
		int side1, side2;
		double hit1, hit2;
		unsigned int full_class;

		/* find out which of the three edges would be intersected: */
		if ((p_side[0] > 0) == (p_side[1] > 0)) {
		    /* first two vertices on the same side, the third
		     * is on the 'other' side of the test edge */
		    side1 = 1;
		    side2 = 2;
		} else if ((p_side[0] > 0) == (p_side[2] > 0)) {
		    /* vertex '1' on the other side */
		    side1 = 0;
		    side2 = 1;
		} else {
		    /* we now know that it must be the first vertex
		     * that is on the 'other' side, as not all three
		     * can be on different sides. */
		    side1 = 0;
		    side2 = 2;
		}

		/* Carry out the classification into those 27 cases,
		 * based upon classification bits precomputed above,
		 * and calculate the intersection parameters, as
		 * appropriate. Start by regarding the polygon's
		 * plane: '1' means point is in front of plane.
		 *
		 * First, some utility macros: */
#define cleanup_hit(hit)			\
		do {				\
		    if (EQ(hit,0))		\
			hit=0;			\
		    else if (EQ(hit,1))		\
			hit=1;			\
		} while (0)

#define hit_in_edge(hit) ((hit >= 0) && (hit <= 1))

		/* find the intersection point through the front
		 * plane, if any: */
		front_hit = 1e10;
		if (classification[3]) {
		    if (SIGN(v1_rel_pplane - v2_rel_pplane)) {
			front_hit = v1_rel_pplane / (v1_rel_pplane - v2_rel_pplane);
			cleanup_hit(front_hit);
		    }
		    if (!hit_in_edge(front_hit)) {
			front_hit = 1e10;
			classification[3] = 0;	/* not really intersecting */
		    }
		}


		/* Find intersection points with the two relevant side
		 * planes of the shadow volume. A macro contains the
		 * code, to be used for both side1 and side2, to
		 * reduce code overhead: */
#define check_out_hit(hit, side)					\
		do {							\
		    hit = 1e10;	/* default is 'way off' */		\
		    if (classification[side]) {				\
			/* Not both inside, i.e. there has to be an	\
			 * intersection point: */			\
			hit_in_poly = - whichside * p_side[side]	\
			    / (e_side[0][side] - e_side[1][side]);	\
			cleanup_hit(hit_in_poly);			\
			if (hit_in_edge(hit_in_poly))	{		\
			    this_hit = e_side[0][side]			\
				/ (e_side[0][side] - e_side[1][side]);	\
			    cleanup_hit(this_hit);			\
			    if (hit_in_edge(this_hit)) {		\
				hit = this_hit;				\
			    }						\
			}						\
		    }							\
		    if (hit == 1e10)					\
			/* doesn't really intersect */			\
			classification[side] = 0; 			\
		} while (0)

		check_out_hit(hit1, side1);
		check_out_hit(hit2, side2);
#undef check_out_hit

		/* OK, now we know the position of all the hits that
		 * might be needed, given by their positions between
		 * the test edge's vertices. */

		/* Macro: combine all the classifications into a
		 * single classification bitpattern, for easier
		 * listing of cases */
#define makeclass(s2, s1, p) (16 * s2 + 4 * s1 + p)

		full_class = makeclass(classification[side2],
				       classification[side1],
				       classification[3]);

		if (!full_class)
		    /* all suspected intersections cleared, already! */
		    continue;

		/* First sort out all cases where one endpoint is
		 * attributed to more than one intersecting
		 * plane. Only one of those can be 'the real'
		 * intersection of the edge with the surface of the
		 * shadow volume surface. The others are intersections
		 * with the planes defining the volumes, but outside
		 * the actual faces.
		 *
		 * This cuts down the 27 to 13 cases (000, 001, 002,
		 * 012, and their permutations). Start with v1: */
		switch (full_class & makeclass(1, 1, 1)) {
		case makeclass(0,1,1):
		    /* v1 is out 'side1' and infront of 'p' */
		    if (front_hit < hit1)
			classification[3] = 0;
		    else
			classification[side1] = 0;
		    break;

		case makeclass(1,0,1):
		    /* v1 is out 'side2' and infront of 'p' */
		    if (front_hit < hit2)
			classification[3] = 0;
		    else
			classification[side2] = 0;
		    break;

		    /* The following two cases never trigger, even in
		     * a full run through all.dem. Neither do the
		     * corresponding 2 cases for the vertex2
		     * classification. I think this is so because the
		     * determination of 'side1' and 'side2' already
		     * took care of this: if the edge vertex is out
		     * two sides, those two will _not_ be chosen as
		     * 'side1' and 'side2', because one of them is
		     * completely on one side of the edge.  */
		case makeclass(1,1,0):
		    /* v1 out both sides, but behind the front */
		    if (hit1 < hit2)
			/* hit2 is closer to the hidden vertex v2, so
			 * it's the relevant intersection: */
			classification[side1] = 0;
		    else
			classification[side2] = 0;
		    break;

		case makeclass(1,1,1):
		    /* v1 out both sides, and in front of p (--> v2 is
		     * hidden) */
		    if (front_hit < hit1) {
			classification[3] = 0;
			if (hit1 < hit2)
			    classification[side1] = 0;
			else
			    classification[side2] = 0;
		    } else {
			classification[side1] = 0;
			if (front_hit < hit2)
			    classification[3] = 0;
			else
			    classification[side2] = 0;
		    }
		    break;

		default:
		    ; /* do nothing */
		} /* switch(v1 classes) */


		/* And the same, for v2 */
		switch (full_class & makeclass(2, 2, 2)) {
		case makeclass(0,2,2):
		    /* v2 is out 'side1' and infront of 'p' */
		    if (front_hit > hit1)
			classification[3] = 0;
		    else
			classification[side1] = 0;
		    break;

		case makeclass(2,0,2):
		    /* v2 out side 'side2' and infront of 'p' */
		    if (front_hit > hit2)
			classification[3] = 0;
		    else
			classification[side2] = 0;
		    break;

		    /* See note in v1 handling about these two cases
		     * being impossible... */
		case makeclass(2,2,0):
		    /* v2 out both sides, but behind the front */
		    if (hit1 > hit2)
			/* hit2 is closer to the hidden vertex v2, so
			 * it's the relevant intersection: */
			classification[side1] = 0;
		    else
			classification[side2] = 0;
		    break;

		case makeclass(2,2,2):
		    /* v2 out both sides, and in front of p (--> v1 is
		     * hidden) */
		    if (front_hit > hit1) {
			classification[3] = 0;
			if (hit1 > hit2)
			    classification[side1] = 0;
			else
			    classification[side2] = 0;
		    } else {
			classification[side1] = 0;
			if (front_hit > hit2)
			    classification[3] = 0;
			else
			    classification[side2] = 0;
		    }
		    break;

		default:
		    ; /* do nothing */
		} /* switch(v2 classes) */


		/* recalculate full_class after all these possible changes: */
		full_class = makeclass(classification[side2],
				       classification[side1],
				       classification[3]);

		/* This monster switch catches all the 13 remaining cases
		 * individually. */
		switch (full_class) {
		default:
		    printf("in_front: (T9) class %d is illegal. Should never happen.\n",
			   full_class);
		    break;
		case makeclass(0,0,0):
		    /* can happen, by resetting of classification bits
		     * inside this test */
		    break;

/*--------- The simplest cases first: only one intersection point -----*/
		    /* Code is the same for all six cases, or nearly so: */
#define handle_singleplane_hit(vert, hitpar)				    \
		    newvert[0] = split_line_at_ratio(vnum1, vnum2, hitpar); \
		    setup_edge(vert, newvert[0]);			    \
		    break;

		case makeclass(0,0,1): /* v1 is in front of poly: */
		    handle_singleplane_hit(vnum1, front_hit);
		case makeclass(0,0,2): /* v2 is in front of poly: */
		    handle_singleplane_hit(vnum2, front_hit);
		case makeclass(0,1,0): /* v1 is out side1: */
		    handle_singleplane_hit(vnum1,hit1);
		case makeclass(0,2,0): /* v2 is out side1: */
		    handle_singleplane_hit(vnum2,hit1);
		case makeclass(1,0,0): /* v1 is out side2 */
		    handle_singleplane_hit(vnum1,hit2);
		case makeclass(2,0,0): /* v2 is out side2 */
		    handle_singleplane_hit(vnum2,hit2);

/*----------- cases with 2 certain intersections: --------------*/
		case makeclass(1,2,0):
		case makeclass(2,1,0):
		    /* Both behind the front, v1 out one side, v2 out
		     * the other.  These two cases can be handled by
		     * common code: */
		    newvert[0] = split_line_at_ratio(vnum1, vnum2, hit1);
		    newvert[1] = split_line_at_ratio(vnum1, vnum2, hit2);
		    if (hit1 < hit2) {
			/* the fragments are v1--hit1 and hit2--v2 */
			handle_edge_fragment(edgenum, newvert[1], vnum2, polynum);
			setup_edge(vnum1, newvert[0]);
		    } else {
			/* the fragments are v2--hit1 and hit2--v1 */
			handle_edge_fragment(edgenum, vnum1, newvert[1], polynum);
			setup_edge(newvert[0], vnum2);
		    }
		    break;

/*----------- cases with either 2 or no intersections: --------------*/

		    /* Mainly identical code block to be used 4 times
		     * --> macro */
		    /* HBB 20001108: up until today, this part of the
		     * was severely buggy. But I do think I've got it
		     * fixed up, this time */
#define handle_outside_behind_vs_infront(hitvar1, hitvar2)			   \
		    /* vnum1 out plane of 'hitvar1' and in back, vnum2		   \
                     * in front */						   \
		    if ((hitvar1) < (hitvar2)) {				   \
			newvert[0] = split_line_at_ratio(vnum1, vnum2, (hitvar1)); \
			newvert[1] = split_line_at_ratio(vnum1, vnum2, (hitvar2)); \
			handle_edge_fragment(edgenum, newvert[1], vnum2, polynum); \
			setup_edge(vnum1, newvert[0]);				   \
		    }								   \
		    /* otherwise, the edge missed the shadow volume		   \
		     * --> nothing to do */					   \
		    break;

		case makeclass(0,1,2): /* v1 out side1 and in back, v2 in front */
		    handle_outside_behind_vs_infront(hit1, front_hit);

		case makeclass(0,2,1): /* v2 out side1 and in back, v1 in front */
		    handle_outside_behind_vs_infront(front_hit, hit1);

		case makeclass(1,0,2): /* v1 out side2 and in back, v2 in front */
		    handle_outside_behind_vs_infront(hit2, front_hit);

		case makeclass(2,0,1): /* v2 out side2 and in back, v1 in front */
		    handle_outside_behind_vs_infront(front_hit, hit2);

#undef handle_outside_behind_vs_infront

		} /* end of switch through the cases */

	    } /* end of part 'T9' */
	} /* for (polygons in list) */

    /* Came here, so there's something left of this edge, which needs
     * to be drawn.  But the vertices are different, now, so copy our
     * new vertices back into 'e' */

    draw_edge(elist + edgenum, vlist + vnum1, vlist + vnum2);

    return 1;
}


/* HBB 20000617: reimplemented this routine from scratch */
/* Externally callable function to draw a line, but hide it behind the
 * visible surface. */
/* NB: The p_vertex arguments are not allowed to be pointers into the
 * hidden3d 'vlist' structure. If they are, they may become invalid
 * before they're used, because of the nextfrom_dynarray() call. */
void
draw_line_hidden(
    p_vertex v1, p_vertex v2,	/* pointers to the end vertices */
    struct lp_style_type *lp)	/* line and point style to draw in */
{
    long int vstore1, vstore2;
    long int edgenum;
    long int temp_pfirst;

    /* If no polygons have been stored, nothing can be hidden, and we
     * can't use in_front() because the datastructures are partly
     * invalid. So just draw the line and be done with it */
    if (!polygons.end) {
	draw3d_line_unconditional(v1, v2, lp, lp->l_type);
	return;
    }

    /* Copy two vertices into hidden3d arrays: */
    nextfrom_dynarray(&vertices);
    vstore1 = vertices.end - 1;
    vlist[vstore1] = *v1;
    if (v2) {
	vlist[vstore1].lp_style = NULL;
	nextfrom_dynarray(&vertices);
	vstore2 = vertices.end - 1;
	vlist[vstore2] = *v2;
	vlist[vstore2].lp_style = NULL;
    } else {
	/* v2 == NULL --> this is a point symbol to be drawn. Make two
	 * vertex pointers the same, and set up the 'style' field */
	vstore2 = vstore1;
	vlist[vstore2].lp_style = lp;
    }

    /* store the edge into the hidden3d datastructures */
    edgenum = make_edge(vstore1, vstore2, lp, lp->l_type, -1);

    /* remove hidden portions of the line, and draw what remains */
    temp_pfirst = pfirst;
    in_front(edgenum, elist[edgenum].v1, elist[edgenum].v2, &temp_pfirst);

    /* release allocated storage slots: */
    droplast_dynarray(&edges);
    droplast_dynarray(&vertices);
    if (v2)
	droplast_dynarray(&vertices);
}

/***********************************************************************
 * and, finally, the 'mother function' that uses all these lots of tools
 ***********************************************************************/
void
plot3d_hidden(struct surface_points *plots, int pcount)
{
    /* make vertices, edges and polygons out of all the plots */
    build_networks(plots, pcount);

    if (! edges.end) {
	/* No drawable edges found. Free all storage and bail out. */
	term_hidden_line_removal();
	graph_error("*All* edges undefined or out of range, thus no plot.");
    }

    if (! polygons.end) {
	/* No polygons anything could be hidden behind... */

	sort_edges_by_z();
	while (efirst >= 0) {
	    draw_edge(elist+efirst, vlist + elist[efirst].v1, vlist + elist[efirst].v2);
	    efirst = elist[efirst].next;
	}
    } else {
	long int temporary_pfirst;

	/* Presort edges in z order */
	sort_edges_by_z();
	/* Presort polygons in z order */
	sort_polys_by_z();

	temporary_pfirst = pfirst;

	while (efirst >=0) {
	    if (elist[efirst].style >= -2) /* skip invisible edges */
		in_front(efirst, elist[efirst].v1, elist[efirst].v2,
			 &temporary_pfirst);
	    efirst = elist[efirst].next;
	}
    }

    /* Free memory */
    /* FIXME: anything to free? */
}

void
reset_hidden3doptions()
{
    hiddenBacksideLinetypeOffset = BACKSIDE_LINETYPE_OFFSET;
    hiddenTriangleLinesdrawnPattern = TRIANGLE_LINESDRAWN_PATTERN;
    hiddenHandleUndefinedPoints = HANDLE_UNDEFINED_POINTS;
    hiddenShowAlternativeDiagonal = SHOW_ALTERNATIVE_DIAGONAL;
    hiddenHandleBentoverQuadrangles = HANDLE_BENTOVER_QUADRANGLES;
}

/* Emacs editing help for HBB:
 * Local Variables: ***
 * c-basic-offset: 4 ***
 * End: ***
 */
