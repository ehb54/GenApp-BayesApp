/*
 * $Id: plot2d.h,v 1.12 2005/03/02 20:35:35 sfeam Exp $
 */

/* GNUPLOT - plot2d.h */

/*[
 * Copyright 1999, 2004   Thomas Williams, Colin Kelley
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

#ifndef GNUPLOT_PLOT2D_H
# define GNUPLOT_PLOT2D_H

#include "syscfg.h"

/* Variables of plot2d.c needed by other modules: */

extern struct curve_points *first_plot;

extern double boxwidth;
extern TBOOLEAN boxwidth_is_absolute;

/* prototypes from plot2d.c */

void plotrequest __PROTO((void));
void cp_free __PROTO((struct curve_points *cp));
void cp_extend __PROTO((struct curve_points *cp, int num));

#ifdef EAM_DATASTRINGS
#include "gp_types.h"
#include "gadgets.h"
void store_label __PROTO((struct text_label *, struct coordinate *, int i,
                          char * string, double colorval));
#endif

#endif /* GNUPLOT_PLOT2D_H */
