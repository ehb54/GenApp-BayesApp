/*
 * $Id: plot.h,v 1.43.2.1 2006/11/10 22:49:37 tlecomte Exp $
 */

/* GNUPLOT - plot.h */

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

#ifndef GNUPLOT_PLOT_H
# define GNUPLOT_PLOT_H

/* #if... / #include / #define collection: */

#include "syscfg.h"
#include "gp_types.h"

#ifdef USE_MOUSE
# include "mouse.h"
#endif

/* Type definitions */

/* Variables of plot.c needed by other modules: */

extern TBOOLEAN interactive;
extern TBOOLEAN persist_cl;

extern const char *user_shell;
#if defined(ATARI) || defined(MTOS)
extern const char *user_gnuplotpath;
#endif

#ifdef OS2
extern TBOOLEAN CallFromRexx;
#endif

/* Prototypes of functions exported by plot.c */

void bail_to_command_line __PROTO((void));
void interrupt_setup __PROTO((void));
void gp_expand_tilde __PROTO((char **));
void get_user_env __PROTO((void));

#ifdef LINUXVGA
void drop_privilege __PROTO((void));
void take_privilege __PROTO((void));
#endif /* LINUXVGA */

#ifdef OS2
int ExecuteMacro __PROTO((char *, int));
#endif

#endif /* GNUPLOT_PLOT_H */
