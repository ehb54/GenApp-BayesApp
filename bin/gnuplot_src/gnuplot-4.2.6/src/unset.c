#ifndef lint
static char *RCSid() { return RCSid("$Id: unset.c,v 1.99.2.7 2008/12/15 03:44:22 sfeam Exp $"); }
#endif

/* GNUPLOT - unset.c */

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

#include "setshow.h"

#include "axis.h"
#include "command.h"
#include "contour.h"
#include "datafile.h"
#include "fit.h"
#include "gp_hist.h"
#include "hidden3d.h"
#include "misc.h"
#include "parse.h"
#include "plot.h"
#include "plot2d.h"
#include "plot3d.h"
#include "tables.h"
#include "term_api.h"
#include "util.h"
#include "variable.h"
#include "pm3d.h"
#ifdef BINARY_DATA_FILE
#  include "datafile.h"
#endif

static void unset_angles __PROTO((void));
static void unset_arrow __PROTO((void));
static void unset_arrowstyles __PROTO((void));
static void free_arrowstyle __PROTO((struct arrowstyle_def *));
static void delete_arrow __PROTO((struct arrow_def *, struct arrow_def *));
static void unset_autoscale __PROTO((void));
static void unset_bars __PROTO((void));
static void unset_border __PROTO((void));

static void unset_boxwidth __PROTO((void));
static void unset_fillstyle __PROTO((void));
static void unset_clabel __PROTO((void));
static void unset_clip __PROTO((void));
static void unset_cntrparam __PROTO((void));
static void unset_contour __PROTO((void));
static void unset_dgrid3d __PROTO((void));
static void unset_dummy __PROTO((void));
static void unset_encoding __PROTO((void));
static void unset_decimalsign __PROTO((void));
static void unset_fit __PROTO((void));
static void unset_format __PROTO((void));
static void unset_grid __PROTO((void));
static void unset_hidden3d __PROTO((void));
#ifdef EAM_HISTOGRAMS
static void unset_histogram __PROTO((void));
#endif
static void unset_historysize __PROTO((void));
static void unset_isosamples __PROTO((void));
static void unset_key __PROTO((void));
static void unset_keytitle __PROTO((void));
static void unset_label __PROTO((void));
static void delete_label __PROTO((struct text_label * prev, struct text_label * this));
#ifdef EAM_OBJECTS
static void unset_object __PROTO((void));
static void delete_object __PROTO((struct object * prev, struct object * this));
static void unset_style_rectangle __PROTO(());
#endif
static void unset_loadpath __PROTO((void));
static void unset_locale __PROTO((void));
static void reset_logscale __PROTO((AXIS_INDEX));
static void unset_logscale __PROTO((void));
#ifdef GP_MACROS
static void unset_macros __PROTO((void));
#endif
static void unset_mapping __PROTO((void));
static void unset_margin __PROTO((t_position *));
static void unset_missing __PROTO((void));
#ifdef USE_MOUSE
static void unset_mouse __PROTO((void));
#endif
#if 0
static void unset_multiplot __PROTO((void));
#endif

static void unset_month_day_tics __PROTO((AXIS_INDEX));
static void unset_minitics __PROTO((AXIS_INDEX));
#ifdef OLDUNSETTICS
static void unset_tics_in __PROTO((void));
#endif /* OLDUNSETTICS */

static void unset_offsets __PROTO((void));
static void unset_origin __PROTO((void));
static void unset_output __PROTO((void));
static void unset_parametric __PROTO((void));
static void unset_pm3d __PROTO((void));
static void unset_palette __PROTO((void));
static void reset_colorbox __PROTO((void));
static void unset_colorbox __PROTO((void));
static void unset_pointsize __PROTO((void));
static void unset_polar __PROTO((void));
static void unset_print __PROTO((void));
static void unset_samples __PROTO((void));
static void unset_size __PROTO((void));
static void unset_style __PROTO((void));
static void unset_surface __PROTO((void));
static void unset_table __PROTO((void));
static void unset_terminal __PROTO((void));
static void unset_tics __PROTO((AXIS_INDEX));
static void unset_ticscale __PROTO((void));
static void unset_ticslevel __PROTO((void));
static void unset_timefmt __PROTO((void));
static void unset_timestamp __PROTO((void));
static void unset_view __PROTO((void));
static void unset_zero __PROTO((void));
static void unset_timedata __PROTO((AXIS_INDEX));
static void unset_range __PROTO((AXIS_INDEX));
static void unset_zeroaxis __PROTO((AXIS_INDEX));
static void unset_all_zeroaxes __PROTO((void));

static void unset_axislabel_or_title __PROTO((text_label *));
static void unset_axislabel __PROTO((AXIS_INDEX));

/******** The 'unset' command ********/
void
unset_command()
{
    static char GPFAR unsetmess[] =
    "valid unset options:  [] = choose one, {} means optional\n\n\
\t'angles', 'arrow', 'autoscale', 'bar', 'border', 'boxwidth', 'clabel',\n\
\t'clip', 'cntrparam', 'colorbox', 'contour', 'dgrid3d', 'decimalsign',\n\
\t'dummy', 'encoding', 'fit', 'format', 'grid', 'hidden3d', 'historysize',\n\
\t'isosamples', 'key', 'label', 'loadpath', 'locale', 'logscale',\n\
\t'[blrt]margin', 'mapping', 'missing', 'mouse', 'multiplot', 'offsets',\n\
\t'origin', 'output', 'palette', 'parametric', 'pm3d', 'pointsize',\n\
\t'polar', 'print', '[rtuv]range', 'samples', 'size', 'style', 'surface',\n\
\t'terminal', 'tics', 'ticscale', 'ticslevel', 'timestamp', 'timefmt',\n\
\t'title', 'view', '[xyz,cb]{2}data', '[xyz,cb]{2}label',\n\
\t'[xyz,cb]{2}range', '{m}[xyz,cb]{2}tics', '[xyz,cb]{2}[md]tics',\n\
\t'{[xyz]{2}}zeroaxis', 'zero'";

    int found_token;

    c_token++;

    found_token = lookup_table(&set_tbl[0],c_token);

    /* HBB 20000506: rationalize occurences of c_token++ ... */
    if (found_token != S_INVALID)
	c_token++;

    switch(found_token) {
    case S_ANGLES:
	unset_angles();
	break;
    case S_ARROW:
	unset_arrow();
	break;
    case S_AUTOSCALE:
	unset_autoscale();
	break;
    case S_BARS:
	unset_bars();
	break;
    case S_BORDER:
	unset_border();
	break;
    case S_BOXWIDTH:
	unset_boxwidth();
	break;
    case S_CLABEL:
	unset_clabel();
	break;
    case S_CLIP:
	unset_clip();
	break;
    case S_CNTRPARAM:
	unset_cntrparam();
	break;
    case S_CONTOUR:
	unset_contour();
	break;
    case S_DGRID3D:
	unset_dgrid3d();
	break;
    case S_DUMMY:
	unset_dummy();
	break;
    case S_ENCODING:
	unset_encoding();
	break;
    case S_DECIMALSIGN:
	unset_decimalsign();
	break;
    case S_FIT:
	unset_fit();
	break;
    case S_FORMAT:
	unset_format();
	break;
    case S_GRID:
	unset_grid();
	break;
    case S_HIDDEN3D:
	unset_hidden3d();
	break;
    case S_HISTORYSIZE:
	unset_historysize();
	break;
    case S_ISOSAMPLES:
	unset_isosamples();
	break;
    case S_KEY:
	unset_key();
	break;
    case S_KEYTITLE:
	unset_keytitle();
	break;
    case S_LABEL:
	unset_label();
	break;
    case S_LOADPATH:
	unset_loadpath();
	break;
    case S_LOCALE:
	unset_locale();
	break;
    case S_LOGSCALE:
	unset_logscale();
	break;
#ifdef GP_MACROS
    case S_MACROS:
	unset_macros();
	break;
#endif
    case S_MAPPING:
	unset_mapping();
	break;
    case S_BMARGIN:
	unset_margin(&bmargin);
	break;
    case S_LMARGIN:
	unset_margin(&lmargin);
	break;
    case S_RMARGIN:
	unset_margin(&rmargin);
	break;
    case S_TMARGIN:
	unset_margin(&tmargin);
	break;
    case S_DATAFILE:
	if (almost_equals(c_token,"fort$ran")) {
	    df_fortran_constants = FALSE;
	    c_token++;
	    break;
	}
	df_fortran_constants = FALSE;
	unset_missing();
	df_separator = '\0';
	free(df_commentschars);
	df_commentschars = gp_strdup(DEFAULT_COMMENTS_CHARS);
#ifdef BINARY_DATA_FILE
	df_unset_datafile_binary();
#endif
	break;
#ifdef USE_MOUSE
    case S_MOUSE:
	unset_mouse();
	break;
#endif
    case S_MULTIPLOT:
/*	unset_multiplot(); */
	term_end_multiplot();
	break;
    case S_OFFSETS:
	unset_offsets();
	break;
    case S_ORIGIN:
	unset_origin();
	break;
    case SET_OUTPUT:
	unset_output();
	break;
    case S_PARAMETRIC:
	unset_parametric();
	break;
    case S_PM3D:
	unset_pm3d();
	break;
    case S_PALETTE:
	unset_palette();
	break;
    case S_COLORBOX:
	unset_colorbox();
	break;
    case S_POINTSIZE:
	unset_pointsize();
	break;
    case S_POLAR:
	unset_polar();
	break;
    case S_PRINT:
	unset_print();
	break;
#ifdef EAM_OBJECTS
    case S_OBJECT:
	unset_object();
	break;
#endif
    case S_SAMPLES:
	unset_samples();
	break;
    case S_SIZE:
	unset_size();
	break;
    case S_STYLE:
	unset_style();
	break;
    case S_SURFACE:
	unset_surface();
	break;
    case S_TABLE:
	unset_table();
	break;
    case S_TERMINAL:
	unset_terminal();
	break;
    case S_TICS:
#ifdef OLDUNSETTICS
	unset_tics_in();
#else
	unset_tics(AXIS_ARRAY_SIZE);
#endif /* OLDUNSETTICS */
	break;
    case S_TICSCALE:
	unset_ticscale();
	break;
    case S_TICSLEVEL:
	unset_ticslevel();
	break;
    case S_TIMEFMT:
	unset_timefmt();
	break;
    case S_TIMESTAMP:
	unset_timestamp();
	break;
    case S_TITLE:
	unset_axislabel_or_title(&title);
	break;
    case S_VIEW:
	unset_view();
	break;
    case S_ZERO:
	unset_zero();
	break;
/* FIXME - are the tics correct? */
    case S_MXTICS:
	unset_minitics(FIRST_X_AXIS);
	break;
    case S_XTICS:
	unset_tics(FIRST_X_AXIS);
	break;
    case S_XDTICS:
    case S_XMTICS:
	unset_month_day_tics(FIRST_X_AXIS);
	break;
    case S_MYTICS:
	unset_minitics(FIRST_Y_AXIS);
	break;
    case S_YTICS:
	unset_tics(FIRST_Y_AXIS);
	break;
    case S_YDTICS:
    case S_YMTICS:
	unset_month_day_tics(FIRST_X_AXIS);
	break;
    case S_MX2TICS:
	unset_minitics(SECOND_X_AXIS);
	break;
    case S_X2TICS:
	unset_tics(SECOND_X_AXIS);
	break;
    case S_X2DTICS:
    case S_X2MTICS:
	unset_month_day_tics(FIRST_X_AXIS);
	break;
    case S_MY2TICS:
	unset_minitics(SECOND_Y_AXIS);
	break;
    case S_Y2TICS:
	unset_tics(SECOND_Y_AXIS);
	break;
    case S_Y2DTICS:
    case S_Y2MTICS:
	unset_month_day_tics(FIRST_X_AXIS);
	break;
    case S_MZTICS:
	unset_minitics(FIRST_Z_AXIS);
	break;
    case S_ZTICS:
	unset_tics(FIRST_Z_AXIS);
	break;
    case S_ZDTICS:
    case S_ZMTICS:
	unset_month_day_tics(FIRST_X_AXIS);
	break;
    case S_MCBTICS:
	unset_minitics(COLOR_AXIS);
	break;
    case S_CBTICS:
	unset_tics(COLOR_AXIS);
	break;
    case S_CBDTICS:
    case S_CBMTICS:
	unset_month_day_tics(FIRST_X_AXIS);
	break;
    case S_XDATA:
	unset_timedata(FIRST_X_AXIS);
	/* FIXME HBB 20000506: does unsetting these axes make *any*
	 * sense?  After all, their content is never displayed, so
	 * what would they need a corrected format for? */
	unset_timedata(T_AXIS);
	unset_timedata(U_AXIS);
	break;
    case S_YDATA:
	unset_timedata(FIRST_Y_AXIS);
	/* FIXME: see above */
	unset_timedata(V_AXIS);
	break;
    case S_ZDATA:
	unset_timedata(FIRST_Z_AXIS);
	break;
    case S_CBDATA:
	unset_timedata(COLOR_AXIS);
	break;
    case S_X2DATA:
	unset_timedata(SECOND_X_AXIS);
	break;
    case S_Y2DATA:
	unset_timedata(SECOND_Y_AXIS);
	break;
    case S_XLABEL:
	unset_axislabel(FIRST_X_AXIS);
	break;
    case S_YLABEL:
	unset_axislabel(FIRST_Y_AXIS);
	break;
    case S_ZLABEL:
	unset_axislabel(FIRST_Z_AXIS);
	break;
    case S_CBLABEL:
	unset_axislabel(COLOR_AXIS);
	break;
    case S_X2LABEL:
	unset_axislabel(SECOND_X_AXIS);
	break;
    case S_Y2LABEL:
	unset_axislabel(SECOND_Y_AXIS);
	break;
    case S_XRANGE:
	unset_range(FIRST_X_AXIS);
	break;
    case S_X2RANGE:
	unset_range(SECOND_X_AXIS);
	break;
    case S_YRANGE:
	unset_range(FIRST_Y_AXIS);
	break;
    case S_Y2RANGE:
	unset_range(SECOND_Y_AXIS);
	break;
    case S_ZRANGE:
	unset_range(FIRST_Z_AXIS);
	break;
    case S_CBRANGE:
	unset_range(COLOR_AXIS);
	break;
    case S_RRANGE:
	unset_range(R_AXIS);
	break;
    case S_TRANGE:
	unset_range(T_AXIS);
	break;
    case S_URANGE:
	unset_range(U_AXIS);
	break;
    case S_VRANGE:
	unset_range(V_AXIS);
	break;
    case S_XZEROAXIS:
	unset_zeroaxis(FIRST_X_AXIS);
	break;
    case S_YZEROAXIS:
	unset_zeroaxis(FIRST_Y_AXIS);
	break;
    case S_ZZEROAXIS:
	unset_zeroaxis(FIRST_Z_AXIS);
	break;
    case S_X2ZEROAXIS:
	unset_zeroaxis(SECOND_X_AXIS);
	break;
    case S_Y2ZEROAXIS:
	unset_zeroaxis(SECOND_Y_AXIS);
	break;
    case S_ZEROAXIS:
	unset_all_zeroaxes();
	break;
    case S_INVALID:
    default:
	int_error(c_token, unsetmess);
	break;
    }
    update_gpval_variables(0); /* update GPVAL_ inner variables */
}


/* process 'unset angles' command */
static void
unset_angles()
{
    ang2rad = 1.0;
}


/* process 'unset arrow' command */
static void
unset_arrow()
{
    struct value a;
    struct arrow_def *this_arrow;
    struct arrow_def *prev_arrow;
    int tag;

    if (END_OF_COMMAND) {
	/* delete all arrows */
	while (first_arrow != NULL)
	    delete_arrow((struct arrow_def *) NULL, first_arrow);
    } else {
	/* get tag */
	tag = (int) real(const_express(&a));
	if (!END_OF_COMMAND)
	    int_error(c_token, "extraneous arguments to unset arrow");
	for (this_arrow = first_arrow, prev_arrow = NULL;
	     this_arrow != NULL;
	     prev_arrow = this_arrow, this_arrow = this_arrow->next) {
	    if (this_arrow->tag == tag) {
		delete_arrow(prev_arrow, this_arrow);
		return;		/* exit, our job is done */
	    }
	}
	int_error(c_token, "arrow not found");
    }
}


/* delete arrow from linked list started by first_arrow.
 * called with pointers to the previous arrow (prev) and the
 * arrow to delete (this).
 * If there is no previous arrow (the arrow to delete is
 * first_arrow) then call with prev = NULL.
 */
static void
delete_arrow(struct arrow_def *prev, struct arrow_def *this)
{
    if (this != NULL) {		/* there really is something to delete */
	if (prev != NULL)	/* there is a previous arrow */
	    prev->next = this->next;
	else			/* this = first_arrow so change first_arrow */
	    first_arrow = this->next;
	free(this);
    }
}

/* delete the whole list of arrow styles */
static void
unset_arrowstyles()
{
    free_arrowstyle(first_arrowstyle);
    first_arrowstyle = NULL;
}

static void
free_arrowstyle(struct arrowstyle_def *arrowstyle)
{
    if (arrowstyle) {
	free_arrowstyle(arrowstyle->next);
	free(arrowstyle);
    }
}

/* process 'unset autoscale' command */
static void
unset_autoscale()
{
    if (END_OF_COMMAND) {
	INIT_AXIS_ARRAY(set_autoscale, FALSE);
    } else if (equals(c_token, "xy") || equals(c_token, "tyx")) {
	axis_array[FIRST_X_AXIS].set_autoscale
	    = axis_array[FIRST_Y_AXIS].set_autoscale = AUTOSCALE_NONE;
	c_token++;
    } else {
	/* HBB 20000506: parse axis name, and unset the right element
	 * of the array: */
	int axis = lookup_table(axisname_tbl, c_token);
	if (axis >= 0) {
	    axis_array[axis].set_autoscale = AUTOSCALE_NONE;
	c_token++;
	}
    }
}


/* process 'unset bars' command */
static void
unset_bars()
{
    bar_size = 0.0;
}


/* process 'unset border' command */
static void
unset_border()
{
    /* this is not the effect as with reset, as the border is enabled,
     * by default */
    draw_border = 0;
}


/* process 'unset boxwidth' command */
static void
unset_boxwidth()
{
    boxwidth = -1.0;
    boxwidth_is_absolute = TRUE;
}


/* process 'unset fill' command */
static void
unset_fillstyle()
{
    default_fillstyle.fillstyle = FS_EMPTY;
    default_fillstyle.filldensity = 100;
    default_fillstyle.fillpattern = 0;
    default_fillstyle.border_linetype = LT_UNDEFINED;
}


/* process 'unset clabel' command */
static void
unset_clabel()
{
    /* FIXME? reset_command() uses TRUE */
    label_contours = FALSE;
}


/* process 'unset clip' command */
static void
unset_clip()
{
    if (END_OF_COMMAND) {
	/* same as all three */
	clip_points = FALSE;
	clip_lines1 = FALSE;
	clip_lines2 = FALSE;
    } else if (almost_equals(c_token, "p$oints"))
	clip_points = FALSE;
    else if (almost_equals(c_token, "o$ne"))
	clip_lines1 = FALSE;
    else if (almost_equals(c_token, "t$wo"))
	clip_lines2 = FALSE;
    else
	int_error(c_token, "expecting 'points', 'one', or 'two'");
    c_token++;
}


/* process 'unset cntrparam' command */
static void
unset_cntrparam()
{
    contour_pts = DEFAULT_NUM_APPROX_PTS;
    contour_kind = CONTOUR_KIND_LINEAR;
    contour_order = DEFAULT_CONTOUR_ORDER;
    contour_levels = DEFAULT_CONTOUR_LEVELS;
    contour_levels_kind = LEVELS_AUTO;
}


/* process 'unset contour' command */
static void
unset_contour()
{
    draw_contour = CONTOUR_NONE;
}


/* process 'unset dgrid3d' command */
static void
unset_dgrid3d()
{
    dgrid3d_row_fineness = 10;
    dgrid3d_col_fineness = 10;
    dgrid3d_norm_value = 1;
    dgrid3d = FALSE;
}


/* process 'unset dummy' command */
static void
unset_dummy()
{
    strcpy(set_dummy_var[0], "x");
    strcpy(set_dummy_var[1], "y");
}


/* process 'unset encoding' command */
static void
unset_encoding()
{
    encoding = S_ENC_DEFAULT;
}


/* process 'unset decimalsign' command */
static void
unset_decimalsign()
{
    if (decimalsign != NULL)
	free(decimalsign);
    decimalsign = NULL;
#ifdef HAVE_LOCALE_H
    setlocale(LC_NUMERIC,"C");
#endif
}


/* process 'unset fit' command */
static void
unset_fit()
{
    if (fitlogfile != NULL)
	free(fitlogfile);
    fitlogfile = NULL;
#if GP_FIT_ERRVARS
    fit_errorvariables = FALSE;
#endif /* GP_FIT_ERRVARS */
}


/* process 'unset format' command */
/* FIXME: compare and merge with set.c::set_format */
static void
unset_format()
{
    TBOOLEAN set_for_axis[AXIS_ARRAY_SIZE] = AXIS_ARRAY_INITIALIZER(FALSE);
    int axis;

    if ((axis = lookup_table(axisname_tbl, c_token)) >= 0) {
	set_for_axis[axis] = TRUE;
    } else if (equals(c_token,"xy") || equals(c_token,"yx")) {
	set_for_axis[FIRST_X_AXIS]
	    = set_for_axis[FIRST_Y_AXIS]
	    = TRUE;
	c_token++;
    } else if (isstring(c_token) || END_OF_COMMAND) {
	/* Assume he wants all */
	for (axis = 0; axis < AXIS_ARRAY_SIZE; axis++)
	    set_for_axis[axis] = TRUE;
    }

    if (END_OF_COMMAND) {
	SET_DEFFORMAT(FIRST_X_AXIS , set_for_axis);
	SET_DEFFORMAT(FIRST_Y_AXIS , set_for_axis);
	SET_DEFFORMAT(FIRST_Z_AXIS , set_for_axis);
	SET_DEFFORMAT(SECOND_X_AXIS, set_for_axis);
	SET_DEFFORMAT(SECOND_Y_AXIS, set_for_axis);
	SET_DEFFORMAT(COLOR_AXIS   , set_for_axis);
    }
}


/* process 'unset grid' command */
static void
unset_grid()
{
    /* FIXME HBB 20000506: there is no command to explicitly reset the
     * linetypes for major and minor gridlines. This function should
     * do that, maybe... */
    AXIS_INDEX i = 0;

    /* grid_selection = GRID_OFF; */
    for (; i < AXIS_ARRAY_SIZE; i++) {
	axis_array[i].gridmajor = FALSE;
	axis_array[i].gridminor = FALSE;
    }
}


/* process 'unset hidden3d' command */
static void
unset_hidden3d()
{
#ifdef LITE
    printf(" Hidden Line Removal Not Supported in LITE version\n");
#else
    hidden3d = FALSE;
#endif
}

#ifdef EAM_HISTOGRAMS
static void
unset_histogram()
{
    histogram_opts.type = HT_CLUSTERED;
    histogram_opts.gap = 2;
}
#endif

/* process 'unset historysize' command */
static void
unset_historysize()
{
#ifdef GNUPLOT_HISTORY
    gnuplot_history_size = -1; /* don't ever truncate the history. */
#else
    int_error(c_token, "Command 'unset historysize' requires history support.");
#endif
}


/* process 'unset isosamples' command */
static void
unset_isosamples()
{
    /* HBB 20000506: was freeing 2D data structures although
     * isosamples are only used by 3D plots. */

    sp_free(first_3dplot);
    first_3dplot = NULL;

    iso_samples_1 = ISO_SAMPLES;
    iso_samples_2 = ISO_SAMPLES;
}


void
reset_key()
{
    legend_key temp_key = DEFAULT_KEY_PROPS;
    memcpy(&keyT, &temp_key, sizeof(keyT));
}

/* process 'unset key' command */
static void
unset_key()
{
    legend_key *key = &keyT;
    key->visible = FALSE;
}


/* process 'unset keytitle' command */
static void
unset_keytitle()
{
    legend_key *key = &keyT;
    key->title[0] = '\0';	/* empty string */
}


/* process 'unset label' command */
static void
unset_label()
{
    struct value a;
    struct text_label *this_label;
    struct text_label *prev_label;
    int tag;

    if (END_OF_COMMAND) {
	/* delete all labels */
	while (first_label != NULL)
	    delete_label((struct text_label *) NULL, first_label);
    } else {
	/* get tag */
	tag = (int) real(const_express(&a));
	if (!END_OF_COMMAND)
	    int_error(c_token, "extraneous arguments to unset label");
	for (this_label = first_label, prev_label = NULL;
	     this_label != NULL;
	     prev_label = this_label, this_label = this_label->next) {
	    if (this_label->tag == tag) {
		delete_label(prev_label, this_label);
		return;		/* exit, our job is done */
	    }
	}
	/* int_warn(c_token, "label not found"); */
    }
}


/* delete label from linked list started by first_label.
 * called with pointers to the previous label (prev) and the
 * label to delete (this).
 * If there is no previous label (the label to delete is
 * first_label) then call with prev = NULL.
 */
static void
delete_label(struct text_label *prev, struct text_label *this)
{
    if (this != NULL) {		/* there really is something to delete */
	if (prev != NULL)	/* there is a previous label */
	    prev->next = this->next;
	else			/* this = first_label so change first_label */
	    first_label = this->next;
	if (this->text) free (this->text);
	if (this->font) free (this->font);
	free (this);
    }
}

#ifdef EAM_OBJECTS
/* process 'unset rectangle' command */
static void
unset_object()
{
    struct value a;
    struct object *this_object;
    struct object *prev_object;
    int tag;

    if (END_OF_COMMAND) {
	/* delete all objects */
	while (first_object != NULL)
	    delete_object((struct object *) NULL, first_object);
    } else {
	/* get tag */
	tag = (int) real(const_express(&a));
	if (!END_OF_COMMAND)
	    int_error(c_token, "extraneous arguments to unset rectangle");
	for (this_object = first_object, prev_object = NULL;
	     this_object != NULL;
	     prev_object = this_object, this_object = this_object->next) {
	    if (this_object->tag == tag) {
		delete_object(prev_object, this_object);
		return;		/* exit, our job is done */
	    }
	}
	int_error(c_token, "object not found");
    }
}


/* delete object from linked list started by first_object.
 * called with pointers to the previous object (prev) and the
 * object to delete (this).
 * If there is no previous object (the object to delete is
 * first_object) then call with prev = NULL.
 */
static void
delete_object(struct object *prev, struct object *this)
{
    if (this != NULL) {		/* there really is something to delete */
	if (prev != NULL)	/* there is a previous rectangle */
	    prev->next = this->next;
	else			/* this = first_object so change first_object */
	    first_object = this->next;
	/* FIXME:  Must free contents as well */
	free (this);
    }
}
#endif


/* process 'unset loadpath' command */
static void
unset_loadpath()
{
    clear_loadpath();
}


/* process 'unset locale' command */
static void
unset_locale()
{
    init_locale();
}

static void
reset_logscale(AXIS_INDEX axis)
{
    axis_array[axis].log = FALSE;
    axis_array[axis].base = 0.0;
}

/* process 'unset logscale' command */
static void
unset_logscale()
{
    int axis;

    if (END_OF_COMMAND) {
	/* clean all the islog flags. This will hit some currently
	 * unused ones, too, but that's actually a good thing, IMHO */
	for(axis = 0; axis < AXIS_ARRAY_SIZE; axis++)
	    reset_logscale(axis);
    } else {
	int i = 0;

	/* do reverse search because of "x", "x1", "x2" sequence in
	 * axisname_tbl */
	while (i < token[c_token].length) {
	    axis = lookup_table_nth_reverse(axisname_tbl, AXIS_ARRAY_SIZE,
					    gp_input_line + token[c_token].start_index + i);
	    if (axis < 0) {
		token[c_token].start_index += i;
		int_error(c_token, "unknown axis");
	    }
	    reset_logscale(axisname_tbl[axis].value);
	    i += strlen(axisname_tbl[axis].key);
	}
	++c_token;
    }
}

#ifdef GP_MACROS
/* process 'unset macros' command */
static void
unset_macros()
{
    expand_macros = FALSE;
}
#endif

/* process 'unset mapping3d' command */
static void
unset_mapping()
{
    /* assuming same as points */
    mapping3d = MAP3D_CARTESIAN;
}

/* process 'unset {blrt}margin' command */
static void
unset_margin(t_position *margin)
{
    margin->scalex = character;
    margin->x = -1;
}

/* process 'unset datafile' command */
static void
unset_missing()
{
    free(missing_val);
    missing_val = NULL;
}

#ifdef USE_MOUSE
/* process 'unset mouse' command */
static void
unset_mouse()
{
    mouse_setting.on = 0;
#ifdef OS2
    PM_update_menu_items();
#endif
    UpdateStatusline(); /* wipe status line */
}
#endif

/* process 'unset mxtics' command */
static void
unset_minitics(AXIS_INDEX axis)
{
    axis_array[axis].minitics = MINI_OFF;
    axis_array[axis].mtic_freq = 10.0;
}


/*process 'unset {x|y|x2|y2|z}tics' command */
static void
unset_tics(AXIS_INDEX axis)
{
    struct position tics_nooffset = { character, character, character, 0., 0., 0.};
    unsigned int istart = 0;
    unsigned int iend = AXIS_ARRAY_SIZE;
    unsigned int i;

    if (axis < AXIS_ARRAY_SIZE) {
	istart = axis;
	iend = axis + 1;
    }
	
    for (i = istart; i < iend; ++i) {
	axis_array[i].ticmode = NO_TICS;

	if (axis_array[i].ticdef.font) {
	    free(axis_array[i].ticdef.font);
	    axis_array[i].ticdef.font = NULL;
	}
	axis_array[i].ticdef.textcolor.type = TC_DEFAULT;
	axis_array[i].ticdef.textcolor.lt = 0;
	axis_array[i].ticdef.textcolor.value = 0;
	axis_array[i].ticdef.offset = tics_nooffset;
	axis_array[i].ticdef.rangelimited = FALSE;
	axis_array[i].tic_rotate = 0;
	axis_array[i].ticscale = 1.0;
	axis_array[i].miniticscale = 0.5;
	axis_array[i].tic_in = TRUE;

	free_marklist(axis_array[i].ticdef.def.user);
	axis_array[i].ticdef.def.user = NULL;
    }
}

static void
unset_month_day_tics(AXIS_INDEX axis)
{
    axis_array[axis].ticdef.type = TIC_COMPUTED;
}

/* process 'unset offsets' command */
static void
unset_offsets()
{
    loff = roff = toff = boff = 0.0;
}


/* process 'unset origin' command */
static void
unset_origin()
{
    xoffset = 0.0;
    yoffset = 0.0;
}


/* process 'unset output' command */
static void
unset_output()
{
    if (multiplot) {
	int_error(c_token, "you can't change the output in multiplot mode");
	return;
    }

    term_set_output(NULL);
    if (outstr) {
	free(outstr);
	outstr = NULL; /* means STDOUT */
    }
}


/* process 'unset print' command */
static void
unset_print()
{
    print_set_output(NULL, FALSE);
}


/* process 'unset parametric' command */
static void
unset_parametric()
{
    if (parametric) {
	parametric = FALSE;
	if (!polar) { /* keep t for polar */
	    unset_dummy();
	    if (interactive)
		(void) fprintf(stderr,"\n\tdummy variable is x for curves, x/y for surfaces\n");
	}
    }
}

/* process 'unset palette' command */
static void
unset_palette()
{
    c_token++;
    fprintf(stderr, "you can't unset the palette.\n");
}


/* reset colorbox to default settings */
static void
reset_colorbox()
{
    color_box = default_color_box;
}


/* process 'unset colorbox' command: reset to default settings and then
 * switch it off */
static void
unset_colorbox()
{
    reset_colorbox();
    color_box.where = SMCOLOR_BOX_NO;
}


/* process 'unset pm3d' command */
static void
unset_pm3d()
{
    pm3d.implicit = PM3D_EXPLICIT;
    /* reset styles, required to 'plot something' after e.g. 'set pm3d map' */
    if (data_style == PM3DSURFACE) data_style = POINTSTYLE;
    if (func_style == PM3DSURFACE) func_style = LINES;
}


/* process 'unset pointsize' command */
static void
unset_pointsize()
{
    pointsize = 1.0;
}


/* process 'unset polar' command */
static void
unset_polar()
{
    if (polar) {
	polar = FALSE;
	if (parametric && axis_array[T_AXIS].set_autoscale) {
	    /* only if user has not set an explicit range */
	    axis_array[T_AXIS].set_min = axis_defaults[T_AXIS].min;
	    axis_array[T_AXIS].set_max = axis_defaults[T_AXIS].min;
	}
	if (!parametric) {
	    strcpy (set_dummy_var[0], "x");
	    if (interactive)
		(void) fprintf(stderr,"\n\tdummy variable is x for curves\n");
	}
    }
}


/* process 'unset samples' command */
static void
unset_samples()
{
    /* HBB 20000506: unlike unset_isosamples(), this one *has* to
     * clear 2D data structues! */
    cp_free(first_plot);
    first_plot = NULL;

    sp_free(first_3dplot);
    first_3dplot = NULL;

    samples_1 = SAMPLES;
    samples_2 = SAMPLES;
}


/* process 'unset size' command */
static void
unset_size()
{
    xsize = 1.0;
    ysize = 1.0;
    zsize = 1.0;
}


/* process 'unset style' command */
static void
unset_style()
{
    if (END_OF_COMMAND) {
	data_style = POINTSTYLE;
	func_style = LINES;
	while (first_linestyle != NULL)
	    delete_linestyle((struct linestyle_def *) NULL, first_linestyle);
	unset_fillstyle();
#ifdef EAM_OBJECTS
	unset_style_rectangle();
#endif
#ifdef EAM_HISTOGRAMS
	unset_histogram();
#endif
	c_token++;
	return;
    }

    switch(lookup_table(show_style_tbl, c_token)){
    case SHOW_STYLE_DATA:
	data_style = POINTSTYLE;
	c_token++;
	break;
    case SHOW_STYLE_FUNCTION:
	func_style = LINES;
	c_token++;
	break;
    case SHOW_STYLE_LINE:
	c_token++;
	if (END_OF_COMMAND) {
	    while (first_linestyle != NULL)
		delete_linestyle((struct linestyle_def *) NULL, first_linestyle);
	} else {
	    struct value a;
	    int tag = (int) real(const_express(&a));
	    struct linestyle_def *this, *prev;
	    for (this = first_linestyle, prev = NULL; this != NULL; 
		 prev = this, this = this->next) {
		if (this->tag == tag) {
		    delete_linestyle(prev, this);
		    break;
		}
	    }
	}
	break;
    case SHOW_STYLE_FILLING:
	unset_fillstyle();
	c_token++;
	break;
#ifdef EAM_HISTOGRAMS
    case SHOW_STYLE_HISTOGRAM:
	unset_histogram();
	c_token++;
	break;
#endif
    case SHOW_STYLE_ARROW:
	unset_arrowstyles();
	c_token++;
	break;
#ifdef EAM_OBJECTS
    case SHOW_STYLE_RECTANGLE:
	unset_style_rectangle();
	c_token++;
	break;
#endif
    default:
	int_error(c_token, "expecting 'data', 'function', 'line', 'fill' or 'arrow'");
    }
}


/* process 'unset surface' command */
static void
unset_surface()
{
    draw_surface = FALSE;
}


/* process 'unset table' command */
static void
unset_table()
{
    if (table_outfile)
	fclose(table_outfile);
    table_outfile = NULL;
    table_mode = FALSE;
}


/* process 'unset terminal' comamnd */
static void
unset_terminal()
{
    /* This is a problematic case */
/* FIXME */
    if (multiplot)
	int_error(c_token, "You can't change the terminal in multiplot mode");

    list_terms();
    screen_ok = FALSE;
}


#ifdef OLDUNSETTICS
/* process 'unset tics' command */
static void
unset_tics_in()
{
    unsigned int i = AXIS_ARRAY_SIZE;
    c_token++;

    if (equals(c_token,"x"))
	i = FIRST_X_AXIS;
    else if (equals(c_token,"y"))
	i = FIRST_Y_AXIS;
    else if (equals(c_token,"z"))
	i = FIRST_Z_AXIS;
    else if (equals(c_token,"x2"))
	i = SECOND_X_AXIS;
    else if (equals(c_token,"y2"))
	i = SECOND_Y_AXIS;
    else if (equals(c_token,"cb"))
	i = COLOR_AXIS;
    if (i < AXIS_ARRAY_SIZE)
	c_token++;

    if (i < AXIS_ARRAY_SIZE)
	axis_array[i].tic_in = TRUE;
    else {
	for (i = 0; i < AXIS_ARRAY_SIZE; ++i)
	    axis_array[i].tic_in = TRUE;
    }
}
#endif /* OLDUNSETTICS */


/* process 'unset ticscale' command */
static void
unset_ticscale()
{
    unsigned int i;

    int_warn(c_token,
	     "Deprecated syntax - please use 'set tics scale default'");

    for (i = 0; i < AXIS_ARRAY_SIZE; ++i) {
	axis_array[i].ticscale = 1.0;
	axis_array[i].miniticscale = 0.5;
    }
}


/* process 'unset ticslevel' command */
static void
unset_ticslevel()
{
    xyplane.ticslevel = 0.5;
    xyplane.absolute = FALSE;
}


/* Process 'unset timefmt' command */
static void
unset_timefmt()
{
    int axis;

    if (END_OF_COMMAND)
	for (axis=0; axis < AXIS_ARRAY_SIZE; axis++)
	    strcpy(axis_array[axis].timefmt,TIMEFMT);
    else if ((axis=lookup_table(axisname_tbl, c_token)) >= 0) {
	strcpy(axis_array[axis].timefmt, TIMEFMT);
	c_token++;
    }
    else
	/* int_error() from inside 'reset' would cause problems */
	int_warn(c_token, "expected optional axis name");

}


/* process 'unset timestamp' command */
static void
unset_timestamp()
{
    unset_axislabel_or_title(&timelabel);
    timelabel_rotate = 0;
    timelabel_bottom = TRUE;
}


/* process 'unset view' command */
static void
unset_view()
{
    splot_map_deactivate();
    splot_map = FALSE;
    aspect_ratio_3D = 0;
    surface_rot_z = 30.0;
    surface_rot_x = 60.0;
    surface_scale = 1.0;
    surface_zscale = 1.0;
}


/* process 'unset zero' command */
static void
unset_zero()
{
    zero = ZERO;
}

/* process 'unset {x|y|z|x2|y2}data' command */
static void
unset_timedata(AXIS_INDEX axis)
{
    axis_array[axis].is_timedata = FALSE;
}


/* process 'unset {x|y|z|x2|y2|t|u|v|r}range' command */
static void
unset_range(AXIS_INDEX axis)
{
    /* FIXME HBB 20000506: do we want to reset the axis autoscale and
     * min/max, too?  */
    axis_array[axis].range_flags = 0;
}

/* process 'unset {x|y|x2|y2|z}zeroaxis' command */
static void
unset_zeroaxis(AXIS_INDEX axis)
{
    axis_array[axis].zeroaxis = default_axis_zeroaxis;
}


/* process 'unset zeroaxis' command */
static void
unset_all_zeroaxes()
{
    AXIS_INDEX axis;

    for(axis = 0; axis < AXIS_ARRAY_SIZE; axis++)
	unset_zeroaxis(axis);
}


/* process 'unset [xyz]{2}label command */
static void
unset_axislabel_or_title(text_label *label)
{
    struct position default_offset = { character, character, character, 
				       0., 0., 0. };
    if (label) {
	free(label->text);
	label->text = NULL;
	free(label->font);
	label->font = NULL;
	label->offset = default_offset;
	label->textcolor.type = TC_DEFAULT;
    }
}

static void
unset_axislabel(AXIS_INDEX axis)
{
    unset_axislabel_or_title(&axis_array[axis].label);
    axis_array[axis].label = default_axis_label;
    if (axis == FIRST_Y_AXIS || axis == SECOND_Y_AXIS || axis == COLOR_AXIS)
	axis_array[axis].label.rotate = TEXT_VERTICAL;
}

/******** The 'reset' command ********/
/* HBB 20000506: I moved this here, from set.c, because 'reset' really
 * is more like a big lot of 'unset' commands, rather than a bunch of
 * 'set's. The benefit is that it can make use of many of the
 * unset_something() contained in this module, i.e. you now have one
 * place less to keep in sync if the semantics or defaults of any
 * option is changed. This is only true for options for which 'unset'
 * state is the default, however, e.g. not for 'surface', 'bars' and
 * some others. */
void
reset_command()
{
    AXIS_INDEX axis;
    TBOOLEAN save_interactive = interactive;
    static const TBOOLEAN set_for_axis[AXIS_ARRAY_SIZE]
	= AXIS_ARRAY_INITIALIZER(TRUE);

    c_token++;

    /* Kludge alert, HBB 20000506: set to noninteractive mode, to
     * suppress some of the commentary output by the individual
     * unset_...() routines. */
    interactive = FALSE;

    unset_samples();
    unset_isosamples();

    /* delete arrows */
    while (first_arrow != NULL)
	delete_arrow((struct arrow_def *) NULL, first_arrow);
    unset_arrowstyles();
    /* delete labels */
    while (first_label != NULL)
	delete_label((struct text_label *) NULL, first_label);
    /* delete linestyles */
    while (first_linestyle != NULL)
	delete_linestyle((struct linestyle_def *) NULL, first_linestyle);
#ifdef EAM_OBJECTS
    /* delete objects */
    while (first_object != NULL)
	delete_object((struct object *) NULL, first_object);
    unset_style_rectangle();
#endif

    /* 'polar', 'parametric' and 'dummy' are interdependent, so be
     * sure to keep the order intact */
    unset_polar();
    unset_parametric();
    unset_dummy();

    unset_axislabel_or_title(&title);

    reset_key();
    unset_keytitle();

    unset_timefmt();
    unset_view(); /* has to be called in advance to reset reversed yrange if splot_map_active */

    for (axis=0; axis<AXIS_ARRAY_SIZE; axis++) {
	SET_DEFFORMAT(axis, set_for_axis);
	unset_timedata(axis);
	unset_zeroaxis(axis);
	unset_range(axis);
	unset_axislabel(axis);

	axis_array[axis].set_autoscale = AUTOSCALE_BOTH;
	axis_array[axis].writeback_min = axis_array[axis].set_min
	    = axis_defaults[axis].min;
	axis_array[axis].writeback_max = axis_array[axis].set_max
	    = axis_defaults[axis].max;

	/* 'tics' default is on for some, off for the other axes: */
	unset_tics(axis);
	axis_array[axis].ticmode = axis_defaults[axis].ticmode;
	unset_minitics(axis);
	axis_array[axis].ticdef = default_axis_ticdef;
	axis_array[axis].minitics = MINI_DEFAULT;

	reset_logscale(axis);
    }

    unset_boxwidth();

    clip_points = FALSE;
    clip_lines1 = TRUE;
    clip_lines2 = FALSE;

    border_lp = default_border_lp;
    draw_border = 31;

    draw_surface = 1.0;

    data_style = POINTSTYLE;
    func_style = LINES;

    bar_size = 1.0;

    unset_grid();
    grid_lp = default_grid_lp;
    mgrid_lp = default_grid_lp;
    polar_grid_angle = 0;
    grid_layer = -1;

    reset_hidden3doptions();
    hidden3d = FALSE;

    label_contours = TRUE;
    strcpy(contour_format, "%8.3g");

    unset_angles();
    unset_mapping();

    unset_size();
    aspect_ratio = 0.0;		/* don't force it */

    unset_origin();
    unset_timestamp();
    unset_offsets();
    unset_contour();
    unset_cntrparam();
    unset_zero();
    unset_dgrid3d();
    unset_ticslevel();
#ifdef OLDUNSETTICS
    unset_tics_in();
#endif /* OLDUNSETTICS */
    unset_margin(&bmargin);
    unset_margin(&lmargin);
    unset_margin(&rmargin);
    unset_margin(&tmargin);
    unset_pointsize();
    unset_encoding();
    unset_decimalsign();
    pm3d_reset();
    reset_colorbox();
    reset_palette();
#ifdef BINARY_DATA_FILE
    df_unset_datafile_binary();
#endif
    unset_fillstyle();
#ifdef EAM_HISTOGRAMS
    unset_histogram();
#endif

    unset_missing();
    df_separator = '\0';
    free(df_commentschars);
    df_commentschars = gp_strdup(DEFAULT_COMMENTS_CHARS);

    unset_locale();
#if 0
    /* 2003-18-04: Don't reset non-graphics settings, like set term, out,
     * loadpath and fontpath. */
    unset_loadpath();
#endif
    unset_fit();

#if 0
    /* 01-Jun-2006: Deleting undefined user variables can break user functions.
     * E.g.    f(x) = a + b;  reset;
     * f(x) is still defined, sort of, but it holds pointers to a and b, which
     * have been deleted. Evaluation after a reset can trigger a segfault.
     */
    /* Garbage collection on space allocated for user variables */
    cleanup_udvlist();
#endif

    update_gpval_variables(0); /* update GPVAL_ inner variables */

    /* HBB 20000506: set 'interactive' back to its real value: */
    interactive = save_interactive;
}

#ifdef EAM_OBJECTS
static void
unset_style_rectangle()
{
    struct object foo = DEFAULT_RECTANGLE_STYLE;
    default_rectangle = foo;
    return;
}
#endif
