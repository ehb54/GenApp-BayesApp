/*
 * $Id: eval.h,v 1.26 2006/07/21 05:19:51 sfeam Exp $
 */

/* GNUPLOT - eval.h */

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

#ifndef GNUPLOT_EVAL_H
# define GNUPLOT_EVAL_H

/* #if... / #include / #define collection: */

#include "syscfg.h"
#include "gp_types.h"

#include <stdio.h>		/* for FILE* */

#define STACK_DEPTH 100		/* maximum size of the execution stack */
#define MAX_AT_LEN 150		/* max number of entries in action table */

/* Type definitions */

/* HBB 20010725: Moved here, from parse.h */
enum operators {
    /* keep this in line with table in eval.c */
    PUSH, PUSHC, PUSHD1, PUSHD2, PUSHD,
    CALL, CALLN, LNOT, BNOT, UMINUS,
    LOR, LAND, BOR, XOR, BAND, EQ, NE, GT, LT, GE, LE, PLUS, MINUS, MULT,
    DIV, MOD, POWER, FACTORIAL, BOOLE,
    DOLLARS, /* for using extension - div */
#ifdef GP_STRING_VARS
    CONCATENATE, EQS, NES, RANGE,
#endif
    /* only jump operators go between jump and sf_start, for is_jump() */
    JUMP, JUMPZ, JUMPNZ, JTERN, SF_START
};
#define is_jump(operator) \
    ((operator) >=(int)JUMP && (operator) <(int)SF_START)

/* user-defined function table entry */
typedef struct udft_entry {
    struct udft_entry *next_udf; /* pointer to next udf in linked list */
    char *udf_name;		 /* name of this function entry */
    struct at_type *at;		 /* pointer to action table to execute */
    char *definition;		 /* definition of function as typed */
    int dummy_num;		 /* required number of input variables */
    t_value dummy_values[MAX_NUM_VAR]; /* current value of dummy variables */
} udft_entry;


/* user-defined variable table entry */
typedef struct udvt_entry {
    struct udvt_entry *next_udv; /* pointer to next value in linked list */
    char *udv_name;		/* name of this value entry */
    TBOOLEAN udv_undef;		/* true if not defined yet */
    t_value udv_value;		/* value it has */
} udvt_entry;

/* p-code argument */
typedef union argument {
	int j_arg;		/* offset for jump */
	struct value v_arg;	/* constant value */
	struct udvt_entry *udv_arg; /* pointer to dummy variable */
	struct udft_entry *udf_arg; /* pointer to udf to execute */
} argument;


/* This type definition has to come after union argument has been declared. */
#ifdef __ZTC__
typedef void (*FUNC_PTR)(...);
#else
typedef void (*FUNC_PTR) __PROTO((union argument *arg));
#endif

/* standard/internal function table entry */
typedef struct ft_entry {
    const char *f_name;		/* pointer to name of this function */
    FUNC_PTR func;		/* address of function to call */
} ft_entry;

/* action table entry */
struct at_entry {
    enum operators index;	/* index of p-code function */
    union argument arg;
};

struct at_type {
    /* count of entries in .actions[] */
    int a_count;
    /* will usually be less than MAX_AT_LEN is malloc()'d copy */
    struct at_entry actions[MAX_AT_LEN];
};


/* Variables of eval.c needed by other modules: */

extern const struct ft_entry GPFAR ft[]; /* The table of builtin functions */
extern struct udft_entry *first_udf; /* user-def'd functions */
extern struct udvt_entry *first_udv; /* user-def'd variables */
extern struct udvt_entry udv_pi; /* 'pi' variable */
extern struct udvt_entry *udv_NaN; /* 'NaN' variable */
extern TBOOLEAN undefined;

/* Prototypes of functions exported by eval.c */

double gp_exp __PROTO((double x));

/* HBB 20010726: Moved these here, from util.h. */
double real __PROTO((struct value *));
double imag __PROTO((struct value *));
double magnitude __PROTO((struct value *));
double angle __PROTO((struct value *));
struct value * Gcomplex __PROTO((struct value *, double, double));
struct value * Ginteger __PROTO((struct value *, int));
#ifdef GP_STRING_VARS
struct value * Gstring __PROTO((struct value *, char *));
struct value * pop_or_convert_from_string __PROTO((struct value *));
#endif
struct value * gpfree_string __PROTO((struct value *a));

void reset_stack __PROTO((void));
void check_stack __PROTO((void));
TBOOLEAN more_on_stack __PROTO((void));
struct value *pop __PROTO((struct value *x));
void push __PROTO((struct value *x));
void int_check __PROTO((struct value * v));

void f_bool __PROTO((union argument *x));
void f_jump __PROTO((union argument *x));
void f_jumpz __PROTO((union argument *x));
void f_jumpnz __PROTO((union argument *x));
void f_jtern __PROTO((union argument *x));

void execute_at __PROTO((struct at_type *at_ptr));
void evaluate_at __PROTO((struct at_type *at_ptr, struct value *val_ptr));
void free_at __PROTO((struct at_type *at_ptr));
#ifdef APOLLO
void apollo_pfm_catch __PROTO((void));
#endif
struct udvt_entry * add_udv_by_name __PROTO((char *key));

/* update GPVAL_ variables available to user */
void update_gpval_variables __PROTO((int from_plot_command));

#endif /* GNUPLOT_EVAL_H */
