#ifndef lint
static char *RCSid() { return RCSid("$Id: scanner.c,v 1.23 2005/11/23 23:33:37 mikulik Exp $"); }
#endif

/* GNUPLOT - scanner.c */

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

#include "scanner.h"

#include "alloc.h"
#include "command.h"
#include "util.h"

static int get_num __PROTO((char str[]));
static void substitute __PROTO((char **strp, size_t *str_lenp, int current));

#ifdef AMIGA_AC_5
#define O_RDONLY	0
int open(const char *_name, int _mode,...);
int close(int);
#endif /* AMIGA_AC_5 */

#ifdef VMS
#include <descrip.h>
#define MAILBOX "PLOT$MAILBOX"
#define pclose(f) fclose(f)
#ifdef __DECC
#include <lib$routines.h>	/* avoid some IMPLICITFNC warnings */
#include <starlet.h>
#endif /* __DECC */
#endif /* VMS */


#define isident(c) (isalnum((unsigned char)c) || (c) == '_')

#define LBRACE '{'
#define RBRACE '}'

#define APPEND_TOKEN {token[t_num].length++; current++;}

#define SCAN_IDENTIFIER while (isident(expression[current + 1]))\
				APPEND_TOKEN

static int t_num;		/* number of token I'm working on */

/*
 * scanner() breaks expression[] into lexical units, storing them in token[].
 *   The total number of tokens found is returned as the function
 *   value.  Scanning will stop when '\0' is found in expression[], or
 *   when token[] is full.  extend_input_line() is called to extend
 *   expression array if needed.
 *
 *       Scanning is performed by following rules:
 *
 *      Current char    token should contain
 *     -------------    -----------------------
 *      1.  alpha,_     all following alpha-numerics
 *      2.  digit       0 or more following digits, 0 or 1 decimal point,
 *                              0 or more digits, 0 or 1 'e' or 'E',
 *                              0 or more digits.
 *      3.  ^,+,-,/     only current char
 *          %,~,(,)
 *          [,],;,:,
 *          ?,comma
 *          $           for using patch (div)
 *      4.  &,|,=,*     current char; also next if next is same
 *      5.  !,<,>       current char; also next if next is =
 *      6.  ", '        all chars up until matching quote
 *      7.  #           this token cuts off scanning of the line (DFK).
 *      8.  `           (command substitution: all characters through the
 *                      matching backtic are replaced by the output of
 *                      the contained command, then scanning is restarted.)
 *
 *                      white space between tokens is ignored
 */
int
scanner(char **expressionp, size_t *expressionlenp)
{
    int current;	/* index of current char in expression[] */
    char *expression = *expressionp;
    int quote;
    char brace;

    for (current = t_num = 0; expression[current] != NUL; current++) {
	if (t_num + 1 >= token_table_size) {
	    /* leave space for dummy end token */
	    extend_token_table();
	}
	if (isspace((unsigned char) expression[current]))
	    continue;		/* skip the whitespace */
	token[t_num].start_index = current;
	token[t_num].length = 1;
	token[t_num].is_token = TRUE;	/* to start with... */

	if (expression[current] == '`') {
	    substitute(expressionp, expressionlenp, current);
	    expression = *expressionp;	/* expression might have moved */
	    current--;
	    continue;
	}
	/* allow _ to be the first character of an identifier */
	if (isalpha((unsigned char) expression[current])
	    || expression[current] == '_') {
	    SCAN_IDENTIFIER;
	} else if (isdigit((unsigned char) expression[current])
		   || expression[current] == '.') {
	    token[t_num].is_token = FALSE;
	    token[t_num].length = get_num(&expression[current]);
	    current += (token[t_num].length - 1);
#ifdef GP_STRING_VARS
	    if (token[t_num].length == 1 && expression[current] == '.')
		token[t_num].is_token = TRUE;
#endif
	} else if (expression[current] == LBRACE) {
	    token[t_num].is_token = FALSE;
	    token[t_num].l_val.type = CMPLX;
#ifdef __PUREC__
	    {
		char l[80];
		if ((sscanf(&expression[++current], "%lf,%lf%[ }]s",
			    &token[t_num].l_val.v.cmplx_val.real,
			    &token[t_num].l_val.v.cmplx_val.imag,
			    &l) != 3) || (!strchr(l, RBRACE)))
		    int_error(t_num, "invalid complex constant");
	    }
#else
	    if ((sscanf(&expression[++current], "%lf , %lf %c",
			&token[t_num].l_val.v.cmplx_val.real,
			&token[t_num].l_val.v.cmplx_val.imag,
			&brace) != 3) || (brace != RBRACE))
		int_error(t_num, "invalid complex constant");
#endif
	    token[t_num].length += 2;
	    while (expression[++current] != RBRACE) {
		token[t_num].length++;
		if (expression[current] == NUL)		/* { for vi % */
		    int_error(t_num, "no matching '}'");
	    }
	} else if (expression[current] == '\'' ||
		   expression[current] == '\"') {
	    token[t_num].length++;
	    quote = expression[current];
	    while (expression[++current] != quote) {
		if (!expression[current]) {
		    expression[current] = quote;
		    expression[current + 1] = NUL;
		    break;
		} else if (quote == '\"'
                           && expression[current] == '\\'
			   && expression[current + 1]) {
		    current++;
		    token[t_num].length += 2;
		} else if (quote == '\"' && expression[current] == '`') {
		    substitute(expressionp, expressionlenp, current);
		    expression = *expressionp;	/* it might have moved */
		    current--;
                } else if (quote == '\'' 
                           && expression[current+1] == '\''
                           && expression[current+2] == '\'') {
                    /* look ahead: two subsequent single quotes 
                     * -> take them in
                     */
                    current += 2;
                    token[t_num].length += 3;
		} else
		    token[t_num].length++;
	    }
	} else
	    switch (expression[current]) {
	    case '#':		/* DFK: add comments to gnuplot */
		goto endline;	/* ignore the rest of the line */
	    case '^':
	    case '+':
	    case '-':
	    case '/':
	    case '%':
	    case '~':
	    case '(':
	    case ')':
	    case '[':
	    case ']':
	    case ';':
	    case ':':
	    case '?':
	    case ',':
	    case '$':		/* div */
		break;
	    case '&':
	    case '|':
	    case '=':
	    case '*':
		if (expression[current] == expression[current + 1])
		    APPEND_TOKEN;
		break;
	    case '!':
	    case '<':
	    case '>':
		if (expression[current + 1] == '=')
		    APPEND_TOKEN;
		break;
	    default:
		int_error(t_num, "invalid character %c",expression[current]);
	    }
	++t_num;		/* next token if not white space */
    }

  endline:			/* comments jump here to ignore line */

/* Now kludge an extra token which points to '\0' at end of expression[].
   This is useful so printerror() looks nice even if we've fallen off the
   line. */

    token[t_num].start_index = current;
    token[t_num].length = 0;
    /* print 3+4  then print 3+  is accepted without
     * this, since string is ignored if it is not
     * a token
     */
    token[t_num].is_token = TRUE;
    return (t_num);
}


static int
get_num(char str[])
{
    int count = 0;
    long lval;

    token[t_num].is_token = FALSE;
    token[t_num].l_val.type = INTGR;	/* assume unless . or E found */
    while (isdigit((unsigned char) str[count]))
	count++;
    if (str[count] == '.') {
	token[t_num].l_val.type = CMPLX;
	/* swallow up digits until non-digit */
	while (isdigit((unsigned char) str[++count]));
	/* now str[count] is other than a digit */
    }
    if (str[count] == 'e' || str[count] == 'E') {
	token[t_num].l_val.type = CMPLX;
/* modified if statement to allow + sign in exponent
   rjl 26 July 1988 */
	count++;
	if (str[count] == '-' || str[count] == '+')
	    count++;
	if (!isdigit((unsigned char) str[count])) {
	    token[t_num].start_index += count;
	    int_error(t_num, "expecting exponent");
	}
	while (isdigit((unsigned char) str[++count]));
    }
    if (token[t_num].l_val.type == INTGR) {
	lval = atol(str);
	if ((token[t_num].l_val.v.int_val = lval) != lval)
	    int_error(t_num, "integer overflow; change to floating point");
    } else {
	token[t_num].l_val.v.cmplx_val.imag = 0.0;
#ifdef HAVE_LOCALE_H
	/* Always read numbers on command line as C locale */
	if (strcmp(localeconv()->decimal_point,".")) {
	    char *save_locale = gp_strdup(setlocale(LC_NUMERIC,NULL));
	    setlocale(LC_NUMERIC,"C");
	    token[t_num].l_val.v.cmplx_val.real = atof(str);
	    setlocale(LC_NUMERIC,save_locale);
	    free(save_locale);
	} else
#endif
	    token[t_num].l_val.v.cmplx_val.real = atof(str);
    }
    return (count);
}

/* substitute output from ` `
 * *strp points to the input string.  (*strp)[current] is expected to
 * be the initial back tic.  Characters through the following back tic
 * are replaced by the output of the command.  extend_input_line()
 * is called to extend *strp array if needed.
 */
static void
substitute(char **strp, size_t *str_lenp, int current)
{
    char *last;
    char c;
    char *str, *pgm, *rest = NULL;
    char *output;
    size_t pgm_len, rest_len = 0;
    int output_pos;

    /* forgive missing closing backquote at end of line */
    str = *strp + current;
    last = str;
    while (*++last) {
	if (*last == '`')
	    break;
    }
    pgm_len = last - str;
    pgm = gp_alloc(pgm_len, "command string");
    safe_strncpy(pgm, str + 1, pgm_len); /* omit ` to leave room for NUL */

    /* save rest of line, if any */
    if (*last) {
	last++;			/* advance past ` */
	rest_len = strlen(last) + 1;
	if (rest_len > 1) {
	    rest = gp_alloc(rest_len, "input line copy");
	    strcpy(rest, last);
	}
    }

    /* do system call */
    (void) do_system_func(pgm, &output);
    free(pgm);

    /* now replace ` ` with output */
    output_pos = 0;
    while ((c = output[output_pos++])) {
	if (c != '\n' && c != '\r')
	    (*strp)[current++] = c;
	if (current == *str_lenp)
	    extend_input_line();
    }
    (*strp)[current] = 0;
    free(output);

    /* tack on rest of line to output */
    if (rest) {
	while (current + rest_len > *str_lenp)
	    extend_input_line();
	strcpy(*strp + current, rest);
	free(rest);
    }
    screen_ok = FALSE;

}


