#ifndef lint
static char *RCSid() { return RCSid("$Id: breaders.c,v 1.3.2.1 2009/01/28 10:39:37 mikulik Exp $"); }
#endif

/* GNUPLOT - breaders.c */

/*[
 * Copyright 2004  Petr Mikulik
 *
 * As part of the program Gnuplot, which is
 *
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

/* AUTHOR : Petr Mikulik */

/*
 * Readers to set up binary data file information for particular formats.
 */

#include "breaders.h"

#ifdef BINARY_DATA_FILE

#include "datafile.h"
#include "alloc.h"
#include "misc.h"

/*
 * Reader for the ESRF Header File format files (EDF / EHF).
 */

/* Inside datafile.c, but kept hidden. */
extern char *df_filename;	/* name of data file */
extern int df_no_bin_cols;	/* cols to read */
extern df_endianess_type df_bin_file_endianess;

/* Reader for the ESRF Header File format files (EDF / EHF).
 */

/* gen_table4 */
struct gen_table4 {
    const char *key;
    int value;
    short signum; /* 0..unsigned, 1..signed, 2..float or double */
    short sajzof; /* sizeof on 32bit architecture */
};
 
/* Exactly like lookup_table_nth from tables.c, but for gen_table4 instead
 * of gen_table.
 */ 
static int
lookup_table4_nth(const struct gen_table4 *tbl, const char *search_str)
{
    int k = -1;
    while (tbl[++k].key)
	if (tbl[k].key && !strncmp(search_str, tbl[k].key, strlen(tbl[k].key)))
	    return k;
    return -1; /* not found */
}

static const struct gen_table4 edf_datatype_table[] =
{
    { "UnsignedByte",	DF_UCHAR,   0, 1 },
    { "SignedByte",	DF_CHAR,    1, 1 },
    { "UnsignedShort",	DF_USHORT,  0, 2 },
    { "SignedShort",	DF_SHORT,   1, 2 },
    { "UnsignedInteger",DF_UINT,    0, 4 },
    { "SignedInteger",	DF_INT,	    1, 4 },
    { "UnsignedLong",	DF_ULONG,   0, 8 },
    { "SignedLong",	DF_LONG,    1, 8 },
    { "FloatValue",	DF_FLOAT,   2, 4 },
    { "DoubleValue",	DF_DOUBLE,  2, 8 },
    { "Float",		DF_FLOAT,   2, 4 }, /* Float and FloatValue are synonyms */
    { "Double",		DF_DOUBLE,  2, 8 }, /* Double and DoubleValue are synonyms */
    { NULL, -1, -1, -1 }
};

static const struct gen_table edf_byteorder_table[] =
{
    { "LowByteFirst",	DF_LITTLE_ENDIAN }, /* little endian */
    { "HighByteFirst",	DF_BIG_ENDIAN },    /* big endian */
    { NULL, -1 }
};

/* Orientation of axes of the raster, as the binary matrix is saved in 
 * the file.
 */
enum EdfRasterAxes {
    EDF_RASTER_AXES_XrightYdown,	/* matricial format: rows, columns */
    EDF_RASTER_AXES_XrightYup		/* cartesian coordinate system */
    /* other 6 combinations not available (not needed until now) */
};

static const struct gen_table edf_rasteraxes_table[] =
{
    { "XrightYdown",	EDF_RASTER_AXES_XrightYdown },
    { "XrightYup",	EDF_RASTER_AXES_XrightYup },
    { NULL, -1 }
};


/* Find value_ptr as pointer to the parameter of the given key in the header.
 * Returns NULL on success.
 */
static char*
edf_findInHeader ( const char* header, const char* key )
{
    char *value_ptr = strstr( header, key );
    if (!value_ptr) return NULL;
    /* an edf line is "key     = value ;" */
    value_ptr = 1 + strchr( value_ptr + strlen(key), '=' );
    while (isspace(*value_ptr)) value_ptr++;
    return value_ptr;
}
 
void
edf_filetype_function(void)
{
    FILE *fp;
    char *header = NULL;
    int header_size = 0;
    char *p;
    int k;
    /* open (header) file */
    fp = loadpath_fopen(df_filename, "rb");
    if (!fp)
	os_error(NO_CARET, "Can't open data file \"%s\"", df_filename);
    /* read header: it is a multiple of 512 B ending by "}\n" */
    while (header_size == 0 || strncmp(&header[header_size-2],"}\n",2)) {
	int header_size_prev = header_size;
	header_size += 512;
	if (!header)
	    header = gp_alloc(header_size+1, "EDF header");
	else
	    header = gp_realloc(header, header_size+1, "EDF header");
	header[header_size_prev] = 0; /* protection against empty file */
	k = fread(header+header_size_prev, 512, 1, fp);
	if (k == 0) { /* protection against indefinite loop */
	    free(header);
	    os_error(NO_CARET, "Damaged EDF header of %s: not multiple of 512 B.\n", df_filename);
	}
	header[header_size] = 0; /* end of string: protection against strstr later on */
    }
    fclose(fp);
    /* make sure there is a binary record structure for each image */
    if (df_num_bin_records < 1)
	df_add_binary_records(1-df_num_bin_records, DF_CURRENT_RECORDS); /* otherwise put here: number of images (records) from this file */
    if ((p = edf_findInHeader(header, "EDF_BinaryFileName"))) {
	int plen = strcspn(p, " ;\n");
	df_filename = gp_realloc(df_filename, plen+1, "datafile name");
	strncpy(df_filename, p, plen);
	df_filename[plen] = '\0';
	if ((p = edf_findInHeader(header, "EDF_BinaryFilePosition")))
	    df_bin_record[0].scan_skip[0] = atoi(p);
	else
	    df_bin_record[0].scan_skip[0] = 0;
    } else
	df_bin_record[0].scan_skip[0] = header_size; /* skip header */
    /* set default values */
    df_bin_record[0].scan_dir[0] = 1;
    df_bin_record[0].scan_dir[1] = -1;
    df_bin_record[0].scan_generate_coord = TRUE;
    df_bin_record[0].cart_scan[0] = DF_SCAN_POINT;
    df_bin_record[0].cart_scan[1] = DF_SCAN_LINE;
    df_extend_binary_columns(1);
    df_set_skip_before(1,0);
    df_set_skip_after(1,0);
    df_no_use_specs = 1;
    use_spec[0].column = 1;
    /* now parse the header */
    if ((p = edf_findInHeader(header, "Dim_1")))
	df_bin_record[0].scan_dim[0] = atoi(p);
    if ((p = edf_findInHeader(header, "Dim_2")))
	df_bin_record[0].scan_dim[1] = atoi(p);
    if ((p = edf_findInHeader(header, "DataType"))) {
	k = lookup_table4_nth(edf_datatype_table, p);
	if (k >= 0) { /* known EDF DataType */
	    int s = edf_datatype_table[k].sajzof; 
	    switch (edf_datatype_table[k].signum) {
		case 0: df_set_read_type(1,SIGNED_TEST(s)); break;
		case 1: df_set_read_type(1,UNSIGNED_TEST(s)); break;
		case 2: df_set_read_type(1,FLOAT_TEST(s)); break;
	    }
	}
    }
    if ((p = edf_findInHeader(header, "ByteOrder"))) {
	k = lookup_table_nth(edf_byteorder_table, p);
	if (k >= 0)
	    df_bin_file_endianess = edf_byteorder_table[k].value;
    }
    /* Origin vs center: EDF specs allows only Center, but it does not hurt if
       Origin is supported as well; however, Center rules if both specified.
    */
    if ((p = edf_findInHeader(header, "Origin_1"))) {
	df_bin_record[0].scan_cen_or_ori[0] = atof(p);
	df_bin_record[0].scan_trans = DF_TRANSLATE_VIA_ORIGIN;
    }
    if ((p = edf_findInHeader(header, "Origin_2"))) {
	df_bin_record[0].scan_cen_or_ori[1] = atof(p);
	df_bin_record[0].scan_trans = DF_TRANSLATE_VIA_ORIGIN;
    }
    if ((p = edf_findInHeader(header, "Center_1"))) {
	df_bin_record[0].scan_cen_or_ori[0] = atof(p);
	df_bin_record[0].scan_trans = DF_TRANSLATE_VIA_CENTER;
    }
    if ((p = edf_findInHeader(header, "Center_2"))) {
	df_bin_record[0].scan_cen_or_ori[1] = atof(p);
	df_bin_record[0].scan_trans = DF_TRANSLATE_VIA_CENTER;
    }
    /* now pixel sizes and raster orientation */
    if ((p = edf_findInHeader(header, "PSize_1")))
	df_bin_record[0].scan_delta[0] = atof(p);
    if ((p = edf_findInHeader(header, "PSize_2")))
	df_bin_record[0].scan_delta[1] = atof(p);
    if ((p = edf_findInHeader(header, "RasterAxes"))) {
	k = lookup_table_nth(edf_rasteraxes_table, p);
	switch (k) {
	    case EDF_RASTER_AXES_XrightYup:
		df_bin_record[0].scan_dir[0] = 1;
		df_bin_record[0].scan_dir[1] = 1;
		df_bin_record[0].cart_scan[0] = DF_SCAN_POINT;
		df_bin_record[0].cart_scan[1] = DF_SCAN_LINE;
		break;
	    default: /* also EDF_RASTER_AXES_XrightYdown */
		df_bin_record[0].scan_dir[0] = 1;
		df_bin_record[0].scan_dir[1] = -1;
		df_bin_record[0].cart_scan[0] = DF_SCAN_POINT;
		df_bin_record[0].cart_scan[1] = DF_SCAN_LINE;
	}
    }

    free(header);
#if 0
    /* Print results. This routine will be completely removed later. */
    fprintf(stderr,"EDF: dim=%ix%i skip=%i datatype=%i datasize=%i dx=%g dy=%g\n",
	df_bin_record[0].scan_dim[0], df_bin_record[0].scan_dim[1],
	df_bin_record[0].scan_skip[0],
	df_get_read_type(1), df_get_read_size(1),
	df_bin_record[0].scan_delta[0], df_bin_record[0].scan_delta[1]);
#endif

}

#endif /* BINARY_DATA_FILE */
