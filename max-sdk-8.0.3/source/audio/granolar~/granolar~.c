/*
	split~: An audio version of the Max 'split' object
	Originally known as tap.split~, from Tap.Tools by Timothy Place
	Copyright 2011 Cycling '74
*/


#include "ext.h"		// standard Max include
#include "ext_obex.h"	// required for "new" (Max 4.5 and later) style objects
#include "z_dsp.h"		// required for audio objects
#include <stdio.h>
#include <sndfile.h>
//#include <math.h>


typedef struct _wavfile {
	SNDFILE* file;
	SF_INFO file_info;
} wavfile;

// struct to represent the object's state
typedef struct _linked_list {
	t_double* buf;
	int length_buf;
	struct _linked_list* next;
} linked_list;

int length_linked_list(const linked_list* l) {
	int length = 0;
	while (l) {
		length += l->length_buf;
		l = l->next;
	}
	return length;
}
t_double* linked_list_to_grain(const linked_list* l, int* grain_length) {
	*grain_length = length_linked_list(l);
	t_double* grain = malloc(*grain_length * sizeof(t_double));
	if (grain)
	{
		int n = 0;
		while (l) {
			for (int i = 0; i < l->length_buf; i++) {
				grain[n] = l->buf[i];
				n++;
			}
			l = l->next;
		}
		return grain;
	}
}

void free_linked_list(linked_list* l) {
	while (l) {
//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Problem here : can't free l->buf !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		//free(l->buf);
		linked_list* next = l->next;
		free(l);
		l = next;
	}
}



typedef struct _granolar {
	t_pxobject	s_ob;			// audio object "base class"
	double		freq;			// length of the grain
	int			start;			// start position of the grain
	int			current_start;	// start if exceeded buffer
	int			current_length;
	bool		record;
	linked_list* recording_buf;
	linked_list* last;
	t_double*	grain;
	int			grain_length;
	//wavfile	grain;			// wav file for the grain

} t_granolar;



// method prototypes
void *granolar_new(t_symbol *msg, long argc, t_atom *argv);
void granolar_assist(t_granolar *x, void *b, long m, long a, char *s);
void granolar_int(t_granolar *x, long n);
void granolar_bang(t_granolar* x);
void granolar_float(t_granolar *x, double val);
// (we can skip the prototypes for the perform methods; we'll define them in the body above the point where they are called)
void granolar_dsp64(t_granolar *x, t_object *dsp64, short *count, double samplerate, long maxvectorsize, long flags);


// global class pointer variable
static t_class *s_granolar_class = NULL;


/*wavfile loadWavFile(char* in_file_name) {
	wavfile new_file;
	int fs;


	new_file.file = sf_open(in_file_name, SFM_READ, &new_file.file_info);
	//sf_close(inFile); this has to be done in another function

	fs = new_file.file_info.samplerate;
	printf("Sample Rate = %d Hz\n", fs);

	return new_file;
}*/

//***********************************************************************************************

void ext_main(void *r)
{
	t_class *c = class_new("granolar~", (method)granolar_new, (method)dsp_free, sizeof(t_granolar), NULL, A_GIMME, 0);

	class_addmethod(c, (method)granolar_int,		"int",		A_LONG, 0);
	class_addmethod(c, (method)granolar_float,		"float",	A_FLOAT, 0);
	class_addmethod(c, (method)granolar_dsp64,		"dsp64",	A_CANT, 0);		// New 64-bit MSP dsp chain compilation for Max 6
	class_addmethod(c, (method)granolar_assist,		"assist",	A_CANT, 0);
	class_addmethod(c, (method)granolar_bang,		"bang",				0);

	class_dspinit(c);
	class_register(CLASS_BOX, c);
	s_granolar_class = c;
}


void *granolar_new(t_symbol *s, long argc, t_atom *argv)
{
	t_granolar* x = (t_granolar*)object_alloc(s_granolar_class);

	if (x) {
		dsp_setup((t_pxobject *)x, 3);				// 2 inlets (freq, start,recording)
		outlet_new((t_object *)x, "signal");	
		float buf = 0;
		atom_arg_getfloat(&buf, 0, argc, argv);	// get typed in args
		x->freq = (double) buf;
		atom_arg_getfloat(&buf, 1, argc, argv);	// ...
		x->start = (int) buf;
		x->record = false;
		x->last = NULL;
		x->grain = NULL;
		x->grain_length = 0;
	}
	return x;
}


//***********************************************************************************************

void granolar_assist(t_granolar *x, void *b, long msg, long arg, char *dst)
{
	if (msg == ASSIST_INLET) {
		switch (arg) {
		case 0: strcpy(dst, "(signal/float) frequency of the output"); break;
		case 1: strcpy(dst, "(signal/int) start of the grain"); break;
		case 2: strcpy(dst, "(signal) this signal is recorded when the recording is activated"); break;
		}
	}
	else if (msg == ASSIST_OUTLET) {
		switch (arg) {
		case 0: strcpy(dst, "(signal) Output"); break;

		}
	}
}


void granolar_float(t_granolar *x, double value)
{
	long inlet_number = proxy_getinlet((t_object *)x);

	if (inlet_number == 1) {
		x->current_length =  (int)round((double)x->current_length * x->freq / value);
		x->freq = value;
	}
	else
		object_error((t_object *)x, "oops -- maybe you sent a number to the wrong inlet?");
		
}


void granolar_int(t_granolar *x, long value)
{
	long inlet_number = proxy_getinlet((t_object*)x);

	if (inlet_number == 1) {
		x->current_length = (int)round((double)x->current_length * x->freq / value);
		x->freq = value;
	}
	else if (inlet_number == 2) {
		x->current_start += x->start - value;
		x->start = value;
	}
	else
		object_error((t_object*)x, "oops -- maybe you sent a number to the wrong inlet?");
}

void granolar_bang(t_granolar* x) {
	x->record = !x->record;
	if (x->record) {

		object_post((t_object*)x, "Recording is on ! ");
	}
	else {
		x->grain = linked_list_to_grain(x->recording_buf,&x->grain_length);
		free_linked_list(x->recording_buf);
		x->last = NULL;
		object_post((t_object*)x, "Recording is off ! ");
	}
}

//***********************************************************************************************

void granolar_update_time_freq(t_granolar* x, double freq, int start) {
	x->current_length = (int)round((double)x->current_length * x->freq / freq);
	x->current_start += x->start - start;
}


// Perform (signal) Method for 3 input signals, 64-bit MSP
void granolar_perform64(t_granolar *x, t_object *dsp64, double **ins, long numins, double **outs, long numouts, long sampleframes, long flags, void *userparam)
{
	t_double	*freq = ins[0];		// Input 1
	t_int		*start = (t_int*) ins[1];
	t_double	*record_in = ins[2];
	t_double	*out = outs[0];	// Output 1
	if (x->record) {
		linked_list* new_buf = malloc(sizeof(linked_list));
		if (new_buf)
		{
			new_buf->buf = calloc(sampleframes , sizeof(t_double));
			new_buf->length_buf = sampleframes;
			new_buf->next = NULL;
			if (!x->last) {
				x->recording_buf = x->last = new_buf;
			}
			else {
				x->last = x->last->next = new_buf;
			}
		}
	}

	int			n = sampleframes;
	t_double	value;

	
	while (n--) {
		value = *freq++;
		if(x->record && x->last && x->last->buf)
			x->last->buf[sampleframes - n - 1] = *record_in++;
		else if (x->grain) {
			*out++ = x->grain[x->current_start++ % x->grain_length];
		}
		*out++ = value;
	}
}



//***********************************************************************************************

void granolar_dsp64(t_granolar *x, t_object *dsp64, short *count, double samplerate, long maxvectorsize, long flags)
{
	object_method(dsp64, gensym("dsp_add64"), x, granolar_perform64, 0, NULL);
}

