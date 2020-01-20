/*
	split~: An audio version of the Max 'split' object
	Originally known as tap.split~, from Tap.Tools by Timothy Place
	Copyright 2011 Cycling '74
*/

// TODO : optimization when not connected

#include "ext.h"		// standard Max include
#include "ext_obex.h"	// required for "new" (Max 4.5 and later) style objects
#include "z_dsp.h"		// required for audio objects
#include <stdio.h>
#include <sndfile.h>
//#include <math.h>

#define PI 3.141592653


typedef struct _wavfile {
	SNDFILE* file;
	SF_INFO file_info;
} wavfile;



unsigned modulo(int value, unsigned m) {
	int mod = value % (int)m;
	if (value < 0) {
		mod += m;
	}
	return mod;
}




typedef struct _granolar {
	t_pxobject	s_ob;			// audio object "base class"

	double		freq;
	double		freq_fonda;		// frequency of the grain when it's read at speed 1x
	double		current_step;
	bool		freq_connected;

	double		start;			// start position of the grain
	double		current_start;	// start cursor
	bool		start_connected;


	t_double*	grain;
	unsigned	grain_length;

	//for the filter
	t_double	previous_y;
	double		alpha;
	double		alpha_opt;

	double		sr;
	bool		sr_set;

	double		cutoff;
	
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
		x->current_start = (double) buf;

		atom_arg_getfloat(&buf, 2, argc, argv);	// ...
		x->cutoff = 1 / buf;

		char* path_to_grain = "E:\\Documents\\ATIAM\\Informatique\\Granolar\\max-sdk-8.0.3\\source\\audio\\granolar~\\grain_test1.wav";
		wavfile new_file;

		new_file.file = sf_open(path_to_grain, SFM_READ, &new_file.file_info);
		
		x->grain = calloc(new_file.file_info.frames, sizeof(t_double));
		if (new_file.file) {
			x->grain_length = sf_readf_double(new_file.file, (double*)x->grain, new_file.file_info.frames); //); //read 1 sec
			sf_close(new_file.file);// this has to be done in another function
		}



		x->start_connected = false;
		x->freq_connected = false;
		x->current_step = 1.;
		x->previous_y = 0.;
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

	if (inlet_number == 0) {
		//x->current_length =  (int)round((double)x->current_length * x->freq / value);
		x->freq = value;
		if (x->sr_set) {
			x->current_step = x->freq / x->freq_fonda;
			double buf = x->cutoff / x->current_step;
			x->alpha = buf / (buf + 1);

		}
	}
	else
		object_error((t_object *)x, "oops -- maybe you sent a number to the wrong inlet?");
	
}


void granolar_int(t_granolar *x, long value)
{
	long inlet_number = proxy_getinlet((t_object*)x);

	if (inlet_number == 0) {
		granolar_float(x, (double)value);
	}
	else if (inlet_number == 1) {
		x->current_start +=  (double)value - x->start ;
		x->start = value;
	}
	else
		object_error((t_object*)x, "oops -- maybe you sent a number to the wrong inlet?");
		
}

void granolar_bang(t_granolar* x) {

}

//***********************************************************************************************



void granolar_update_time_freq(t_granolar* x, double freq, int start) {

	//x->current_length = (int)round((double)x->current_length * x->freq / freq);
	if (x->start_connected && start < x->grain_length && start >= 0) {
		x->current_start += start - x->start;
		
		x->start = start;
		x->freq_fonda = x->sr / (x->grain_length - x->start);
	}
	if (x->freq_connected) {
		x->current_step = freq / x->freq_fonda;
		double buf = 2 * PI / x->current_step;
		x->alpha = buf / (buf + 1);
	}

}


void granolar_perform64(t_granolar *x, t_object *dsp64, double **ins, long numins, double **outs, long numouts, long sampleframes, long flags, void *userparam)
{
	t_double	*freq = ins[0];		// Input 1
	t_double	*start =  ins[1];
	t_double	*out = outs[0];	// Output 1
	

	int			n = sampleframes;

	
	while (n--) {

		granolar_update_time_freq(x, *freq++, (int)*start++);
		if (x->current_start > (double)x->grain_length)
			x->current_start += x->start - x->grain_length ;
		int floor_index = (int)x->current_start;
		int next_index = floor_index + 1;
		if(next_index == x->grain_length)
			next_index += x->start - x->grain_length;
		double mantissa = x->current_start - floor_index;

		double now_y = (x->alpha * x->grain[next_index] + \
			(1 - x->alpha) * x->previous_y);

		*out++ = (1. - mantissa) * x->previous_y + mantissa * now_y;
		x->previous_y = now_y;

		x->current_start += x->current_step;
		int next_floor_index = (int)x->current_start;
		for (int i = next_index + 1; i < next_floor_index ; i++) {
			int current_index = modulo(i - x->start, x->grain_length - x->start);
			x->previous_y = x->alpha * x->grain[current_index] + \
				(1 - x->alpha) * x->previous_y;
		}

		
	}
}



//***********************************************************************************************

void granolar_dsp64(t_granolar *x, t_object *dsp64, short *count, double samplerate, long maxvectorsize, long flags)
{
	x->freq_connected = count[0];
	x->start_connected = count[1];
	if (!x->sr_set) {
		x->sr = samplerate;
		x->sr_set = true;

		x->freq_fonda = x->sr / (x->grain_length - x->start);
		x->current_step = x->freq / x->freq_fonda;
		double buf = x->cutoff / x->current_step;
		x->alpha = buf / (buf + 1);

	}
	object_method(dsp64, gensym("dsp_add64"), x, granolar_perform64, 0, NULL);
}

