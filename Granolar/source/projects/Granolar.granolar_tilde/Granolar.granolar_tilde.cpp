/// @file
///	@ingroup 	minexamples
///	@copyright	Copyright 2018 The Min-DevKit Authors. All rights reserved.
///	@license	Use of this source code is governed by the MIT License found in the License.md file.



// TODO how to know if start and freq are connected?

#include "c74_min.h"
#include <typeinfo>
#include "AudioFile.h"

using namespace c74::min;

#ifndef M_PI
	#define M_PI 3.14159265358979323846
#endif

unsigned modulo(int value, unsigned m) {
	int mod = value % (int)m;
	if (value < 0) {
		mod += m;
	}
	return mod;
}


class granolar : public object<granolar>, public sample_operator<2, 1> {
private:
	double		freq_fonda;		// frequency of the grain when it's read at speed 1x
	double		current_step;

	double		current_start;	// start cursor

	unsigned	grain_length;
	double		alpha; // filter coefficient

	sample		previous_y;

	vector<sample> grain;
	AudioFile<double> audioFile;

	lib::sync m_oscillator;    // note: must be created prior to any attributes that might set parameters below


public:
	MIN_DESCRIPTION {"A machine learning based granular stynthsizer."};
	MIN_TAGS {"audio, oscillator"};
	MIN_AUTHOR {"Alice Rixte"};
	MIN_RELATED {""};

	granolar(const atoms& args = {}) {
		char* path_to_grain = "E:\\Documents\\Max 8\\Packages\\Granolar\\resources\\Kick Basic_2.wav";// "E:\\Documents\\ATIAM\\Informatique\\Granolar\\max-sdk-8.0.3\\source\\audio\\granolar~\\grain_test1.wav";
		audioFile.load(path_to_grain);
		grain_length = audioFile.getNumSamplesPerChannel();
		grain = audioFile.samples[0];
		previous_y = 0;


		if (args.size() >= 1) {
			if (args.size() >= 2) {
				start = static_cast<int>(args[1]);
			}
			else 
				start = 0;
			freq = static_cast<double>(args[0]);
		}
		else
			freq = 440;

		
	}


	inlet<>  in_freq {this, "(signal) frequency", "signal"};
	inlet<>  in_start {this, "(signal) start", "signal"};
	outlet<> out1 {this, "(signal) output signal", "signal"};


	argument<number> frequency_arg {
		this, "freq", "Initial frequency in hertz.", MIN_ARGUMENT_FUNCTION { freq = arg; }};

	argument<number> frequency_start{
		this, "start", "Initial start in samples.", MIN_ARGUMENT_FUNCTION { start = arg; } };


	message<> m_number {
		this, "number", "Set the frequency in Hz.", MIN_FUNCTION {
			if(inlet == 1)
				start = static_cast<int>(args[0]);
			if (inlet == 0)
				freq = static_cast<double>(args[0]);
			return {};
		}};

	

	attribute<number> freq{ this, "freq", 1.0, description {"Frequency in Hz."},
		setter { MIN_FUNCTION {
		double new_freq = static_cast<double>(args[0]);
			current_step = new_freq / freq_fonda;
			double buf = 2 * M_PI / current_step;
			alpha = buf / (buf + 1);
			return args;
		}}};

	attribute<number> start{ this, "start", 1.0, description {"Start in samples."},
		setter { MIN_FUNCTION {
			int new_start = static_cast<int>(args[0]);
			current_start += new_start - start;
			freq_fonda = samplerate() / (grain_length - new_start);
			return args;
		}} };


	sample operator()(sample freq_in, sample start_in) {
		
		sample out;
		if(in_start.has_signal_connection())
			start = start_in;
		if(in_freq.has_signal_connection())
			freq = freq_in;
		int current_length = grain_length - static_cast<int>(start);

		if (current_start > static_cast<double>(grain_length))
			current_start -= current_length;
		int floor_index = static_cast<int>(current_start);
		int next_index = floor_index + 1;
		if (next_index == grain_length)
			next_index -= current_length;
		double mantissa = current_start - floor_index;
		double now_y = (alpha * grain[next_index] + \
			(1 - alpha) * previous_y);

		// return the mean between the two nearest samples
		out = (1. - mantissa) * previous_y + mantissa * now_y;
		previous_y = now_y;

		/***** Lp Filter*******/
		current_start += current_step;
		int next_floor_index = static_cast<int>(current_start);
		for (int i = next_index + 1; i < next_floor_index; i++) {
			previous_y = alpha * grain[modulo(i - static_cast<int>(start), current_length)] + \
				(1 - alpha) * previous_y;
		}

		return out;
	}
};

MIN_EXTERNAL(granolar);
