/*
Copyright (c) 2022.  The Regents of the University of California (Regents).
All Rights Reserved.

Permission to use, copy, modify, and distribute this software and its
documentation for educational, research, and not-for-profit purposes, without
fee and without a signed licensing agreement, is hereby granted, provided that
the above copyright notice, this paragraph and the following two paragraphs
appear in all copies, modifications, and distributions.  Contact The Office of
Technology Licensing, UC Berkeley, 2150 Shattuck Avenue, Suite 510, Berkeley,
CA 94720-1620, (510) 643-7201, for commercial licensing opportunities.

Written by Jeremy L. Wagner, The Center for New Music and Audio Technologies,
University of California, Berkeley.

     IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT,
     SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS,
     ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF
     REGENTS HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

     REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT
     LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
     FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING
     DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS".
     REGENTS HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES,
     ENHANCEMENTS, OR MODIFICATIONS.
*/
//
//  qrm~.c
//  vindex_tilde
//  This is a sandbox attempt at returning a vector from a buffer at some specified (int) index.
//  Load that buffer into FFTW, perform FFT, then do some peak finding and rough estimation/refinement of
//  bin frequency. Output as a list of likely frequencies.
//
//  Created by Jeremy Wagner on 8/11/22.
//

#include "qrm_tilde.h" //probably not needed

#include "ext.h"
#include "ext_obex.h"
#include "ext_common.h" // contains CLAMP macro
#include "z_dsp.h"
#include "ext_buffer.h"
#include "fftw3.h"
#include "time.h"

#define NUMSLICES 5
#define EPSILON 0.0001

//struct to contain analysis window
typedef struct _Slice {
    long index_in_buffer;
    fftw_plan p;
    double *in;
    double *outs;
    double *mag_spec;
    double *phase_spec;
    double sum;
    double max_peak;
    int num_peaks;
//    long *peaks;
}t_Slice;

//struct for object
typedef struct _qrm {
    t_pxobject l_obj;
    long l_vector;
    t_buffer_ref *l_buffer_reference;
    long l_chan;
    //t_buffer_ref *o_buffer_reference;
    //long o_chan;
    double *window_function;
    long sample_vector_size;    //length of vector we will pull from the buffer
    long cursor;                //cursor in buffer (the analysis point)
    long cursor2;               //cursor2 in buffer (the second analysis point)
    long *analysis_points;      //an array of analysis points for determining decay rates
    double *ap0;
    double *ap1;
    double *ap2;
    double *ap3;
    double *ap4;
    void *out;                  //outlet
//    void *f_out;
    void *slice_out;                //dump outlet
    void *model_out;
    long fft_size;
    fftw_plan p;                //sinusoidal fftw plan
    double *in;                 //sinusoidal model analysis input
    double *outs;               //sinusoidal model analysis inputs
    double *mag_spec;           //sinusoidal model magnitude spectrum
    double *phase_spec;         //sinusoidal model phase spectrum
    double thresh;
    double sum;
    double max_peak;
    int num_peaks;
    long *peaks;
    float sr;
    double *cooked;
    float* tab;                 //variable for buffer access
    t_buffer_obj* buffer;       //pointer to buffer
    long region_max_ind;        //index for max functions
    float max_val;              //value for max functions
    struct _Slice slices[NUMSLICES];    //an array of analysis windows for resonant model computation
    long* idxs;                 //list of slice indexes
    double* tempY;              //array for amplitude values in exponential fitting
    double* tempAB;             //array to hold result of exponential fitting
    double* amps;               //output amplitudes
    double* dr;                 //output decay rates
    double* model;              //output model
    
} t_qrm;



//prototypes
void qrm_perform64(t_qrm *x, t_object *dsp64, double **ins, long numins, double **outs, long numouts, long sampleframes, long flags, void *userparam);
void qrm_dsp64(t_qrm *x, t_object *dsp64, short *count, double samplerate, long maxvectorsize, long flags);
void qrm_int(t_qrm *x, long n);
void qrm_set(t_qrm *x, t_symbol *s);
void qrm_setvsize(t_qrm *x, long n);
void qrm_getvsize(t_qrm *x);
void qrm_set_fft_size(t_qrm *x, long n);
void *qrm_new(t_symbol *s, long chan);
void qrm_free(t_qrm *x);
t_max_err qrm_notify(t_qrm *x, t_symbol *s, t_symbol *msg, void *sender, void *data);
void qrm_in1(t_qrm *x, long n);
void qrm_assist(t_qrm *x, void *b, long m, long a, char *s);
void qrm_dblclick(t_qrm *x);
void qrm_set_thresh(t_qrm *x, double n);
void qrm_bang(t_qrm *x);
void qrm_list_out(t_qrm *x, double* a, long l, void* outlet);
void qrm_list(t_qrm *x, t_symbol *msg, long argc, t_atom *argv);
void exp_fit(long *xVals, double *yVals, long n, double* out, double wt);
void hann_window(t_qrm *x, double *a);
void hann_window_gen(t_qrm *x);
void findMaxInBuffer(t_qrm* x);

//class
static t_class *qrm_class;

C74_EXPORT void ext_main(void *r)
{
    t_class *c = class_new("qrm~", (method)qrm_new, (method)qrm_free, sizeof(t_qrm), 0L, A_SYM, A_DEFLONG, 0);
    class_addmethod(c, (method)qrm_dsp64, "dsp64", A_CANT, 0);
    class_addmethod(c, (method)qrm_set, "set", A_SYM, 0);
    class_addmethod(c, (method)qrm_in1, "in1", A_LONG, 0);
    class_addmethod(c, (method)qrm_int, "int", A_LONG, 0);
    class_addmethod(c, (method)qrm_assist, "assist", A_CANT, 0);
    class_addmethod(c, (method)qrm_dblclick, "dblclick", A_CANT, 0);
    class_addmethod(c, (method)qrm_notify, "notify", A_CANT, 0);
    class_addmethod(c, (method)qrm_setvsize, "set_vector_size", A_LONG, 0);
    class_addmethod(c, (method)qrm_getvsize, "get_vector_size", A_DEFSYM, 0);
    class_addmethod(c, (method)qrm_set_fft_size, "fft_size", A_LONG, 0);
    class_addmethod(c, (method)qrm_set_thresh, "set_thresh", A_FLOAT, 0);
    class_addmethod(c, (method)qrm_bang, "bang", A_CANT, 0);                  //for experimental purposes
    class_addmethod(c, (method)qrm_list, "list", A_CANT, 0);
    class_dspinit(c);
    class_register(CLASS_BOX, c);
    qrm_class = c;
}

void qrm_perform64(t_qrm *x, t_object *dsp64, double **ins, long numins, double **outs, long numouts, long sampleframes, long flags, void *userparam)
{
    t_double    *in = ins[0];
    t_double    *out = outs[0];
    long            n = sampleframes;
    t_float        *tab;
    double        temp;
    double        f;
    long        index, chan, frames, nc;
    t_buffer_obj    *buffer = buffer_ref_getobject(x->l_buffer_reference);

    tab = buffer_locksamples(buffer);
    if (!tab)
        goto zero;

    frames = buffer_getframecount(buffer);
    nc = buffer_getchannelcount(buffer);
    x->sr = buffer_getsamplerate(buffer);
    chan = MIN(x->l_chan, nc);
    while (n--) {
        temp = *in++;
        f = temp + 0.5;
        index = f;
        if (index < 0)
            index = 0;
        else if (index >= frames)
            index = frames - 1;
        if (nc > 1)
            index = index * nc + chan;
        *out++ = tab[index];
    }
    buffer_unlocksamples(buffer);
    return;
zero:
    while (n--)
        *out++ = 0.0;
}

//UNUSED
//static void print_result(float bw, t_qrm *x) {
//    for(int k=0;k<x->fft_size / 2;k++)
//        post("%d: (%f) (%f, %fi), %f, %f",
//             k,
//             k*bw,
//             x->outs[2*k],
//             x->outs[2*k+1],
//             sqrt(pow(x->outs[2*k],2)+pow(x->outs[2*k+1],2)),  //could remove the sqrt here
//             atan2(x->outs[2*k+1],x->outs[2*k])
//             );
//}

//when we get an int, set the cursor and print the fft input at that point in the target buffer, then the fft of that window
//needs to be refactored
void qrm_int(t_qrm *x, long n)
{
    if(n>=0)
    {
        x->cursor = n;
        
        t_float *tab;
        t_buffer_obj    *buffer = buffer_ref_getobject(x->l_buffer_reference);
        x->sr = buffer_getsamplerate(buffer);
        tab = buffer_locksamples(buffer);
        if(!tab)
            goto zero;
        //get buffer length. If window at cursor exceeds buffer length, truncate window.
        long frames = buffer_getframecount(buffer);
        long i = MIN(x->cursor, frames - x->fft_size);
        
        //get channels
        long nc = buffer_getchannelcount(buffer);
        long chan = MIN(x->l_chan, nc);
        
        //load window into fft input
        for(int j=0; j< x->fft_size;j++){
            x->in[j] = tab[(j+i)*nc+chan];
            //x->in[2*j+1] = 0;  //no imaginary component
            //post("%d: %f", j, x->in[j]);
            
        }
        buffer_unlocksamples(buffer);

        
        //perform fft
        hann_window(x, x->in);  //window the input
        clock_t t1, t2;         //timing variables
        t1=clock();             //start the clock
        fftw_execute(x->p);     //do that FFT
        t2 = clock();           //stop the clock
//        post("qrm: fft took %f s", (double)(t2-t1)/CLOCKS_PER_SEC);
        
        //find bin width based on window size and sample rate
        float bw = x->sr / x->fft_size;
        //print_result(bw, x);
        
        //derive magnitude & phase, find sum and max
        x->sum = 0;
        double temp=0;
        for(int i=0;i<x->fft_size;i++){
            temp = sqrt(pow(x->outs[2*i],2) + pow(x->outs[2*i+1],2));
            x->sum+=temp;
            if(temp>=x->max_peak) x->max_peak = temp;
            x->mag_spec[i]=temp;
            x->phase_spec[i]=atan2(x->outs[2*i+1],x->outs[2*i]);
        }
        
        
        
        //find peaks
        int c = 0;
        for(int i=1;i<x->fft_size-1;i++)
        {
            if((x->mag_spec[i-1]<x->mag_spec[i]) && (x->mag_spec[i+1]<x->mag_spec[i]))
            {
                //now check if the detected peak is within the threshold of our max peak
                if(20*log10(x->mag_spec[i] / x->max_peak) > x->thresh)
                {
                    //only write if it is loud enough
                    x->peaks[c]=i;
                    c++;
                }
            }
            x->peaks[c]=-1;
        }
        //let's track the number of peaks we've found (+1)
        x->num_peaks=c;
        c=0;

//        while(x->peaks[c]>=0){
//            post("qrm: peak at bin %ld: (%f Hz)", x->peaks[c], x->peaks[c]*bw);
//            c++;
//        }
        
        //cook the pitch with a fractional bin analysis
        // /fractional_bins = [ 0, log(/spectrum[[/i+1]] / /spectrum[[/i -1]]) / (2 * log(pow(/spectrum[[/i]],2) / (/spectrum[[/i-1]] * /spectrum[[/i+1]]))), 0 ],
        for(int i=0; i<x->num_peaks;i++){
            long ind = x->peaks[i];
            double f = ind;
            //the following can be found in the literature on fractional bin extraction
            //the log functions improve accuracy, but could be omitted or other numerical methods tried for efficiency
            f += log(x->mag_spec[ind+1]/x->mag_spec[ind - 1]) / (2*log(pow(x->mag_spec[ind],2) / (x->mag_spec[ind+1] * x->mag_spec[ind - 1])) );
            x->cooked[2*i] = f*bw;      //add cooked frequency to output list
            x->cooked[2*i+1] = x->mag_spec[ind] / x->max_peak;  //add normalized amplitude to output list (for now)
//            post("qrm: cooked bin %f: (%f Hz)", f, f*bw);
        }
        qrm_list_out(x, x->cooked, x->num_peaks * 2, x->slice_out);     //list the cooked (frequency, amplitude) pairs out the outlet
//        buffer_unlocksamples(buffer); //maybe move this earlier
        return;
        
        
    zero:
//        outlet_float(x->f_out, 0.0);
        object_error((t_object*)x, "Did not get buffer.");
    } else {
        x->cursor = 0;
        object_warn((t_object*)x, "Cursor values must be integers greater than zero. Setting to zero.");
    }
}

void qrm_list(t_qrm *x, t_symbol *msg, long argc, t_atom *argv){
    
    if(argc != 2){
        object_error((t_object*)x,"Only supports list length of 2: (cursor1, cursor2)");
        return;
    }
    
    //first, let's double check that the buffer sample rate and channel count hasn't changed
    x->sr = buffer_getsamplerate(buffer_ref_getobject(x->l_buffer_reference));
    qrm_in1(x,buffer_getchannelcount(buffer_ref_getobject(x->l_buffer_reference)));
    
    long buffer_len = buffer_getframecount(buffer_ref_getobject(x->l_buffer_reference));
    //error("qrm: buffer framecount: %ld", buffer_len);

    long c1 = atom_getlong(argv);
    long c2 = atom_getlong(argv +1);
    if(c1<0 || c1 >= buffer_len)
    {
        //buffer has no negative indices
        if(c1<0) c1=0;
        //need at least 1 fft frame of runway before end of buffer
        if(c1>buffer_len - 1 - x->fft_size) c1 = buffer_len - 1 - x->fft_size;
        object_warn((t_object*)x,"cursor1 position must be between zero and buffer length. Setting to %ld", c1);
        
    } else if(c2<0 || c2 >= buffer_len)
    {
        if(c2<0) c2=0;
        if(c2>buffer_len - 1 - x->fft_size) c2 = buffer_len - 1 - x->fft_size;
        object_warn((t_object*)x,"cursor2 position must be between zero and buffer length. Setting to %ld", c2);

    }
    
    x->cursor = c1;
    x->cursor2 = c2;
//    post("qrm: received cursor positions %ld, %ld", x->cursor, x->cursor2);  //this works
    
    //check that our cursors are in order
    if(c1>c2){
        object_error((t_object*)x,"cursor position 2 must be greater than cursor position 1");
        return;
    }
    
    //adjust cursor 1 to first peak in buffer region
    findMaxInBuffer(x);
    c1 = x->region_max_ind;
//    post("qrm: resetting cursor to attack at index %d", c1);
    outlet_int(x->out, c1);
    
    //set analysis points; needs some error checking
    long step = (c2-c1)/4;
//    post("qrm: setting slice indexes at:");
    x->idxs[0]=x->slices[0].index_in_buffer;
    for(int i=0;i<NUMSLICES; i++)
    {
        x->slices[i].index_in_buffer = c1+i*step;
//        post("qrm: \t %d", x->slices[i].index_in_buffer);
        
        x->idxs[i]=x->slices[i].index_in_buffer - x->idxs[0];
        
    }
    
    //TODO:
    //conduct qrm_int operations at c1, then perform ffts at the remaining 4 points, logging the amplitudes at bins identified as peaks at c1
    
        
        t_float *tab;
        t_buffer_obj    *buffer = buffer_ref_getobject(x->l_buffer_reference);
        x->sr = buffer_getsamplerate(buffer);
        tab = buffer_locksamples(buffer);
        if(!tab)
            goto zero;
        //get buffer length. If window at cursor exceeds buffer length, truncate window.
//        long frames = buffer_getframecount(buffer);
//        long i = x->cursor + x->fft_size;
//        i = MIN(x->cursor, frames - x->fft_size);
        
        //get channels
        long nc = buffer_getchannelcount(buffer);
        long chan = MIN(x->l_chan, nc);
        
        
        //load window into slice input buffers; window as we go
    for(int k=0;k<x->fft_size;k++){
        for(int j=0; j<NUMSLICES;j++){
            x->slices[j].in[k]=tab[(k+x->slices[j].index_in_buffer) * nc+chan] * x->window_function[k];
        }
    }
    buffer_unlocksamples(buffer);
        
        //perform ffts
    for(int i=0;i<NUMSLICES; i++){
        fftw_execute(x->slices[i].p);
    }

        //find bin width based on window size and sample rate (move this to set_fft_size
    float bw = x->sr / x->fft_size;   //TODO: put this in new and set_fft_size

        //derive magnitude & phase of slice[0], find sum and max
    x->slices[0].sum = 0;
    x->slices[0].max_peak=0;
        
        double temp=0;
        for(int i=0;i<x->fft_size;i++){
            temp = sqrt(pow(x->slices[0].outs[2*i],2) + pow(x->slices[0].outs[2*i+1],2));
            x->slices[0].sum += temp;
            if(temp>=x->slices[0].max_peak)
            {
                x->slices[0].max_peak = temp;
                //now we also need mag and phase spec for the other slices for this bin
                for(int j=1;j<NUMSLICES;j++){
                    x->slices[j].mag_spec[i]=sqrt(pow(x->slices[j].outs[2*i],2) + pow(x->slices[j].outs[2*i+1],2));
                    x->slices[j].phase_spec[i]=atan2(x->slices[j].outs[2*i+1],x->slices[j].outs[2*i]);
                }
            }
            x->slices[0].mag_spec[i]=temp;
            x->slices[0].phase_spec[i]=atan2(x->slices[0].outs[2*i+1],x->slices[0].outs[2*i]);
            

        }
        
        


        //find peaks in slice 0
        int c = 0;
        for(int i=1;i<x->fft_size-1;i++)
        {
            if((x->slices[0].mag_spec[i-1] < x->slices[0].mag_spec[i]) && (x->slices[0].mag_spec[i+1] < x->slices[0].mag_spec[i]))
            {
                //now check if the detected peak is within the threshold of our max peak
                if(20*log10(x->slices[0].mag_spec[i] / x->slices[0].max_peak) > x->thresh)
                {
                    //only write if it is loud enough
                    x->peaks[c]=i;
                    c++;
                }
               
            }
            x->peaks[c]=-1;
        }
        //let's track the number of peaks we've found (+1)
        x->num_peaks=c;
        c=0;


        
        
        
//        while(x->peaks[c]>=0){
//            post("qrm: peak at bin %ld: (%f Hz)", x->peaks[c], x->peaks[c]*bw);
//            c++;
//        }
    
        //work out decay rates from peak bins
    temp=0;
    for(int i=0; i<x->num_peaks; i++){
        for(int j = 0;j<NUMSLICES;j++) {
            x->tempY[j]=x->slices[j].mag_spec[x->peaks[i]]+EPSILON;
//            post("qrm: Peak %d: Amp %d: %f", i, j, x->tempY[j]);
        }
        exp_fit(x->idxs, x->tempY, NUMSLICES, x->tempAB, 10 );
        x->amps[i] = x->tempAB[0];
        temp=MAX(temp, x->tempAB[0]);
        x->dr[i] = x->tempAB[1]*x->sr; //we multiply by sampling rate here to correct for scaling
        
    }
    
//    //normalize amps
//    for(int i=0; i<x->num_peaks; i++) x->amps[i] /= temp;
    
    
        
        //cook the pitch with a fractional bin analysis
        // /fractional_bins = [ 0, log(/spectrum[[/i+1]] / /spectrum[[/i -1]]) / (2 * log(pow(/spectrum[[/i]],2) / (/spectrum[[/i-1]] * /spectrum[[/i+1]]))), 0 ],
        for(int i=0; i<x->num_peaks;i++){
            long ind = x->peaks[i];
            double f = ind;
            //the following can be found in the literature on fractional bin extraction
            //the log functions improve accuracy, but could be omitted or other numerical methods tried for efficiency
            f += log(x->slices[0].mag_spec[ind+1]/x->slices[0].mag_spec[ind - 1]) / (2*log(pow(x->slices[0].mag_spec[ind],2) / (x->slices[0].mag_spec[ind+1] * x->slices[0].mag_spec[ind - 1])) );
            x->model[3*i] = f*bw;      //add cooked frequency to output list
            x->model[3*i+1] = x->amps[i] / x->slices[0].max_peak;  //add normalized amplitude to output list;
            x->model[3*i+2] = MAX(2.0, ABS(x->dr[i])); //impose a constraint of positive and greater than threshold
            //catch NaNs
            if(x->model[3*i+1] != x->model[3*i+1] || x->model[3*i+1] != x->model[3*i+1]){
                x->model[3*i+1] = 0.0;
                x->model[3*i+2] = 10;
            }
//            post("qrm: cooked bin %f: (%f Hz)", f, f*bw);
            
        }
        qrm_list_out(x, x->model, x->num_peaks * 3, x->model_out);     //list the cooked (frequency, amplitude) pairs out the outlet
//        buffer_unlocksamples(buffer); //maybe move this earlier
        return;
        
        
    zero:
//        outlet_float(x->f_out, 0.0);
        object_error((t_object*)x,"Did not get buffer.");
}




void qrm_set(t_qrm *x, t_symbol *s)
{
    if (!x->l_buffer_reference)
        x->l_buffer_reference = buffer_ref_new((t_object *)x, s);
    else
        buffer_ref_set(x->l_buffer_reference, s);
    
    //the buffer may have a different sample rate.  Let's find out what it is and reset our SR to match.
    t_buffer_obj    *buffer = buffer_ref_getobject(x->l_buffer_reference);
    x->sr = buffer_getsamplerate(buffer);
    long nc = buffer_getchannelcount(buffer);
    qrm_in1(x,nc);
}

void qrm_setvsize(t_qrm *x, long n)
{
    if(n>0){
        x->sample_vector_size =n;
        object_warn((t_object*)x,"Vector size will be set to %d", n);
    } else {
        x->sample_vector_size = 1;
        object_warn((t_object*)x,"Sample Vector Size must be a positive integer. Setting to 1");
    }
}

void qrm_set_fft_size(t_qrm *x, long n)
{
    if(n>0){
        //bitwise-& checks for power of 2
        if((n & (n-1)) == 0){
            x->fft_size = n;
            
            //when resetting fft size we need to free and re-allocate memory
            //otherwise, we get a crash
            if(x->in != NULL) fftw_free((char *)x->in); x->in = NULL;
            if(x->outs !=NULL) fftw_free((char *)x->outs); x->outs = NULL;
            x->in = (double *) fftw_malloc(sizeof(double) * (x->fft_size));
            x->outs = (double *) fftw_malloc(sizeof(double) * (x->fft_size)*2);
            memset(x->in, '\0', x->fft_size * sizeof(double));
            memset(x->outs, '\0', x->fft_size * 2 * sizeof(double));
            
            //reset window function
            if(x->window_function !=NULL) free(x->window_function);
            x->window_function = malloc(sizeof(double)*((int)x->fft_size));
            hann_window_gen(x);
            
            
            //make a new plan. If one already exists, kill it and build a new one for the new fft size.
            //note, using the FFTW_MEASURE flag adds a beat, but optimizes for fast execution.
            //one could also use the FFTW_PATIENT flag here to really optimize at the expense an even longer wait.
            if(x->p !=NULL) fftw_destroy_plan(x->p);
            x->p = fftw_plan_dft_r2c_1d((int)x->fft_size, x->in, (fftw_complex *)x->outs, FFTW_MEASURE);
            
            if(x->mag_spec !=NULL) free(x->mag_spec); x->mag_spec = NULL;
            if(x->phase_spec !=NULL) free(x->phase_spec); x->phase_spec = NULL;
            x->mag_spec = malloc(sizeof(double)*(x->fft_size));     //these could be half as long
            x->phase_spec = malloc(sizeof(double)*(x->fft_size));
            
            if(x->peaks !=NULL) free(x->peaks); x->peaks=NULL;
            x->peaks = malloc(sizeof(long)*(x->fft_size / 2));
            
            //we also need to change the size of our cooked array:
            if(x->cooked !=NULL) free(x->cooked); x->cooked = NULL;
            x->cooked = malloc(sizeof(double) * (x->fft_size));
            
            if(x->model !=NULL) free(x->model); x->model = NULL;
            x->model = malloc(sizeof(double) * (x->fft_size * 3));
           
            if(x->ap0 !=NULL) free(x->ap0);
            x->ap0 = malloc(sizeof(double)*(x->fft_size / 2));
            if(x->ap1 !=NULL) free(x->ap1);
            x->ap1 = malloc(sizeof(double)*(x->fft_size / 2));
            if(x->ap2 !=NULL) free(x->ap2);
            x->ap2 = malloc(sizeof(double)*(x->fft_size / 2));
            if(x->ap3 !=NULL) free(x->ap3);
            x->ap3 = malloc(sizeof(double)*(x->fft_size / 2));
            if(x->ap4 !=NULL) free(x->ap4);
            x->ap4 = malloc(sizeof(double)*(x->fft_size / 2));
            
            if(x->amps !=NULL) free(x->amps);
            x->amps = malloc(sizeof(double) * x->fft_size);
            
            if(x->dr !=NULL) free(x->dr);
            x->dr=malloc(sizeof(double) * x->fft_size);
            
            for(int i=0; i<NUMSLICES; i++){
                if(x->slices[i].in !=NULL) fftw_free((char * )x->slices[i].in);
                if(x->slices[i].outs !=NULL) fftw_free((char *)x->slices[i].outs);
                x->slices[i].in = (double *) fftw_malloc(sizeof(double) * (x->fft_size));
                x->slices[i].outs = (double *) fftw_malloc(sizeof(double) * (x->fft_size)*2);
                memset(x->slices[i].in, '\0', x->fft_size * sizeof(double));
                memset(x->slices[i].outs, '\0', x->fft_size * 2 * sizeof(double));
                if(x->slices[i].mag_spec !=NULL) free(x->slices[i].mag_spec);
                x->slices[i].mag_spec = malloc(sizeof(double) * x->fft_size);
                if(x->slices[i].phase_spec !=NULL) free(x->slices[i].phase_spec);
                x->slices[i].phase_spec = malloc(sizeof(double) * x->fft_size);
                
                if(x->slices[i].p !=NULL) fftw_destroy_plan(x->slices[i].p);
                x->slices[i].p = fftw_plan_dft_r2c_1d((int)x->fft_size, x->slices[i].in, (fftw_complex *)x->slices[i].outs, FFTW_MEASURE);
            }
            

            object_post((t_object*)x,"FFT size set to %d", n);
            
        } else {
            object_error((t_object*)x,"FFT size must be a power of 2");
        }
    } else {
        object_error((t_object*)x,"FFT size must be a positive integer");
    }
}

void qrm_getvsize(t_qrm *x)
{
    outlet_int(x->out, x->sample_vector_size);
    object_post((t_object*)x,"Sample Vector Size is %d", x->sample_vector_size);
}

void qrm_in1(t_qrm *x, long n)
{
    if (n)
        x->l_chan = MAX(n, 1) - 1;
    else
        x->l_chan = 0;
}


void qrm_dsp64(t_qrm *x, t_object *dsp64, short *count, double samplerate, long maxvectorsize, long flags)
{
    dsp_add64(dsp64, (t_object *)x, (t_perfroutine64)qrm_perform64, 0, NULL);
}


// this lets us double-click on index~ to open up the buffer~ it references
void qrm_dblclick(t_qrm *x)
{
    buffer_view(buffer_ref_getobject(x->l_buffer_reference));
}

void qrm_assist(t_qrm *x, void *b, long m, long a, char *s)
{
    if (m == ASSIST_OUTLET)
        switch(a){
            case 0: sprintf(s,"(signal) Placeholder for impulse function"); break;
            case 1: sprintf(s,"Slice Out (list)"); break;
            case 2: sprintf(s,"Model Out (list)"); break;
            case 3: sprintf(s,"Buffer Index of Attack (int)"); break;
        }
    else if(m==ASSIST_INLET) {
        switch (a) {
        case 0:    sprintf(s,"(signal) Sample Index");    break;
        case 1:    sprintf(s,"Audio Channel In buffer~");    break;
        }
    }
}

void qrm_set_thresh(t_qrm *x, double n)
{
    if(n<=0){
        x->thresh = n;
        object_post((t_object*)x,"qrm: Threshold set to %f dB relative to peak value.", n);
    } else {
        object_error((t_object*)x,"qrm: Threshold should be a negative number (dB relative to peak value)");
    }
}

//eventually, I'd like the bang method to find the first peak in the buffer and return the resonance at that point. For now, it outputs the last frame
void qrm_bang(t_qrm *x)
{
    //    t_atom myList[3];
    //    double theNumbers[3];
    //    short i;
    //
    //    theNumbers[0] = 23.01;
    //    theNumbers[1] = 12.02;
    //    theNumbers[2] = 5.03;
    //    for (i=0; i < 3; i++) {
    //        atom_setfloat(myList+i,theNumbers[i]);
    //    }
    //    outlet_list(x->d_out, 0L, 3, &myList);
    qrm_list_out(x, x->cooked, x->num_peaks*2, x->slice_out);     //list the cooked frequencies out the left outlet
    qrm_list_out(x, x->model, x->num_peaks * 3, x->model_out);    //list the most recent model out the second left outlet
}

void qrm_list_out(t_qrm *x, double* a, long l, void* outlet)
{
    t_atom list[l];
    for(int i = 0; i<l; i++){
        atom_setfloat(list+i, a[i]);
    }
    outlet_list(outlet, 0L, l, list);
}

void *qrm_new(t_symbol *s, long chan)
{
    t_qrm *x = object_alloc(qrm_class);
    dsp_setup((t_pxobject *)x, 1);
    intin((t_object *)x,1);
    x->out = outlet_new((t_object *)x, "int");  //right outlet
//    x->f_out = outlet_new((t_object *)x, "float");
    x->model_out = outlet_new((t_object *)x, NULL); //outlet for models
    x->slice_out = outlet_new((t_object *)x, NULL);     //outlet for slices
    outlet_new((t_object *)x, "signal");        //left outlet
    qrm_set(x, s);
    qrm_in1(x,chan);
    x->sr = sys_getsr();                        //initially, adopt the system sample rate. We will reset later.
    post("qrm: SR = %f", x->sr);
    x->fft_size = 4096;
    
    x->in = (double *) fftw_malloc(sizeof(double) * (x->fft_size));
    x->outs = (double *) fftw_malloc(sizeof(double) * (x->fft_size)*2);     //output is twice the size of input since we are going real->complex
    memset(x->in, '\0', x->fft_size * sizeof(double));  //initialize to zero
    memset(x->outs, '\0', x->fft_size * 2 * sizeof(double));
    x->p = fftw_plan_dft_r2c_1d((int)x->fft_size, x->in, (fftw_complex *)x->outs, FFTW_MEASURE); //plan for sinusoidal model extraction
    x->mag_spec = malloc(sizeof(double)*(x->fft_size));     //these could be half as long
    x->phase_spec = malloc(sizeof(double)*(x->fft_size));
    x->peaks = malloc(sizeof(long) * (x->fft_size / 2));
    x->cooked = malloc(sizeof(double) * (x->fft_size));
    x->model = malloc(sizeof(double) * (x->fft_size * 3));
    x->analysis_points = malloc(sizeof(long) * 5);      //we are going to analyze just 5 points to extract decay rates
    x->ap0 = malloc(sizeof(double) * x->fft_size / 2);
    x->ap1 = malloc(sizeof(double) * x->fft_size / 2);
    x->ap2 = malloc(sizeof(double) * x->fft_size / 2);
    x->ap3 = malloc(sizeof(double) * x->fft_size / 2);
    x->ap4 = malloc(sizeof(double) * x->fft_size / 2);
    x->idxs = malloc(sizeof(long) * NUMSLICES);
    x->tempY = malloc(sizeof(double) * NUMSLICES);
    x->tempAB = malloc(sizeof(double) * 2);
    x->amps = malloc(sizeof(double) * x->fft_size);
    x->dr = malloc(sizeof(double) * x->fft_size);
    
    //allocate slice fft plans and input/output buffers
    for(int i=0;i<NUMSLICES;i++){
        x->slices[i].in = (double *) fftw_malloc(sizeof(double) * (x->fft_size));
        x->slices[i].outs = (double *) fftw_malloc(sizeof(double) * (x->fft_size)*2);
        memset(x->slices[i].in, '\0', x->fft_size * sizeof(double));  //initialize to zero
        memset(x->slices[i].outs, '\0', x->fft_size * 2 * sizeof(double));
        x->slices[i].p = fftw_plan_dft_r2c_1d((int)x->fft_size, x->slices[i].in, (fftw_complex *)x->slices[i].outs, FFTW_MEASURE);
        x->slices[i].mag_spec = malloc(sizeof(double)*x->fft_size);
        x->slices[i].phase_spec = malloc(sizeof(double)*x->fft_size);
    }
    
    x->thresh = -32;
    x->num_peaks = 0;
    x->window_function = malloc(sizeof(double)*((int)x->fft_size));
    hann_window_gen(x);
    
    
//    //test exponential fitting
//    long *xVals = malloc(sizeof(long)*5);
//    double *yVals = malloc(sizeof(double)*5);
//
//    for(long i=1;i<6;i++){
//        xVals[i-1]=i;
//        yVals[i-1]= 3 * exp(-0.002 * (double)i)+0.005*random();
//    }
//
//    double *eftest = exp_fit(xVals,yVals, 5);
//    post("qrm: exp_fit test: y=A*exp(B*x) for vals (%d, %lf), (%d, %lf), (%d, %lf), (%d, %f), (%d, %f)",
//         xVals[0], yVals[0],
//         xVals[1], yVals[1],
//         xVals[2], yVals[2],
//         xVals[3], yVals[3],
//         xVals[4], yVals[4]
//         );
//    post("A: %f", eftest[0]);
//    post("B: %f", eftest[1]);   //works
    
    return (x);     //return our new object struct to the caller
}


void qrm_free(t_qrm *x)
{
    dsp_free((t_pxobject *)x);
    fftw_destroy_plan(x->p);
    for(int i=0;i<5;i++){
        fftw_destroy_plan(x->slices[i].p);
        if(x->slices[i].mag_spec !=NULL) free(x->slices[i].mag_spec);
        if(x->slices[i].phase_spec !=NULL) free(x->slices[i].phase_spec);
    }
    if(x->in != NULL) fftw_free((char *)x->in);
    if(x->outs !=NULL) fftw_free((char *)x->outs);
    if(x->mag_spec !=NULL) free(x->mag_spec);
    if(x->phase_spec !=NULL) free(x->phase_spec);
    if(x->peaks !=NULL) free(x->peaks);
    if(x->cooked !=NULL) free(x->cooked);
    if(x->model !=NULL) free(x->model);
    if(x->idxs !=NULL) free(x->idxs);
    if(x->tempY !=NULL) free(x->tempY);
    if(x->tempAB !=NULL) free(x->tempAB);
    if(x->amps !=NULL) free(x->amps);
    if(x->dr !=NULL) free(x->dr);
    
    if(x->window_function !=NULL) free(x->window_function);
    if(x->analysis_points !=NULL) free(x->analysis_points);
    if(x->ap0 !=NULL) free(x->ap0);
    if(x->ap1 !=NULL) free(x->ap1);
    if(x->ap2 !=NULL) free(x->ap2);
    if(x->ap3 !=NULL) free(x->ap3);
    if(x->ap4 !=NULL) free(x->ap4);
    object_free(x->l_buffer_reference);
}


t_max_err qrm_notify(t_qrm *x, t_symbol *s, t_symbol *msg, void *sender, void *data)
{
    return buffer_ref_notify(x->l_buffer_reference, s, msg, sender, data);
}

//TODO: need to rethink this function to scale with multiple ffts
//hann_window function
void hann_window(t_qrm *x, double *a){
    for(int i=0;i<x->fft_size;i++){
        a[i] *= pow(sin(PI*i / x->fft_size),2);
    }
}

void hann_window_gen(t_qrm *x)
{
    for(int i=0;i<x->fft_size;i++){
        x->window_function[i] = pow(sin(PI*i / x->fft_size),2);
    }
}

//method to perform exponential fitting via least squares
//the bias term 'wt' allows for weighting the initial value to ensure closer approximation of amplitude
void exp_fit(long *xVals, double *yVals, long n, double* out, double wt)
{
//    double *out = malloc(sizeof(double)*2);
    out[0] =0;
    out[1] = 0;
    double sum_Y=0;
    double sum_XY=0;
    double sum_X2Y=0;
    double sum_YlnY=0;
    double sum_XYlnY=0;
    double bias = wt;
    
    for(int i=0;i<n;i++){
        if(i==0 || i==n-1){ bias=wt;}else{bias = 1.0;}
        double XY = bias*(double)(xVals[i] * yVals[i]);
        double X2Y = bias*((double)xVals[i]) * XY;
        double YlnY = bias*(yVals[i] * log(yVals[i]));
        double XYlnY = bias*(double)xVals[i] * YlnY;
        
        sum_Y += bias*yVals[i];
        sum_XY += XY;
        sum_X2Y += X2Y;
        sum_YlnY += YlnY;
        sum_XYlnY += XYlnY;
    }
//    post("qrm:XY: %f, %f, %f, %f, %f", sum_Y, sum_XY, sum_X2Y, sum_YlnY, sum_XYlnY);
    
    double den = sum_Y * sum_X2Y - sum_XY * sum_XY + EPSILON;
    
    out[0] = exp((sum_X2Y * sum_YlnY - sum_XY * sum_XYlnY)/(den));
    out[1] = (sum_Y * sum_XYlnY - sum_XY * sum_YlnY)/(den);
    
//    post("%f", out[0]);
//    post("%f", out[1]);
//    return out;
}

void findMaxInBuffer(t_qrm* x){
    t_float *tab;
    t_buffer_obj    *buffer = buffer_ref_getobject(x->l_buffer_reference);
    x->sr = buffer_getsamplerate(buffer);
    tab = buffer_locksamples(buffer);
    if(!tab){
        goto zero;
    }
    //get buffer length. If window at cursor exceeds buffer length, truncate window.
    long frames = buffer_getframecount(buffer);
    long i = x->cursor + x->fft_size;
    i = MIN(x->cursor, frames - x->fft_size);
    
    //get channels
    long nc = buffer_getchannelcount(buffer);
    long chan = MIN(x->l_chan, nc);
    
    x->region_max_ind = x->cursor;
    x->max_val = 0.0;
    double t = 0;
    //start at the end and work back
    for(long j = x->cursor; j< x->cursor2;j++){
        t = ABS(tab[(j)*nc+chan]);
        if(t > x->max_val){
            x->max_val = t;
            x->region_max_ind = j;
        };
        
    }
    buffer_unlocksamples(buffer);
    return;
    
zero:
//    outlet_float(x->f_out, 0.0);
    object_error((t_object*)x,"qrm:findMaxInBuffer: Error: did not get buffer.");
}

