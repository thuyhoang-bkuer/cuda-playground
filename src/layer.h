#ifndef __LAYER_H__
#define __LAYER_H__

typedef struct  nnlayer cnnlayer_t;
typedef struct  cnn cnn_t;
typedef float   real_t;
typedef int     bool_t;

struct cnn
{
	int nlayers;
	int pool_type;

	int*		act_maps;
	int*    fmap_size;
	int*    no_fmaps; 
	int*    fkernel; 
};

struct nnlayer
{
	int no_of_fmaps; 
	int no_of_neurons;
	int fmap_width;
	int fmap_height;
	int fkernel;			// Kernel size
	int layer_type;			// activation (0 = None, 1 = sigmoid, 2 = tangent, 3 = reLU)
	int pool_type;			// pooling (1 = average, 2 = max, 3 = stochastic)
	int subsampling;

	/* Host (HOST CPU) Variables */
	real_t* neurons_input;          // neurons input on host device
	real_t* neurons_output;         // for host, output = transfer function (input)
	real_t* dy_output;              // derivative d/dx (output)
	real_t* error_deltas;           // errors (an deltas) for each neuron
	real_t* weights_matrix;         // weights matrix on host
	real_t* delta_weights;   		// weight updates
	real_t* biases;
	real_t* delta_biases;
	int*	gradientMap;

	/* Device (GPU CUDA) variables */
	real_t* d_neurons_output;       // for device, output = transfer function (input)
	real_t* d_biases;
	real_t* d_delta_biases;
	real_t* d_error_deltas;         // errors (an deltas) for each neuron
	real_t* d_weights;       		// weights matrix on device (GPU) 
	real_t* d_delta_weights;   		// weight updates
	int*	d_gradientMap;

	struct nnlayer* next;           // points to next layer, null for last layer
	struct nnlayer* previous;       // points to prev layer, null for first layer
};

#ifdef __cplusplus
extern "C" 
{
#endif
	cnnlayer_t* create_cnn(cnn_t* cnn_specs);
	void getDetail(struct nnlayer *headlayer);
	int initialize_weights_matrices(struct nnlayer *headlayer, bool_t generate_random);
	int initialize_qweights(struct nnlayer *headlayer);
	int reset_parameters_to_high_precision(cnnlayer_t* headlayer);
	void destroy_cnn(cnnlayer_t* layer);
	void free_cnn_specs(cnn_t* cnn);

#ifdef __cplusplus
}
#endif

#endif
