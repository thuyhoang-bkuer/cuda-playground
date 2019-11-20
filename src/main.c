/*
    <A simulation envoirnment for Deep Convolution Neural Networks>
    Copyright (C) <2012-213>  <Sajid Anwar> <engrsajidanwar@gmail.com>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
#define __freadDebug

#include <stdio.h>
#include <stdlib.h>
#include "readMNISTDB.h"
#include "readCIFAR.h"
#include "layer.h"
#include "debug_extra.h"
#include "dataset.h"
#include "cnn.h"
#include "batch.h"

int main(int argc, char *argv[])
{
	//reading random seed from the OS (can be part of the execution script)
	int seed = rand();
	if (argc > 1)
		seed = atoi(argv[1]);

	srand(seed);


	 /*
	 *	Architecture
	 * 								(relu)					(relu)							(relu)						(relu)							(tanh)					(tanh)			(sigmoid)
	 * 	Input ------> Conv1 ------> Subsampling2 -------> Conv3 -------> Subsampling4 -------> Flatten --------> FC ------> Output 
	*/

	cnn_t*  cnn_specs = (cnn_t *) malloc(sizeof(cnn_t));
	cnn_specs->nlayers = 8;
	cnn_specs->pool_type = 2;    //avg = 1 max = 2 stochastic pooling = 3


	/*	Activation Type
	 * 	1 - sigmoid
	 * 	2 - tanh
	 * 	3 - relu
	*/
	// cnn_specs->act_maps = (int *) calloc(cnn_specs->nlayers, sizeof(int));
	// cnn_specs->act_maps[2] = 3;		// relu
	// cnn_specs->act_maps[4] = 3;		// relu
	// cnn_specs->act_maps[5] = 2;		// relu
	// cnn_specs->act_maps[6] = 2;		// relu
	// cnn_specs->act_maps[7] = 1;		// relu

	// cnn_specs->no_fmaps = (int *) malloc(sizeof(int) * cnn_specs->nlayers);
	// cnn_specs->no_fmaps[0] = 1;			//input
	// cnn_specs->no_fmaps[1] = 16;		//Conv-ReLU 1
	// cnn_specs->no_fmaps[2] = 16;		//Subsample 2
	// cnn_specs->no_fmaps[3] = 24;   	//Conv-ReLU 3
	// cnn_specs->no_fmaps[4] = 24; 		//Subsample 4
	// cnn_specs->no_fmaps[5] = 48;  	//Flatten
	// cnn_specs->no_fmaps[6] = 128; 	//
	// cnn_specs->no_fmaps[7] = 10; 		//final

	// cnn_specs->fmap_size = (int *) malloc(sizeof(int) * cnn_specs->nlayers);

	// cnn_specs->fmap_size[0] = 1024;	//32 x 32 x 1
	// cnn_specs->fmap_size[1] = 784;	//28 x 28 
	// cnn_specs->fmap_size[2] = 196; 	//14 x 14 
	// cnn_specs->fmap_size[3] = 100; 	//10 x 10 
	// cnn_specs->fmap_size[4] = 25; 	// 5 x  5 x 24
	// cnn_specs->fmap_size[5] = 1; 	// 1 x  1 x 48
	// cnn_specs->fmap_size[6] = 1; 	// 1 x  1 x 128
	// cnn_specs->fmap_size[7] = 1; 	// 1 x  1 x 10

	// cnn_specs->fkernel = (int *) malloc(sizeof(int) * cnn_specs->nlayers);
	// cnn_specs->fkernel[0] = 5;
	// cnn_specs->fkernel[1] = 1;
	// cnn_specs->fkernel[2] = 5;
	// cnn_specs->fkernel[3] = 1;
	// cnn_specs->fkernel[4] = 5;
	// cnn_specs->fkernel[5] = 1;
	// cnn_specs->fkernel[6] = 1;
	// cnn_specs->fkernel[7] = 1;


	cnn_specs->act_maps = (int *) calloc(cnn_specs->nlayers, sizeof(int));
	cnn_specs->act_maps[2] = 3;		// relu
	cnn_specs->act_maps[4] = 3;		// relu
	cnn_specs->act_maps[5] = 2;		// tanh
	cnn_specs->act_maps[6] = 2;		// tanh
	cnn_specs->act_maps[7] = 1;		// sigmoid

	cnn_specs->no_fmaps = (int *) malloc(sizeof(int) * cnn_specs->nlayers);
	cnn_specs->no_fmaps[0] = 1;			//input
	cnn_specs->no_fmaps[1] = 6;			//Conv-ReLU 1
	cnn_specs->no_fmaps[2] = 6;			//Subsample 2
	cnn_specs->no_fmaps[3] = 10;   		//Conv-ReLU 3
	cnn_specs->no_fmaps[4] = 10; 		//Subsample 4
	cnn_specs->no_fmaps[5] = 120;  		//Flatten
	cnn_specs->no_fmaps[6] = 84; 		// Fc
	cnn_specs->no_fmaps[7] = 10; 		//final

	cnn_specs->fmap_size = (int *) malloc(sizeof(int) * cnn_specs->nlayers);

	cnn_specs->fmap_size[0] = 1024;	//32 x 32 x 1
	cnn_specs->fmap_size[1] = 784;	//28 x 28 
	cnn_specs->fmap_size[2] = 196; 	//14 x 14 
	cnn_specs->fmap_size[3] = 100; 	//10 x 10 
	cnn_specs->fmap_size[4] = 25; 	// 5 x  5 x 24
	cnn_specs->fmap_size[5] = 1; 	// 1 x  1 x 48
	cnn_specs->fmap_size[6] = 1; 	// 1 x  1 x 128
	cnn_specs->fmap_size[7] = 1; 	// 1 x  1 x 10

	cnn_specs->fkernel = (int *) malloc(sizeof(int) * cnn_specs->nlayers);
	cnn_specs->fkernel[0] = 5;
	cnn_specs->fkernel[1] = 1;
	cnn_specs->fkernel[2] = 5;
	cnn_specs->fkernel[3] = 1;
	cnn_specs->fkernel[4] = 5;
	cnn_specs->fkernel[5] = 1;
	cnn_specs->fkernel[6] = 1;
	cnn_specs->fkernel[7] = 1;

	cnnlayer_t* headlayer = create_cnn(cnn_specs);
	getDetail(headlayer);

	initialize_weights_matrices(headlayer, true);
    	//display_weights_matrices(headlayer);
	save_trained_network_weights(headlayer, "big_nw_init");

	dataset_t* train_samples = NULL;
	dataset_t* test_samples = NULL;

	/* Exeriment: Read MNIST handwritten digit recognition dataset */
	train_samples = create_data_container(60000, 32 * 32, 10);
	test_samples = create_data_container(10000, 32 * 32, 10);
	readMNISTDB(train_samples, test_samples);	

    /* Exeriment: Read CIFAR-10 handwritten digit recognition dataset */
	//train_samples = create_data_container(50000, 3072, 10);
	//test_samples = create_data_container(10000, 3072, 10);
	//readCIFAR(train_samples, test_samples);	

	/* train floating point CNN	*/
	train_cnn(headlayer, train_samples, test_samples);

	/* Debugging networks */
	//display_cnn_layers(headlayer);
	
	#if 1
	double mcr = 0;	
	// printf("\n Computing Missclassification Rate on Test Set");
	fprintf(stderr,"\n Computing Missclassification Rate on Test Set");
	mcr = d_compute_missclassification_rate(headlayer, test_samples, 3);	
	// printf("\n missclassification rate is : %6.3f\n", mcr);
	fprintf(stderr,"\n missclassification rate is : %6.3f\n", mcr);

	// printf("\n Computing Missclassification Rate on Training Set");
	fprintf(stderr,"\n Computing Missclassification Rate on Training Set");
	mcr = d_compute_missclassification_rate(headlayer, train_samples, 0);	
	fprintf(stderr,"\n missclassification rate on training is : %6.3f\n", mcr);
	// printf("\n missclassification rate on training is : %6.3f\n", mcr);
	#endif

	/* save network parameters (weights and biases) */
	save_trained_network_weights(headlayer, "big_nw_finally");
	
	/* Release dynamically allocated memory */
	destroy_cnn(headlayer);
	free_cnn_specs(cnn_specs);
	free_data_container(train_samples);
	free_data_container(test_samples);

	return 0;
}
