pipeline {
    triggers { pollSCM('') }  // Run tests whenever a new commit is detected.
    agent { dockerfile {args '--gpus all'}} // Use the Dockerfile defined in the root Flash-X directory
    environment {
		// Get rid of Read -1, expected <someNumber>, errno =1 error
    	// See https://github.com/open-mpi/ompi/issues/4948
        OMPI_MCA_btl_vader_single_copy_mechanism = 'none'
    }
    stages {

        //=============================//
    	// Set up submodules and amrex //
        //=============================//
    	stage('Prerequisites'){ steps{
	    sh 'mpicc -v'
	    sh 'nvidia-smi'
	    sh 'nvcc -V'
	    sh 'git submodule update --init'
	}}



	//=======//
	// Tests //
	//=======//
	// create_database.py writes dummy_asymptotic.h5, which is only an intermediate.
	// split_database.py turns it into the chunk files that parms["*_database_list"]
	stage('data generation'){ steps{
	    dir('model_training'){ dir('data'){
		    sh 'python3 create_database.py'
		}
		sh 'python3 data/split_database.py --n_chunks 3 data/dummy_asymptotic.h5'
	    }
	}}
	stage('training'){ steps{
            dir('model_training'){
		sh 'python3 ml_pytorch.py'
		sh 'python3 convert_model_to_cpu.py model10_cuda.pt model10_cpu.pt'
	    }
	}}
	stage('Python Interface'){ steps{
            dir('model_training'){
		sh 'python3 example_use_model.py ../model_training/model10_cuda.pt'
	    }
	}}
	stage('C++ Interface'){ steps{
	    dir('cpp_interface'){
		sh 'make'
		sh './test_torch_model ../model_training/model10_cpu.pt'
	    }
	}}

    } // stages{

    post {
        always {
	    cleanWs(
	        cleanWhenNotBuilt: true,
		deleteDirs: true,
		disableDeferredWipeout: false,
		notFailBuild: true,
		patterns: [[pattern: 'submodules', type: 'EXCLUDE']] ) // allow submodules to be cached
	}
    }

} // pipeline{
