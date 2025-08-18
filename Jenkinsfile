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
	stage('data generation'){ steps{
            dir('model_training'){ dir('data'){
		sh 'python3 maxentropy.py'
		sh 'python3 generate.py'
		sh 'python3 create_database.py'
	    }}
	}}
	stage('training'){ steps{
            dir('model_training'){
		sh 'python3 ml_pytorch.py'
	    }
	}}
	stage('Python Interface'){ steps{
            dir('model_training'){
		sh 'python3 example_use_model.py ../model_training/model10_cpu.pt'
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
