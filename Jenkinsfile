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
	    dir('cpp_interface'){
	        sh 'mkdir -p build'
			dir('build'){
				sh 'cmake -DCMAKE_PREFIX_PATH=/usr/local/libtorch ..'
				sh 'cmake --build . --config Debug'
			}
	    }
	}}



	//=======//
	// Tests //
	//=======//
	stage('Does it run?'){ steps{
		dir('cpp_interface/build'){
			sh './test_cpp_interface'
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
