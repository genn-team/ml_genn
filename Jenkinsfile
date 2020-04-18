#!groovy​

//--------------------------------------------------------------------------
// Helper functions
//--------------------------------------------------------------------------

// Wrapper around setting of GitHUb commit status curtesy of https://groups.google.com/forum/#!topic/jenkinsci-issues/p-UFjxKkXRI
// **NOTE** since that forum post, stage now takes a Closure as the last argument hence slight modification 
void buildStage(String message, Closure closure) {
    stage(message) {
        try {
            setBuildStatus(message, "PENDING");
            closure();
        }
	catch (Exception e) {
            setBuildStatus(message, "FAILURE");
        }
    }
}

void setBuildStatus(String message, String state) {
    // **NOTE** ManuallyEnteredCommitContextSource set to match the value used by bits of Jenkins outside pipeline control
    step([
        $class: "GitHubCommitStatusSetter",
        reposSource: [$class: "ManuallyEnteredRepositorySource", url: "https://github.com/genn-team/tensor_genn/"],
        contextSource: [$class: "ManuallyEnteredCommitContextSource", context: "continuous-integration/jenkins/branch"],
        errorHandlers: [[$class: "ChangingBuildStatusErrorHandler", result: "UNSTABLE"]],
        statusResultSource: [ $class: "ConditionalStatusResultSource", results: [[$class: "AnyBuildResult", message: message, state: state]] ]
    ]);
}


//--------------------------------------------------------------------------
// Entry point
//--------------------------------------------------------------------------

// All the types of build we'll ideally run if suitable nodes exist
def desiredBuilds = [
    ["cpu_only", "linux", "x86_64", "python27"] as Set,
    ["cpu_only", "linux", "x86_64", "python3"] as Set,
    ["cuda10", "linux", "x86_64", "python27"] as Set,
    ["cuda10", "linux", "x86_64", "python3"] as Set,
    ["cpu_only", "mac", "python27"] as Set,
    ["cpu_only", "mac", "python3"] as Set
]

// Build dictionary of available nodes and their labels
def availableNodes = [:]
for (node in jenkins.model.Jenkins.instance.nodes) {
    if (node.getComputer().isOnline() && node.getComputer().countIdle() > 0) {
        availableNodes[node.name] = node.getLabelString().split() as Set
    }
}

// // Add master if it has any idle executors
// if (jenkins.model.Jenkins.instance.toComputer().countIdle() > 0) {
//     availableNodes["master"] = jenkins.model.Jenkins.instance.getLabelString().split() as Set
// }

// Loop through the desired builds
def builderNodes = []
for (b in desiredBuilds) {
    // Loop through all available nodes
    for (n in availableNodes) {
        // If, after subtracting this node's labels, all build properties are satisfied
        if ((b - n.value).size() == 0) {
            // Add node's name to list of builders and remove it from dictionary of available nodes
            // **YUCK** for some reason tuples aren't serializable so need to add an arraylist
            builderNodes.add([n.key, n.value])
            availableNodes.remove(n.key)
            break
        }
    }
}

//  desiredBuilds:  list of desired node feature sets
// availableNodes:  dict of node feature sets, keyed by node name
//   builderNodes:  list of [node_name, node_features] satisfying desiredBuilds entries


//--------------------------------------------------------------------------
// Parallel build step
//--------------------------------------------------------------------------

// **YUCK** need to do a C style loop here - probably due to JENKINS-27421 
def builders = [:]
for (b = 0; b < builderNodes.size(); b++) {
    // **YUCK** meed to bind the label variable before the closure - can't do 'for (label in labels)'
    def nodeName = builderNodes.get(b).get(0)
    def nodeLabel = builderNodes.get(b).get(1)
   
    // Create a map to pass in to the 'parallel' step so we can fire all the builds at once
    builders[nodeName] = {
        node(nodeName) {
            stage("Checkout (${NODE_NAME})") {
                // Checkout Tensor GeNN
                echo "Checking out Tensor GeNN";
                sh "rm -rf tensor_genn";
		checkout scm
            }

	    buildStage("Setup virtualenv (${NODE_NAME})") {
		// Set up new virtualenv
		echo "Creating virtualenv";
                sh "rm -rf ${WORKSPACE}/venv";
                sh "pip install virtualenv";
                sh "virtualenv ${WORKSPACE}/venv";
		sh """
                    source ${WORKSPACE}/venv/bin/activate
                    pip install -U pip
                    pip install numpy pytest pytest-cov
                """;
	    }

            buildStage("Installing PyGeNN (${NODE_NAME})") {
		// Checkout GeNN
		echo "Checking out GeNN";
		sh "rm -rf genn";
		sh "git clone --branch tensor_genn https://github.com/genn-team/genn.git";

		dir("genn") {
		    // Build dynamic LibGeNN
		    echo "Building LibGeNN";
		    def messages_libGeNN = "libgenn_${NODE_NAME}";
                    sh "rm -f ${messages_libGeNN}";
		    def commands_libGeNN = """
                        make DYNAMIC=1 LIBRARY_DIRECTORY=`pwd`/pygenn/genn_wrapper/  1>>\"${messages_libGeNN}\" 2>&1
                    """;
		    def status_libGeNN = sh script:commands_libGeNN, returnStatus:true;
		    archive messages_libGeNN;
		    if (status_libGeNN != 0) {
			setBuildStatus("Building PyGeNN (${NODE_NAME})", "FAILURE");
		    }

		    // Build PyGeNN module
		    echo "Building and installing PyGeNN";
		    def messages_PyGeNN = "pygenn_${NODE_NAME}";
                    sh "rm -f ${messages_PyGeNN}";
		    def commands_PyGeNN = """
                        source ${WORKSPACE}/venv/bin/activate
                        python setup.py install  1>>\"${messages_PyGeNN}\" 2>&1
                        python setup.py install  1>>\"${messages_PyGeNN}\" 2>&1
                    """;
		    def status_PyGeNN = sh script:commands_PyGeNN, returnStatus:true;
		    archive messages_PyGeNN;
		    if (status_PyGeNN != 0) {
			setBuildStatus("Building PyGeNN (${NODE_NAME})", "FAILURE");
		    }
		}
            }

            buildStage("Running tests (${NODE_NAME})") {
                dir("tensor_genn") {
                    // Install TensorGeNN
                    echo "Installing TensorGeNN";
                    sh """
                        source ${WORKSPACE}/venv/bin/activate
                        pip install .
                    """;

		    dir("tests") {
                        // Run TensorGeNN test suite
			def messages_TensorGeNN = "tensorgenn_${NODE_NAME}";
                        sh "rm -f ${messages_TensorGeNN}";
			def commands_TensorGeNN = """
                            source ${WORKSPACE}/venv/bin/activate
                            pytest -v  1>>\"${messages_TensorGeNN}\" 2>&1
                        """;
			def status_TensorGeNN = sh script:commands_TensorGeNN, returnStatus:true;
			archive messages_TensorGeNN;
			if (status_TensorGeNN != 0) {
			    setBuildStatus("Running tests (${NODE_NAME})", "UNSTABLE");
			}
		    }
		}
	    }
        }
    }
}

// Run builds in parallel
parallel builders;
