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
        reposSource: [$class: "ManuallyEnteredRepositorySource", url: "https://github.com/genn-team/ml_genn/"],
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
    ["cpu_only", "linux", "x86_64"] as Set,
    ["cuda10", "linux", "x86_64"] as Set,
    ["cuda11", "linux", "x86_64"] as Set,
    ["cuda12", "linux", "x86_64"] as Set,
    ["cpu_only", "mac"] as Set,
]

// Build list of available nodes and their labels
def availableNodes = []
for (node in jenkins.model.Jenkins.instance.nodes) {
    if (node.getComputer().isOnline() && node.getComputer().countIdle() > 0) {
        availableNodes.add([node.name, node.getLabelString().split() as Set])
    }
}

// Shuffle nodes so multiple compatible machines get used
Collections.shuffle(availableNodes)

// Loop through the desired builds
def builderNodes = []
for (b in desiredBuilds) {
    // Loop through all available nodes
    for (n = 0; n < availableNodes.size(); n++) {
        // If this node has all desired properties
        if(availableNodes[n][1].containsAll(b)) {
            // Add node's name to list of builders and remove it from dictionary of available nodes
            builderNodes.add(availableNodes[n])
            availableNodes.remove(n)
            break
        }
    }
}

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
                // Checkout ML GeNN
                echo "Checking out ML GeNN";
                sh "rm -rf ml_genn";
                checkout scm
            }

            buildStage("Setup virtualenv (${NODE_NAME})") {
                // Set up new virtualenv
                echo "Creating virtualenv";
                sh """
                rm -rf ${WORKSPACE}/venv
                ${env.PYTHON} -m venv ${WORKSPACE}/venv
                . ${WORKSPACE}/venv/bin/activate
                pip install -U pip
                pip install numpy pybind11 pytest pytest-cov wheel flake8 psutil
                """;
            }

            buildStage("Installing PyGeNN (${NODE_NAME})") {
                // Checkout GeNN
                echo "Checking out GeNN";
                sh "rm -rf genn";
                sh "git clone --branch master https://github.com/genn-team/genn.git";

                dir("genn") {
                    // Build PyGeNN module
                    echo "Building and installing PyGeNN";
                    def messagesPyGeNN = "pygenn_${NODE_NAME}.txt";
                    def commandsPyGeNN = """
                    . ${WORKSPACE}/venv/bin/activate
                    rm -f ${messagesPyGeNN}
                    pip install -e . 1>>\"${messagesPyGeNN}\" 2>&1
                    """;
                    def statusPyGeNN = sh script:commandsPyGeNN, returnStatus:true;
                    archive messagesPyGeNN;
                    if (status_PyGeNN != 0) {
                        setBuildStatus("Building PyGeNN (${NODE_NAME})", "FAILURE");
                    }
                }
            }

            buildStage("Installing mlGeNN (${NODE_NAME})") {
                def messagesMLGeNN = "ml_genn_${NODE_NAME}.txt";
                sh "rm -f ${messagesMLGeNN}";

                dir("ml_genn") {
                    // Install ML GeNN
                    sh """
                    . ${WORKSPACE}/venv/bin/activate
                    pip install -e .
                    """;
                }

                dir("ml_genn_tf") {
                    // Install ML GeNN
                    sh """
                    . ${WORKSPACE}/venv/bin/activate
                    pip install -e .
                    """;
                }
            }

            def coverageMLGeNN = "${WORKSPACE}/coverage_${NODE_NAME}.xml";
            buildStage("Running tests (${NODE_NAME})") {
                dir("tests") {
                    // Run ML GeNN test suite
                    def messagesTests = "tests_${NODE_NAME}.txt";
                    def commandsTest = """
                    . ${WORKSPACE}/venv/bin/activate
                    rm -f ${messagesTests}
                    pytest -v --cov ../ml_genn --cov ../ml_genn_tf --cov-report=xml:${coverageMLGeNN} --junitxml ml_genn_${NODE_NAME}.xml 1>>\"${messagesTests}\" 2>&1
                    """;
                    def statusTests = sh script:commandsTest, returnStatus:true;
                    archive messagesTests;
                    if (statusTests != 0) {
                        setBuildStatus("Running tests (${NODE_NAME})", "FAILURE");
                    }
                }
            }

            buildStage("Gathering test results (${NODE_NAME})") {
                // Process JUnit test output
                dir("tests") {
                    junit "ml_genn_${NODE_NAME}.xml";
                }
            }

            buildStage("Uploading coverage (${NODE_NAME})") {
                withCredentials([string(credentialsId: "codecov_token_ml_genn", variable: "CODECOV_TOKEN")]) {

                    // **NOTE** groovy would expand ${CODECOV_TOKEN} which would mean it could, for example, be strace'd
                    // see https://www.jenkins.io/doc/book/pipeline/jenkinsfile/#string-interpolation
                    sh 'curl -s https://codecov.io/bash | bash -s - -n ' + env.NODE_NAME + ' -f ' + coverageMLGeNN + ' -t $CODECOV_TOKEN';
                }
            }

            buildStage("Running Flake8 (${NODE_NAME})") {
                // Run flake8
                def logFlake8MLGeNN = "flake8_${NODE_NAME}.log";
                def commandsFlake8MLGeNN = """
                . ${WORKSPACE}/venv/bin/activate
                rm -f ${logFlake8MLGeNN}
                flake8 --format pylint --output-file ${logFlake8MLGeNN} ml_genn/ml_genn ml_genn_tf/ml_genn_tf
                """
                def statusFlake8MLGeNN = sh script:commandsFlake8MLGeNN, returnStatus:true;

                // Record any issues
                recordIssues enabledForFailure: true, tool: flake8(pattern: logFlake8MLGeNN);

                if (statusFlake8MLGeNN != 0) {
                    setBuildStatus("Running Flake8 (${NODE_NAME})", "FAILURE");
                }
            }
        }
    }
}

// Run builds in parallel
parallel builders;
