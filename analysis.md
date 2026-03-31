# Exhaustive Analysis Report: Distributed ANN Search Experiments

## 1. Overview
The provided bash script `run_experiment.sh` is an automated orchestration tool designed to run distributed Approximate Nearest Neighbor (ANN) search experiments. The script initializes server instances across multiple hosts (or on localhost), dispatches an experimental workload via a singular client binary, manages output logs/statistics, and cleans up the resulting processes gracefully.

## 2. Main Script Analysis (`run_experiment.sh`)

### Core Execution Flow
1. **Configuration Loading:**
   The script loads configurations by sourcing a relative helper file `setup_exp_vars.sh`. This sets experimental bounds, hardware IPs, metric choices, dataset configurations (e.g. `bigann`), and logical variables needed for the binaries.
   
2. **Setup and Directory Management:**
   Establishes local and remote workspace directories. It employs SSH multiplexing/command wrappers (`sshCommandAsync`, `sshCommandSync`, `sshStopCommand`) with stringent host-checking disabled to silently launch detached background jobs on CloudLab setups.
   
3. **Server Initialization (`state_send_server`):**
   The script iterates through configured peers and runs the server binary `build/src/state_send/state_send_server` concurrently.
   It passes in parameters configuring search thread distribution, memory vs. SSD index toggles, batching metrics, dataset formats (`--data_type`), index prefixes, query balance limits, network schemas, and more. 
   
4. **Client Initialization (`run_benchmark_state_send_tcp`):**
   It triggers the client benchmark driver on the designated final peer node `build/benchmark/state_send/run_benchmark_state_send_tcp`.
   Arguments control the throughput, metric boundaries (e.g., beam width, $K$-nearest values), search mode, and result extraction logic. 

5. **Wait and Cleanup Mechanism:**
   The process iteratively polls the client's PID until expiration. Upon normal or interrupted (via `SIGTERM / SIGINT`) termination, it leverages the `sshStopCommand` to elegantly teardown the corresponding server daemons, guaranteeing complete cleanup and zero orphaned processes hanging on compute nodes.
   
6. **Log Management:**
   Aggregates traces, latency distributions, and query metrics from CloudLab back to the local host directory utilizing `scp` and `tar`.

## 3. External Dependencies & Child Executables

### 3.1 `setup_exp_vars.sh`
**Path**: `~/workspace/rdma_anns/scripts/setup_exp_vars.sh`
- Validates the primary CLI arguments passed from `run_experiment.sh`.
- Acts as an orchestrating namespace for hyperparameters scaling. It determines local versus distributed prefix paths for graphical index partitions (e.g. `$ANNGRAHPS_PREFIX`), and defines logical mappings for differing vector modalities like float/int8 `bigann`, `deep1b`, etc.

### 3.2 `state_send_server.cpp`
**Path**: `~/workspace/rdma_anns/src/state_send/state_send_server.cpp`
- The backend engine of the distributed vector database partition. It links against internal modules like `ssd_partition_index.h` and `communicator.h`.
- Based on startup arguments, it caches the local slice of vector data, maintains asynchronous search threads, and fulfills similarity search queries. 

### 3.3 `run_benchmark_state_send_tcp.cpp`
**Path**: `~/workspace/rdma_anns/benchmark/state_send/run_benchmark_state_send_tcp.cpp`
- Serving as the client orchestrator, it links with `state_send_client.h`.
- The executable sends streaming batches of vector embeddings or query bounds to listening servers. It compiles ground-truth verifications (`write_results_csv`), records networking latencies via precise timeline tracking, and serializes responses corresponding to recall/RPS outputs. 

## 4. Summary & Purpose
The `run_experiment.sh` module is the backbone for reproducible large-scale benchmarking in a distributed, likely RDMA or TCP-backed, approximate nearest neighbor pipeline. It cleanly bridges configuration layers (`setup_exp_vars.sh`) to low-level C++ drivers, ensuring proper bootstrapping, process isolation, data telemetry, and system stability under rigorous analytical metrics.
