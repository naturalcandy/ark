# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import ark
import numpy as np
import multiprocessing
import os
from pathlib import Path
import time

world_size = 8

tensor_len = 8192 * 1024
tensor_size = tensor_len * ark.fp16.element_size()


def allreduce_test_function(rank, np_inputs, plan_path, ground_truth):
    print("rank:", rank)
    ark.set_rank(rank)
    ark.set_world_size(world_size)

    # Run `plan_path` file if exists
    if not Path(plan_path).is_file():
        print(f"File {plan_path} does not exist. Exiting...")
        return

    input = ark.tensor([tensor_len], ark.fp16)
    output = ark.all_reduce(input, rank, world_size, input)
    with ark.Runtime.get_runtime() as rt:
        plan = ark.Plan.from_file(plan_path)
        rt.launch(plan=plan, device_id=rank)
        input.from_numpy(np_inputs)
        rt.run()
        # Copy data back to host and calculate errors
        host_output = output.to_numpy()
        np.testing.assert_allclose(
            host_output, ground_truth, rtol=1e-2, atol=1e-2
        )

        rt.barrier()
        # Measure throughput
        iter = 10000
        ts = time.time()
        rt.run(iter)
        elapsed_ms = (time.time() - ts) * 1e3
        print(
            f"Current plan elapsed time: total {elapsed_ms:.6f} ms, {elapsed_ms/iter:.6f} ms/iter"
        )


def allreduce_test(plan_path: str, plan_prefix: str):
    num_processes = world_size  # number of processes
    processes = []
    np_inputs = []
    for i in range(world_size):
        np_inputs.append(np.random.uniform(0, 1, tensor_len).astype(np.float16))
    ground_truth = np.sum(np_inputs, axis=0)

    # Create a process for each GPU
    for i in range(num_processes):
        process = multiprocessing.Process(
            target=allreduce_test_function,
            args=(
                i,
                np_inputs[i],
                os.path.join(plan_path, plan_prefix + str(i) + ".json"),
                ground_truth,
            ),
        )
        process.start()
        processes.append(process)

    # Join the processes after completion
    for process in processes:
        process.join()


if __name__ == "__main__":
    ark.init()
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan_dir", type=str, default="examples/tutorial")
    parser.add_argument("--plan_prefix", type=str, default="plan_gpu")

    args = parser.parse_args()
    allreduce_test(args.plan_dir, args.plan_prefix)
