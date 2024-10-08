// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "unittest/unittest_utils.h"

#include <signal.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <vector>

#include "file_io.h"
#include "logging.hpp"

// Grep SIGALRM and exit.
static void sigalrm_timeout_handler(int) {
    signal(SIGALRM, SIG_IGN);
    UNITTEST_FAIL("timeout");
}

namespace ark {
namespace unittest {

// Temporal unittest states.
struct TempStates {
    std::vector<int> pids;
    std::vector<std::thread *> threads;
};

TempStates GLOBAL_TEMP_STATES_;

// Set a timeout of the current process.
Timeout::Timeout(int timeout) {
    signal(SIGALRM, sigalrm_timeout_handler);
    alarm(timeout);
}

// Remove the timeout.
Timeout::~Timeout() {
    alarm(0);
    signal(SIGALRM, SIG_DFL);
}

// Spawn a thread that runs the given function.
std::thread *spawn_thread(std::function<State()> func) {
    std::thread *t = new std::thread(func);
    GLOBAL_TEMP_STATES_.threads.emplace_back(t);
    return t;
}

// Wait for all threads to finish.
void wait_all_threads() {
    for (std::thread *t : GLOBAL_TEMP_STATES_.threads) {
        if (t->joinable()) {
            t->join();
        }
        delete t;
    }
    GLOBAL_TEMP_STATES_.threads.clear();
}

// Spawn a process that runs the given function.
int spawn_process(std::function<State()> func) {
    pid_t pid = fork();
    if (pid < 0) {
        UNITTEST_UNEXPECTED("fork() failed");
    } else if (pid == 0) {
        State ret = func();
        std::exit(ret);
    }
    GLOBAL_TEMP_STATES_.pids.push_back(pid);
    return (int)pid;
}

// Wait for all processes to finish.
void wait_all_processes() {
    size_t nproc = GLOBAL_TEMP_STATES_.pids.size();
    for (size_t i = 0; i < nproc; ++i) {
        pid_t pid;
        int status;
        do {
            pid = wait(&status);
            if (pid == -1) {
                UNITTEST_UNEXPECTED("wait() failed");
            }
        } while (!WIFEXITED(status));
        status = WEXITSTATUS(status);
        if (status != State::SUCCESS) {
            UNITTEST_EXIT((State)status, "process " + std::to_string(pid));
        }
    }
    GLOBAL_TEMP_STATES_.pids.clear();
}

// Run the given test function.
State test(std::function<State()> test_func) { return test_func(); }

//
std::string get_kernel_code(const std::string &name) {
    return ark::read_file(ark::get_dir(std::string{__FILE__}) +
                          "/../ops/kernels/" + name + ".h");
}

}  // namespace unittest
}  // namespace ark
