#ifndef ARK_UNITTEST_UNITTEST_UTILS_H_
#define ARK_UNITTEST_UNITTEST_UTILS_H_

#include <cstdlib>
#include <functional>
#include <iomanip>
#include <string>
#include <thread>

#include "ark/cpu_timer.h"
#include "ark/logging.h"

namespace ark {
namespace unittest {

typedef enum
{
    SUCCESS = 0,
    FAILURE,
    UNEXPECTED
} State;

void exit(State s, const std::string &errmsg);
void fexit(const std::string &errmsg = "");
void uexit(const std::string &errmsg = "");
void sexit(const std::string &errmsg = "");

//
class Timeout
{
  public:
    Timeout(int timeout);
    ~Timeout();
};

std::thread *spawn_thread(std::function<State()> func);
void wait_all_threads();

int spawn_process(std::function<State()> func);
void wait_all_processes();

State test(std::function<State()> test_func);

} // namespace unittest
} // namespace ark

// Run the given test function.
#define UNITTEST(test_func)                                                    \
    do {                                                                       \
        LOG(ark::INFO, "unittest start: " #test_func);                         \
        double _s = ark::cpu_timer();                                          \
        ark::unittest::State _ret = ark::unittest::test(test_func);            \
        double _e = ark::cpu_timer() - _s;                                     \
        if (_ret != ark::unittest::SUCCESS) {                                  \
            UNITTEST_EXIT(_ret, "Unexpected exit");                            \
        }                                                                      \
        LOG(ark::INFO, "unittest succeed: " #test_func " (elapsed ",           \
            std::setprecision(4), _e, "s)");                                   \
    } while (0)

// Exit with proper error messages and return values.
#define UNITTEST_EXIT(state, ...)                                              \
    do {                                                                       \
        if ((state) == ark::unittest::FAILURE) {                               \
            LOG(ark::ERROR, "unittest failed: ", __VA_ARGS__);                 \
        } else if ((state) == ark::unittest::UNEXPECTED) {                     \
            LOG(ark::ERROR,                                                    \
                "Unexpected error during unittest: ", __VA_ARGS__);            \
        } else if ((state) == ark::unittest::SUCCESS) {                        \
            LOG(ark::INFO, "unittest succeed");                                \
        }                                                                      \
        std::exit(state);                                                      \
    } while (0)

// Fail the test.
#define UNITTEST_FEXIT(...) UNITTEST_EXIT(ark::unittest::FAILURE, __VA_ARGS__)
// Unexpected error during test.
#define UNITTEST_UEXIT(...)                                                    \
    UNITTEST_EXIT(ark::unittest::UNEXPECTED, __VA_ARGS__)
// Success.
#define UNITTEST_SEXIT() UNITTEST_EXIT(ark::unittest::SUCCESS, "")

// Check if the given condition is true.
#define UNITTEST_TRUE(cond)                                                    \
    do {                                                                       \
        if (cond) {                                                            \
            break;                                                             \
        }                                                                      \
        UNITTEST_FEXIT("condition `" #cond "` failed");                        \
    } while (0)
// Check if the given experssions are equal.
#define UNITTEST_EQ(exp0, exp1)                                                \
    do {                                                                       \
        auto _v0 = (exp0);                                                     \
        auto _v1 = (exp1);                                                     \
        if (_v0 == _v1) {                                                      \
            break;                                                             \
        }                                                                      \
        UNITTEST_FEXIT("`" #exp0 "` (value: ", _v0,                            \
                       ") != `" #exp1 "` (value: ", _v1, ")");                 \
    } while (0)
// Check if the given experssions are not equal.
#define UNITTEST_NE(exp0, exp1)                                                \
    do {                                                                       \
        auto _v0 = (exp0);                                                     \
        auto _v1 = (exp1);                                                     \
        if (_v0 != _v1) {                                                      \
            break;                                                             \
        }                                                                      \
        UNITTEST_FEXIT("`" #exp0 "` (value: ", _v0,                            \
                       ") == `" #exp1 "` (value: ", _v1, ")");                 \
    } while (0)

#endif // ARK_UNITTEST_UNITTEST_UTILS_H_