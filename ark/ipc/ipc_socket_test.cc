// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ark/env.h"
#include "ark/init.h"
#include "ark/ipc/ipc_hosts.h"
#include "ark/ipc/ipc_socket.h"
#include "ark/logging.h"
#include "ark/process.h"
#include "ark/unittest/unittest_utils.h"

using namespace ark;

struct TestIpcSocketItem
{
    int a;
    int b;
    int c;
};

unittest::State test_ipc_socket_simple()
{
    int pid0 = proc_spawn([] {
        unittest::Timeout timeout{5};
        int port_base = get_env().ipc_listen_port_base;
        IpcSocket is{get_host(0), port_base};

        struct TestIpcSocketItem item;
        item.a = 1;
        item.b = 2;
        item.c = 3;
        is.add_item("test_item_name", &item, sizeof(item));

        struct TestIpcSocketItem remote_item;
        IpcSocket::State s =
            is.query_item(get_host(0), port_base + 1, "test_item_name",
                          &remote_item, sizeof(remote_item), true);
        UNITTEST_TRUE(s == IpcSocket::SUCCESS);
        UNITTEST_EQ(remote_item.a, 4);
        UNITTEST_EQ(remote_item.b, 5);
        UNITTEST_EQ(remote_item.c, 6);

        const IpcSocket::Item *item_ptr = is.get_item("test_item_name");
        UNITTEST_TRUE(item_ptr != nullptr);
        while (item_ptr->cnt == 0) {
            sched_yield();
        }
        return 0;
    });
    UNITTEST_NE(pid0, -1);

    int pid1 = proc_spawn([] {
        unittest::Timeout timeout{5};
        int port_base = get_env().ipc_listen_port_base;
        IpcSocket is{get_host(0), port_base + 1};

        struct TestIpcSocketItem item;
        item.a = 4;
        item.b = 5;
        item.c = 6;
        is.add_item("test_item_name", &item, sizeof(item));

        struct TestIpcSocketItem remote_item;
        IpcSocket::State s =
            is.query_item(get_host(0), port_base, "test_item_name",
                          &remote_item, sizeof(remote_item), true);
        UNITTEST_TRUE(s == IpcSocket::SUCCESS);
        UNITTEST_EQ(remote_item.a, 1);
        UNITTEST_EQ(remote_item.b, 2);
        UNITTEST_EQ(remote_item.c, 3);

        const IpcSocket::Item *item_ptr = is.get_item("test_item_name");
        UNITTEST_TRUE(item_ptr != nullptr);
        while (item_ptr->cnt == 0) {
            sched_yield();
        }
        return 0;
    });
    UNITTEST_NE(pid1, -1);

    int ret = proc_wait({pid0, pid1});
    UNITTEST_EQ(ret, 0);
    return unittest::SUCCESS;
}

unittest::State test_ipc_socket_no_item()
{
    int pid0 = proc_spawn([] {
        unittest::Timeout timeout{10};
        int port_base = get_env().ipc_listen_port_base;
        IpcSocket is{get_host(0), port_base};

        struct TestIpcSocketItem remote_item;
        IpcSocket::State s =
            is.query_item(get_host(0), port_base + 1, "test_item_name",
                          &remote_item, sizeof(remote_item), true);
        UNITTEST_TRUE(s == IpcSocket::SUCCESS);
        UNITTEST_EQ(remote_item.a, 4);
        UNITTEST_EQ(remote_item.b, 5);
        UNITTEST_EQ(remote_item.c, 6);
        return 0;
    });
    UNITTEST_NE(pid0, -1);

    int pid1 = proc_spawn([] {
        unittest::Timeout timeout{10};
        int port_base = get_env().ipc_listen_port_base;
        IpcSocket is{get_host(0), port_base + 1};

        // Sleep for a while to make the remote experience NO_ITEM
        cpu_timer_sleep(2);

        struct TestIpcSocketItem item;
        item.a = 4;
        item.b = 5;
        item.c = 6;
        is.add_item("test_item_name", &item, sizeof(item));

        const IpcSocket::Item *item_ptr = is.get_item("test_item_name");
        UNITTEST_TRUE(item_ptr != nullptr);
        while (item_ptr->cnt == 0) {
            sched_yield();
        }
        return 0;
    });
    UNITTEST_NE(pid1, -1);

    int ret = proc_wait({pid0, pid1});
    UNITTEST_EQ(ret, 0);
    return unittest::SUCCESS;
}

int main()
{
    ark::init();
    UNITTEST(test_ipc_socket_simple);
    UNITTEST(test_ipc_socket_no_item);
    return 0;
}
