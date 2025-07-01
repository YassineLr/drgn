# Copyright (c) 2025 Oracle and/or its affiliates
# SPDX-License-Identifier: LGPL-2.1-or-later

"""An implementation of ps(crash command) using drgn"""

import sys
import argparse
from typing import Any, Dict, Iterable, NamedTuple, Optional, Tuple

import drgn
from drgn import Object, TypeKind, Program
from drgn.helpers.common.format import escape_ascii_string
from drgn.helpers.linux.cpumask import for_each_online_cpu
from drgn.helpers.linux.list import list_for_each_entry
from drgn.helpers.linux.percpu import per_cpu, percpu_counter_sum
from drgn.helpers.linux.pid import find_task, for_each_task
from drgn.helpers.linux.sched import cpu_curr, task_state_to_char
from drgn.helpers.linux.mm import totalram_pages

from drgn.commands import argument, drgn_argument
from drgn.commands.crash import crash_command
from drgn.helpers.common.format import print_table

ByteToKB = 1024

# Linux kernel commit 06eb61844d84("sched/debug: Add explicit
# TASK_IDLE printing") (in v4.14) introduced printing of TASKs in idle
# state and hence changed size of task_state_array. Further Linux kernel
# commit 8ef9925b02c2("sched/debug: Add explicit TASK_PARKED printing)
# (in v4.14) introduced printing of parked tasks and further changed
# the size of task_state_array. This change also changed values of
# some states.
# Since size of task_state_array and value of some task states changed
# in same (v4.14) kernel, we can also use length of task_state_array
# to distnguish between old and new values of task states, whose values
# changed.
# Lastly since newer kernels add new task states without changing the
# values of pre-existing task states, we can safely assume that the task
# state values given below are valid for newer kernels as well, even though
# the newer kernels may have added some task states of their own.


# The following task states have same values in
# all currently supported kernel versions.

def has_member(obj: Object, name: str) -> bool:
    """
    Return true if a given object has a member with the given name.
    :param obj: Drgn object to check
    :param name: string member name to check
    :returns: whether the object has a member by that name
    """
    try:
        obj.member_(name)
        return True
    except LookupError:
        return False


def nanosecs_to_secs(nanosecs: int) -> float:
    """
    Convert from nanosecs to secs

    :param nanosecs: time duration in nano secs
    :returns: time duration in secs
    """
    val = nanosecs // 1000000
    return val / 1000


def get_task_arrival_time(task: Object) -> int:
    """
    Get a task's arrival time on cpu

    A task's arrival time is only updated when the task is put ON a cpu via
    context_switch.

    :param task: ``struct task_struct *``
    :returns: arrival time instance in ns granularity
    """

    if has_member(task, "last_run"):
        arrival_time = task.last_run.value_()
    elif has_member(task, "timestamp"):
        arrival_time = task.timestamp.value_()
    else:
        arrival_time = task.sched_info.last_arrival.value_()

    return arrival_time

def get_current_run_time(prog: drgn.Program, cpu: int) -> int:
    """
    Get running duration of the current task on some cpu

    :param prog: drgn program
    :param cpu: cpu index
    :returns: duration in ns granularity
    """
    return task_lastrun2now(cpu_curr(prog, cpu))


def task_thread_info(task: drgn.Object) -> drgn.Object:
    """
    Return a task's ``thread_info``

    This is an equivalent to the kernel function / inline / macro
    ``task_thread_info()``, but it must cover a wide variety of versions and
    configurations.

    :param task: Object of type ``struct task_struct *``
    :returns: The ``struct thread_info *`` for this task
    """
    if has_member(task, "thread_info"):
        return task.thread_info.address_of_()
    return drgn.cast("struct thread_info *", task.stack)


def task_cpu(task: drgn.Object) -> int:
    """
    Return the CPU on which a task is running.

    This is an equivalent to the kernel function ``task_cpu()``, but it covers
    a wide variety of variations in kernel version and configuration. It would
    be a bit impractical to spell out all the variants, but essentially, if
    there's a "cpu" field in ``struct task_struct``, then we can just use that.
    Otherwise, we need to get it from the ``thread_info``.

    :param task: Object of type ``struct task_struct *``
    :retruns: The cpu as a Python int
    """
    if has_member(task, "cpu"):
        return task.cpu.value_()
    return task_thread_info(task).cpu.value_()


def runq_clock(prog: drgn.Program, cpu: int) -> int:
    """
    Get clock of cpu runqueue ``struct rq``

    :param prog: drgn program
    :param cpu: cpu index
    :returns: cpu runqueue clock in ns granularity
    """
    rq = per_cpu(prog["runqueues"], cpu)
    return rq.clock.value_()


def task_lastrun2now(task: drgn.Object) -> int:
    """
    Get the duration from task last run timestamp to now

    The return duration will cover task's last run time on cpu and also
    the time staying in current status, usually the time slice for task
    on cpu will be short, so this can roughly tell how long this task
    has been staying in current status.
    For task status in "RU" status, if it's still on cpu, then this return
    the duration time this task has been running, otherwise it roughly tell
    how long this task has been staying in runqueue.

    :param prog: drgn program
    :param task: ``struct task_struct *``
    :returns: duration in ns granularity
    """
    prog = task.prog_
    arrival_time = get_task_arrival_time(task)
    rq_clock = runq_clock(prog, task_cpu(task))

    return rq_clock - arrival_time

def format_nanosecond_duration(nanosecs: int) -> str:
    """
    :returns: conversion of nanoseconds to [dd hh:mm:ss.ms] format
    """
    secs = nanosecs_to_secs(nanosecs)
    dd, rem = divmod(secs, 86400)
    hh, rem = divmod(rem, 3600)
    mm, secs = divmod(rem, 60)
    return "%02ld %02ld:%02ld:%06.3f" % (dd, hh, mm, secs)


def get_pid(task: Object) -> int:
    """
    :returns: PID of the task
    """
    return task.pid.value_()


def get_command(task: Object) -> str:
    """
    :returns: name of the command
    """
    return escape_ascii_string(task.comm.string_())


def get_ppid(task: Object) -> int:
    """
    :returns: Parent PID of the task
    """
    return task.parent.pid.value_()


class TaskRss(NamedTuple):
    """
    Represent's a task's resident set size in pages. See task_rss().
    """

    rss_file: int
    rss_anon: int
    rss_shmem: int
    swapents: int

    @property
    def total(self) -> int:
        return self.rss_file + self.rss_anon + self.rss_shmem

def get_task_rss(task: Object, cache: Optional[Dict[int, TaskRss]]) -> TaskRss:
    """
    Return the task's resident set size (RSS) in pages

    The task's RSS is the number of pages which are currently resident in
    memory. The RSS values can be broken down into anonymous pages (not bound to
    any file), file pages (those associated with memory mapped files), and
    shared memory pages (those which aren't associated with on-disk files, but
    belonging to shared memory mappings). This function returns a tuple
    containing each category, but the common behavior is to use the "total"
    value which sums them up.

    :param task: ``struct task_struct *`` for which to compute RSS
    :param cache: if provided, we can use this to cache the mapping of
      "mm_struct" to RSS. This helps avoid re-computing the RSS value for
      processes with many threads, but note that it could result in out of date
      values on a live system.
    :returns: the file, anon, and shmem page values
    """
    mmptr = task.mm.value_()
    if mmptr and cache and mmptr in cache:
        return cache[mmptr]

    # Kthreads have a NULL mm, simply skip them, returning 0.
    if not task.mm:
        return TaskRss(0, 0, 0, 0)

    prog = task.prog_
    rss_stat = task.mm.rss_stat

    MM_FILEPAGES = prog.constant("MM_FILEPAGES").value_()
    MM_ANONPAGES = prog.constant("MM_ANONPAGES").value_()
    MM_SWAPENTS = prog.constant("MM_SWAPENTS").value_()
    try:
        MM_SHMEMPAGES = prog.constant("MM_SHMEMPAGES").value_()
    except LookupError:
        MM_SHMEMPAGES = -1

    # Start with the counters from the mm_struct
    filerss = anonrss = shmemrss = swapents = 0

    if rss_stat.type_.kind == TypeKind.ARRAY:
        # Since v6.2, f1a7941243c10 ("mm: convert mm's rss stats into
        # percpu_counter"), the "rss_stat" object is an array of percpu
        # counters. Simply sum them up!
        filerss = percpu_counter_sum(rss_stat[MM_FILEPAGES].address_of_())
        anonrss = percpu_counter_sum(rss_stat[MM_ANONPAGES].address_of_())
        swapents = percpu_counter_sum(rss_stat[MM_SWAPENTS].address_of_())
        shmemrss = 0
        if MM_SHMEMPAGES >= 0:
            shmemrss = percpu_counter_sum(
                rss_stat[MM_SHMEMPAGES].address_of_()
            )
    else:
        # Prior to this, the "rss_stat" was a structure containing counters that
        # were cached on each task_struct and periodically updated into the
        # mm_struct. We start with the counter values from the mm_struct and
        # then sum up the cached copies from each thread.
        filerss += rss_stat.count[MM_FILEPAGES].counter.value_()
        anonrss += rss_stat.count[MM_ANONPAGES].counter.value_()
        if MM_SHMEMPAGES >= 0:
            shmemrss += rss_stat.count[MM_SHMEMPAGES].counter.value_()

        for gtask in for_each_task_in_group(task, include_self=True):
            filerss += gtask.rss_stat.count[MM_FILEPAGES].value_()
            anonrss += gtask.rss_stat.count[MM_ANONPAGES].value_()
            swapents += gtask.rss_stat.count[MM_SWAPENTS].value_()
            if MM_SHMEMPAGES >= 0:
                shmemrss += gtask.rss_stat.count[MM_SHMEMPAGES].value_()
    rss = TaskRss(filerss, anonrss, shmemrss, swapents)

    if cache is not None:
        cache[mmptr] = rss

    return rss


def get_vmem(task: Object) -> float:
    """
    Return virtual memory size of the task
    """
    prog = task.prog_
    page_size = prog["PAGE_SIZE"].value_()
    try:
        vmem = (task.mm.total_vm.value_() * page_size) // ByteToKB
    except drgn.FaultError:
        vmem = 0
    return vmem


def show_tasks_last_runtime(tasks: Iterable[Object]) -> None:
    """
    Display task information in their last arrival order.
    """
    rows = [["LAST_ARRIVAL", "ST", "PID", "TASK", "CPU", "COMMAND"]]
    tasks = list(tasks)
    tasks.sort(key=task_lastrun2now)
    for t in tasks:
        cpu = str(task_cpu(t))
        pid = str(get_pid(t))
        state = task_state_to_char(t)
        command = get_command(t)
        time_nanosec = task_lastrun2now(t)
        last_arrival = format_nanosecond_duration(time_nanosec)
        print(f"[{last_arrival}] [{state}]  PID: {pid:<8} TASK: {t.value_()}  CPU: {cpu:<2}  COMMAND: \"{command}\"")


def show_tasks_last_runtime_timestamp(tasks: Iterable[Object]) -> None:
    """
    Display task information sorted by last_run/timestamp/last_arrival, showing the raw timestamp.
    """
    rows = [["LAST_ARRIVAL_TIMESTAMP", "ST", "PID", "TASK", "CPU", "COMMAND"]]
    tasks = list(tasks)
    tasks.sort(key=get_task_arrival_time)
    for t in tasks:
        cpu = str(task_cpu(t))
        pid = str(get_pid(t))
        state = task_state_to_char(t)
        command = get_command(t)
        last_arrival = str(get_task_arrival_time(t))
        print(f"[{last_arrival}] [{state}]  PID: {pid:<8} TASK: {t.value_()}  CPU: {cpu:<2}  COMMAND: \"{command}\"")


def for_each_task_in_group(
    task: Object, include_self: bool = False
) -> Iterable[Object]:
    """
    Iterate over all tasks in the thread group

    Or, in the more common userspace terms, iterate over all threads of a
    process.

    :param task: a task whose group to iterate over
    :param include_self: should ``task`` itself be returned
    :returns: an iterable of every thread in the thread group
    """
    if include_self:
        yield task
    if hasattr(task, "thread_group"):
        yield from list_for_each_entry(
            "struct task_struct",
            task.thread_group.address_of_(),
            "thread_group",
        )
    else:
        # Since commit 8e1f385104ac0 ("kill task_struct->thread_group") from
        # 6.7, the thread_group list is gone, replaced by a list inside the
        # task.signal struct. This has an explicit list_head (unlike the
        # thread_group which just linked each task together with no explicit
        # head node).
        for other in list_for_each_entry(
            "struct task_struct",
            task.signal.thread_head.address_of_(),
            "thread_node",
        ):
            # We've already yielded "task" (or not, depending on the caller's
            # preference) so skip it here.
            if other != task:
                yield other

def show_taskinfo(prog: Program, tasks: Iterable[Object]) -> None:
    """
    Display task information.
    """
    rows = [["PID", "PPID", "CPU", "TASK", "ST", "%MEM", "VSZ", "RSS", "COMM"]]
    tasks = list(tasks)
    tasks.sort(key=get_pid)
    page_size = int(prog["PAGE_SIZE"])
    total_mem = int(totalram_pages(prog))
    rss_cache: Dict = {}
    for t in tasks:
        task_rss = get_task_rss(t, rss_cache)
        rss_kb = task_rss.total * page_size // ByteToKB
        pct_mem = task_rss.total * 100 / total_mem
        rows.append(
            [
                str(get_pid(t)),
                str(get_ppid(t)),
                str(task_cpu(t)),
                hex(t.value_()),
                task_state_to_char(t),
                str("%.1f" % pct_mem),
                str(get_vmem(t)),
                str(rss_kb),
                f"[{get_command(t)}]"
            ]
        )
    print_table(rows)


def is_kthread(t: Object) -> bool:
    """
    Check if a task is kernel thread.
    """
    if not t.mm and not task_state_to_char(t) == "Z":
        return True
    return False


def is_user(t: Object) -> bool:
    """
    Check if a task is user thread.
    """
    if t.mm:
        return True
    elif task_state_to_char(t) == "Z":
        return True
    return False


def is_group_leader(t: Object) -> bool:
    """
    Check if a task is thread group leader.
    """
    if get_pid(t) == t.tgid.value_():
        return True
    return False


def check_arg_type(arg: Optional[str]) -> Tuple[str, Any]:
    """
    Check the filter type of the argument
    """
    if arg is not None:
        try:
            return ("pid", int(arg, 10))
        except ValueError:
            pass
        try:
            return ("task", int(arg, 16))
        except ValueError:
            return ("comm", str(arg))
    else:
        return ("none", None)


@crash_command(
    description="process information",
    long_description="display process status information",
    arguments=(
        argument(
            "-u",
            dest="user",
            action="store_true",
            default=False,
            help="display only user threads information",
        ),
        argument(
            "-k",
            dest="kernel",
            action="store_true",
            default=False,
            help="display only kernel threads information",
        ),
        argument(
            "-G",
            dest="group_leader",
            action="store_true",
            default=False,
            help="display only the thread group leader in a thread group",
        ),
        argument(
            "-m",
            dest="last_run",
            action="store_true",
            default=False,
            help="show last run information",
        ),
        argument(
            "-l",
            dest="last_run_timestamp",
            action="store_true",
            default=False,
            help="show last run information (raw timestamp)",
        ),
        argument(
            "arg",
            nargs="?",
            type=check_arg_type,
            metavar="pid | task | command",
            help="pid is a process PID. task is hexadecimal task_struct pointer. command is a command name.",
        ),
        drgn_argument,
    ),
)

def _crash_cmd_ps(
    prog: Program, name: str, args: argparse.Namespace, **kwargs: Any
) -> None:
    if args.drgn: #--TO DO
#         print("""
# from drgn.helpers.linux.pid import for_each_task
# from drgn.helpers.linux.percpu import percpu_counter_sum
# from drgn.helpers.common.format import escape_ascii_string

#     rss_cache: Dict = {}
#     for t in tasks:
#         rss_kb = task_rss.total * page_size // ByteToKB
# for task in for_each_task(prog):
#     total_mem = int(totalram_pages(prog))
#     page_size = prog["PAGE_SIZE"].value_()
#     cpu = task.cpu.value_()
#     pid = task.pid.value_()
#     state = task_state_to_char(task)
#     command = escape_ascii_string(task.comm.string_())
#     task = hex(task.value_())
#     try:
#         vmem = (task.mm.total_vm.value_() * page_size) // ByteToKB
#     except drgn.FaultError:
#         vmem = 0
#     task_rss = get_task_rss(t, rss_cache)
#     ppid = task.parent.pid.value_()
#     pct_mem = task_rss.total * 100 / total_mem
# """)
        return
    tasks = for_each_task(prog)
    if args.arg is not None:
        if args.arg[0] == "pid":
            tasks = [find_task(prog, args.arg[1])]
        elif args.arg[0] == "task":
            tasks = [
                Object(prog, "struct task_struct *", value=args.arg[1])
            ]
        elif args.arg[0] == "comm":
            tasks = (t for t in tasks if get_command(t) == args.arg[1])

    if args.user:
        tasks = filter(is_user, tasks)
    elif args.kernel:
        tasks = filter(is_kthread, tasks)
    if args.group_leader:
        tasks = filter(is_group_leader, tasks)
    if args.last_run:
        show_tasks_last_runtime(tasks)
    elif args.last_run_timestamp:
        show_tasks_last_runtime_timestamp(tasks)
    else:
        show_taskinfo(prog, tasks)
