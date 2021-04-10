# encoding: utf-8
"""
Created by PyCharm.
File:               LinuxBashShellScriptForOps:pyScheduleTask.py
User:               Guodong
Create Date:        2017/4/6
Create Time:        22:33
 """
import codecs
import locale, ctypes
import sched
import threading
import time
import psutil
import sys
import os

locale.setlocale(locale.LC_ALL, '')


def get_system_encoding():
    """
    The encoding of the default system locale but falls back to the given
    fallback encoding if the encoding is unsupported by python or could
    not be determined.  See tickets #10335 and #5846
    """
    try:
        encoding = locale.getdefaultlocale()[1] or 'ascii'
        codecs.lookup(encoding)
    except LookupError:
        encoding = 'ascii'
    return encoding


DEFAULT_LOCALE_ENCODING = get_system_encoding()


def shutdown_NetEaseCloudMusic(name):
    """
    根据进程名关闭进程
    :param name: 进程名
    :return:
    """
    # define NetEaseCloudMusic process name
    ProcessNameToKill = name  # 要关闭的进程名

    print()

    # learn from getpass.getuser()
    def getuser():
        """
        Get the username from the environment or password database.
        First try various environment variables, then the password
        database.  This works on Windows as long as USERNAME is set.
        从环境或密码数据库获取用户名。
        首先尝试各种环境变量，然后尝试密码数据库。
        只要设置了USERNAME，它就可以在Windows上运行。
        :return:
        """
        for username in ('LOGNAME', 'USER', 'LNAME', 'USERNAME'):
            user = os.environ.get(username)  # 从环境变量中获取用户名
            if user:
                return user

    currentUserName = getuser()  # 获取当前用户名

    if ProcessNameToKill in [x.name() for x in psutil.process_iter()]:  # 获取当前所有正在运行的进程对象(Process对象),
        # 获取进程名并判断目标进程名是否在其中
        # 如果目标进程在当前进程列表中
        print("[I] Process \"%s\" is found!" % ProcessNameToKill)  # 输出获取到了进程
    else:  # 如果目标进程不在当前进程列表中
        print("[E] Process \"%s\" is NOT running!" % ProcessNameToKill)  # 输出进程不在运行状态中
        sys.exit(1)  # 直接退出

    for process in psutil.process_iter():  # 遍历正在运行的所有进程
        if process.name() == ProcessNameToKill:  # 如果进程名符合,关闭该进程
            try:
                # root user can only kill its process, but can NOT kill other users process
                # root用户只能杀死其进程，而不能杀死其他用户进程
                if process.username().endswith(currentUserName):  # 如果目标进程是当前用户创建的
                    process.kill()  # 关闭进程
                    print(
                        "[I] Process \"%s(pid=%s)\" is killed successfully!" % (process.name(), process.pid))
            except Exception as e:
                print(e)


def display_countdown(sec, name):
    """
    第一个进程,用于计时器显示
    :param sec: 设置的关闭倒计时(秒),int类型
    :param name: 要关闭的进程名
    :return:
    """
    print("The process will be shutdown is %s" % name)  # 输出要关闭的进程信息

    def countdown(secs):
        """
        blocking process 1
        计时器
        :param secs: seconds, int 剩余秒数 int类型
        :return:
        """
        current_time = time.strftime("%Y-%m-%d %H:%M:%S %Z")  # 获取当前时间
        print("Time current: %s" % current_time)  # 输出当前时间

        while secs:  # 剩余秒数不为零时
            now = time.strftime("%Y-%m-%d %H:%M:%S %Z")  # 获取当前时间
            hours, seconds = divmod(secs, 3600)  # 计算剩余秒数对应的小时和秒 剩余秒数对一个小时的秒数（3600）取余
            minutes, seconds = divmod(seconds, 60)  # 计算小时外剩余秒数对应的分钟和秒 剩余秒数对一分钟的秒数（60）取余
            clock_format = '{:02d}:{:02d}:{:02d}'.format(hours, minutes, seconds)  # 格式化剩余时间（时：分：秒）
            sys.stdout.write('\rTime now: %s Time left: %s' % (now, clock_format))  # 不换行的输出时间信息
            # ‘\r’回到行首，下次打印从行首开始 实现刷新
            sys.stdout.flush()  # 刷新缓存区
            time.sleep(1)  # 等待1秒
            secs -= 1  # 剩余秒数减一

    # set a human readable timer here, such as display how much time left to shutdown
    # 在此处设置人类可读的计时器，例如显示还剩多少时间关闭
    countdown(int(sec))


def display_scheduler(name):
    """
    blocking process 2
    第二个进程,用于关闭进程
    :param name:要关闭的进程名
    :return:
    """
    s = sched.scheduler(time.time, time.sleep)  # 生成延迟事件调度器
    # https://docs.python.org/2/library/sched.html#sched.scheduler.enter
    s.enter(seconds_to_shutdown, 1, shutdown_NetEaseCloudMusic,
            (name,))  # 设置调度器（间隔时间，优先级（数字小的大），要执行的方法，要执行方法的参数）
    s.run()  # 运行调度器

    """上面调度器执行完之后执行 输出时间信息"""
    now = time.strftime("%Y-%m-%d %H:%M:%S %Z")  # 获取当前时间
    print("Time finished: %s\nGood bye!" % now)  # 输出


if __name__ == '__main__':
    # seconds_to_shutdown = 10  # after 10s shutdown cloudmusic.exe
    seconds_to_shutdown = 5  # after 10s shutdown cloudmusic.exe
    # seconds_to_shutdown = 7200  # after 10s shutdown cloudmusic.exe
    # seconds_to_shutdown = 4500  # after 10s shutdown cloudmusic.exe
    # process_name_to_shutdown = "cloudmusic.exe"
    process_name_to_shutdown = "QQMusic.exe"

    threadingPool = list()
    threading_1 = threading.Thread(target=display_countdown,
                                   args=(seconds_to_shutdown, process_name_to_shutdown,))
    threading_2 = threading.Thread(target=display_scheduler, args=(process_name_to_shutdown,))
    threadingPool.append(threading_1)
    threadingPool.append(threading_2)

    for thread in threadingPool:
        thread.setDaemon(False)
        thread.start()

    thread.join()
