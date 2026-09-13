import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = ROOT / "data" / "phase3" / "sft"


examples = [

    {
        "instruction": "服务器 CPU 使用率不高，但是 load average 很高，应该如何排查？",
        "output": """这种情况说明系统中可能存在大量等待资源的任务，而不一定是 CPU 计算压力导致的。

建议按以下顺序排查：

1. 使用 uptime 或 top 确认 load average 的变化趋势。
2. 使用 ps aux 或 ps -eo state,pid,cmd 检查是否存在大量 D 状态进程。
3. 使用 vmstat 1 查看 CPU、运行队列以及 IO wait 情况。
4. 使用 iostat -x 1 检查磁盘是否存在高延迟、高利用率或长队列。
5. 如果使用 NFS、Ceph、云盘等远程存储，还需要检查网络和存储后端。
6. 检查是否存在大量锁等待、内核线程阻塞或文件系统异常。

Linux 的 load average 不只统计正在使用 CPU 的进程，也会统计部分处于不可中断睡眠状态的任务，因此 CPU 利用率低并不代表系统没有负载。"""
    },

    {
        "instruction": "一个 Kubernetes Pod 一直处于 Pending 状态，但是集群中还有空闲节点，应该怎么排查？",
        "output": """Pod 处于 Pending 状态，说明调度器暂时无法为它选择合适的节点。

首先执行 kubectl describe pod 查看 Events，这是最重要的一步。

重点检查：

1. CPU 和内存 requests 是否超过节点可分配资源。
2. 是否存在 nodeSelector 或 nodeAffinity 限制。
3. 节点是否存在 taint，而 Pod 没有对应 toleration。
4. Pod 是否依赖尚未绑定的 PVC。
5. 是否使用了特定 GPU 或其他扩展资源。
6. PodTopologySpread、PodAffinity 或 PodAntiAffinity 是否导致调度条件过于严格。
7. kube-scheduler 是否正常运行。

不要只看节点的实时 CPU 使用率。Kubernetes 调度主要依据 requests 和节点的 allocatable 资源，而不是当前实际利用率。"""
    },

    {
        "instruction": "Pod 出现 CrashLoopBackOff 应该怎么排查？",
        "output": """CrashLoopBackOff 表示容器启动后不断退出，Kubernetes 正在逐渐增加重启等待时间。

建议按照以下顺序排查：

1. kubectl describe pod 查看容器退出原因和 Events。
2. kubectl logs POD 查看当前容器日志。
3. 使用 kubectl logs POD --previous 查看上一次崩溃实例的日志。
4. 检查容器 command、args 和启动脚本。
5. 检查环境变量、ConfigMap 和 Secret 是否正确。
6. 检查依赖的数据库、Redis、DNS 或外部服务。
7. 检查 livenessProbe 是否配置错误导致容器被反复杀死。
8. 查看容器退出码，例如 137 通常需要关注 OOM 或强制终止。

核心思路是先确认“容器为什么退出”，而不是直接重启 Pod。"""
    },

    {
        "instruction": "Kubernetes Pod 被 OOMKilled 了，应该如何分析？",
        "output": """OOMKilled 通常表示容器使用的内存超过了 cgroup 的内存限制。

可以按照以下步骤排查：

1. kubectl describe pod 查看 Last State 和退出原因。
2. 检查容器 memory limit。
3. 查看 Prometheus 中容器历史内存使用曲线。
4. 判断内存增长是瞬时峰值还是持续泄漏。
5. 检查应用是否存在大对象、缓存无限增长、大请求或批处理任务。
6. 对 Java 等运行时还应检查堆大小是否和容器 limit 匹配。
7. 如果应用正常需要更多内存，再评估提高 memory limit。

不要简单地认为节点内存还有剩余，容器仍然可能因为自己的 memory limit 被 OOMKill。"""
    },

    {
        "instruction": "Kubernetes 中应用访问 Service 很慢，应该怎么排查？",
        "output": """建议把问题分成 DNS、Service 转发、网络链路和后端应用四层排查。

首先确认：

1. Pod 直接访问目标 Pod IP 是否正常。
2. 访问 Service ClusterIP 是否变慢。
3. 使用 nslookup 或 dig 检查 DNS 延迟。
4. 检查 CoreDNS 的 CPU、内存、错误率和日志。
5. 检查 kube-proxy 或 CNI 网络状态。
6. 查看是否存在 conntrack 表耗尽。
7. 检查目标 Service 后端 Endpoint 是否正常。
8. 对比节点之间和节点内部的网络延迟。

通过逐层绕过组件，可以快速判断问题位于 DNS、Service 转发、CNI 还是应用本身。"""
    },

    {
        "instruction": "线上接口延迟突然从 50ms 上升到 2 秒，但是应用 CPU 和内存都正常，应该怎么排查？",
        "output": """CPU 和内存正常并不能排除应用性能问题。接口延迟还可能来自下游依赖、网络、磁盘、线程池或连接池。

建议：

1. 先确认影响范围，是全部接口还是部分接口。
2. 查看 P50、P95、P99 延迟，而不是只看平均值。
3. 检查数据库、Redis、消息队列和外部 API 的响应时间。
4. 检查连接池是否耗尽。
5. 检查线程池、协程池或请求队列是否存在排队。
6. 检查 DNS 和网络延迟。
7. 查看磁盘 IO、日志写入和文件系统延迟。
8. 结合 Trace 找出请求耗时最长的 Span。
9. 对比问题发生前后的发布、配置和流量变化。

排障时应先确定延迟具体消耗在哪一层，而不是只盯着 CPU 和内存。"""
    },

    {
        "instruction": "磁盘空间明明还有很多，但是提示 No space left on device，可能是什么原因？",
        "output": """最常见的原因之一是 inode 用完，而不是磁盘容量本身耗尽。

可以先执行：

df -h

查看磁盘容量，再执行：

df -i

查看 inode 使用率。

如果 inode 达到 100%，通常说明存在大量小文件。

另外还需要检查：

1. 文件系统是否存在用户或项目 quota。
2. Docker 或容器运行时是否占用了大量 inode。
3. /tmp、日志目录或缓存目录是否生成大量小文件。
4. 删除中的文件是否仍被进程持有，可以使用 lsof +L1 检查。
5. 文件系统是否以只读或异常状态挂载。

因此遇到 No space left on device 时，不能只检查 df -h。"""
    },

    {
        "instruction": "Linux 服务器内存使用率很高，是不是一定存在问题？",
        "output": """不一定。

Linux 会主动利用空闲内存作为 page cache，因此仅仅看到 used memory 很高不能判断内存不足。

建议重点关注：

1. MemAvailable。
2. swap 是否持续增长。
3. 是否发生 major page fault。
4. 是否存在 OOMKill。
5. 应用 RSS 是否异常增长。
6. PSI memory pressure 是否持续升高。

可以使用 free -h、vmstat、sar、ps 以及 /proc/meminfo 综合判断。

Linux 中“空闲内存少”很多时候是正常现象，真正重要的是系统还能否快速回收缓存并满足新的内存申请。"""
    },

    {
        "instruction": "GPU 显存占用很高，但是 GPU 利用率很低，可能是什么原因？",
        "output": """GPU 显存占用高并不意味着 GPU 一直在执行计算。

常见原因包括：

1. 模型权重已经加载到显存，但当前请求量不足。
2. batch size 太小，GPU 没有足够并行任务。
3. CPU 数据预处理或 DataLoader 成为瓶颈。
4. 请求存在大量同步操作。
5. Python 或框架频繁启动小 kernel，导致 GPU 无法持续满载。
6. 网络、磁盘或远程存储导致 GPU 等待数据。
7. 推理服务受到并发数限制。
8. attention 或其他算子规模太小，无法充分利用 GPU。

分析时需要同时观察 GPU utilization、显存、SM 利用率、吞吐量和 CPU 情况，而不能只看显存占用。"""
    },

    {
        "instruction": "vLLM 显存占用很高，但是当前没有多少请求，这是为什么？",
        "output": """这是 vLLM 中比较常见的现象。

vLLM 启动时除了加载模型权重，还会根据 gpu-memory-utilization 等配置预留或使用显存用于 KV Cache。

因此即使当前请求很少，显存占用仍可能很高。

应该区分：

1. 模型权重占用。
2. KV Cache 占用。
3. CUDA Context 和运行时开销。
4. 临时计算 buffer。

如果设置了较高的 gpu-memory-utilization，例如 0.9 或 0.95，vLLM 会尽可能利用显存，以提高后续并发推理能力。

因此不能简单通过 nvidia-smi 中的显存占用判断当前业务负载。"""
    },

    {
        "instruction": "Prometheus 突然出现大量 target down，应该如何排查？",
        "output": """首先不要立即假设所有业务实例同时故障，需要判断是目标服务问题还是 Prometheus 自身采集链路问题。

建议：

1. 打开 Prometheus Targets 页面查看具体错误。
2. 判断失败目标是否集中在同一网段、集群或服务发现来源。
3. 从 Prometheus 所在机器直接 curl 对应 metrics 地址。
4. 检查 DNS、网络策略、防火墙和 Service。
5. 检查 Prometheus 的 CPU、内存和文件描述符。
6. 检查服务发现配置是否发生变化。
7. 检查证书、认证信息或 scrape 配置。
8. 查看 Prometheus 日志。

如果大量不同业务在同一时间 down，通常应该优先怀疑公共网络、DNS、服务发现或 Prometheus 自身，而不是同时怀疑所有业务实例。"""
    },

    {
        "instruction": "服务器出现大量 TIME_WAIT 会有什么影响？",
        "output": """TIME_WAIT 是 TCP 主动关闭连接一方的正常状态，本身并不一定代表故障。

大量 TIME_WAIT 通常意味着系统存在大量短连接。

需要关注：

1. TIME_WAIT 数量是否持续异常增长。
2. 本地临时端口是否接近耗尽。
3. 是否存在频繁创建 HTTP 或数据库短连接的应用。
4. 客户端是否没有使用连接池或 Keep-Alive。
5. NAT、负载均衡或代理设备是否受到连接跟踪压力。

不要看到 TIME_WAIT 就直接修改内核参数。首先应该确定是否真的导致端口耗尽或连接失败，并优先从应用连接复用方面解决。"""
    },

    {
        "instruction": "服务器 ping 正常，但是访问 TCP 服务超时，应该怎么排查？",
        "output": """ping 正常只能说明 ICMP 链路基本可达，并不能证明 TCP 服务正常。

建议：

1. 使用 ss -lntp 确认服务是否监听正确端口。
2. 使用 nc 或 telnet 测试 TCP 端口连通性。
3. 检查服务是否只监听 127.0.0.1。
4. 检查主机防火墙和云安全组。
5. 检查 Kubernetes NetworkPolicy 或其他网络策略。
6. 使用 tcpdump 观察 SYN、SYN-ACK 和 ACK。
7. 如果 SYN 到达服务器但没有 SYN-ACK，需要检查本机协议栈、防火墙或监听状态。
8. 如果 TCP 建连成功但应用仍超时，再检查应用层。

网络排障时需要区分 ICMP、TCP 建连和应用层响应三个不同阶段。"""
    },

    {
        "instruction": "systemd 服务启动失败应该怎么排查？",
        "output": """建议首先使用 systemctl status 服务名 查看服务状态和最近的错误信息。

然后：

1. journalctl -u 服务名 查看详细日志。
2. 检查 ExecStart 配置的程序路径是否存在。
3. 检查运行用户和文件权限。
4. 检查 Environment 或 EnvironmentFile。
5. 手工执行 ExecStart 中的命令，看是否可以正常启动。
6. 检查端口是否已经被占用。
7. 修改 unit 文件后执行 systemctl daemon-reload。
8. 检查 WorkingDirectory、依赖文件和 SELinux 等安全限制。

systemd 启动问题的核心是先得到真实退出原因，而不是反复执行 systemctl restart。"""
    },

    {
        "instruction": "Docker 容器启动后立刻退出，应该如何排查？",
        "output": """Docker 容器是否持续运行取决于它的 PID 1 主进程。

如果主进程结束，容器也会退出。

建议：

1. docker ps -a 查看容器退出状态。
2. docker logs 查看应用日志。
3. docker inspect 查看 ExitCode。
4. 检查 ENTRYPOINT 和 CMD。
5. 确认启动程序是否以前台模式运行。
6. 检查环境变量和挂载文件。
7. 检查权限和依赖服务。

不要通过在容器中人为运行一个无意义的永久进程来掩盖真正的启动问题。"""
    },

    {
        "instruction": "数据库本身 CPU 很低，但是应用访问数据库很慢，应该检查什么？",
        "output": """数据库 CPU 低并不代表数据库访问链路没有问题。

建议检查：

1. 应用到数据库之间的网络延迟和丢包。
2. DNS 解析耗时。
3. 数据库连接池是否耗尽。
4. 是否存在锁等待。
5. 慢 SQL 和磁盘 IO。
6. 最大连接数是否接近限制。
7. TLS 建连或认证是否耗时。
8. 应用是否频繁创建短连接。
9. 数据库代理或中间件是否存在瓶颈。

最好结合应用 Trace，把连接获取、网络传输、SQL 执行和结果返回几个阶段分别测量。"""
    },

    {
        "instruction": "Redis 延迟突然升高，但是 CPU 使用率不高，可能有哪些原因？",
        "output": """Redis 延迟升高不一定来自 CPU。

常见原因包括：

1. 执行了大 key 相关命令。
2. 使用 KEYS、SMEMBERS 等高复杂度操作。
3. RDB fork 或 AOF rewrite 带来延迟。
4. 磁盘 IO 抖动。
5. 内存压力导致 swap。
6. 网络延迟或丢包。
7. 客户端连接数过多。
8. Redis 单线程被某个慢命令阻塞。

可以使用 SLOWLOG、latency doctor、INFO、客户端监控和系统 IO 指标共同分析。"""
    },

    {
        "instruction": "Nginx 返回 502 Bad Gateway，应该怎么排查？",
        "output": """502 通常表示 Nginx 作为代理时无法从上游获得有效响应。

建议：

1. 查看 Nginx error.log。
2. 检查 upstream 服务是否存活。
3. 从 Nginx 主机直接访问 upstream 地址。
4. 检查 upstream 的 IP 和端口配置。
5. 检查 DNS。
6. 检查连接超时和连接数。
7. 检查后端是否主动关闭连接或崩溃。
8. 检查容器、Service 或负载均衡配置。

502 的重点是分析 Nginx 到 upstream 这一段，而不是首先检查浏览器到 Nginx 的连接。"""
    },

    {
        "instruction": "服务器磁盘 IO 利用率很高，应该如何定位是哪个进程导致的？",
        "output": """可以从设备和进程两个层面排查。

首先使用 iostat -x 1 确认哪个块设备存在高利用率、长 await 或较深队列。

然后可以使用：

iotop

或者：

pidstat -d 1

定位读写量较大的进程。

还可以结合 lsof 确认进程正在访问哪些文件。

如果是数据库或日志服务，还需要继续判断是正常业务流量、后台任务、日志刷盘还是异常请求造成的。

需要注意，磁盘 utilization 高不一定代表性能已经到极限，还需要结合 await、队列长度和实际吞吐分析。"""
    },

    {
        "instruction": "Kubernetes Node 变成 NotReady，应该优先检查什么？",
        "output": """Node NotReady 时建议先查看节点 Condition。

执行：

kubectl describe node NODE

重点关注 Ready、MemoryPressure、DiskPressure、PIDPressure 和 NetworkUnavailable。

然后登录节点检查：

1. kubelet 是否运行正常。
2. containerd 或其他容器运行时是否正常。
3. 磁盘是否满。
4. 内存是否严重不足。
5. 节点到 API Server 的网络是否正常。
6. CNI 插件是否异常。
7. kubelet 日志中是否存在证书、运行时或网络错误。

Node NotReady 只是最终状态，需要通过 Condition 和 kubelet 日志定位真正原因。"""
    },

]


def write_jsonl(path, rows):

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:

        for row in rows:

            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    random.seed(42)

    random.shuffle(
        examples
    )

    split = max(
        1,
        int(
            len(examples) * 0.9
        ),
    )

    train = examples[:split]
    val = examples[split:]

    write_jsonl(
        OUTPUT_DIR / "train.jsonl",
        train,
    )

    write_jsonl(
        OUTPUT_DIR / "val.jsonl",
        val,
    )

    print(
        "Total:",
        len(examples),
    )

    print(
        "Train:",
        len(train),
    )

    print(
        "Val:",
        len(val),
    )

    print(
        "Saved:",
        OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
