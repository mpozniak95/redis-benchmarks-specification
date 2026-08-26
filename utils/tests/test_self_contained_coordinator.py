import os

import docker
import redis
import yaml
from pathlib import Path

from redisbench_admin.utils.remote import get_overall_dashboard_keynames
from redisbench_admin.utils.utils import get_ts_metric_name
import logging

from redis_benchmarks_specification.__common__.env import (
    STREAM_KEYNAME_NEW_BUILD_EVENTS,
)
from redis_benchmarks_specification.__common__.spec import (
    extract_client_cpu_limit,
    extract_client_container_image,
    extract_client_tool,
)
from redis_benchmarks_specification.__self_contained_coordinator__.self_contained_coordinator import (
    self_contained_coordinator_blocking_read,
    stop_and_remove_container_safe,
)
from redis_benchmarks_specification.__self_contained_coordinator__.docker import (
    start_redis_container,
)

from redis_benchmarks_specification.__self_contained_coordinator__.runners import (
    build_runners_consumer_group_create,
    get_runners_consumer_group_name,
)
from redis_benchmarks_specification.__self_contained_coordinator__.cpuset import (
    generate_cpuset_cpus,
)
from redis_benchmarks_specification.__setups__.topologies import get_topologies
from utils.tests.test_data.api_builder_common import flow_1_and_2_api_builder_checks


from redis_benchmarks_specification.__self_contained_coordinator__.docker import (
    collect_replication_stream_stats,
    generate_standalone_redis_server_args,
    generate_cluster_redis_server_args,
    inject_replication_stream_metrics,
    inject_replication_sync_metrics,
    spin_up_redis_replicas,
    spin_docker_cluster_redis,
)


def test_inject_replication_sync_metrics_with_replicas():
    """Both ReplicationFullSyncSeconds and ReplicationFullSyncCountDuringBench
    should be injected when replicas exist and sync times are non-empty."""
    results = {"ALL STATS": {"Totals": {"Ops/sec": 100000.0}}}
    ok = inject_replication_sync_metrics(results, [3.5, 4.2, 2.1], 2)
    assert ok is True
    totals = results["ALL STATS"]["Totals"]
    # Max sync time across replicas (slowest replica gates the topology)
    assert totals["ReplicationFullSyncSeconds"] == 4.2
    assert totals["ReplicationFullSyncCountDuringBench"] == 2
    # Existing metrics not clobbered
    assert totals["Ops/sec"] == 100000.0


def test_inject_replication_sync_metrics_no_replicas():
    """When no replicas were spun up, ReplicationFullSyncSeconds is omitted
    but ReplicationFullSyncCountDuringBench is still set to 0."""
    results = {"ALL STATS": {"Totals": {"Ops/sec": 50000.0}}}
    ok = inject_replication_sync_metrics(results, [], 0)
    assert ok is True
    totals = results["ALL STATS"]["Totals"]
    assert "ReplicationFullSyncSeconds" not in totals
    assert totals["ReplicationFullSyncCountDuringBench"] == 0
    assert totals["Ops/sec"] == 50000.0


def test_inject_replication_sync_metrics_creates_missing_keys():
    """The function should create ALL STATS / Totals if they don't exist."""
    results = {}
    ok = inject_replication_sync_metrics(results, [1.5], 1)
    assert ok is True
    assert "ALL STATS" in results
    assert "Totals" in results["ALL STATS"]
    assert results["ALL STATS"]["Totals"]["ReplicationFullSyncSeconds"] == 1.5
    assert results["ALL STATS"]["Totals"]["ReplicationFullSyncCountDuringBench"] == 1


def test_inject_replication_sync_metrics_invalid_input():
    """Non-dict results should be rejected gracefully (return False)."""
    assert inject_replication_sync_metrics(None, [1.0], 0) is False
    assert inject_replication_sync_metrics("not a dict", [1.0], 0) is False
    assert inject_replication_sync_metrics([], [1.0], 0) is False


def test_inject_replication_sync_metrics_count_only_during_bench():
    """A backlog overflow during a write-heavy benchmark on an existing
    replica should bump ReplicationFullSyncCountDuringBench even when the
    initial sync time was already captured."""
    results = {"ALL STATS": {"Totals": {}}}
    ok = inject_replication_sync_metrics(results, [2.0], 5)
    assert ok is True
    totals = results["ALL STATS"]["Totals"]
    assert totals["ReplicationFullSyncSeconds"] == 2.0
    assert totals["ReplicationFullSyncCountDuringBench"] == 5


class _FakeRedisConn:
    """Minimal stand-in for a redis connection returning canned INFO sections."""

    def __init__(self, sections, raises=False):
        self.sections = sections
        self.raises = raises

    def info(self, section):
        if self.raises:
            raise ConnectionError("server is gone")
        return self.sections.get(section, {})


def _conn(
    repl_out=0, repl_in=0, uncompressed=0, decompressed=0, cpu_sys=0.0, cpu_user=0.0
):
    return _FakeRedisConn(
        {
            "stats": {
                "total_net_repl_output_bytes": repl_out,
                "total_net_repl_input_bytes": repl_in,
                "total_net_repl_uncompressed_bytes": uncompressed,
                "total_net_repl_decompressed_bytes": decompressed,
            },
            "cpu": {"used_cpu_sys": cpu_sys, "used_cpu_user": cpu_user},
        }
    )


def test_collect_replication_stream_stats_sums_and_tolerates_failures():
    snapshot = collect_replication_stream_stats(
        [
            _conn(repl_out=100, uncompressed=1000, cpu_sys=1.0, cpu_user=2.0),
            _conn(repl_out=50, uncompressed=500, cpu_sys=0.5, cpu_user=1.0),
            _FakeRedisConn({}, raises=True),
        ]
    )
    assert snapshot["total_net_repl_output_bytes"] == 150
    assert snapshot["total_net_repl_uncompressed_bytes"] == 1500
    assert snapshot["used_cpu_sys"] == 1.5
    assert snapshot["used_cpu_user"] == 3.0
    # Counters redis omits when compression is off default to 0, not KeyError.
    assert snapshot["total_net_repl_decompressed_bytes"] == 0


def test_inject_replication_stream_metrics_compression_on():
    """With compression active the ratio comes from the master's own counters and
    the CPU deltas are normalized per replica and by the benchmark duration."""
    results = {"ALL STATS": {"Totals": {"Ops/sec": 200000.0}}}
    primary_before = collect_replication_stream_stats([_conn()])
    primary_after = collect_replication_stream_stats(
        [_conn(repl_out=1000, uncompressed=10000, cpu_sys=4.0, cpu_user=6.0)]
    )
    replica_before = collect_replication_stream_stats([_conn()])
    replica_after = collect_replication_stream_stats(
        [_conn(repl_in=1000, decompressed=10000, cpu_sys=1.0, cpu_user=3.0)]
    )
    ok = inject_replication_stream_metrics(
        results,
        primary_before,
        primary_after,
        replica_before,
        replica_after,
        1,
        20.0,
        3,
    )
    assert ok is True
    totals = results["ALL STATS"]["Totals"]
    assert totals["MasterReplicationBytesSent"] == 1000
    assert totals["MasterReplicationBytesUncompressed"] == 10000
    assert totals["ReplicaReplicationBytesReceived"] == 1000
    assert totals["ReplicaReplicationBytesDecompressed"] == 10000
    assert totals["ReplicationCompressionRatio"] == 10.0
    assert totals["MasterCPUSeconds"] == 10.0
    assert totals["ReplicaCPUSecondsPerReplica"] == 4.0
    assert totals["MasterCPUUtilizationPct"] == 50.0
    assert totals["ReplicaCPUUtilizationPct"] == 20.0
    assert totals["Ops/sec"] == 200000.0


def test_inject_replication_stream_metrics_compression_off():
    """With compression off redis never emits the compressed byte counters, so
    the ratio must be reported as 1.0 rather than dividing by zero."""
    results = {}
    before = collect_replication_stream_stats([_conn()])
    primary_after = collect_replication_stream_stats(
        [_conn(repl_out=5000, cpu_sys=1.0, cpu_user=1.0)]
    )
    replica_after = collect_replication_stream_stats([_conn(repl_in=5000)])
    ok = inject_replication_stream_metrics(
        results, before, primary_after, before, replica_after, 1, 10.0, 0
    )
    assert ok is True
    totals = results["ALL STATS"]["Totals"]
    assert totals["MasterReplicationBytesSent"] == 5000
    assert totals["MasterReplicationBytesUncompressed"] == 0
    assert totals["ReplicationCompressionRatio"] == 1.0


def test_inject_replication_stream_metrics_counter_reset_is_clamped():
    """A CONFIG RESETSTAT or a restarted server mid-window must not produce a
    negative bandwidth datapoint."""
    results = {}
    before = collect_replication_stream_stats([_conn(repl_out=9000)])
    after = collect_replication_stream_stats([_conn(repl_out=10)])
    ok = inject_replication_stream_metrics(
        results, before, after, before, after, 1, 10.0, 0
    )
    assert ok is True
    assert results["ALL STATS"]["Totals"]["MasterReplicationBytesSent"] == 0.0


def test_inject_replication_stream_metrics_no_duration_and_no_replicas():
    """Without a known benchmark duration the utilization metrics are omitted,
    and a zero replica count must not divide by zero."""
    results = {}
    before = collect_replication_stream_stats([_conn()])
    after = collect_replication_stream_stats([_conn(repl_out=10, cpu_sys=1.0)])
    ok = inject_replication_stream_metrics(
        results, before, after, before, after, 0, None, 0
    )
    assert ok is True
    totals = results["ALL STATS"]["Totals"]
    assert "MasterCPUUtilizationPct" not in totals
    assert "ReplicaCPUUtilizationPct" not in totals
    assert totals["ReplicaCPUSecondsPerReplica"] == 0.0


def test_inject_replication_stream_metrics_invalid_input():
    assert inject_replication_stream_metrics(None, {}, {}, {}, {}, 1) is False
    assert inject_replication_stream_metrics("not a dict", {}, {}, {}, {}, 1) is False


def test_repl_compression_specs_config_requirements():
    """The repl-compression specs only measure anything if three preconditions
    hold, and all three are silent no-ops when broken:

    1. repl-compression must be set in dbconfig configuration-parameters, so the
       coordinator passes it as a startup argument to BOTH primary and replicas
       (it is an immutable config and the master rejects a level mismatch).
    2. every topology must be an io-threads replica topology — the master
       refuses the compression handshake when io-threads <= 1.
    3. build-variants must only list the *-compression builders, since a server
       built without BUILD_COMPRESSION=yes forces repl-compression to 0 at
       startup (and the dockerhub images do not know the config at all).
    """
    import glob

    spec_paths = sorted(
        glob.glob(
            "./redis_benchmarks_specification/test-suites/*-repl-compression-*.yml"
        )
    )
    # The A/B/C steady-state triplet plus the full-sync variant.
    assert len(spec_paths) == 4, f"unexpected repl-compression specs: {spec_paths}"

    topologies = get_topologies(
        "./redis_benchmarks_specification/setups/topologies/topologies.yml"
    )
    levels_seen = set()
    for path in spec_paths:
        with open(path, "r") as yml_file:
            cfg = yaml.safe_load(yml_file)
        name = os.path.basename(path)
        assert cfg["name"] == name[: -len(".yml")], f"{name}: name must match filename"

        params = cfg["dbconfig"]["configuration-parameters"]
        assert (
            "repl-compression" in params
        ), f"{name}: must pin repl-compression explicitly"
        level = int(params["repl-compression"])
        levels_seen.add(level)

        for topology in cfg["redis-topologies"]:
            assert topology in topologies, f"{name}: unknown topology {topology}"
            redis_arguments = topologies[topology].get("redis_arguments", "")
            assert (
                "replicas" in topology
            ), f"{name}: repl-compression specs need a replica topology, got {topology}"
            assert "--io-threads" in redis_arguments, (
                f"{name}: topology {topology} has io-threads <= 1, the master will "
                "reject the compression handshake"
            )

        for build_variant in cfg["build-variants"]:
            assert build_variant.endswith("-compression"), (
                f"{name}: build variant {build_variant} is not compression-enabled; "
                "repl-compression would be silently forced to 0"
            )

        exported = cfg["exporter"]["redistimeseries"]["metrics"]
        for metric in [
            "MasterReplicationBytesSent",
            "MasterReplicationBytesUncompressed",
            "ReplicaReplicationBytesReceived",
            "ReplicationCompressionRatio",
            "MasterCPUUtilizationPct",
            "ReplicaCPUUtilizationPct",
            "ReplicationFullSyncSeconds",
        ]:
            assert any(
                metric in jsonpath for jsonpath in exported
            ), f"{name}: {metric} is not exported, the KPI would be dropped"

    # A control arm (level 0) is what the compressed arms' bandwidth/CPU deltas
    # are measured against — without it the suite reports absolutes only.
    assert 0 in levels_seen, "repl-compression specs must include a level-0 control"
    assert len(levels_seen) >= 3, f"expected a level curve, got {sorted(levels_seen)}"


def test_compression_builders_enable_build_compression():
    """The *-compression builders must actually pass BUILD_COMPRESSION=yes on the
    make command line: the builder starts the build container with
    command=build_command and no environment, so the `env` block cannot carry it.
    """
    import glob

    builder_paths = sorted(
        glob.glob("./redis_benchmarks_specification/setups/builders/*-compression.yml")
    )
    assert len(builder_paths) == 2, f"expected amd64+arm64, got {builder_paths}"
    for path in builder_paths:
        with open(path, "r") as yml_file:
            build_config = yaml.safe_load(yml_file)
        assert build_config["id"] == os.path.basename(path)[: -len(".yml")]
        assert "BUILD_COMPRESSION=yes" in build_config.get(
            "build_command", ""
        ), f"{path}: build_command must pass BUILD_COMPRESSION=yes"


def test_preload_before_replica_flag_in_20m_spec():
    """The 20M-keys replica-only test spec must set preload_before_replica=true.

    Without this flag the coordinator spins up replicas BEFORE preload, which
    means the full sync transfers an empty primary (sync time ~0s) and the
    20 GB dataset propagates via the replication stream instead. The whole
    point of this benchmark is to measure full-sync time on the loaded
    dataset, so the flag must be present.
    """
    spec_path = (
        "./redis_benchmarks_specification/test-suites/"
        "memtier_benchmark-20Mkeys-load-string-with-1KiB-values-replica-only.yml"
    )
    with open(spec_path, "r") as yml_file:
        benchmark_config = yaml.safe_load(yml_file)
    assert (
        benchmark_config["dbconfig"].get("preload_before_replica") is True
    ), "20M replica-only spec must set preload_before_replica: true"
    # Sanity-check that the spec has a preload_tool — the flag is meaningless
    # without one.
    assert "preload_tool" in benchmark_config["dbconfig"]
    # And that it ONLY targets replica topologies
    for topology in benchmark_config["redis-topologies"]:
        assert (
            "replicas" in topology
        ), f"20M replica-only spec must use only replica topologies, got {topology}"


def test_preload_before_replica_default_off():
    """Existing replica test specs must not have preload_before_replica set.

    This guards against accidentally enabling the new ordering for tests
    that were tuned for the historical "preload after replica" behavior.
    """
    import glob

    spec_files = glob.glob("./redis_benchmarks_specification/test-suites/*.yml")
    # The only spec that should have the flag is the 20M one we just added.
    enabled_specs = []
    for path in spec_files:
        with open(path, "r") as yml_file:
            try:
                cfg = yaml.safe_load(yml_file)
            except yaml.YAMLError:
                continue
        if not isinstance(cfg, dict):
            continue
        dbconfig = cfg.get("dbconfig") or {}
        if isinstance(dbconfig, list):
            # Some legacy specs use a list-of-dicts format for dbconfig
            merged = {}
            for entry in dbconfig:
                if isinstance(entry, dict):
                    merged.update(entry)
            dbconfig = merged
        if dbconfig.get("preload_before_replica") is True:
            enabled_specs.append(os.path.basename(path))
    expected = {
        "memtier_benchmark-20Mkeys-load-string-with-1KiB-values-replica-only.yml",
        "memtier_benchmark-20Mkeys-load-string-with-1KiB-values-replica-only-no-rdbcomp.yml",
        "memtier_benchmark-20Mkeys-load-string-with-1KiB-values-replica-only-parallel-fullsync-02.yml",
        "memtier_benchmark-20Mkeys-load-string-with-1KiB-values-replica-only-parallel-fullsync-04.yml",
        "memtier_benchmark-20Mkeys-load-string-with-1KiB-values-replica-only-parallel-fullsync-04-16cpu.yml",
        "memtier_benchmark-20Mkeys-load-string-with-1KiB-values-replica-only-parallel-fullsync-08-16cpu.yml",
        "memtier_benchmark-3Mkeys-string-set-1KiB-pipeline-10.yml",
        "memtier_benchmark-3Mkeys-string-set-1KiB-pipeline-10-rate-limited-40Kops.yml",
        # Replication-compression specs: the full-sync one needs the flag for the
        # same reason as the 20M parent, and the steady-state triplet needs the
        # dataset to arrive via one initial full sync so the measured window is
        # pure steady-state stream traffic.
        "memtier_benchmark-20Mkeys-load-string-with-1KiB-values-replica-only-repl-compression-03.yml",
        "memtier_benchmark-3Mkeys-string-set-1KiB-pipeline-10-repl-compression-00.yml",
        "memtier_benchmark-3Mkeys-string-set-1KiB-pipeline-10-repl-compression-03.yml",
        "memtier_benchmark-3Mkeys-string-set-1KiB-pipeline-10-repl-compression-09.yml",
    }
    assert (
        set(enabled_specs) == expected
    ), f"Unexpected specs with preload_before_replica enabled: {enabled_specs}"


def test_extract_client_cpu_limit():
    with open(
        "./utils/tests/test_data/test-suites/redis-benchmark-full-suite-1Mkeys-100B.yml",
        "r",
    ) as yml_file:
        benchmark_config = yaml.safe_load(yml_file)
        client_cpu_limit = extract_client_cpu_limit(benchmark_config)
        # we use a benchmark spec with smaller CPU limit for client given github machines only contain 2 cores
        # and we need 1 core for DB and another for CLIENT
        assert client_cpu_limit == 1


def test_extract_client_container_image():
    with open(
        "./utils/tests/test_data/test-suites/redis-benchmark-full-suite-1Mkeys-100B.yml",
        "r",
    ) as yml_file:
        benchmark_config = yaml.safe_load(yml_file)
        client_container_image = extract_client_container_image(benchmark_config)
        assert client_container_image == "redis:6.2.4"

    with open(
        "./redis_benchmarks_specification/test-suites/memtier_benchmark-1Mkeys-100B-expire-use-case.yml",
        "r",
    ) as yml_file:
        benchmark_config = yaml.safe_load(yml_file)
        client_container_image = extract_client_container_image(benchmark_config)
        assert client_container_image == "redislabs/memtier_benchmark:edge"


def test_extract_client_tool():
    with open(
        "./utils/tests/test_data/test-suites/redis-benchmark-full-suite-1Mkeys-100B.yml",
        "r",
    ) as yml_file:
        benchmark_config = yaml.safe_load(yml_file)
        client_tool = extract_client_tool(benchmark_config)
        assert client_tool == "redis-benchmark"

    with open(
        "./redis_benchmarks_specification/test-suites/memtier_benchmark-1Mkeys-100B-expire-use-case.yml",
        "r",
    ) as yml_file:
        benchmark_config = yaml.safe_load(yml_file)
        client_tool = extract_client_tool(benchmark_config)
        assert client_tool == "memtier_benchmark"


def test_generate_cpuset_cpus():
    db_cpuset_cpus, current_cpu_pos = generate_cpuset_cpus(2.0, 0)
    assert db_cpuset_cpus == "0,1"

    db_cpuset_cpus, current_cpu_pos = generate_cpuset_cpus(3, current_cpu_pos)
    assert db_cpuset_cpus == "2,3,4"


def run_coordinator_tests():
    run_coordinator = True
    TST_RUNNER_X = os.getenv("TST_RUNNER_X", "0")
    if TST_RUNNER_X == "0":
        run_coordinator = False
    return run_coordinator


def test_self_contained_coordinator_blocking_read():
    try:
        if run_coordinator_tests():
            conn = redis.StrictRedis(port=6379)
            conn.ping()
            expected_datapoint_ts = None
            conn.flushall()
            build_variant_name, reply_fields = flow_1_and_2_api_builder_checks(conn)
            if b"git_timestamp_ms" in reply_fields:
                expected_datapoint_ts = int(reply_fields[b"git_timestamp_ms"].decode())
            if b"git_timestamp_ms" in reply_fields:
                expected_datapoint_ts = int(reply_fields[b"git_timestamp_ms"].decode())
            if "git_timestamp_ms" in reply_fields:
                expected_datapoint_ts = int(reply_fields["git_timestamp_ms"])

            assert conn.exists(STREAM_KEYNAME_NEW_BUILD_EVENTS)
            assert conn.xlen(STREAM_KEYNAME_NEW_BUILD_EVENTS) > 0
            running_platform = "fco-ThinkPad-T490"

            build_runners_consumer_group_create(conn, running_platform, "0")
            datasink_conn = redis.StrictRedis(port=6379)
            rts = datasink_conn.ts()
            docker_client = docker.from_env()
            home = str(Path.home())
            stream_id = ">"
            topologies_map = get_topologies(
                "./redis_benchmarks_specification/setups/topologies/topologies.yml"
            )
            # we use a benchmark spec with smaller CPU limit for client given github machines only contain 2 cores
            # and we need 1 core for DB and another for CLIENT
            testsuite_spec_files = [
                "./utils/tests/test_data/test-suites/redis-benchmark-full-suite-1Mkeys-100B.yml"
            ]
            (
                result,
                stream_id,
                number_processed_streams,
                _,
            ) = self_contained_coordinator_blocking_read(
                conn,
                True,
                docker_client,
                home,
                stream_id,
                datasink_conn,
                testsuite_spec_files,
                topologies_map,
                running_platform,
                False,
                [],
            )
            assert result == True
            assert number_processed_streams == 1
            # ensure we're able to aknowledge the consumed message
            assert (
                conn.xinfo_consumers(
                    STREAM_KEYNAME_NEW_BUILD_EVENTS,
                    get_runners_consumer_group_name(running_platform),
                )[0]["pending"]
                == 0
            )
            tf_github_org = "redis"
            tf_github_repo = "redis"
            test_name = "redis-benchmark-full-suite-1Mkeys-100B"
            tf_triggering_env = "ci"
            deployment_type = "oss-standalone"
            deployment_name = "oss-standalone"
            metric_name = "rps"
            use_metric_context_path = True
            metric_context_path = "MSET"

            ts_key_name = get_ts_metric_name(
                "by.branch",
                "unstable",
                tf_github_org,
                tf_github_repo,
                deployment_name,
                deployment_type,
                test_name,
                tf_triggering_env,
                metric_name,
                metric_context_path,
                use_metric_context_path,
                build_variant_name,
                running_platform,
            )

            assert ts_key_name.encode() in conn.keys()
            assert len(rts.range(ts_key_name, 0, "+")) == 1
            if expected_datapoint_ts is not None:
                assert rts.range(ts_key_name, 0, "+")[0][0] == expected_datapoint_ts
            (
                prefix,
                testcases_setname,
                deployment_name_setname,
                tsname_project_total_failures,
                tsname_project_total_success,
                running_platforms_setname,
                build_variant_setname,
                testcases_metric_context_path_setname,
                testcases_and_metric_context_path_setname,
                project_archs_setname,
                project_oss_setname,
                project_branches_setname,
                project_versions_setname,
                project_compilers_setname,
            ) = get_overall_dashboard_keynames(
                tf_github_org,
                tf_github_repo,
                tf_triggering_env,
                build_variant_name,
                running_platform,
                test_name,
            )

            assert datasink_conn.exists(testcases_setname)
            assert datasink_conn.exists(running_platforms_setname)
            assert datasink_conn.exists(build_variant_setname)
            assert datasink_conn.exists(testcases_and_metric_context_path_setname)
            assert datasink_conn.exists(testcases_metric_context_path_setname)
            assert build_variant_name.encode() in datasink_conn.smembers(
                build_variant_setname
            )
            assert test_name.encode() in datasink_conn.smembers(testcases_setname)
            assert running_platform.encode() in datasink_conn.smembers(
                running_platforms_setname
            )
            testcases_and_metric_context_path_members = [
                x.decode()
                for x in datasink_conn.smembers(
                    testcases_and_metric_context_path_setname
                )
            ]
            metric_context_path_members = [
                x.decode()
                for x in datasink_conn.smembers(testcases_metric_context_path_setname)
            ]
            assert len(testcases_and_metric_context_path_members) == len(
                metric_context_path_members
            )

            assert [x.decode() for x in datasink_conn.smembers(testcases_setname)] == [
                test_name
            ]

            assert "amd64".encode() in datasink_conn.smembers(project_archs_setname)
            assert "debian-bookworm".encode() in datasink_conn.smembers(
                project_oss_setname
            )
            assert "gcc".encode() in datasink_conn.smembers(project_compilers_setname)
            assert build_variant_name.encode() in datasink_conn.smembers(
                build_variant_setname
            )
            assert running_platform.encode() in datasink_conn.smembers(
                running_platforms_setname
            )

            assert len(datasink_conn.smembers(project_archs_setname)) == 1
            assert len(datasink_conn.smembers(project_oss_setname)) == 1
            assert len(datasink_conn.smembers(project_compilers_setname)) == 1
            assert len(datasink_conn.smembers(build_variant_setname)) == 1
            assert len(datasink_conn.smembers(running_platforms_setname)) == 1
            assert len(datasink_conn.smembers(testcases_setname)) == 1
            assert len(datasink_conn.smembers(project_branches_setname)) == 1
            assert len(datasink_conn.smembers(project_versions_setname)) == 0

    except redis.exceptions.ConnectionError:
        pass


def test_start_redis_container():
    temporary_dir = os.getenv("TST_BINARY_REDIS_DIR", "")
    if temporary_dir == "":
        return

    mnt_point = "/mnt/redis/"
    executable = f"{mnt_point}redis-server"
    redis_proc_start_port = 6379
    current_cpu_pos = 0
    ceil_db_cpu_limit = 1
    redis_configuration_parameters = None
    redis_arguments = ""
    docker_client = docker.from_env()
    redis_containers = []

    redis_password = "test_password_123"
    command = generate_standalone_redis_server_args(
        executable,
        redis_proc_start_port,
        mnt_point,
        redis_configuration_parameters,
        redis_arguments,
        redis_password,
    )
    command_str = " ".join(command)
    db_cpuset_cpus, current_cpu_pos = generate_cpuset_cpus(
        ceil_db_cpu_limit, current_cpu_pos
    )
    run_image = "gcc:8.5"
    redis_container = start_redis_container(
        command_str,
        db_cpuset_cpus,
        docker_client,
        mnt_point,
        redis_containers,
        run_image,
        temporary_dir,
    )
    r = redis.StrictRedis(port=redis_proc_start_port, password=redis_password)
    try:
        r.ping()
    except redis.exceptions.ConnectionError:
        # Access and print the logs
        logs = redis_container.logs().decode("utf-8")
        logging.error("Container failed. Here are the logs:")
        logging.error(logs)
        raise
    redis_container.remove()


def test_spin_up_redis_replicas():
    """Test replica deployment using the official redis:8.6 Docker image."""
    run_image = "redis:8.6"
    mnt_point = ""
    executable = "redis-server"
    primary_port = 6399
    current_cpu_pos = 0
    redis_configuration_parameters = None
    redis_arguments = ""
    docker_client = docker.from_env()
    redis_containers = []
    redis_password = "test_password_123"
    temporary_dir = ""

    # Start primary
    command = generate_standalone_redis_server_args(
        executable,
        primary_port,
        mnt_point,
        redis_configuration_parameters,
        redis_arguments,
        redis_password,
    )
    command_str = " ".join(command)
    db_cpuset_cpus, current_cpu_pos = generate_cpuset_cpus(1, current_cpu_pos)
    primary_container = start_redis_container(
        command_str,
        db_cpuset_cpus,
        docker_client,
        mnt_point,
        redis_containers,
        run_image,
        temporary_dir,
    )
    try:
        r = redis.StrictRedis(port=primary_port, password=redis_password)
        r.ping()

        # Start 1 replica
        (
            replica_conns,
            current_cpu_pos,
            sync_times_seconds,
        ) = spin_up_redis_replicas(
            1,
            primary_port,
            current_cpu_pos,
            docker_client,
            redis_containers,
            run_image,
            temporary_dir,
            mnt_point,
            1,
            redis_configuration_parameters,
            redis_arguments,
            redis_password,
        )
        assert len(replica_conns) == 1
        replica_info = replica_conns[0].info("replication")
        assert replica_info["role"] == "slave"
        assert replica_info["master_link_status"] == "up"

        primary_info = r.info("replication")
        assert primary_info["connected_slaves"] == 1

        # Validate the sync_times_seconds return value
        assert isinstance(sync_times_seconds, list)
        assert len(sync_times_seconds) == 1
        assert isinstance(sync_times_seconds[0], float)
        # Empty primary, sync should complete in well under 30 seconds
        assert 0.0 <= sync_times_seconds[0] < 30.0
        logging.info(
            "Replica sync time captured: {:.3f}s".format(sync_times_seconds[0])
        )

        # Validate that the master exposes sync_full counter
        # (used by the coordinator's ReplicationFullSyncCountDuringBench metric).
        # Note: sync_full is in the "stats" section (server.stat_sync_full),
        # not the replication section.
        primary_stats = r.info("stats")
        assert (
            "sync_full" in primary_stats
        ), "Master stats info should expose sync_full counter"
        assert (
            int(primary_stats["sync_full"]) >= 1
        ), "Master should report at least 1 full sync after replica connect"

        # Shutdown replicas then primary
        for rc in replica_conns:
            try:
                rc.shutdown(nosave=True)
            except redis.exceptions.ConnectionError:
                pass
        r.shutdown(nosave=True)
    except Exception:
        # Print logs on failure
        for c in redis_containers:
            try:
                logs = c.logs().decode("utf-8")
                logging.error(f"Container logs: {logs}")
            except Exception:
                pass
        raise
    finally:
        for c in redis_containers:
            try:
                c.stop()
                c.remove()
            except Exception:
                pass


def test_generate_cluster_redis_server_args():
    """Cluster server args must include --cluster-enabled yes and a unique
    nodes config file per port."""
    command = generate_cluster_redis_server_args(
        "/mnt/redis/redis-server",
        6379,
        "/mnt/redis/",
        None,
        "",
        None,
    )
    assert "/mnt/redis/redis-server" in command
    assert "--cluster-enabled" in command
    idx = command.index("--cluster-enabled")
    assert command[idx + 1] == "yes"
    assert "--cluster-config-file" in command
    idx2 = command.index("--cluster-config-file")
    assert command[idx2 + 1] == "nodes-6379.conf"
    assert "--cluster-node-timeout" in command
    idx3 = command.index("--cluster-node-timeout")
    assert command[idx3 + 1] == "5000"
    # Must also contain the standalone args (port, dir)
    assert "--port" in command
    port_idx = command.index("--port")
    assert command[port_idx + 1] == "6379"


def test_generate_cluster_redis_server_args_with_password():
    """Cluster args with a password must include --requirepass."""
    command = generate_cluster_redis_server_args(
        "redis-server",
        6380,
        "",
        None,
        "",
        "secret123",
    )
    assert "--cluster-enabled" in command
    assert "--requirepass" in command
    pw_idx = command.index("--requirepass")
    assert command[pw_idx + 1] == "secret123"
    cfg_idx = command.index("--cluster-config-file")
    assert command[cfg_idx + 1] == "nodes-6380.conf"


def test_generate_cluster_redis_server_args_unique_per_port():
    """Each port must get a unique cluster config filename."""
    ports = [6379, 6380, 6381]
    config_files = []
    for port in ports:
        cmd = generate_cluster_redis_server_args(
            "redis-server", port, "", None, "", None
        )
        idx = cmd.index("--cluster-config-file")
        config_files.append(cmd[idx + 1])
    # All config files must be unique
    assert len(set(config_files)) == len(ports)
    assert config_files == ["nodes-6379.conf", "nodes-6380.conf", "nodes-6381.conf"]


def test_cluster_yaml_spec_get():
    """The cluster GET test suite YAML must reference the correct topology."""
    spec_path = (
        "./redis_benchmarks_specification/test-suites/"
        "memtier_benchmark-1Mkeys-string-get-10B-cluster.yml"
    )
    with open(spec_path, "r") as yml_file:
        benchmark_config = yaml.safe_load(yml_file)
    assert "oss-cluster-3-primaries" in benchmark_config["redis-topologies"]
    assert "get" in benchmark_config["tested-commands"]
    assert benchmark_config["clientconfig"]["tool"] == "memtier_benchmark"
    # --cluster-mode is injected programmatically, not in the YAML arguments
    assert (
        "--cluster-mode"
        not in benchmark_config["dbconfig"]["preload_tool"]["arguments"]
    )


def test_cluster_yaml_spec_set():
    """The cluster SET test suite YAML must reference the correct topology
    and should NOT have a preload tool (it IS the preload)."""
    spec_path = (
        "./redis_benchmarks_specification/test-suites/"
        "memtier_benchmark-1Mkeys-load-string-with-10B-values-cluster.yml"
    )
    with open(spec_path, "r") as yml_file:
        benchmark_config = yaml.safe_load(yml_file)
    assert "oss-cluster-3-primaries" in benchmark_config["redis-topologies"]
    assert "set" in benchmark_config["tested-commands"]
    assert benchmark_config["clientconfig"]["tool"] == "memtier_benchmark"
    # SET workload doesn't need preload
    assert "preload_tool" not in benchmark_config.get("dbconfig", {})


def test_cluster_mode_injected_programmatically():
    """--cluster-mode must NOT be in cluster YAML arguments but MUST be
    injected by prepare_memtier_benchmark_parameters when
    oss_cluster_api_enabled is True."""
    from redis_benchmarks_specification.__self_contained_coordinator__.clients import (
        prepare_memtier_benchmark_parameters,
    )

    cluster_specs = [
        "./redis_benchmarks_specification/test-suites/"
        "memtier_benchmark-1Mkeys-string-get-10B-cluster.yml",
        "./redis_benchmarks_specification/test-suites/"
        "memtier_benchmark-10Mkeys-string-get-set-1-10-512B-cluster.yml",
    ]
    for spec_path in cluster_specs:
        with open(spec_path, "r") as yml_file:
            benchmark_config = yaml.safe_load(yml_file)

        # YAML must not contain --cluster-mode (neither client nor preload)
        client_args = benchmark_config["clientconfig"].get("arguments", "")
        assert (
            "--cluster-mode" not in client_args
        ), f"--cluster-mode should not be in clientconfig.arguments of {spec_path}"
        preload_args = (
            benchmark_config.get("dbconfig", {})
            .get("preload_tool", {})
            .get("arguments", "")
        )
        assert (
            "--cluster-mode" not in preload_args
        ), f"--cluster-mode should not be in preload_tool.arguments of {spec_path}"

        # When oss_cluster_api_enabled=True, --cluster-mode IS added
        _, cmd_str = prepare_memtier_benchmark_parameters(
            benchmark_config["clientconfig"],
            "memtier_benchmark",
            6379,
            "localhost",
            "out.json",
            True,
        )
        assert "--cluster-mode" in cmd_str, (
            f"--cluster-mode must be injected when oss_cluster_api_enabled=True "
            f"for {spec_path}"
        )

        # When oss_cluster_api_enabled=False, --cluster-mode is NOT added
        _, cmd_str = prepare_memtier_benchmark_parameters(
            benchmark_config["clientconfig"],
            "memtier_benchmark",
            6379,
            "localhost",
            "out.json",
            False,
        )
        assert "--cluster-mode" not in cmd_str, (
            f"--cluster-mode must NOT be injected when oss_cluster_api_enabled=False "
            f"for {spec_path}"
        )


def test_standalone_mode_no_cluster_flag():
    """Standalone test suites must never get --cluster-mode injected."""
    from redis_benchmarks_specification.__self_contained_coordinator__.clients import (
        prepare_memtier_benchmark_parameters,
    )

    spec_path = (
        "./redis_benchmarks_specification/test-suites/"
        "memtier_benchmark-1Mkeys-100B-expire-use-case.yml"
    )
    with open(spec_path, "r") as yml_file:
        benchmark_config = yaml.safe_load(yml_file)

    _, cmd_str = prepare_memtier_benchmark_parameters(
        benchmark_config["clientconfig"],
        "memtier_benchmark",
        6379,
        "localhost",
        "out.json",
        False,
    )
    assert "--cluster-mode" not in cmd_str


def test_spin_docker_cluster_redis():
    """Integration test: spin up a 3-primary Redis cluster using Docker,
    verify cluster_state:ok, slot coverage, and CLUSTER MEET."""
    run_image = "redis:8.0"
    mnt_point = ""
    primary_port = 7379
    current_cpu_pos = 0
    ceil_db_cpu_limit = 3
    redis_configuration_parameters = None
    redis_containers = []
    redis_password = None
    temporary_dir = ""
    primary_count = 3
    docker_client = docker.from_env()

    try:
        (
            cluster_conns,
            current_cpu_pos,
        ) = spin_docker_cluster_redis(
            primary_count,
            ceil_db_cpu_limit,
            current_cpu_pos,
            docker_client,
            redis_configuration_parameters,
            redis_containers,
            primary_port,
            run_image,
            temporary_dir,
            mnt_point=mnt_point,
            redis_arguments="",
            password=redis_password,
        )

        # Should have 3 connections
        assert len(cluster_conns) == 3

        # All nodes should be reachable
        for conn in cluster_conns:
            conn.ping()

        # Check cluster_state:ok on all nodes
        for i, conn in enumerate(cluster_conns):
            cluster_info = conn.execute_command("CLUSTER", "INFO")
            if isinstance(cluster_info, bytes):
                cluster_info = cluster_info.decode()
            assert (
                "cluster_state:ok" in cluster_info
            ), f"Node {i} (port {primary_port + i}) not in cluster_state:ok"

        # Verify all 16384 slots are covered
        cluster_info = cluster_conns[0].execute_command("CLUSTER", "INFO")
        if isinstance(cluster_info, bytes):
            cluster_info = cluster_info.decode()
        assert "cluster_slots_ok:16384" in cluster_info

        # Verify each node knows about all 3 nodes
        nodes_raw = cluster_conns[0].execute_command("CLUSTER", "NODES")
        if isinstance(nodes_raw, bytes):
            nodes_raw = nodes_raw.decode()
        node_lines = [l for l in nodes_raw.strip().split("\n") if l.strip()]
        assert (
            len(node_lines) == 3
        ), f"Expected 3 nodes in CLUSTER NODES, got {len(node_lines)}"

        # Verify we can write a key using a cluster-aware client
        rc = redis.RedisCluster(
            host="localhost", port=primary_port, password=redis_password
        )
        rc.set("test_cluster_key", "value")
        assert rc.get("test_cluster_key") == b"value"
        rc.close()

        # Shutdown all nodes
        for conn in cluster_conns:
            try:
                conn.shutdown(nosave=True)
            except redis.exceptions.ConnectionError:
                pass
    except Exception:
        for c in redis_containers:
            try:
                logs = c.logs().decode("utf-8")
                logging.error(f"Container logs: {logs}")
            except Exception:
                pass
        raise
    finally:
        for c in redis_containers:
            try:
                c.stop()
                c.remove()
            except Exception:
                pass


def test_stop_and_remove_container_safe_not_found():
    """NotFound (container already gone) is treated as success."""

    class FakeContainer:
        id = "fake-id"
        image = "fake-image"

        def stop(self):
            raise docker.errors.NotFound("container gone")

        def remove(self):
            raise AssertionError("remove() should not be called after NotFound")

    # Must not raise.
    stop_and_remove_container_safe(FakeContainer(), "Test")


def test_stop_and_remove_container_safe_409_already_in_progress():
    """Regression test for v0.3.0 oss-cluster-3-primaries teardown race.

    When a container is started with ``auto_remove=True`` (which the
    cluster spin-up helpers do for primaries and replicas), Docker begins
    removing the container automatically when it stops. The explicit
    ``.remove()`` call that this coordinator makes a moment later hits a
    409 Conflict with body "removal of container X is already in
    progress". docker-py raises ``docker.errors.APIError`` rather than
    ``NotFound``, and previously this aborted the entire stream
    processing — the coordinator logged the traceback and silently
    ACK'd the stream as "filtered/skipped", so 0/N benchmark runs
    actually produced data.

    The helper must swallow that specific 409 path and let teardown
    complete.
    """

    class FakeAPIError(docker.errors.APIError):
        def __init__(self):
            super().__init__(
                "409 Client Error: Conflict",
                response=None,
                explanation=(
                    "removal of container abc123 is already in progress"
                ),
            )

        def __str__(self):
            return (
                "409 Client Error for http+docker://localhost/v1.47/"
                "containers/abc123?v=False&link=False&force=False: "
                'Conflict ("removal of container abc123 is already in '
                'progress")'
            )

    calls = {"stop": 0, "remove": 0}

    class FakeContainer:
        id = "abc123"
        image = "fake-image"

        def stop(self):
            calls["stop"] += 1

        def remove(self):
            calls["remove"] += 1
            raise FakeAPIError()

    # Must not raise.
    stop_and_remove_container_safe(FakeContainer(), "DB")
    assert calls["stop"] == 1
    assert calls["remove"] == 1


def test_stop_and_remove_container_safe_other_api_error_is_swallowed():
    """Unexpected APIError is logged but not raised — teardown is best-effort."""

    class FakeAPIError(docker.errors.APIError):
        def __init__(self):
            super().__init__(
                "500 Internal Server Error",
                response=None,
                explanation="boom",
            )

        def __str__(self):
            return "500 Internal Server Error: boom"

    class FakeContainer:
        id = "zzz"
        image = "fake-image"

        def stop(self):
            pass

        def remove(self):
            raise FakeAPIError()

    # Must not raise — teardown must never abort the stream.
    stop_and_remove_container_safe(FakeContainer(), "Client")
