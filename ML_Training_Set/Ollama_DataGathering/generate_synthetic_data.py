#!/usr/bin/env python3
"""
generate_synthetic_data.py

Generates realistic synthetic GitHub Actions log snippets for underrepresented
training classes: infrastructure and flaky_test.

Output: data/final/synthetic_training_data.jsonl
        (append this to github_actions_training_dataset.jsonl before training)

Usage:
    python generate_synthetic_data.py
"""

import json
import random
import uuid
from pathlib import Path
from datetime import datetime, timedelta

OUT_PATH = Path("data/final/synthetic_training_data.jsonl")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

random.seed(42)

# ── How many to generate ──────────────────────────────────────────────────────
N_INFRASTRUCTURE = 1500
N_FLAKY_TEST     = 1000

# ── Fake metadata pools ───────────────────────────────────────────────────────
JAVA_REPOS = [
    "apache/kafka", "apache/camel", "apache/zookeeper", "elastic/elasticsearch",
    "quarkusio/quarkus", "hazelcast/hazelcast", "redisson/redisson",
    "fabric8io/kubernetes-client", "testcontainers/testcontainers-java",
    "docker-java/docker-java", "apache/flink", "spring-projects/spring-framework",
    "hibernate/hibernate-orm", "netty/netty", "junit-team/junit5",
]

PYTHON_REPOS = [
    "docker/compose", "ansible/ansible", "apache/airflow", "mlflow/mlflow",
    "pre-commit/pre-commit", "astral-sh/ruff", "adrienverge/yamllint",
    "pypa/pipenv", "kubernetes-client/python", "celery/celery",
    "aio-libs/aiohttp", "scrapy/scrapy", "django/django",
]

WORKFLOW_NAMES_JAVA  = ["Java CI", "CI", "Build and Test", "Maven CI", "Gradle CI"]
WORKFLOW_NAMES_PYTHON = ["CI", "Tests", "Python CI", "Tox", "pytest"]

JAVA_FAILING_STEPS   = ["Build with Maven", "Build with Gradle", "Test", "Run tests"]
PYTHON_FAILING_STEPS = ["Run tests", "pytest", "tox", "Test", "Run tox"]

def fake_date() -> str:
    base = datetime(2025, 1, 1)
    delta = timedelta(days=random.randint(0, 450))
    return (base + delta).strftime("%Y-%m-%dT%H:%M:%SZ")

def fake_run_id() -> int:
    return random.randint(18_000_000_000, 26_000_000_000)

def fake_job_id() -> int:
    return random.randint(50_000_000_000, 80_000_000_000)

def fake_source(repo: str, run_id: int, job_id: int) -> str:
    safe = repo.replace("/", "__")
    return f"data\\raw_logs\\{safe}\\run_{run_id}_job_{job_id}.log"

def make_row(text: str, label: str, primary_label: str,
             repo: str, lang: str, workflow: str,
             job_name: str, failing_step: str) -> dict:
    run_id = fake_run_id()
    job_id = fake_job_id()
    return {
        "text":          text,
        "label":         label,
        "confidence":    0.95,
        "label_source":  "synthetic",
        "reason":        f"synthetic:{primary_label}",
        "primary_label": primary_label,
        "repo":          repo,
        "lang":          lang,
        "run_id":        run_id,
        "run_number":    random.randint(100, 3000),
        "job_id":        job_id,
        "workflow_name": workflow,
        "job_name":      job_name,
        "failing_step":  failing_step,
        "created_at":    fake_date(),
        "source_file":   fake_source(repo, run_id, job_id),
        "rule_label":    label,
        "rule_confidence": 0.95,
        "rule_reason":   f"synthetic:{primary_label}",
        "ollama_label":  None,
        "ollama_confidence": None,
        "ollama_reason": None,
        "ollama_model":  None,
    }


# ════════════════════════════════════════════════════════════════════════════
#  INFRASTRUCTURE TEMPLATES
# ════════════════════════════════════════════════════════════════════════════

def _infra_runner_shutdown(lang):
    repo = random.choice(JAVA_REPOS if lang == "java" else PYTHON_REPOS)
    wf   = random.choice(WORKFLOW_NAMES_JAVA if lang == "java" else WORKFLOW_NAMES_PYTHON)
    step = random.choice(JAVA_FAILING_STEPS if lang == "java" else PYTHON_FAILING_STEPS)
    text = f"""Run {step}
shell: /usr/bin/bash -e {{0}}

{"[INFO] Running tests..." if lang == "java" else "collecting ..."}
{"[INFO] Tests run: 47, Failures: 0, Errors: 0" if lang == "java" else "collected 142 items"}

##[error]The runner has received a shutdown signal. This can happen when the runner service is stopped, or a manually started runner is canceled.
##[error]Process completed with exit code 1.
Error: Process completed with exit code 1."""
    return make_row(text, "infrastructure", "infra.runner_lost",
                    repo, lang, wf, step, step)

def _infra_oom_exit137(lang):
    repo = random.choice(JAVA_REPOS if lang == "java" else PYTHON_REPOS)
    wf   = random.choice(WORKFLOW_NAMES_JAVA if lang == "java" else WORKFLOW_NAMES_PYTHON)
    step = random.choice(JAVA_FAILING_STEPS if lang == "java" else PYTHON_FAILING_STEPS)
    heap = random.choice(["Java heap space", "GC overhead limit exceeded",
                          "unable to create new native thread"])
    text = f"""Run {step}
shell: /usr/bin/bash -e {{0}}

{"[INFO] Running org.example.LargeIntegrationTest" if lang == "java" else "running large test suite..."}
{"java.lang.OutOfMemoryError: " + heap if lang == "java" else ""}
{"	at java.util.Arrays.copyOf(Arrays.java:3210)" if lang == "java" else "Killed"}
{"	at java.util.ArrayList.grow(ArrayList.java:265)" if lang == "java" else ""}

##[error]Process completed with exit code 137.
Error: Process completed with exit code 137."""
    return make_row(text, "infrastructure", "infra.resources.oom",
                    repo, lang, wf, step, step)

def _infra_disk_full(lang):
    repo = random.choice(JAVA_REPOS if lang == "java" else PYTHON_REPOS)
    wf   = random.choice(WORKFLOW_NAMES_JAVA if lang == "java" else WORKFLOW_NAMES_PYTHON)
    step = random.choice(JAVA_FAILING_STEPS if lang == "java" else PYTHON_FAILING_STEPS)
    text = f"""Run {step}
shell: /usr/bin/bash -e {{0}}

{"[INFO] Compiling source files..." if lang == "java" else "installing dependencies..."}
{"[ERROR] BUILD FAILURE" if lang == "java" else "ERROR: Could not install packages due to an OSError"}
{"[ERROR] Failed to execute goal org.apache.maven.plugins:maven-compiler-plugin:3.11.0:compile" if lang == "java" else ""}
IOError: [Errno 28] No space left on device
OSError: [Errno 28] No space left on device: '/home/runner/work/{repo.split("/")[1]}'

##[error]Process completed with exit code 1.
Error: Process completed with exit code 1."""
    return make_row(text, "infrastructure", "infra.resources.disk_full",
                    repo, lang, wf, step, step)

def _infra_docker_rate_limit(lang):
    repo = random.choice(JAVA_REPOS if lang == "java" else PYTHON_REPOS)
    wf   = random.choice(WORKFLOW_NAMES_JAVA if lang == "java" else WORKFLOW_NAMES_PYTHON)
    step = "Set up Docker"
    image = random.choice(["ubuntu:22.04", "python:3.11-slim", "openjdk:17-slim",
                            "postgres:15", "redis:7-alpine", "nginx:latest"])
    text = f"""Run docker pull {image}
shell: /usr/bin/bash -e {{0}}

{image}: Pulling from library/{image.split(":")[0]}
toomanyrequests: You have reached your pull rate limit. You may increase the limit by authenticating and upgrading: https://www.docker.com/increase-rate-limit

##[error]Process completed with exit code 1.
Error: Process completed with exit code 1."""
    return make_row(text, "infrastructure", "infra.container.registry_server_error",
                    repo, lang, wf, step, step)

def _infra_docker_manifest_unknown(lang):
    repo = random.choice(JAVA_REPOS if lang == "java" else PYTHON_REPOS)
    wf   = random.choice(WORKFLOW_NAMES_JAVA if lang == "java" else WORKFLOW_NAMES_PYTHON)
    step = "Build Docker image"
    tag  = f"v{random.randint(1,9)}.{random.randint(0,20)}.{random.randint(0,10)}"
    text = f"""Run docker build .
shell: /usr/bin/bash -e {{0}}

#0 building with "default" instance using docker driver
#1 [internal] load build definition from Dockerfile
#1 DONE 0.1s
#2 [internal] load metadata for gcr.io/distroless/java17:latest
#2 ERROR: failed to solve: gcr.io/distroless/java17:{tag}: failed to resolve source metadata for gcr.io/distroless/java17:{tag}: unexpected status code 503 Service Unavailable

##[error]Process completed with exit code 1.
Error: Process completed with exit code 1."""
    return make_row(text, "infrastructure", "infra.container.registry_server_error",
                    repo, lang, wf, step, step)

def _infra_network_dns(lang):
    repo = random.choice(JAVA_REPOS if lang == "java" else PYTHON_REPOS)
    wf   = random.choice(WORKFLOW_NAMES_JAVA if lang == "java" else WORKFLOW_NAMES_PYTHON)
    step = random.choice(["Install dependencies", "Set up build", "Download dependencies"])
    host = random.choice(["repo1.maven.org", "plugins.gradle.org", "pypi.org",
                          "registry.npmjs.org", "packages.confluent.io"])
    text = f"""Run {step}
shell: /usr/bin/bash -e {{0}}

{"Downloading from central: https://" + host + "/maven2/org/apache/..." if lang == "java" else "Collecting requests"}
{"[WARNING] Could not transfer metadata org.apache:apache:pom:30 from/to central (https://" + host + "): Transfer failed for https://" + host + "/maven2/org/apache/apache/30/apache-30.pom: " if lang == "java" else "WARNING: Retrying (Retry(total=4)) after connection broken by 'NewConnectionError":}
{"Could not resolve host: " + host if lang == "java" else "Could not resolve host: " + host}
{"Temporary failure in name resolution" if lang == "java" else "socket.gaierror: [Errno -2] Name or service not known"}
{"[ERROR] BUILD FAILURE" if lang == "java" else ""}

##[error]Process completed with exit code 1."""
    return make_row(text, "infrastructure", "infra.network.dns",
                    repo, lang, wf, step, step)

def _infra_network_timeout(lang):
    repo = random.choice(JAVA_REPOS if lang == "java" else PYTHON_REPOS)
    wf   = random.choice(WORKFLOW_NAMES_JAVA if lang == "java" else WORKFLOW_NAMES_PYTHON)
    step = random.choice(["Install dependencies", "Download artifacts", "Resolve dependencies"])
    host = random.choice(["repo1.maven.org", "jcenter.bintray.com", "pypi.org",
                          "files.pythonhosted.org", "registry.npmjs.org"])
    text = f"""Run {step}
shell: /usr/bin/bash -e {{0}}

{"[INFO] Downloading from central: https://" + host + "/maven2/..." if lang == "java" else "Collecting " + random.choice(["numpy", "pandas", "requests", "boto3", "pydantic"])}
{"[WARNING] Retrying (1/3): https://" + host + "/..." if lang == "java" else "WARNING: Retrying (Retry(total=4, connect=None, read=None, redirect=None, status=None))"}
{"after connection broken by 'ReadTimeoutError" if lang == "java" else "after connection broken by 'ReadTimeoutError(\"HTTPSConnectionPool(host='" + host + "', port=443): Read timed out. (read timeout=15)\")'"}
{"Read timed out (read timeout=60)" if lang == "java" else ""}
{"[ERROR] Failed to execute goal: Could not resolve dependencies" if lang == "java" else "ERROR: Could not install packages due to an OSError: HTTPSConnectionPool"}

##[error]Process completed with exit code 1."""
    return make_row(text, "infrastructure", "infra.network.registry",
                    repo, lang, wf, step, step)

def _infra_network_502(lang):
    repo = random.choice(JAVA_REPOS if lang == "java" else PYTHON_REPOS)
    wf   = random.choice(WORKFLOW_NAMES_JAVA if lang == "java" else WORKFLOW_NAMES_PYTHON)
    step = random.choice(["Set up Java", "Set up Python", "Set up Node.js"])
    text = f"""Run actions/setup-{'java@v4' if lang == "java" else 'python@v5'}

Unexpected HTTP response: 502
Error: HTTPError: Unexpected HTTP response: 502
    at <url:api.github.com>
waiting 10 seconds before trying again
Unexpected HTTP response: 502
waiting 20 seconds before trying again
Unexpected HTTP response: 502

##[error]Unhandled error: Error: Unexpected HTTP response: 502
##[error]Process completed with exit code 1."""
    return make_row(text, "infrastructure", "infra.network.service_5xx",
                    repo, lang, wf, step, step)

def _infra_runner_lost_comms(lang):
    repo = random.choice(JAVA_REPOS if lang == "java" else PYTHON_REPOS)
    wf   = random.choice(WORKFLOW_NAMES_JAVA if lang == "java" else WORKFLOW_NAMES_PYTHON)
    step = random.choice(JAVA_FAILING_STEPS if lang == "java" else PYTHON_FAILING_STEPS)
    text = f"""Run {step}
shell: /usr/bin/bash -e {{0}}

{"[INFO] Running tests..." if lang == "java" else "running pytest..."}
{"[INFO] Tests run: 23, Failures: 0, Errors: 0" if lang == "java" else "passed 45 items"}

The runner has received a shutdown signal. This can happen when the runner service is stopped, or a manually started runner is canceled.
The self-hosted runner: runner-abc123 lost communication with the server.

##[error]Process completed with exit code 1."""
    return make_row(text, "infrastructure", "infra.runner_lost",
                    repo, lang, wf, step, step)

def _infra_tls_error(lang):
    repo = random.choice(JAVA_REPOS if lang == "java" else PYTHON_REPOS)
    wf   = random.choice(WORKFLOW_NAMES_JAVA if lang == "java" else WORKFLOW_NAMES_PYTHON)
    step = "Install dependencies"
    host = random.choice(["repo1.maven.org", "pypi.org", "registry.npmjs.org"])
    text = f"""Run {step}
shell: /usr/bin/bash -e {{0}}

{"[WARNING] Could not transfer artifact from/to central (https://" + host + "):" if lang == "java" else "WARNING: pip is configured with locations that require TLS/SSL"}
{"javax.net.ssl.SSLHandshakeException: PKIX path building failed" if lang == "java" else "ssl.SSLError: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate (_ssl.c:1006)"}
{"	sun.security.validator.ValidatorException: PKIX path building failed: sun.security.provider.certpath.SunCertPathBuilderException: unable to find valid certification path to requested target" if lang == "java" else ""}
{"[ERROR] BUILD FAILURE" if lang == "java" else "ERROR: Could not install packages due to an OSError"}

##[error]Process completed with exit code 1."""
    return make_row(text, "infrastructure", "infra.network.tls_ssl",
                    repo, lang, wf, step, step)

def _infra_cache_service_error(lang):
    repo = random.choice(JAVA_REPOS if lang == "java" else PYTHON_REPOS)
    wf   = random.choice(WORKFLOW_NAMES_JAVA if lang == "java" else WORKFLOW_NAMES_PYTHON)
    step = "Cache dependencies"
    text = f"""Run actions/cache@v4

Cache not found for input keys: {random.choice(["Linux-maven-abc123", "pip-linux-python3.11", "gradle-linux-"])}

##[warning]Cache service responded with 503
##[error]Failed to restore cache: Cache service responded with 503 Service Unavailable
##[error]Process completed with exit code 1.
Error: Process completed with exit code 1."""
    return make_row(text, "infrastructure", "infra.network.service_5xx",
                    repo, lang, wf, step, step)

def _infra_docker_daemon_not_running(lang):
    repo = random.choice(JAVA_REPOS if lang == "java" else PYTHON_REPOS)
    wf   = random.choice(WORKFLOW_NAMES_JAVA if lang == "java" else WORKFLOW_NAMES_PYTHON)
    step = "Build and push Docker image"
    text = f"""Run docker build -t myapp:latest .
shell: /usr/bin/bash -e {{0}}

Cannot connect to the Docker daemon at unix:///var/run/docker.sock. Is the docker daemon running?
ERROR: Cannot connect to the Docker daemon at unix:///var/run/docker.sock.
error during connect: this error may indicate that the docker daemon is not running:
Get "http://%2Fvar%2Frun%2Fdocker.sock/v1.24/info": dial unix /var/run/docker.sock: connect: connection refused

##[error]Process completed with exit code 1."""
    return make_row(text, "infrastructure", "infra.container.runtime",
                    repo, lang, wf, step, step)

def _infra_apt_network(lang):
    repo = random.choice(PYTHON_REPOS)
    wf   = random.choice(WORKFLOW_NAMES_PYTHON)
    step = "Install system dependencies"
    text = f"""Run sudo apt-get update && sudo apt-get install -y libpq-dev
shell: /usr/bin/bash -e {{0}}

Hit:1 https://packages.microsoft.com/ubuntu/22.04/prod jammy InRelease
Get:2 https://archive.ubuntu.com/ubuntu jammy InRelease
Err:2 https://archive.ubuntu.com/ubuntu jammy InRelease
  Connection timed out [IP: 91.189.91.81 80]
E: Failed to fetch https://archive.ubuntu.com/ubuntu/dists/jammy/InRelease  Connection timed out [IP: 91.189.91.81 80]
E: Some index files failed to download. They have been ignored, or old ones used instead.

##[error]Process completed with exit code 100.
Error: Process completed with exit code 100."""
    return make_row(text, "infrastructure", "infra.network.registry",
                    repo, lang, wf, step, step)

def _infra_connection_reset(lang):
    repo = random.choice(JAVA_REPOS if lang == "java" else PYTHON_REPOS)
    wf   = random.choice(WORKFLOW_NAMES_JAVA if lang == "java" else WORKFLOW_NAMES_PYTHON)
    step = "Install dependencies"
    text = f"""Run {step}
shell: /usr/bin/bash -e {{0}}

{"[WARNING] Could not transfer artifact:" if lang == "java" else "WARNING: Retrying (Retry(total=3))"}
{"org.eclipse.aether.transfer.TransferCancelledException: Connection reset" if lang == "java" else "after connection broken by 'ConnectionResetError(104, 'Connection reset by peer')'"}
{"	at org.apache.maven.wagon.providers.http.AbstractHttpClientWagon.fillInputData" if lang == "java" else "  raise ProxyError(e, request=request)"}
{"Connection reset by peer" if lang == "java" else "requests.exceptions.ConnectionError: ('Connection aborted.', ConnectionResetError(104, 'Connection reset by peer'))"}
{"[ERROR] BUILD FAILURE" if lang == "java" else "ERROR: Could not install packages"}

##[error]Process completed with exit code 1."""
    return make_row(text, "infrastructure", "infra.network.socket",
                    repo, lang, wf, step, step)

def _infra_pod_oom(lang):
    repo = random.choice(JAVA_REPOS)
    wf   = "Kubernetes Integration Tests"
    step = "Run integration tests"
    text = """Run kubectl apply -f k8s/test-job.yaml && kubectl wait --for=condition=complete job/test-job
shell: /usr/bin/bash -e {0}

job.batch/test-job created
Waiting for job/test-job...
Error from server: pods "test-job-xyz123" is forbidden: [maximum memory usage per Container is 512Mi, but limit is 1Gi]
pod/test-job-xyz123 failed
OOMKilled
Container test-runner in pod test-job-xyz123 exceeded its memory limit.
The node had condition: [MemoryPressure]

##[error]Process completed with exit code 1."""
    return make_row(text, "infrastructure", "infra.resources.oom",
                    repo, "java", wf, step, step)


# ════════════════════════════════════════════════════════════════════════════
#  FLAKY TEST TEMPLATES
# ════════════════════════════════════════════════════════════════════════════

def _flaky_test_timeout_java():
    repo = random.choice(JAVA_REPOS)
    wf   = random.choice(WORKFLOW_NAMES_JAVA)
    step = random.choice(JAVA_FAILING_STEPS)
    pkg  = random.choice([
        "org.apache.kafka.streams.integration.StreamsUpgradeTest",
        "org.elasticsearch.index.replication.ESIndexLevelReplicationTestCase",
        "com.hazelcast.map.impl.operation.MapOperationTest",
        "org.apache.camel.component.http.HttpProducerTest",
        "io.quarkus.it.panache.reactive.PanacheTestResource",
    ])
    secs = random.randint(30, 300)
    text = f"""[INFO] Running {pkg}
[ERROR] Tests run: 1, Failures: 0, Errors: 1, Skipped: 0, Time elapsed: {secs}.{random.randint(100,999)} s <<< ERROR! - in {pkg}
[ERROR] {pkg.split(".")[-1]}  Time elapsed: {secs}.{random.randint(100,999)} s  <<< ERROR!

org.junit.runners.model.TestTimedOutException: test timed out after {secs} seconds
\tat sun.misc.Unsafe.park(Native Method)
\tat java.util.concurrent.locks.LockSupport.parkNanos(LockSupport.java:215)
\tat java.util.concurrent.locks.AbstractQueuedSynchronizer$ConditionObject.awaitNanos(AbstractQueuedSynchronizer.java:2078)
\tat java.util.concurrent.LinkedBlockingQueue.poll(LinkedBlockingQueue.java:467)

[INFO] Results:
[INFO]
[ERROR] Errors:
[ERROR]   {pkg.split(".")[-1]}.{random.choice(["testConcurrentAccess", "testAsyncOperation", "testEventDriven", "testNetworkCall"])}:
org.junit.runners.model.TestTimedOutException: test timed out after {secs} seconds
[INFO]
[ERROR] Tests run: 1, Failures: 0, Errors: 1, Skipped: 0
[INFO]
[INFO] BUILD FAILURE
[ERROR] Failed to execute goal org.apache.maven.plugins:maven-surefire-plugin:3.1.2:test"""
    return make_row(text, "flaky_test", "flaky.time_or_async",
                    repo, "java", wf, step, step)

def _flaky_test_timeout_python():
    repo = random.choice(PYTHON_REPOS)
    wf   = random.choice(WORKFLOW_NAMES_PYTHON)
    step = random.choice(PYTHON_FAILING_STEPS)
    test = random.choice([
        "tests/test_client.py::TestAsyncClient::test_concurrent_requests",
        "tests/integration/test_broker.py::TestBrokerConnection::test_reconnect",
        "tests/test_worker.py::TestWorker::test_task_timeout",
        "tests/test_http.py::TestHTTPClient::test_streaming_response",
    ])
    secs = random.randint(10, 120)
    text = f"""collecting ... collected {random.randint(200, 800)} items

{test} FAILED                                          [{random.randint(10,90)}%]

================================ FAILURES ================================
_________________ {test.split("::")[-1]} _________________

    @pytest.mark.timeout({secs})
    async def {test.split("::")[-1]}(self):
>       result = await asyncio.wait_for(client.fetch(), timeout={secs})

E   pytest.fail.Exception: Timeout >={secs}s
E   asyncio.exceptions.TimeoutError

tests/test_client.py:{random.randint(50,300)}: asyncio.exceptions.TimeoutError
=========================== short test summary info ===========================
FAILED {test} - asyncio.exceptions.TimeoutError
============================== 1 failed in {secs}.{random.randint(10,99)}s =============================="""
    return make_row(text, "flaky_test", "flaky.time_or_async",
                    repo, "python", wf, step, step)

def _flaky_concurrency_java():
    repo = random.choice(JAVA_REPOS)
    wf   = random.choice(WORKFLOW_NAMES_JAVA)
    step = random.choice(JAVA_FAILING_STEPS)
    cls  = random.choice([
        "ConcurrentHashMapTest", "BlockingQueueTest",
        "ExecutorServiceTest", "ThreadPoolTest", "AsyncHandlerTest",
    ])
    text = f"""[INFO] Running org.example.concurrent.{cls}
[ERROR] Tests run: 1, Failures: 0, Errors: 1, Skipped: 0 <<< ERROR!

java.util.ConcurrentModificationException
\tat java.util.ArrayList$Itr.checkForComodification(ArrayList.java:911)
\tat java.util.ArrayList$Itr.next(ArrayList.java:861)
\tat org.example.concurrent.{cls}.{random.choice(["testConcurrentRead", "testParallelWrite", "testSharedState"])}({cls}.java:{random.randint(50,200)})

[ERROR] Tests run: 1, Failures: 0, Errors: 1, Skipped: 0
[INFO] BUILD FAILURE
[ERROR] Failed to execute goal org.apache.maven.plugins:maven-surefire-plugin:3.1.2:test"""
    return make_row(text, "flaky_test", "flaky.order_or_race",
                    repo, "java", wf, step, step)

def _flaky_data_race_go():
    repo = random.choice(["apache/kafka", "elastic/elasticsearch"])
    wf   = "Go CI"
    step = "Run tests"
    func = random.choice(["TestConcurrentWrite", "TestParallelRead", "TestSharedCounter"])
    text = f"""$ go test -race ./...
==================
WARNING: DATA RACE
Write at 0x00c000124000 by goroutine 7:
  main.{func}.func1()
      /home/runner/work/project/main_test.go:{random.randint(50,200)} +0x44

Previous read at 0x00c000124000 by goroutine 6:
  main.{func}()
      /home/runner/work/project/main_test.go:{random.randint(20,49)} +0x88

Goroutine 7 (running) created at:
  main.{func}()
      /home/runner/work/project/main_test.go:{random.randint(30,60)} +0x66
==================
Found 1 data race(s)
FAIL\tmain\t0.{random.randint(100,999)}s

##[error]Process completed with exit code 1."""
    return make_row(text, "flaky_test", "flaky.order_or_race",
                    repo, "java", wf, step, step)

def _flaky_rerun_marker_java():
    repo = random.choice(JAVA_REPOS)
    wf   = random.choice(WORKFLOW_NAMES_JAVA)
    step = random.choice(JAVA_FAILING_STEPS)
    pkg  = random.choice([
        "org.apache.kafka.streams.KafkaStreamsTest",
        "com.hazelcast.cache.CacheExpiryTest",
        "org.elasticsearch.cluster.ClusterStateIT",
    ])
    text = f"""[INFO] Running {pkg}
[WARNING] Tests run: 1, Failures: 1, Errors: 0, Skipped: 0, Flakes: 1, Time elapsed: {random.randint(5,60)}.{random.randint(100,999)} s

[WARNING] Flakes:
[WARNING]   {pkg.split(".")[-1]}.testEventualConsistency  Run {random.randint(1,3)}: FAILED, Run {random.randint(4,5)}: PASSED

[INFO] Tests run: 1, Failures: 0, Errors: 0, Skipped: 0, Flakes: 1

[INFO] BUILD SUCCESS"""
    return make_row(text, "flaky_test", "flaky.suspected_retry_sensitive",
                    repo, "java", wf, step, step)

def _flaky_rerun_marker_python():
    repo = random.choice(PYTHON_REPOS)
    wf   = random.choice(WORKFLOW_NAMES_PYTHON)
    step = random.choice(PYTHON_FAILING_STEPS)
    test = random.choice([
        "tests/test_tasks.py::TestAsyncTask::test_retry_on_failure",
        "tests/test_broker.py::TestBroker::test_connection_pool",
        "tests/integration/test_api.py::TestAPI::test_concurrent_requests",
    ])
    text = f"""collecting ... collected {random.randint(100, 500)} items

RERUN {test}
RERUN {test}
FAILED {test}

================================ FAILURES ================================
Flaky test detected. This test passed {random.randint(1,4)} times and failed {random.randint(1,3)} times.

{test.split("::")[-1]} - AssertionError: Expected response within {random.randint(1,5)}s but got timeout
=========================== short test summary info ===========================
FAILED {test}
======================== 1 failed, {random.randint(50, 200)} passed in {random.randint(30,180)}.{random.randint(10,99)}s ========================="""
    return make_row(text, "flaky_test", "flaky.suspected_retry_sensitive",
                    repo, "python", wf, step, step)

def _flaky_selenium_stale():
    repo = random.choice(PYTHON_REPOS)
    wf   = "E2E Tests"
    step = "Run Selenium tests"
    text = f"""collecting ... collected {random.randint(20, 80)} items

FAILED tests/e2e/test_ui.py::TestUI::test_form_submission

================================ FAILURES ================================
_________________ TestUI.test_form_submission _________________

    def test_form_submission(self):
        driver.get("https://example.com/form")
>       submit_button = driver.find_element(By.ID, "submit")
>       submit_button.click()

selenium.common.exceptions.StaleElementReferenceException: Message: stale element reference: element is not attached to the page document
  (Session info: chrome={random.randint(110,120)}.0.{random.randint(5000,6000)}.0)

tests/e2e/test_ui.py:{random.randint(30,100)}: StaleElementReferenceException
=========================== short test summary info ===========================
FAILED tests/e2e/test_ui.py::TestUI::test_form_submission - selenium.common.exceptions.StaleElementReferenceException
======================== 1 failed, {random.randint(10,40)} passed in {random.randint(5,30)}.{random.randint(10,99)}s ========================="""
    return make_row(text, "flaky_test", "flaky.time_or_async",
                    repo, "python", wf, step, step)

def _flaky_deadlock_java():
    repo = random.choice(JAVA_REPOS)
    wf   = random.choice(WORKFLOW_NAMES_JAVA)
    step = random.choice(JAVA_FAILING_STEPS)
    text = f"""[INFO] Running org.example.db.TransactionTest
[ERROR] Tests run: 1, Failures: 0, Errors: 1, Skipped: 0 <<< ERROR!

com.mysql.cj.jdbc.exceptions.MySQLTransactionRollbackException: Deadlock found when trying to get lock; try restarting transaction
\tat com.mysql.cj.jdbc.exceptions.SQLError.createSQLException(SQLError.java:123)
\tat org.example.db.TransactionTest.testConcurrentUpdates(TransactionTest.java:{random.randint(50,200)})

[ERROR] Tests run: 1, Failures: 0, Errors: 1, Skipped: 0
[INFO] BUILD FAILURE"""
    return make_row(text, "flaky_test", "flaky.order_or_race",
                    repo, "java", wf, step, step)

def _flaky_async_python():
    repo = random.choice(PYTHON_REPOS)
    wf   = random.choice(WORKFLOW_NAMES_PYTHON)
    step = random.choice(PYTHON_FAILING_STEPS)
    text = f"""collecting ... collected {random.randint(50, 300)} items

FAILED tests/test_async.py::TestAsync::test_concurrent_handler

================================ FAILURES ================================
_________________ TestAsync.test_concurrent_handler _________________

RuntimeWarning: coroutine 'AsyncHandler.process' was never awaited
  RuntimeWarning: Enable tracemalloc to get the object allocation traceback

E   asyncio.exceptions.CancelledError

tests/test_async.py:{random.randint(40,150)}: CancelledError

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ===========================
FAILED tests/test_async.py::TestAsync::test_concurrent_handler
======================== 1 failed, {random.randint(30,100)} passed in {random.randint(5,30)}.{random.randint(10,99)}s ========================="""
    return make_row(text, "flaky_test", "flaky.time_or_async",
                    repo, "python", wf, step, step)

def _flaky_socket_timeout_java():
    repo = random.choice(JAVA_REPOS)
    wf   = random.choice(WORKFLOW_NAMES_JAVA)
    step = random.choice(JAVA_FAILING_STEPS)
    text = f"""[INFO] Running org.example.integration.NetworkIntegrationTest
[ERROR] Tests run: 1, Failures: 0, Errors: 1, Skipped: 0 <<< ERROR!

java.net.SocketTimeoutException: Read timed out
\tat java.net.SocketInputStream.socketRead0(Native Method)
\tat java.net.SocketInputStream.socketRead(SocketInputStream.java:116)
\tat org.example.integration.NetworkIntegrationTest.testRemoteCall(NetworkIntegrationTest.java:{random.randint(50,200)})

[ERROR] Tests run: 1, Failures: 0, Errors: 1, Skipped: 0
[INFO]
[INFO] BUILD FAILURE
[ERROR] Failed to execute goal org.apache.maven.plugins:maven-surefire-plugin:3.1.2:test"""
    return make_row(text, "flaky_test", "flaky.time_or_async",
                    repo, "java", wf, step, step)

def _flaky_jest_open_handles():
    repo = random.choice(PYTHON_REPOS)
    wf   = "Node CI"
    step = "Run Jest tests"
    text = f"""PASS src/__tests__/api.test.js
FAIL src/__tests__/server.test.js
  ● server › should handle concurrent requests

    expect(received).toBe(expected)
    Expected: 200
    Received: 503

      {random.randint(10,50)} |   const response = await request(app).get('/api/data');
    > {random.randint(51,100)} |   expect(response.status).toBe(200);
         |                           ^

Jest did not exit one second after the test run has completed.
'This usually means that there are asynchronous operations that weren''t stopped in your tests.'
Jest has detected the following 2 open handles potentially keeping Jest from exiting:

  ●  TCPSERVEROWRAP

      {random.randint(5,20)} | const server = app.listen(3000);

Tests: 1 failed, {random.randint(10,50)} passed, {random.randint(51,100)} total"""
    return make_row(text, "flaky_test", "flaky.time_or_async",
                    repo, "python", wf, step, step)


# ── Generator dispatcher ──────────────────────────────────────────────────────

INFRA_GENERATORS = [
    (_infra_runner_shutdown,      0.12),
    (_infra_oom_exit137,          0.12),
    (_infra_disk_full,            0.07),
    (_infra_docker_rate_limit,    0.08),
    (_infra_docker_manifest_unknown, 0.07),
    (_infra_network_dns,          0.10),
    (_infra_network_timeout,      0.10),
    (_infra_network_502,          0.08),
    (_infra_runner_lost_comms,    0.08),
    (_infra_tls_error,            0.06),
    (_infra_cache_service_error,  0.05),
    (_infra_docker_daemon_not_running, 0.04),
    (_infra_apt_network,          0.04),
    (_infra_connection_reset,     0.05),
    (_infra_pod_oom,              0.04),  # Java only
]

FLAKY_GENERATORS = [
    (_flaky_test_timeout_java,    0.15),
    (_flaky_test_timeout_python,  0.15),
    (_flaky_concurrency_java,     0.12),
    (_flaky_data_race_go,         0.08),
    (_flaky_rerun_marker_java,    0.10),
    (_flaky_rerun_marker_python,  0.10),
    (_flaky_selenium_stale,       0.08),
    (_flaky_deadlock_java,        0.08),
    (_flaky_async_python,         0.07),
    (_flaky_socket_timeout_java,  0.05),
    (_flaky_jest_open_handles,    0.02),
]


def weighted_choice(generators):
    funcs, weights = zip(*generators)
    return random.choices(funcs, weights=weights, k=1)[0]


def main():
    rows = []

    print(f"Generating {N_INFRASTRUCTURE} infrastructure rows...")
    for _ in range(N_INFRASTRUCTURE):
        gen  = weighted_choice(INFRA_GENERATORS)
        lang = random.choice(["java", "python"])
        try:
            row = gen(lang)
        except TypeError:
            row = gen()  # generators that don't take lang
        rows.append(row)

    print(f"Generating {N_FLAKY_TEST} flaky_test rows...")
    for _ in range(N_FLAKY_TEST):
        gen = weighted_choice(FLAKY_GENERATORS)
        try:
            row = gen()
        except TypeError:
            row = gen("java")
        rows.append(row)

    random.shuffle(rows)

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    infra_count = sum(1 for r in rows if r["label"] == "infrastructure")
    flaky_count = sum(1 for r in rows if r["label"] == "flaky_test")

    print(f"\nDone.")
    print(f"  infrastructure : {infra_count}")
    print(f"  flaky_test     : {flaky_count}")
    print(f"  Total          : {len(rows)}")
    print(f"  Saved to       : {OUT_PATH}")
    print(f"\nTo add to your dataset:")
    print(f"  type {OUT_PATH} >> data\\final\\github_actions_training_dataset.jsonl")


if __name__ == "__main__":
    main()
