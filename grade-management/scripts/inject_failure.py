#!/usr/bin/env python3
"""
Failure Injection Script
========================
Injects one of five failure scenarios into the Student Grade Management
pipeline for self-healing mechanism evaluation.

Usage:
    python inject_failure.py <failure_type>

Failure types:
    compilation     - Introduces a syntax error into StudentController.java
    test            - Introduces a failing assertion into StudentControllerTest.java
    flaky           - Introduces stateful marker-file flakiness into StudentControllerTest.java
    configuration   - Corrupts application.properties with an invalid value
    infrastructure  - Introduces a non-existent dependency into pom.xml

To restore the original state after injection, use:
    python inject_failure.py restore
"""

import sys
import os
import re
import shutil

# ─── File paths ───────────────────────────────────────────────────────────────

CONTROLLER      = "src/main/java/com/university/grades/controller/StudentController.java"
CONTROLLER_TEST = "src/test/java/com/university/grades/controller/StudentControllerTest.java"
SERVICE_TEST    = "src/test/java/com/university/grades/service/StudentServiceTest.java"
APP_PROPS       = "src/main/resources/application.properties"
POM_XML         = "pom.xml"

BACKUP_SUFFIX = ".backup"

# ─── Utilities ────────────────────────────────────────────────────────────────

def backup(path):
    """Create a backup of the original file before injection."""
    backup_path = path + BACKUP_SUFFIX
    if not os.path.exists(backup_path):
        shutil.copy2(path, backup_path)
        print(f"  Backed up: {path} -> {backup_path}")
    else:
        print(f"  Backup already exists: {backup_path}")


def restore(path):
    """Restore the original file from backup."""
    backup_path = path + BACKUP_SUFFIX
    if os.path.exists(backup_path):
        shutil.copy2(backup_path, path)
        os.remove(backup_path)
        print(f"  Restored: {backup_path} -> {path}")
    else:
        print(f"  No backup found for: {path}")


def read_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_file(path, content):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

# ─── Failure 1: Compilation Failure ───────────────────────────────────────────

def inject_compilation_failure():
    """
    Removes the closing brace of the getAllStudents method in
    StudentController.java, producing a Java compilation error.
    Failure category: Compilation
    Expected mechanism response: M1 (retry), M3 (notification + lockout)
    """
    print("\n[INJECT] Compilation failure -> StudentController.java")
    backup(CONTROLLER)

    content = read_file(CONTROLLER)

    # Insert a deliberate syntax error — unclosed string literal
    injected = content.replace(
        "public ResponseEntity<List<Student>> getAllStudents() {",
        "public ResponseEntity<List<Student>> getAllStudents() {\n        String broken = \"unclosed string;"
    )

    if injected == content:
        print("  WARNING: Injection target not found. File may have changed.")
        return False

    write_file(CONTROLLER, injected)
    print("  Injected: unclosed string literal into getAllStudents()")
    print("  Expected build error: reached end of file while parsing")
    return True

# ─── Failure 2: Test Failure ───────────────────────────────────────────────────

def inject_test_failure():
    """
    Replaces a correct assertion value with an incorrect one in
    StudentControllerTest.java, causing a JUnit assertion failure.
    Failure category: Test failure
    Expected mechanism response: M4 (retry), M6 (trend analysis)
    """
    print("\n[INJECT] Test failure -> StudentControllerTest.java")
    backup(CONTROLLER_TEST)

    content = read_file(CONTROLLER_TEST)

    # Replace correct expected list size with wrong value
    injected = content.replace(
        '.andExpect(jsonPath("$.length()").value(2))',
        '.andExpect(jsonPath("$.length()").value(999))'
    )

    if injected == content:
        print("  WARNING: Injection target not found. File may have changed.")
        return False

    write_file(CONTROLLER_TEST, injected)
    print("  Injected: incorrect assertion value (expected 999, actual 2)")
    print("  Expected test error: AssertionError in getAllStudents_shouldReturn200WithStudentList")
    return True

# ─── Failure 3: Flaky Test ────────────────────────────────────────────────────

def inject_flaky_test():
    """
    Introduces stateful marker-file flakiness into the
    getAllStudents_shouldReturn200WithStudentList test in
    StudentControllerTest.java.

    On the first attempt in a fresh runner workspace the marker file
    $TMPDIR/flaky_marker.tmp does not exist, so the test throws a
    RuntimeException simulating cold-start / transient instability.
    The marker is written on that same attempt, so every subsequent
    retry within the same job passes the real assertion.

    The marker file lives in the JVM temp directory, which is preserved
    across retries within the same GitHub Actions job (same runner
    workspace) but is absent on a fresh pipeline run (new VM).

    Failure category: Flaky test
    Expected mechanism response:
        M4 — Test retry succeeds on attempt 2
        M5 — Flaky candidate flagged (fail then pass within one job)
        M6 — TREND OK (job ultimately passes)
        M13 — May detect 'Simulated transient failure' keyword in logs
    """
    print("\n[INJECT] Flaky test -> StudentControllerTest.java")
    backup(CONTROLLER_TEST)

    content = read_file(CONTROLLER_TEST)

    # ── Injected block that replaces the original method body prefix ──────────
    # Strategy: insert the marker-file guard BEFORE the existing mockMvc call
    # by replacing the opening line of the method body.
    # We match both "() throws Exception {" and "()" forms so it works
    # whether or not the original method already declares throws.

    guard_block = """\
        // INJECTED: Stateful marker-file flakiness (Experiment 3)
        // Simulates cold-start / transient environmental instability.
        // Fails on attempt 1 (marker absent), passes on attempt 2+ (marker present).
        java.io.File _flaky_marker = new java.io.File(
                System.getProperty("java.io.tmpdir"), "flaky_marker.tmp");
        if (!_flaky_marker.exists()) {
            try { _flaky_marker.createNewFile(); } catch (java.io.IOException _ignored) {}
            throw new RuntimeException(
                "Simulated transient failure: cold-start instability detected");
        }
        // END INJECTED
"""

    # Try to insert the guard after the opening brace of the target method.
    # Pattern matches the @Test annotation line + void line + opening brace.
    pattern = (
        r"(@Test\s*\n"
        r"\s*void getAllStudents_shouldReturn200WithStudentList\(\)"
        r"(?:\s*throws\s+\w+(?:\s*,\s*\w+)*)?"
        r"\s*\{)"
    )

    def replacer(m):
        return m.group(1) + "\n" + guard_block

    injected, n = re.subn(pattern, replacer, content, flags=re.DOTALL)

    if n == 0:
        print("  WARNING: Injection target not found. File may have changed.")
        return False

    write_file(CONTROLLER_TEST, injected)
    print("  Injected: marker-file guard into getAllStudents_shouldReturn200WithStudentList")
    print("  Expected behaviour: RuntimeException on attempt 1, passes on attempt 2")
    return True

# ─── Failure 4: Configuration Failure ────────────────────────────────────────

def inject_configuration_failure():
    """
    Corrupts application.properties by setting an invalid value for
    server.port, causing the Spring Boot application context to fail
    to initialise during the test stage.
    Failure category: Configuration failure
    Expected mechanism response: M7 (rollback), M8 (validation gate), M9 (env verify)
    """
    print("\n[INJECT] Configuration failure -> application.properties")
    backup(APP_PROPS)

    content = read_file(APP_PROPS)

    # Replace valid port with invalid non-numeric value
    injected = content.replace(
        "server.port=8080",
        "server.port=INVALID_PORT_VALUE"
    )

    if injected == content:
        print("  WARNING: Injection target not found. File may have changed.")
        return False

    write_file(APP_PROPS, injected)
    print("  Injected: invalid value 'INVALID_PORT_VALUE' for server.port")
    print("  Expected error: ApplicationContext failure — port must be a valid integer")
    return True

# ─── Failure 5: Infrastructure Failure ───────────────────────────────────────

def inject_infrastructure_failure():
    """
    Introduces a non-existent dependency into pom.xml, simulating
    the unavailability of an external Maven artifact repository and
    causing the build stage to fail with a dependency resolution error.
    Failure category: Infrastructure failure
    Expected mechanism response: M1 (retry), M10 (fresh container), M11 (cache invalidation)
    """
    print("\n[INJECT] Infrastructure failure -> pom.xml")
    backup(POM_XML)

    content = read_file(POM_XML)

    # Insert a non-existent dependency before the closing </dependencies> tag
    fake_dependency = """
        <!-- INJECTED: Non-existent dependency to simulate infrastructure failure -->
        <dependency>
            <groupId>com.nonexistent.library</groupId>
            <artifactId>does-not-exist</artifactId>
            <version>9.9.9</version>
        </dependency>
"""

    injected = content.replace(
        "    </dependencies>",
        fake_dependency + "    </dependencies>"
    )

    if injected == content:
        print("  WARNING: Injection target not found. File may have changed.")
        return False

    write_file(POM_XML, injected)
    print("  Injected: non-existent dependency com.nonexistent.library:does-not-exist:9.9.9")
    print("  Expected error: Could not resolve dependencies — artifact not found in repository")
    return True

# ─── Restore All ─────────────────────────────────────────────────────────────

def restore_all():
    """Restores all injected files to their original state from backups."""
    print("\n[RESTORE] Restoring all files to original state...")
    files = [CONTROLLER, CONTROLLER_TEST, SERVICE_TEST, APP_PROPS, POM_XML]
    for f in files:
        restore(f)
    print("\nAll files restored. Pipeline is back to clean baseline.")

# ─── Main ─────────────────────────────────────────────────────────────────────

FAILURE_TYPES = {
    "compilation":    inject_compilation_failure,
    "test":           inject_test_failure,
    "flaky":          inject_flaky_test,
    "configuration":  inject_configuration_failure,
    "infrastructure": inject_infrastructure_failure,
}

def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)

    failure_type = sys.argv[1].lower()

    if failure_type == "restore":
        restore_all()
        return

    if failure_type not in FAILURE_TYPES:
        print(f"ERROR: Unknown failure type '{failure_type}'")
        print(f"Valid types: {', '.join(FAILURE_TYPES.keys())}, restore")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  FAILURE INJECTION: {failure_type.upper()}")
    print(f"{'='*60}")

    success = FAILURE_TYPES[failure_type]()

    if success:
        print(f"\nInjection complete.")
        print(f"Commit and push to trigger the pipeline and observe self-healing.")
        print(f"To restore: python scripts/inject_failure.py restore")
    else:
        print(f"\nInjection failed — check the WARNING above.")

if __name__ == "__main__":
    main()
