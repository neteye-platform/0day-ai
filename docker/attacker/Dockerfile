FROM kalilinux/kali-rolling:amd64@sha256:cd030992fed90891eb7de0e2b1b7b1972ab46fcfaf75d969af0b94304214b6e3

# Dedicated attacker box for the validator agent. kali-linux-headless is the
# Kali meta-package with the full tool surface minus the heavyweight
# desktop/WiFi stack. First build pulls a large image and can take a while;
# the build-hash stamp in attacker/.build.hash makes subsequent runs reuse it.

# kali-rolling is a rolling distro: pinned package versions vanish from the
# mirror within weeks and would break the build, so DL3008 is waived here.
# hadolint ignore=DL3008
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends kali-linux-headless \
    && rm -rf /var/lib/apt/lists/*

# Working directory for validator shell activity; file tools confine reads
# and writes to this tree so PoC scripts and evidence stay predictable.
WORKDIR /work
