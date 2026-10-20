FROM kalilinux/kali-rolling:amd64@sha256:459f87b760ca9c2d8e953126db1847ae329d5b0660d113e4e3a9fb0806509fcd

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
