FROM kalilinux/kali-rolling:amd64

# Dedicated attacker box for the validator agent. kali-linux-headless is the
# Kali meta-package with the full tool surface minus the heavyweight
# desktop/WiFi stack. First build pulls a large image and can take a while;
# the build-hash stamp in attacker/.build.hash makes subsequent runs reuse it.

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends kali-linux-headless \
    && rm -rf /var/lib/apt/lists/*

# Working directory for validator shell activity; file tools confine reads
# and writes to this tree so PoC scripts and evidence stay predictable.
WORKDIR /work
