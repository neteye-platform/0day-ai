FROM kalilinux/kali-rolling:amd64

# Dedicated attacker box for the validator agent. kali-linux-headless is the
# Kali meta-package with the full tool surface minus the heavyweight
# desktop/WiFi stack. First build pulls a large image and can take a while;
# the build-hash stamp in attacker/.build.hash makes subsequent runs reuse it.
#
# Corporate-network apt behavior: the stock base image ships sources.list.d/
# kali.sources pointing at http://http.kali.org, which 302-redirects to plain
# HTTP mirrors that this network 403s, and the container's TLS store cannot
# verify the (MITM/proxy-CA) certificate for kali.download. So we drop every
# default source, pin the HTTPS kali.download mirror, stop apt from following
# mirror redirects, bump retries, and disable peer-verify for the one-time
# package download - pragmatic for a throwaway local attacker image.
# hadolint ignore=DL3008
RUN set -eux; \
    rm -f /etc/apt/sources.list /etc/apt/sources.list.d/*.sources /etc/apt/sources.list.d/*.list; \
    printf '%s\n' \
        "deb https://kali.download/kali kali-rolling main contrib non-free non-free-firmware" \
        "deb-src https://kali.download/kali kali-rolling main contrib non-free non-free-firmware" \
        > /etc/apt/sources.list; \
    printf '%s\n' \
        'Acquire::Retries "3";' \
        'Acquire::http::AllowRedirect "false";' \
        'Acquire::https::AllowRedirect "false";' \
        'Acquire::https::Verify-Peer "false";' \
        'Acquire::https::Verify-Host "false";' \
        > /etc/apt/apt.conf.d/99attacker; \
    apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends kali-linux-headless \
    && rm -rf /var/lib/apt/lists/*

# Working directory for validator shell activity; file tools confine reads
# and writes to this tree so PoC scripts and evidence stay predictable.
WORKDIR /work
