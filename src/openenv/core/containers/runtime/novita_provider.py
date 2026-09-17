# SPDX-License-Identifier: BSD-3-Clause

"""
Novita AI Sandbox container provider for running OpenEnv environments.

Requires the ``novita-sandbox`` SDK: ``pip install openenv[novita]``

The provider boots an OpenEnv server inside a Novita sandbox, exposes port 8000
on the sandbox's public host, and returns an ``https://`` URL that ``EnvClient``
connects to over ``wss://``.

Note: the ``RFC 002 security invariant S<n>`` references in this module point to
the **proposed** "Cloud Sandbox Providers" amendment to RFC 002, which is pending
review/sign-off by the RFC authors. They describe the security properties this
provider enforces; treat them as proposed (not yet ratified) until that sign-off.
"""

from __future__ import annotations

import re
import shlex
import time
from typing import Any, Callable, Dict, List, Optional

from ._server_config import parse_openenv_app_field
from .providers import ContainerProvider

_DEFAULT_NOVITA_PORT = 8000

# Novita defaults a sandbox to a 300s hard lifetime, which would kill an RL
# rollout mid-episode. One hour is the documented ceiling for Hobby accounts, so
# it is the largest default that cannot make ``create`` fail on plan limits;
# longer-running work passes ``timeout=`` explicitly.
_DEFAULT_SANDBOX_TIMEOUT_S = 3600

# The sandbox is created with this as its start command so the image's own CMD
# is not launched (see `_launch_server`). `sleep infinity` keeps the sandbox
# alive with nothing bound to port 8000, leaving the port free for the server
# this provider starts itself.
_KEEPALIVE_CMD = "sleep infinity"
_KEEPALIVE_READY_MS = 5_000

# Resource defaults for a template built from a Dockerfile. The SDK's own
# defaults, restated here so they are visible at the call site.
_DEFAULT_TEMPLATE_CPU = 2
_DEFAULT_TEMPLATE_MEMORY_MB = 1024

# User the sandbox runs as. Root because the environment installs under /app,
# which the image creates as root and the Novita parser's default user ("user")
# cannot write to -- see `build_template`. Matches what a plain `docker run` of
# the same image gives, which is what the other providers' users observe.
_SANDBOX_USER = "root"

_FROM_RE = re.compile(r"^\s*FROM\s+(?P<rest>.+?)\s*$", re.IGNORECASE)
_COPY_FROM_RE = re.compile(
    r"^\s*COPY\s+--from=(?P<stage>\S+)\s+(?P<src>\S+)\s+(?P<dst>\S+)\s*$",
    re.IGNORECASE,
)
_RUN_RE = re.compile(r"^\s*RUN\s+(?P<rest>.*)$", re.IGNORECASE)
_ARG_DEFAULT_RE = re.compile(
    r"^\s*ARG\s+(?P<name>\w+)=(?P<value>\S+)\s*$", re.IGNORECASE
)


def _strip_mount_flags(content: str) -> str:
    """Remove BuildKit ``--mount=...`` flags from ``RUN`` instructions.

    Novita's template parser copies a ``RUN`` body verbatim into the build
    command, so a leading ``--mount=type=cache,...`` would be handed to the
    shell as an argument rather than honored as BuildKit syntax. Every in-repo
    OpenEnv Dockerfile uses one to cache ``uv`` downloads, so the flags are
    stripped rather than left to fail the build.
    """
    lines = content.split("\n")
    out: List[str] = []
    in_run = False
    for line in lines:
        if _RUN_RE.match(line):
            in_run = True
        if in_run:
            # Strip leading --mount flags from the RUN line itself and from any
            # continuation lines that follow it.
            stripped = line
            prefix = ""
            run_match = _RUN_RE.match(line)
            if run_match:
                prefix = line[: run_match.start("rest")]
                stripped = run_match.group("rest")
            while True:
                mount = re.match(r"\s*--mount=\S+\s*", stripped)
                if not mount:
                    break
                stripped = stripped[mount.end() :]
            line = prefix + stripped
            if not line.rstrip().endswith("\\"):
                in_run = False
        out.append(line)
    return "\n".join(out)


def _resolve_from_references(content: str) -> str:
    """Substitute ``ARG`` defaults into ``FROM`` lines and drop ``--platform``.

    Two parser limitations, handled together because both corrupt the base
    image reference:

    - The parser stores a ``FROM ${BASE_IMAGE}`` line verbatim, so the template
      would be built from a literal image named ``${BASE_IMAGE}``. Only global
      ARGs (declared before the first ``FROM``) are in scope for a ``FROM``
      line, which is the form every in-repo Dockerfile uses.
    - ``FROM --platform=linux/amd64 python:3.10-slim`` likewise keeps the flag
      as part of the name. Novita builds for its own platform, so the flag is
      dropped rather than propagated.
    """
    arg_defaults: Dict[str, str] = {}
    for line in content.split("\n"):
        if _FROM_RE.match(line):
            break
        match = _ARG_DEFAULT_RE.match(line)
        if match:
            arg_defaults[match.group("name")] = match.group("value")

    out: List[str] = []
    for line in content.split("\n"):
        match = _FROM_RE.match(line)
        if not match:
            out.append(line)
            continue

        reference = match.group("rest").strip()
        reference = re.sub(r"^(--platform=\S+\s*)+", "", reference).strip()
        for name, value in arg_defaults.items():
            reference = reference.replace(f"${{{name}}}", value)
            reference = reference.replace(f"${name}", value)
        out.append(f"FROM {reference}")
    return "\n".join(out)


def _split_stages(content: str) -> List[Dict[str, Any]]:
    """Split a Dockerfile into its ``FROM``-delimited stages."""
    stages: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    for line in content.split("\n"):
        match = _FROM_RE.match(line)
        if match:
            rest = match.group("rest").strip()
            alias_match = re.match(r"(?P<base>.+?)\s+[Aa][Ss]\s+(?P<alias>\S+)$", rest)
            if alias_match:
                base = alias_match.group("base").strip()
                alias = alias_match.group("alias")
            else:
                base, alias = rest, None
            current = {"base": base, "alias": alias, "body": []}
            stages.append(current)
        elif current is not None:
            current["body"].append(line)
    return stages


def _flatten_multistage(content: str) -> str:
    """Collapse a multi-stage Dockerfile into a single stage.

    Novita's parser rejects multi-stage Dockerfiles outright, and that is the
    layout every in-repo OpenEnv environment uses: a ``builder`` stage that runs
    ``uv sync`` into ``/app/env/.venv``, then a runtime stage that copies the
    result across. Both stages name the *same* base image, so the build can be
    replayed linearly in one stage -- the builder's ``RUN`` steps execute, then
    each ``COPY --from=builder`` becomes an in-place ``cp``.

    A copy whose source and destination are equal is dropped: after flattening,
    the file is already at its destination, and Novita rejects an absolute
    source path.

    Raises:
        ValueError: If the stages do not share one base image, or a copy pulls
            from a stage other than the builder.
    """
    stages = _split_stages(content)
    if len(stages) < 2:
        return content

    bases = {stage["base"] for stage in stages}
    if len(bases) != 1:
        raise ValueError(
            "Novita templates cannot express a multi-stage Dockerfile whose "
            f"stages use different base images: {sorted(bases)}. Build the "
            "image with `openenv build` and push it with "
            "`openenv push --registry`, then pass the registry reference to "
            "NovitaSandboxProvider(image=...)."
        )

    builder_alias = stages[0]["alias"]
    if builder_alias is None:
        raise ValueError(
            "The first stage of a multi-stage Dockerfile must be named with "
            "`AS <alias>` so its output can be referenced after flattening."
        )

    out: List[str] = [f"FROM {stages[0]['base']}"]
    for index, stage in enumerate(stages):
        for line in stage["body"]:
            copy_match = _COPY_FROM_RE.match(line)
            if copy_match is None:
                out.append(line)
                continue
            if index == 0:
                # A copy within the builder stage is already valid as written.
                out.append(line)
                continue
            source_stage = copy_match.group("stage")
            if source_stage != builder_alias:
                raise ValueError(
                    f"COPY --from={source_stage} does not reference the builder "
                    f"stage ({builder_alias}), so it cannot be flattened into a "
                    "single-stage template."
                )
            src = copy_match.group("src")
            dst = copy_match.group("dst")
            if src == dst:
                continue
            out.append(f"RUN mkdir -p $(dirname {dst}) && cp -a {src} {dst}")

    return "\n".join(out)


def _prepare_dockerfile(content: str) -> str:
    """Rewrite an OpenEnv Dockerfile into what Novita's template parser accepts.

    Applies, in order: BuildKit ``--mount`` stripping, ``ARG``/``--platform``
    resolution in ``FROM``, then multi-stage flattening. Each transform
    compensates for a parser limitation rather than changing the build's
    meaning.
    """
    return _flatten_multistage(_resolve_from_references(_strip_mount_flags(content)))


def _print_build_log(entry: Any) -> None:
    """Default ``on_build_logs``: print each template build-log entry.

    A first build resolves and builds the image server-side and can take
    minutes, so silence looks like a hang. Named (rather than a lambda default)
    so the parameter's default renders readably in the generated API docs.
    """
    print(entry)


def _require_secure_url(url: str) -> str:
    """Enforce https/wss transport (RFC 002 security invariant S1).

    ``EnvClient`` derives its WebSocket URL from this base URL, so a plaintext
    URL would become a cleartext ``ws://`` connection. The offending URL is
    deliberately omitted from the error: a sandbox host is reachable only
    through the account that owns it, so the address is treated as
    account-scoped rather than pasted into logs.
    """
    if not isinstance(url, str) or not url.lower().startswith("https://"):
        raise RuntimeError(
            "Novita sandbox returned a non-HTTPS host URL. OpenEnv requires an "
            "https/wss base_url so EnvClient traffic is encrypted. Refusing to "
            "connect over plaintext."
        )

    # With NOVITA_DEBUG set, the SDK's `get_host` returns `localhost:<port>`
    # instead of the sandbox host. Adding the scheme would yield a URL that
    # looks valid and silently targets whatever is listening on the caller's
    # machine, so reject it explicitly rather than connect to the wrong thing.
    hostname = url.split("://", 1)[1].split("/", 1)[0].rsplit(":", 1)[0]
    if hostname in ("localhost", "127.0.0.1", "::1"):
        raise RuntimeError(
            "Novita sandbox returned a loopback host URL, which means the SDK is "
            "running with NOVITA_DEBUG enabled. Debug mode bypasses the sandbox "
            "host and would connect to the local machine instead; unset "
            "NOVITA_DEBUG to reach the sandbox."
        )

    return url


def _raise_install_error(exc: ImportError) -> None:
    raise RuntimeError(
        "Novita sandbox support requires optional dependencies. "
        "Install them with `pip install openenv[novita]`."
    ) from exc


class _DefaultNovitaAdapter:
    """Thin adapter over the ``novita-sandbox`` SDK.

    The provider talks to this private adapter instead of spreading SDK details
    through its own logic; tests inject a duck-typed fake in its place. Keeping
    the SDK surface here is what makes the provider testable without a Novita
    account and localizes API churn (RFC 002 implementation-hygiene invariant).
    """

    def __init__(
        self,
        *,
        api_key: Optional[str],
        domain: Optional[str],
    ):
        try:
            from novita_sandbox import Novita
        except ImportError as exc:  # pragma: no cover - exercised via provider tests
            _raise_install_error(exc)

        # `api_key`/`domain` fall back to NOVITA_API_KEY / NOVITA_DOMAIN inside
        # the SDK's ConnectionConfig, so `None` is the documented "use the env
        # var" spelling rather than a missing value.
        self._novita = Novita(api_key=api_key, domain=domain)

    def create_sandbox(
        self,
        *,
        image: Optional[str],
        template: Optional[str],
        env_vars: Optional[Dict[str, str]],
        timeout: int,
        metadata: Optional[Dict[str, str]],
        secure: Optional[bool],
        allow_internet_access: bool,
    ) -> Any:
        # Exactly one source: a registry image (resolved and cached by the SDK
        # itself) or an already-built template id from `image_from_dockerfile`.
        kwargs: Dict[str, Any] = {
            "timeout": timeout,
            "allow_internet_access": allow_internet_access,
        }

        if template is not None:
            kwargs["template"] = template
        else:
            from novita_sandbox import wait_for_timeout

            # `image` is an OCI reference; the SDK resolves it to a template
            # itself (Template.from_image -> Template.build) and caches the
            # result by image fingerprint, so repeated starts do not rebuild.
            #
            # The `build` block exists solely to pin the template's start
            # command to a keepalive. `from_image` defaults to
            # `inherit_config=True`, which restores the image's ENV/WORKDIR
            # *and* its ENTRYPOINT/CMD as the template start command -- for an
            # OpenEnv image that CMD is the uvicorn server, so leaving it in
            # place would bind port 8000 before this provider launches its own
            # copy. Setting `start_cmd` here wins over the inherited value (the
            # caller always overrides the image) while the image's ENV still
            # comes through, which is what the PATH needs.
            kwargs["image"] = image
            kwargs["build"] = {
                "cmd": _KEEPALIVE_CMD,
                "ready_cmd": wait_for_timeout(_KEEPALIVE_READY_MS),
            }

        if env_vars:
            kwargs["envs"] = dict(env_vars)
        if metadata:
            kwargs["metadata"] = dict(metadata)
        if secure is not None:
            kwargs["secure"] = secure

        return self._novita.sandbox.create(**kwargs)

    def exec(self, sandbox: Any, command: str, *, timeout: float = 10) -> str:
        """Run *command* through a login shell inside *sandbox*, returning stdout.

        The SDK runs commands as ``/bin/bash -l -c <command>`` and raises
        ``CommandExitException`` on a non-zero exit. The discovery and liveness
        probes below are *expected* to exit non-zero ("no openenv.yaml", "process
        is dead"), so the exit code is swallowed and stdout returned; callers
        distinguish states by the marker the command echoes, not by the status.
        """
        from novita_sandbox import CommandExitException

        try:
            result = sandbox.commands.run(command, timeout=timeout)
        except CommandExitException as exc:
            return exc.stdout or ""
        except ImportError:  # pragma: no cover - SDK always exports this
            result = sandbox.commands.run(command, timeout=timeout)

        return getattr(result, "stdout", "") or ""

    def host(self, sandbox: Any, port: int) -> str:
        """Return the public host (no scheme) exposing *port* on *sandbox*."""
        return str(sandbox.get_host(port))

    def kill(self, sandbox: Any) -> None:
        sandbox.kill()

    def build_template(
        self,
        *,
        dockerfile_content: str,
        context_dir: str,
        name: str,
        cpu_count: int,
        memory_mb: int,
        on_build_logs: Optional[Callable[[Any], None]],
    ) -> str:
        """Build a Novita template from Dockerfile *content* and return its id.

        Uses ``Template.from_dockerfile`` to translate the Dockerfile into
        builder instructions, pins the template's start command to a keepalive
        (so the image's own ``CMD`` does not bind port 8000 before the provider
        launches the server), and builds it. ``file_context_path`` is what makes
        relative ``COPY`` sources resolve against the build context.

        Returns:
            `str`: The built template id, to pass to ``Sandbox.create``.
        """
        from novita_sandbox import Template, wait_for_timeout

        template = Template(file_context_path=context_dir)
        builder = template.from_dockerfile(dockerfile_content)

        # The parser rewrites USER to "user" when the Dockerfile declares none
        # (dockerfile_parser.py: `if not user_changed: set_user("user")`). For an
        # OpenEnv image that is wrong: `WORKDIR /app`, created by root, stays
        # root-owned, so the server and every command the environment later runs
        # in-process (`TB2_MODE=local`, and any env doing the same) fail with
        # "Permission denied" on the install tree. Providers that do not rewrite
        # USER -- Daytona, for one -- keep the image's default and never see
        # this, which is why the same Dockerfile works there unchanged.
        #
        # Set explicitly rather than relying on the Dockerfile: a Dockerfile that
        # DOES declare a USER would otherwise be honored, leaving this provider
        # inconsistent about the identity its commands run as.
        builder.set_user(_SANDBOX_USER)

        # `set_start_cmd` is the only way to pin the start command and it
        # requires a readiness check, so the keepalive gets a short timer. The
        # provider's own `wait_for_ready` does the real health polling.
        builder.set_start_cmd(_KEEPALIVE_CMD, wait_for_timeout(_KEEPALIVE_READY_MS))

        build_kwargs: Dict[str, Any] = {
            "cpu_count": cpu_count,
            "memory_mb": memory_mb,
        }
        if on_build_logs is not None:
            build_kwargs["on_build_logs"] = on_build_logs

        info = Template.build(builder, name, **build_kwargs)
        return str(info.template_id)


class NovitaSandboxProvider(ContainerProvider):
    """
    Container provider that runs environments in Novita AI sandboxes.

    ``start_container`` accepts either form of source:

    - A Docker/OCI registry reference (e.g. ``"ghcr.io/org/env:tag"``), which is
      the form the ``ContainerProvider`` contract specifies. The SDK resolves it
      into a Novita template internally and caches it by image fingerprint.
    - A ``"template:<id>"`` reference returned by
      :meth:`image_from_dockerfile`, which builds a template from a local
      Dockerfile.

    :meth:`image_from_dockerfile` rewrites the Dockerfile for Novita's template
    parser, which does not accept multi-stage build definitions. The rewrite is
    mechanical and preserves the build's meaning: BuildKit ``--mount`` flags are
    stripped, ``ARG``/``--platform`` in ``FROM`` lines are resolved, and a
    two-stage build whose stages share one base image is replayed as a single
    stage (see `_prepare_dockerfile`). A Dockerfile that does not fit those
    rules raises `ValueError` with the registry route as the alternative.

    The environment runs untrusted code, so the provider is secure by default: it
    enforces https/wss transport (S1) and never surfaces raw sandbox output
    unless ``surface_server_logs=True`` (S4).

    Only one sandbox is active per provider: calling ``start_container`` again
    before ``stop_container()``/``close()`` raises ``RuntimeError`` rather than
    orphaning the running sandbox.

    Examples:

        ```python
        # From a pre-built registry image
        with NovitaSandboxProvider(image="ghcr.io/org/echo-env:latest") as provider:
            base_url = provider.start_container()
            provider.wait_for_ready(base_url)

        # From a local Dockerfile (builds a template on first use)
        image = NovitaSandboxProvider.image_from_dockerfile(
            "envs/echo_env/server/Dockerfile"
        )
        with NovitaSandboxProvider(image=image) as provider:
            base_url = provider.start_container()
        # sandbox killed on exit
        ```
    """

    # Prepared Dockerfile content, keyed by "template:<abs_path>". Populated by
    # `image_from_dockerfile` so `start_container` can build the template later,
    # mirroring DaytonaProvider / ModalProvider.
    _dockerfile_registry: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def image_from_dockerfile(
        cls,
        dockerfile_path: str,
        context_dir: Optional[str] = None,
    ) -> str:
        """Validate a Dockerfile and return a ``template:`` reference for
        :meth:`start_container`.

        Eagerly validates the Dockerfile (existence, COPY sources) and stores
        the rewritten content in an internal registry. The Novita template is
        built later, inside ``start_container``, by the adapter's
        ``build_template`` — building needs credentials and network, which this
        class-level helper deliberately does not.

        The rewrite compensates for Novita's Dockerfile parser, which rejects
        multi-stage build definitions (the layout every in-repo OpenEnv
        environment uses). BuildKit ``--mount`` flags are stripped, ``ARG`` and
        ``--platform`` in ``FROM`` lines are resolved, and a two-stage build
        whose stages share one base image is replayed as a single stage.

        Args:
            dockerfile_path (`str`):
                Path to the Dockerfile on disk.
            context_dir (`str`, *optional*):
                Build context directory, used to resolve relative ``COPY``
                sources. Defaults to the Dockerfile's grandparent directory,
                matching the `openenv init` convention where Dockerfiles live in
                `<env>/server/Dockerfile` and the build context is `<env>/`.

        Returns:
            `str`: A `"template:<abs_path>"` reference to pass to
            `start_container` or the constructor's `image`.

        Raises:
            FileNotFoundError: If *dockerfile_path* does not exist.
            ValueError: If *context_dir* does not exist, if COPY sources cannot
                be found under the resolved context directory, or if the
                Dockerfile cannot be expressed as a Novita template (see
                `_prepare_dockerfile`).

        Examples:

        ```python
        image = NovitaSandboxProvider.image_from_dockerfile(
            "envs/echo_env/server/Dockerfile"
        )
        provider = NovitaSandboxProvider(image=image)
        base_url = provider.start_container()
        ```
        """
        import pathlib

        src = pathlib.Path(dockerfile_path).resolve()
        if not src.is_file():
            raise FileNotFoundError(f"Dockerfile not found: {dockerfile_path}")

        if context_dir is not None:
            ctx = pathlib.Path(context_dir)
            if not ctx.is_dir():
                raise ValueError(f"context_dir does not exist: {context_dir}")
        else:
            # Default: grandparent of the Dockerfile, matching the openenv init
            # layout (<env>/server/Dockerfile -> <env>/).
            ctx = src.parent.parent

        prepared = _prepare_dockerfile(src.read_text())

        # Validate that COPY sources exist under the context directory. This
        # catches mismatches early (e.g. a Dockerfile expecting the repo root as
        # context when we defaulted to the env directory).
        for line in prepared.splitlines():
            match = re.match(r"^\s*COPY\s+(?!--from=)(\S+)\s+", line, re.IGNORECASE)
            if not match:
                continue
            copy_src = match.group(1)
            if copy_src.startswith("/"):
                continue
            resolved = ctx / copy_src
            if not resolved.exists() and not any(ctx.glob(copy_src)):
                raise ValueError(
                    f"Dockerfile COPY source '{copy_src}' not found under "
                    f"context_dir '{ctx}'. This Dockerfile may expect a "
                    "different build context (e.g. the repo root). Pass "
                    "context_dir explicitly."
                )

        cls._dockerfile_registry[str(src)] = {
            "content": prepared,
            "context_dir": str(ctx),
        }

        return f"template:{src}"

    def _resolve_template_ref(self, reference: str) -> str:
        """Build the template for a ``template:<path>`` reference; return its id."""
        path = reference[len("template:") :]
        meta = self._dockerfile_registry.get(path)
        if meta is None:
            raise ValueError(
                f"No registered Dockerfile metadata for {path}. Call "
                "NovitaSandboxProvider.image_from_dockerfile() first."
            )

        return self._adapter.build_template(
            dockerfile_content=meta["content"],
            context_dir=meta["context_dir"],
            name=self._template_name or self._default_template_name(path),
            cpu_count=self._cpu_count,
            memory_mb=self._memory_mb,
            on_build_logs=self._on_build_logs,
        )

    @staticmethod
    def _default_template_name(dockerfile_path: str) -> str:
        """Derive a stable template name from the Dockerfile's env directory.

        The env directory keeps the template recognizable in the Novita
        dashboard; the content hash keeps two different Dockerfiles from
        colliding on one name, while a rebuild of identical content still hits
        the build cache.
        """
        import hashlib
        import pathlib

        src = pathlib.Path(dockerfile_path)
        env_dir = src.parent.parent.name or src.parent.name
        slug = re.sub(r"[^a-z0-9-]+", "-", env_dir.lower().replace("_", "-"))
        digest = hashlib.sha256(src.read_bytes()).hexdigest()[:8]
        return f"openenv-{slug.strip('-')}-{digest}"

    def __init__(
        self,
        *,
        image: Optional[str] = None,
        env_vars: Optional[Dict[str, str]] = None,
        api_key: Optional[str] = None,
        domain: Optional[str] = None,
        timeout: int = _DEFAULT_SANDBOX_TIMEOUT_S,
        metadata: Optional[Dict[str, str]] = None,
        secure: Optional[bool] = None,
        allow_internet_access: bool = True,
        cmd: Optional[str] = None,
        working_directory: Optional[str] = None,
        surface_server_logs: bool = False,
        cpu_count: int = _DEFAULT_TEMPLATE_CPU,
        memory_mb: int = _DEFAULT_TEMPLATE_MEMORY_MB,
        template_name: Optional[str] = None,
        on_build_logs: Optional[Callable[[Any], None]] = _print_build_log,
        _adapter: Any = None,
    ):
        """
        Args:
            image (`str`, *optional*):
                Registry image reference (e.g. `"ghcr.io/org/env:latest"`) or a
                `"template:<id>"` reference from
                [`~image_from_dockerfile`], to use when `start_container()` is
                called without an image.
            env_vars (`dict`, *optional*):
                Environment variables to use when `start_container()` is called
                without explicit `env_vars`.
            api_key (`str`, *optional*):
                Novita API key. Falls back to the `NOVITA_API_KEY` environment
                variable.
            domain (`str`, *optional*):
                Novita region domain. Falls back to the `NOVITA_DOMAIN`
                environment variable, then to the SDK default (`us-phx-1`).
                Features marked modern-domain-only (snapshots, secrets) are
                unavailable on the legacy `us-virginia-1` domain.
            timeout (`int`, *optional*, defaults to `3600`):
                Hard sandbox lifetime in seconds. Counts down from creation
                regardless of activity and always fires, so it must exceed the
                longest expected episode.
            metadata (`dict`, *optional*):
                Arbitrary key/value labels on the sandbox. The `idle_timeout`
                key is meaningful to Novita and takes seconds as a **string**
                (e.g. `{"idle_timeout": "900"}`); omitting it disables the idle
                timer, which is the conservative default for RL rollouts — an
                automatic transition can drop a live WebSocket mid-episode
                (RFC 002 invariant 6).
            secure (`bool`, *optional*):
                Whether envd requires its access token. `None` uses the SDK
                default (secured on modern domains). This governs the sandbox
                control plane, not the exposed server port.
            allow_internet_access (`bool`, *optional*, defaults to `True`):
                When `False`, blocks all outbound traffic from the sandbox —
                equivalent to a deny-all egress policy. Prefer `False` for
                untrusted environments that do not need network access.
            cmd (`str`, *optional*):
                Shell command to start the server inside the sandbox. When
                omitted, the command is auto-discovered from `openenv.yaml`.
            working_directory (`str`, *optional*):
                Directory to `cd` into before running `cmd`.
            surface_server_logs (`bool`, *optional*, defaults to `False`):
                When `False` (default), captured sandbox output is withheld from
                raised errors so secrets the workload printed cannot leak into
                orchestrator/CI logs. When `True`, a best-effort redacted,
                length-bounded excerpt is included in startup-crash errors (S4).
            cpu_count (`int`, *optional*, defaults to `2`):
                vCPU cores for sandboxes built from a Dockerfile via
                [`~image_from_dockerfile`]. Ignored for registry images, whose
                resources come from the image's own template.
            memory_mb (`int`, *optional*, defaults to `1024`):
                Memory in MiB for sandboxes built from a Dockerfile. Ignored for
                registry images.
            template_name (`str`, *optional*):
                Template name to register when building from a Dockerfile.
                Defaults to a name derived from the Dockerfile's directory plus a
                content hash, so re-running with the same Dockerfile reuses the
                build cache.
            on_build_logs (`Callable`, *optional*, defaults to printing each entry):
                Callback receiving build-log entries while a Dockerfile template
                is built. Defaults to printing the entries, so a first build
                (which can take minutes) is not silent. Pass `None` to suppress
                output, or your own callable to route it elsewhere.
            _adapter (`Any`, *optional*):
                Injection seam for tests; a duck-typed fake replaces the SDK.
        """
        self._image = image
        self._env_vars = env_vars
        self._timeout = timeout
        self._metadata = metadata
        self._secure = secure
        self._allow_internet_access = allow_internet_access
        self._cmd = cmd
        self._working_directory = working_directory
        self.surface_server_logs = surface_server_logs
        self._cpu_count = cpu_count
        self._memory_mb = memory_mb
        self._template_name = template_name
        self._on_build_logs = on_build_logs

        self._sandbox: Any = None
        self._base_url: Optional[str] = None
        # Registry paths from `image_from_dockerfile`, keyed by "dockerfile:<path>".
        self._dockerfile_registry = dict(type(self)._dockerfile_registry)
        # Injected env-var values, used to scrub captured server output before
        # it is ever surfaced in an error.
        self._redact_values: set[str] = set()

        if _adapter is None:
            # Build the client eagerly so credential/region problems surface at
            # construction rather than mid-episode, matching ModalProvider.
            self._adapter: Any = _DefaultNovitaAdapter(api_key=api_key, domain=domain)
        else:
            self._adapter = _adapter

    def _discover_server_cmd(self, port: int = _DEFAULT_NOVITA_PORT) -> str:
        """Discover the server command from ``openenv.yaml`` inside the sandbox.

        Finds the file, reads the ``app`` field, and constructs a command of the
        form ``cd <env_root> && python -m uvicorn <app> --host 0.0.0.0 --port <port>``.

        Raises:
            ValueError: If ``openenv.yaml`` is not found or lacks an ``app`` field.
        """
        yaml_path = self._find_openenv_yaml()
        if yaml_path is None:
            raise ValueError(
                "Could not find openenv.yaml inside the sandbox. Pass an "
                "explicit cmd= to NovitaSandboxProvider or start_container()."
            )

        content = self._adapter.exec(self._sandbox, f"cat {shlex.quote(yaml_path)}")
        app = self._parse_app_field(content)
        if app is None:
            raise ValueError(
                f"openenv.yaml at {yaml_path} does not contain an 'app' field. "
                "Pass an explicit cmd= to NovitaSandboxProvider or start_container()."
            )

        # The directory containing openenv.yaml is the env root
        env_root = yaml_path.rsplit("/", 1)[0]
        return (
            f"cd {shlex.quote(env_root)} && "
            f"python -m uvicorn {shlex.quote(app)} --host 0.0.0.0 --port {port}"
        )

    def _find_openenv_yaml(self) -> Optional[str]:
        """Locate ``openenv.yaml`` inside the sandbox.

        Tries the modern layout path ``/app/env/openenv.yaml`` first, then falls
        back to a ``find`` command for the old layout.
        """
        # Fast path: modern Dockerfile layout
        out = self._adapter.exec(
            self._sandbox, "test -f /app/env/openenv.yaml && echo found"
        )
        if "found" in (out or ""):
            return "/app/env/openenv.yaml"

        # Fallback: search for it (redirect stderr so error messages like
        # "No such file or directory" don't get mistaken for paths).
        path = self._adapter.exec(
            self._sandbox,
            "find /app -maxdepth 4 -name openenv.yaml -print -quit 2>/dev/null",
        ).strip()
        if path and path.startswith("/"):
            return path

        return None

    @staticmethod
    def _parse_app_field(yaml_content: str) -> Optional[str]:
        """Extract the ``app`` value from raw openenv.yaml content.

        Uses PyYAML to handle comments, quotes, and nested keys correctly.
        """
        return parse_openenv_app_field(yaml_content)

    def start_container(
        self,
        image: Optional[str] = None,
        port: Optional[int] = None,
        env_vars: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> str:
        """
        Create a Novita sandbox and start an OpenEnv server inside it.

        Args:
            image (`str`, *optional*):
                Either a registry image reference (e.g.
                `"ghcr.io/org/env:latest"`) or a `"template:<path>"` reference
                returned by [`~image_from_dockerfile`]. May be omitted when
                supplied to the constructor.
            port (`int`, *optional*):
                Must be `None` or `8000`. Novita exposes port 8000 on the
                sandbox host; other ports raise `ValueError`.
            env_vars (`dict`, *optional*):
                Environment variables forwarded to the sandbox, overriding the
                constructor's.
            **kwargs:
                `cmd` (`str`) to override the server command. Unknown options
                raise `ValueError` so typos cannot silently change sandbox
                behavior.

        Returns:
            `str`: HTTPS sandbox URL for the exposed port (base_url).

        Raises:
            RuntimeError: If a sandbox is already active on this provider.
            ValueError: If no image is available, the port is unsupported, or an
                unknown option is passed.
        """
        if self._sandbox is not None:
            raise RuntimeError(
                "NovitaSandboxProvider already has an active sandbox. Call "
                "stop_container() (or close()) before starting another — a "
                "second start would orphan the running sandbox."
            )

        if port is not None and port != _DEFAULT_NOVITA_PORT:
            raise ValueError(
                f"NovitaSandboxProvider only supports port {_DEFAULT_NOVITA_PORT} "
                f"(got {port}). The sandbox host routes to port "
                f"{_DEFAULT_NOVITA_PORT} inside the sandbox."
            )

        effective_image = image if image is not None else self._image
        if effective_image is None:
            raise ValueError(
                "NovitaSandboxProvider requires an image. Pass it to the "
                "constructor or start_container()."
            )
        effective_env_vars = self._env_vars if env_vars is None else env_vars

        cmd = kwargs.pop("cmd", None) or self._cmd
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise ValueError(
                f"Unsupported NovitaSandboxProvider start options: {unknown}"
            )

        # A "template:<path>" reference builds a Novita template from a local
        # Dockerfile; anything else is a registry image the SDK resolves itself.
        # The build runs before `create_sandbox` so a build failure never leaves
        # a sandbox behind, and outside the `_redact_values` window below so its
        # recorded secrets do not outlive a failed build.
        if effective_image.startswith("template:"):
            template_id: Optional[str] = self._resolve_template_ref(effective_image)
            sandbox_image: Optional[str] = None
        else:
            template_id = None
            sandbox_image = effective_image

        # Record injected secret values so captured server output can be scrubbed
        # before it is ever surfaced in an error (security invariant S4).
        self._redact_values = {
            value for value in (effective_env_vars or {}).values() if value
        }

        # A create failure created nothing, so just drop the recorded secrets and
        # re-raise (the single-sandbox guard above guarantees there is no
        # pre-existing sandbox to delete).
        try:
            self._sandbox = self._adapter.create_sandbox(
                image=sandbox_image,
                template=template_id,
                env_vars=effective_env_vars,
                timeout=self._timeout,
                metadata=self._metadata,
                secure=self._secure,
                allow_internet_access=self._allow_internet_access,
            )
        except Exception:
            self._redact_values = set()
            raise

        try:
            # Discovery runs after creation, when the filesystem can be read.
            if cmd is None:
                cmd = self._discover_server_cmd()

            self._launch_server(cmd)
            self._base_url = _require_secure_url(
                f"https://{self._adapter.host(self._sandbox, _DEFAULT_NOVITA_PORT)}"
            )
        except Exception:
            # A cleanup failure here must not mask the original error: swallow
            # any exception from stop_container() so the root cause propagates.
            try:
                self.stop_container()
            except Exception:
                pass
            raise

        return self._base_url

    def _launch_server(self, cmd: str) -> None:
        """Start the OpenEnv server inside the sandbox as a background process.

        This is the one place the provider assumes something about the image, so
        it is kept to a single method.

        The server is launched with ``nohup`` and its PID written to
        ``/tmp/openenv-server.pid``, so ``wait_for_ready`` can tell "still
        booting" apart from "crashed" instead of waiting out the full timeout.
        The sandbox was created with a ``sleep infinity`` start command rather
        than the image's own CMD (see ``_DefaultNovitaAdapter.create_sandbox``),
        which is what leaves port 8000 free for this process.

        An alternative is to let the platform own it: pass the server command as
        the template's start command via the SDK's ``build={"cmd": ...}`` and
        drop this method's ``nohup``. That trades the PID-based crash detection
        below for platform-managed lifecycle and readiness. It is not the
        default because a start command can only be set when the image is first
        resolved into a template, so it would not apply to an already-cached
        image.

        ``cmd`` is trusted orchestrator configuration, not agent or environment
        input — callers must not pass agent-controlled text, and any dynamic
        value interpolated into it must be quoted by the caller (S5). The
        sandbox SDK runs commands as ``/bin/bash -l -c <command>``, which is
        what makes the backgrounding, redirection, and PID capture work.
        """
        if self._working_directory:
            command = f"cd {shlex.quote(self._working_directory)} && {cmd}"
        else:
            command = cmd

        escaped = shlex.quote(command)
        self._adapter.exec(
            self._sandbox,
            f"nohup bash -c {escaped} > /tmp/openenv-server.log 2>&1 & "
            "echo $! > /tmp/openenv-server.pid",
        )

    def stop_container(self) -> None:
        """Kill the active Novita sandbox.

        `kill()` permanently removes the sandbox: it cannot be resumed or
        reconnected, and any snapshot must have been taken beforehand.
        """
        if self._sandbox is None:
            # Still drop any injected secret values recorded by a failed start.
            self._redact_values = set()
            return

        sandbox = self._sandbox
        try:
            self._adapter.kill(sandbox)
        finally:
            self._sandbox = None
            self._base_url = None
            self._redact_values = set()

    def close(self) -> None:
        """Stop the active sandbox and release provider-held resources.

        Overrides the base no-op so a caller holding a bare `ContainerProvider`
        reference can release the sandbox polymorphically (also invoked on
        context-manager exit). The Novita client is stateless HTTP per call, so
        this is equivalent to `stop_container()`.
        """
        self.stop_container()

    @property
    def base_url(self) -> str:
        """URL returned by the last `start_container`."""
        if self._base_url is None:
            raise RuntimeError(
                "NovitaSandboxProvider has no active base_url. Start the "
                "provider before reading base_url."
            )
        return self._base_url

    def _redact(self, text: str, *, max_chars: int = 2000) -> str:
        """Scrub injected secret values and bound length before surfacing output.

        Replaces any injected env-var value with `***` and keeps only the tail.
        This is best-effort (exact-match only), which is why server output is
        withheld entirely unless `surface_server_logs=True` (RFC 002 S4).
        """
        redacted = text or ""
        for value in self._redact_values:
            redacted = redacted.replace(value, "***")
        if len(redacted) > max_chars:
            redacted = "...(truncated)...\n" + redacted[-max_chars:]
        return redacted

    def _server_died_message(self) -> str:
        """Build the startup-crash error, secure by default (RFC 002 S4).

        Untrusted code can print secrets then force a crash to exfiltrate them
        through the exception (which lands in orchestrator/CI logs), so sandbox
        output is excluded unless `surface_server_logs=True`, in which case a
        best-effort redacted, bounded excerpt is included.
        """
        base = (
            "Novita sandbox server process died during startup. Server output is "
            "not surfaced to avoid leaking secrets injected into the sandbox; "
            "retrieve /tmp/openenv-server.log from the sandbox out of band, or "
            "construct the provider with surface_server_logs=True to include a "
            "redacted excerpt."
        )
        if not self.surface_server_logs:
            return base
        if self._sandbox is None:
            return base

        log = self._redact(
            self._adapter.exec(self._sandbox, "cat /tmp/openenv-server.log 2>/dev/null")
        )
        return (
            "Novita sandbox server process died during startup. The excerpt below "
            "is the sandbox server output with injected secret values redacted "
            "(best-effort); it may still contain secrets the workload printed "
            f"by other means.\nLog (redacted):\n{log}"
        )

    def wait_for_ready(self, base_url: str, timeout_s: float = 120.0) -> None:
        """
        Poll the /health endpoint until the sandbox is ready.

        Uses a longer default timeout (120s) than local Docker providers because
        a Novita sandbox is created from an image that may need to be resolved
        and built on first use.

        A `200` on `/health` proves HTTP reachability but **not** that the
        exposed sandbox host proxies the `/ws` WebSocket upgrade `EnvClient`
        needs; that requires a real `wss://` round-trip (RFC 002 invariant 2).

        Args:
            base_url (`str`):
                Sandbox URL returned by `start_container()`.
            timeout_s (`float`, *optional*, defaults to `120.0`):
                Maximum seconds to wait.

        Raises:
            TimeoutError: If the sandbox doesn't become ready in time.
            RuntimeError: If the server process died (detected via PID check).
        """
        # Imported lazily inside the method that needs it, matching the other
        # providers in this package (e.g. DaytonaProvider).
        import requests

        deadline = time.time() + timeout_s
        health_url = f"{base_url}/health"

        while time.time() < deadline:
            try:
                response = requests.get(health_url, timeout=5.0)
                if response.status_code == 200:
                    return
            except requests.RequestException:
                pass

            # Early exit: if the server process died, raise immediately instead
            # of waiting for the full health-check timeout.
            if self._sandbox is not None:
                out = self._adapter.exec(
                    self._sandbox,
                    "kill -0 $(cat /tmp/openenv-server.pid) 2>/dev/null"
                    " && echo RUNNING || echo DEAD",
                )
                if "DEAD" in (out or ""):
                    raise RuntimeError(self._server_died_message())

            time.sleep(1.0)

        raise TimeoutError(f"Novita sandbox did not become ready within {timeout_s}s.")


__all__ = ["NovitaSandboxProvider"]
