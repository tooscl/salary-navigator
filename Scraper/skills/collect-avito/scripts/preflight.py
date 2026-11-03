"""Read-only local checks for the agent's Aside workflow; never visits Avito."""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG = ROOT / "Scraper/collect-avito.local.json"


def saved_config_path():
    if os.name == "nt":
        directory = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "T&L/collect-avito"
    else:
        directory = Path(os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local/share"))) / "tl-avito"
    return directory / "collector.local.json"


def default_config_path():
    """Reuse the known shared file across fresh checkouts, without scanning disk."""
    explicit = os.environ.get("TL_AVITO_CONFIG")
    if explicit:
        return Path(explicit).expanduser()
    if DEFAULT_CONFIG.exists():
        return DEFAULT_CONFIG
    shared = saved_config_path()
    return shared if shared.exists() else DEFAULT_CONFIG


def check_xray(config):
    """Conservative static check: explicit domain rules, order and direct outbound.

    This cannot verify the configuration loaded by the running tunnel, DNS/IP
    matching or all Xray routing semantics. Ambiguous earlier rules need review.
    """
    if not isinstance(config, dict):
        return ["Корень Xray-конфигурации должен быть объектом."]
    routing = config.get("routing", {})
    rules = routing.get("rules", []) if isinstance(routing, dict) else None
    outbounds = config.get("outbounds", [])
    if (not isinstance(rules, list) or not all(isinstance(r, dict) for r in rules)
            or not isinstance(outbounds, list)
            or not all(isinstance(o, dict) for o in outbounds)):
        return ["Неизвестная структура routing.rules или outbounds."]
    tags = [o.get("tag") for o in outbounds]
    if not all(isinstance(tag, str) and tag for tag in tags) or len(tags) != len(set(tags)):
        return ["У выходов должны быть непустые уникальные теги; проверьте outbounds."]
    direct = {o["tag"] for o in outbounds if o.get("protocol") == "freedom"}
    errors = []
    for domain in ("avito.ru", "avito.st"):
        found = False
        for index, rule in enumerate(rules):
            network = rule.get("network", "tcp,udp")
            if not isinstance(network, str) or not set(network.split(",")) <= {"tcp", "udp"}:
                errors.append(f"{domain}: проверьте network в правиле {index + 1}.")
                break
            if network == "udp":
                continue  # A UDP-only block does not shadow a TCP direct rule.
            domains = rule.get("domain", [])
            if not isinstance(domains, list) or not all(isinstance(d, str) for d in domains):
                errors.append(f"{domain}: проверьте domain в правиле {index + 1}.")
                break
            explicit = f"domain:{domain}" in domains
            # Additional conditions may narrow an otherwise matching domain rule.
            conditions = set(rule) - {"type", "domain", "outboundTag", "ruleTag", "network"}
            if explicit and not conditions and rule.get("type") == "field":
                if rule.get("outboundTag") in direct and not rule.get("balancerTag"):
                    found = True
                else:
                    errors.append(f"{domain}: первое явное правило не ведёт в freedom/direct.")
                break
            # Only clearly unrelated full/domain rules can be skipped safely.
            unrelated = bool(domains) and all(
                (d.startswith("full:") and not (d[5:] == domain or d[5:].endswith("." + domain)))
                or (d.startswith("domain:") and not (
                    domain == d[7:] or domain.endswith("." + d[7:])
                    or d[7:].endswith("." + domain)))
                for d in domains
            )
            if not unrelated:
                errors.append(f"{domain}: правило {index + 1} перед direct требует проверки порядка/условий.")
                break
        else:
            errors.append(f"{domain}: нет явного доменного правила direct с покрытием поддоменов.")
        if not found and not any(e.startswith(domain + ":") for e in errors):
            errors.append(f"{domain}: прямой маршрут не подтверждён конфигурацией.")
    return errors


def check_happ(config):
    """Only validates a routing profile, never its activation or subscription."""
    if not isinstance(config, dict) or not isinstance(config.get("Name"), str) or not config["Name"].strip():
        return ["Нужен именованный JSON-профиль маршрутизации Happ."]
    sites = config.get("DirectSites", [])
    order = config.get("RouteOrder", "block-proxy-direct")
    if not isinstance(order, str) or order not in (
            "direct-block-proxy", "direct-proxy-block"):
        return ["Direct должен проверяться первым; иначе proxy/block могут перекрыть Авито."]
    if not isinstance(sites, list) or not all(
            d in sites for d in ("domain:avito.ru", "domain:avito.st")):
        return ["DirectSites должен содержать domain:avito.ru и domain:avito.st."]
    return []


def find_browse_skill(root=ROOT, explicit=None):
    codex = Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex")))
    candidates = [root / ".agents/skills/gstack/browse/SKILL.md",
                  root / ".claude/skills/gstack/browse/SKILL.md",
                  root / ".claude/skills/browse/SKILL.md",
                  codex / "skills/gstack/browse/SKILL.md",
                  codex / "skills/gstack-browse/SKILL.md",
                  Path.home() / ".claude/skills/gstack/browse/SKILL.md",
                  Path.home() / ".claude/skills/browse/SKILL.md",
                  Path.home() / ".agents/skills/gstack/browse/SKILL.md"]
    if os.environ.get("GSTACK_ROOT"):
        candidates.insert(0, Path(os.environ["GSTACK_ROOT"]) / "browse/SKILL.md")
    override = explicit or os.environ.get("TL_BROWSE_SKILL")
    if override:
        # An invalid explicit choice must not silently validate another installation.
        if not isinstance(override, (str, os.PathLike)):
            return None
        selected = Path(override).expanduser()
        return selected if selected.is_file() else None
    return next((p for p in candidates if p.is_file()), None)


def find_aside():
    executable = shutil.which("aside")
    for name in ("aside", "aside.exe", "aside.cmd"):
        fallback = Path.home() / ".local/bin" / name
        if not executable and fallback.is_file() and os.access(fallback, os.X_OK):
            executable = str(fallback)
    return executable


def aside_command(executable):
    """Select a connected profile without changing Aside's persistent default."""
    host = os.environ.get("TL_ASIDE_HOST", "local")
    prefix = [executable, "--host", host]
    account = os.environ.get("TL_ASIDE_ACCOUNT")
    if not account:
        try:
            status = subprocess.run([executable, "account", "status"],
                                    capture_output=True, text=True, timeout=20)
            match = re.search(r"(?m)^\*?\s*(u\d+)\s", status.stdout)
            if status.returncode or not match:
                raise RuntimeError("Cannot determine Aside account; specify TL_ASIDE_ACCOUNT from aside account status")
            account = match[1]
        except (OSError, subprocess.TimeoutExpired) as error:
            raise RuntimeError("Cannot determine Aside account; inspect aside account status") from error
    prefix += ["--account", account]
    try:
        result = subprocess.run(prefix + ["host", "profiles", "--json"],
                                capture_output=True, text=True, timeout=20)
        if result.returncode:
            raise RuntimeError("Aside profiles unavailable; inspect host/account, not CLI installation")
        data = json.loads(result.stdout)
        profiles = data.get("profiles") if isinstance(data, dict) else None
        if not isinstance(profiles, list):
            raise ValueError("Invalid profiles list")
    except (ValueError, OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError("Cannot read Aside profiles; inspect aside host profiles --json") from error
    connected = [p["profileId"] for p in profiles if isinstance(p, dict)
                 and p.get("connected") is True and isinstance(p.get("profileId"), str)
                 and p["profileId"]]
    selected = os.environ.get("TL_ASIDE_PROFILE")
    if selected:
        if selected not in connected:
            raise RuntimeError("Selected TL_ASIDE_PROFILE is not connected; open that profile in Aside")
    elif len(connected) == 1:
        selected = connected[0]
    elif not connected:
        raise RuntimeError("No connected Aside profiles; open a browser profile, do not reinstall CLI")
    else:
        raise RuntimeError("Multiple connected Aside profiles; select one with TL_ASIDE_PROFILE")
    return prefix + ["--profile", selected]


def aside_ready(executable, *, strict=False):
    try:
        result = subprocess.run(
            aside_command(executable) + ["repl", 'console.log("TNL_ASIDE_READY")'],
            capture_output=True, text=True, timeout=20,
        )
        # Only static reasons are exposed: CLI output may contain private paths.
        ready = (result.returncode == 0
                 and "TNL_ASIDE_READY" in result.stdout.splitlines()
                 and "[error" not in (result.stdout + result.stderr).lower())
        if not ready and strict:
            raise RuntimeError("Aside REPL probe failed for the selected profile; inspect host/profile connection")
        return ready
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
        if strict:
            if isinstance(error, RuntimeError):
                raise
            raise RuntimeError("Aside REPL probe unavailable or timed out") from None
        return False


def run_checks(config_path=None, probe_aside=False, strict_routing=False):
    config_path = Path(config_path) if config_path is not None else default_config_path()
    checks = []

    def add(name, ok, detail, fix="", *, blocking=True):
        checks.append({"check": name, "ok": ok, "detail": detail, "blocking": blocking,
                       **({"next_step": fix} if not ok else {})})

    add("python", sys.version_info >= (3, 10), "Нужен Python 3.10+.",
        "Установите Python 3.10+ и повторите команду этим интерпретатором.")
    required = ["Scraper/extract.js", "Scraper/skills/collect-avito/SKILL.md",
                "Scraper/ASIDE_WORKFLOW.md", "Scraper/templates/manifest.template.json"]
    add("project", all((ROOT / p).is_file() for p in required),
        "Проверка наличия файлов навыка, экстрактора и паспорта.",
        "Обновите checkout проекта; не копируйте один SKILL.md отдельно от файлов.")
    try:
        settings = json.loads(Path(config_path).read_text(encoding="utf-8"))
        if not isinstance(settings, dict):
            settings = None
    except (OSError, ValueError):
        settings = None
    skill = find_browse_skill(explicit=settings.get("browse_skill_path") if settings else None)
    add("gstack_browse", skill is not None, "Проверено наличие инструкции gstack /browse.",
        "Откройте Scraper/skills/collect-avito/references/FIRST_RUN.md. Установите gstack /browse для своего агента; нестандартный путь задаётся через TL_BROWSE_SKILL.")
    aside = find_aside()
    add("aside_cli", bool(aside), "Проверено наличие Aside CLI.",
        "Установите Aside для своей ОС с https://aside.com/download и CLI по инструкции приложения; см. FIRST_RUN.md.")
    if probe_aside and aside:
        detail = "Проверка REPL без открытия вкладок и сайта."
        try:
            ready = aside_ready(aside, strict=True)
        except RuntimeError as error:
            ready = False
            detail += " " + str(error)
        add("aside_session", ready, detail,
            "Проверьте aside host profiles --host local --json. Если подключённых нет, откройте нужный профиль; если их несколько, задайте TL_ASIDE_PROFILE. TL_ASIDE_HOST/TL_ASIDE_ACCOUNT задают иной host/account. Повторная установка/login нужны только при подтверждённой ошибке установки/авторизации. См. FIRST_RUN.md.")

    if settings is None:
        add("local_config", False, "Локальные настройки отсутствуют или не читаются как JSON-объект.",
            "Для первого запуска: setup_workspace.py --mode disabled (без VPN) или --mode happ (профиль для импорта). Режим выбирается по вашей среде; см. FIRST_RUN.md.",
            blocking=strict_routing)
    if settings is not None:
        mode = settings.get("tunnel_mode")
        add("tunnel_mode", mode in ("disabled", "happ", "xray"),
            "Режим туннеля должен быть указан явно: disabled, happ или xray.",
            "Запустите setup_workspace.py с выбранным режимом по FIRST_RUN.md.",
            blocking=strict_routing)
        if mode == "disabled":
            add("tunnel_config", True, "В локальных настройках заявлена работа без туннеля; состояние ОС и маршрут не проверены.")
        elif mode == "happ":
            try:
                profile_path = Path(settings["happ_routing_path"]).expanduser()
                if not profile_path.is_absolute() or profile_path.resolve().is_relative_to(ROOT):
                    raise ValueError
                errors = check_happ(json.loads(profile_path.read_text(encoding="utf-8")))
                add("tunnel_config", not errors,
                    "Статическая проверка профиля Happ; импорт и работающий маршрут не проверяются.",
                    " ".join(errors) + " См. VPN_HAPP.md.", blocking=strict_routing)
            except (OSError, ValueError, TypeError, KeyError):
                add("tunnel_config", False, "Профиль Happ отсутствует или не читается.",
                    "Создайте профиль через setup_workspace.py --mode happ; затем импортируйте по VPN_HAPP.md.", blocking=strict_routing)
        elif mode == "xray":
            path = settings.get("xray_config_path")
            try:
                if not isinstance(path, str) or not path.strip():
                    raise ValueError
                private_path = Path(path).expanduser()
                if not private_path.is_absolute() or private_path.resolve().is_relative_to(ROOT):
                    raise ValueError
                config = json.loads(private_path.read_text(encoding="utf-8"))
                errors = check_xray(config)
                add("tunnel_config", not errors,
                    "Проверка явных TCP-правил avito.ru/avito.st и freedom-outbound в файле; работающий туннель не проверяется.",
                    " ".join(errors) + " См. Scraper/VPN_HAPP.md; проверьте актуальный профиль и повторите команду.",
                    blocking=strict_routing)
            except (OSError, ValueError, TypeError):
                add("tunnel_config", False, "Не удалось прочитать локальную Xray-конфигурацию.",
                    "Укажите абсолютный xray_config_path к актуальному JSON вне репозитория. Полный конфиг не публикуйте.",
                    blocking=strict_routing)
    blocked = any(not c["ok"] and c["blocking"] for c in checks)
    warnings = any(not c["ok"] and not c["blocking"] for c in checks)
    status = "NEEDS_SETUP" if blocked else "READY_WITH_WARNINGS" if warnings else "LOCAL_CHECKS_OK"
    return {"status": status, "checks": checks,
            "aside_session_checked": bool(probe_aside and aside),
            "can_collect": not blocked and bool(probe_aside and aside),
            "live_route_verified": False,
            "strict_routing": strict_routing,
            "next": "Перед сбором: --probe-aside, собственная сессия /browse, одна обычная загрузка сайта. Предупреждение о конфиге не означает отказ браузера. При блокировке сайта остановитесь; при resume сначала checkpoint. Маршрут остаётся неподтверждённым без отдельного измерения."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=default_config_path())
    parser.add_argument("--probe-aside", action="store_true",
                        help="Readiness-only REPL; no tabs, navigation or Avito requests")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--strict-routing", action="store_true",
                        help="Require a valid static routing config when auditing network setup")
    for target in ('host', 'account', 'profile'): parser.add_argument('--'+target, help='Explicit Aside target for this check')
    args = parser.parse_args()
    for target in ('host', 'account', 'profile'):
        value=getattr(args,target); key='TL_ASIDE_'+target.upper()
        if value:
            if os.environ.get(key) and os.environ[key]!=value: parser.error(f'--{target} differs from {key}')
            os.environ[key]=value
    result = run_checks(args.config, args.probe_aside, args.strict_routing)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(result["status"])
        for check in result["checks"]:
            label = "OK" if check["ok"] else "НУЖНО ДЕЙСТВИЕ" if check["blocking"] else "ПРЕДУПРЕЖДЕНИЕ"
            print(f"{label} · {check['check']}: {check['detail']}")
            if "next_step" in check:
                print("  " + check["next_step"])
        print(result["next"])
    return 2 if result["status"] == "NEEDS_SETUP" else 0


if __name__ == "__main__":
    raise SystemExit(main())
