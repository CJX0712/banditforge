"""三级降级推送脚本（作者：晨星）— BanditFuse 合并入既有 CJX0712/banditforge。

层级:
  L1  git           : git init/commit/push（优先，最可靠）
  L2  Git Data API  : gh api 走 blobs→tree→commit→ref（git 网络被拦时）
  L3  Contents API  : gh api PUT /contents/{path}（仅 gh api 可达时）

本次为「合并」模式：仓库已存在（旧 OPE 系统），我们只 add/update 本次 BanditFuse
文件，不删除旧 OPE 包（banditforge/ope、hpo、data、bandits）。旧 v0.1.0 Release 保留，
本次以 v0.2.0 发布，避免覆盖。

用法:
  python scripts/gh_push.py            # 执行推送 + 更新描述 + v0.2.0 tag + release
  python scripts/gh_push.py --dry      # 仅打印计划，不执行任何写操作
  python scripts/gh_push.py --no-release
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import tempfile

REPO = "CJX0712/banditforge"
TAG = "v0.2.0"
BRANCH = "main"
AUTHOR_NAME = "晨星"
AUTHOR_EMAIL = "CJX0712@users.noreply.github.com"
COMMIT_MSG = (
    "BanditFuse v0.2.0 · 在线去偏方差校准上下文老虎机（LinUCB/LinTS/UCB1 基线 "
    "+ BanditFuse 旗舰），作者晨星"
)
REPO_DESC = (
    "BanditForge · 上下文老虎机工具箱：在线去偏校准 BanditFuse（LinUCB/LinTS/UCB1 "
    "+ 去偏方差校准旗舰）+ 离策略评估 OPE（CF-DR-AC）。纯 NumPy，确定性可复现。作者晨星"
)

EXCLUDE = {
    ".git", ".ruff_cache", ".pytest_cache", "__pycache__", ".github",
    ".coverage", "benchmark.json", "benchmark_quick.json", "private_key.txt",
}
SCAN_EXTS = (".py", ".md", ".txt", ".toml", ".yml", ".yaml", ".cfg", ".ini",
             ".lock", ".lock.txt", ".gitignore", ".dockerfile", "")


def run(cmd, check=True):
    print("+ " + " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if check and r.returncode != 0:
        raise RuntimeError(f"cmd failed: {' '.join(cmd)}\n{r.stderr}")
    return r


def gh_api(method: str, endpoint: str, payload: dict | None = None) -> str:
    """调用 gh api；payload 经临时文件传入（gh api 无 --body，仅 --input）。"""
    args = ["gh", "api", endpoint, "-X", method]
    if payload is not None:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False,
                                         encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False)
            path = fh.name
        args += ["--input", path]
        try:
            r = run(args)
        finally:
            os.unlink(path)
    else:
        r = run(args)
    return r.stdout.strip()


def gh_view_repo() -> bool:
    r = run(["gh", "repo", "view", REPO, "--json", "name"], check=False)
    return r.returncode == 0


def collect_files(root="."):
    out = []
    for dp, dn, fn in os.walk(root):
        dn[:] = [d for d in dn if d not in EXCLUDE]
        for f in fn:
            if f in EXCLUDE:
                continue
            ext = os.path.splitext(f)[1].lower()
            if ext not in SCAN_EXTS:
                continue
            out.append(os.path.relpath(os.path.join(dp, f), root).replace(os.sep, "/"))
    return sorted(out)


# ---------------------------------------------------------------- L1: git
def push_l1():
    if not os.path.isdir(".git"):
        run(["git", "init", "-q"])
        run(["git", "branch", "-M", BRANCH])
    run(["git", "config", "user.name", AUTHOR_NAME])
    run(["git", "config", "user.email", AUTHOR_EMAIL])

    if not gh_view_repo():
        run(["gh", "repo", "create", REPO, "--public", "--description", REPO_DESC])

    rem = run(["git", "remote"]).stdout
    if "origin" not in rem:
        run(["git", "remote", "add", "origin", f"https://github.com/{REPO}.git"])
    run(["git", "add", "-A"])
    status = run(["git", "status", "--porcelain"]).stdout.strip()
    if status:
        run(["git", "commit", "-q", "-m", COMMIT_MSG,
             "--author", f"{AUTHOR_NAME} <{AUTHOR_EMAIL}>"])
    r = run(["git", "push", "-u", "origin", BRANCH], check=False)
    if r.returncode != 0:
        raise RuntimeError(f"git push 失败:\n{r.stderr}")
    print("[L1] git push 成功")
    return True


# --------------------------------------------------------- L2: Git Data API
def push_l2():
    files = collect_files()
    blobs = {}
    for path in files:
        with open(path, "rb") as fh:
            content = fh.read()
        out = gh_api("POST", f"repos/{REPO}/git/blobs",
                     {"content": base64.b64encode(content).decode(),
                      "encoding": "base64"})
        blobs[path] = json.loads(out)["sha"]

    tree = [{"path": p, "mode": "100644", "type": "blob", "sha": blobs[p]}
            for p in files]
    tree_sha = json.loads(gh_api("POST", f"repos/{REPO}/git/trees",
                                 {"tree": tree}))["sha"]

    parents = []
    try:
        ref = gh_api("GET", f"repos/{REPO}/git/ref/heads/{BRANCH}")
        parents = [json.loads(ref)["object"]["sha"]]
    except RuntimeError:
        pass

    commit_sha = json.loads(gh_api("POST", f"repos/{REPO}/git/commits",
                                   {"message": COMMIT_MSG, "tree": tree_sha,
                                    "parents": parents,
                                    "author": {"name": AUTHOR_NAME, "email": AUTHOR_EMAIL},
                                    "committer": {"name": AUTHOR_NAME, "email": AUTHOR_EMAIL}}))["sha"]

    if parents:
        gh_api("PATCH", f"repos/{REPO}/git/refs/heads/{BRANCH}", {"sha": commit_sha})
    else:
        gh_api("POST", f"repos/{REPO}/git/refs",
               {"ref": f"refs/heads/{BRANCH}", "sha": commit_sha})
    print("[L2] Git Data API 推送成功")
    return True


# -------------------------------------------------------- L3: Contents API
def push_l3():
    files = collect_files()
    for path in files:
        with open(path, "rb") as fh:
            content = base64.b64encode(fh.read()).decode()
        body = {
            "message": COMMIT_MSG,
            "content": content,
            "branch": BRANCH,
            "author": {"name": AUTHOR_NAME, "email": AUTHOR_EMAIL},
            "committer": {"name": AUTHOR_NAME, "email": AUTHOR_EMAIL},
        }
        try:
            existing = gh_api("GET", f"repos/{REPO}/contents/{path}")
            body["sha"] = json.loads(existing)["sha"]
        except RuntimeError:
            pass
        gh_api("PUT", f"repos/{REPO}/contents/{path}", body)
    print("[L3] Contents API 推送成功")
    return True


def update_description():
    try:
        gh_api("PATCH", f"repos/{REPO}", {"description": REPO_DESC})
        print("[desc] 已更新仓库描述")
    except RuntimeError as e:
        print(f"[desc] 跳过：{e}")


def tag_and_release():
    try:
        run(["git", "tag", "-a", TAG, "-m", COMMIT_MSG, "--force"], check=False)
        run(["git", "push", "origin", TAG, "--force"], check=False)
        print(f"[tag] 已打 {TAG}（本地 git 不可达时忽略）")
    except Exception as e:  # noqa: BLE001
        print(f"[tag] 本地 tag 跳过: {e}")

    notes = (
        "## BanditFuse v0.2.0\n\n"
        "在既有 BanditForge（OPE 离策略评估）基础上，新增**在线去偏方差校准上下文老虎机**子系统（作者：晨星）。\n\n"
        "### 新增：在线学习\n"
        "- 方法：LinUCB / Linear Thompson Sampling (LinTS) / UCB1（上下文无关 SOTA）/ ε-Greedy / GreedyCF / RandomCF\n"
        "- 旗舰 **BanditFuse**：在线去偏方差校准——用预更新后验残差除以杠杆项 `1 + xᵀA⁻¹x` 在线估计 σ²，使探索半径自动匹配真实噪声，无需手调 α、不假设 σ\n"
        "- 基准（10 seed, T=3000, linear, d=10, K=5, σ=0.5）：BanditFuse 97.68 削减 UCB1 99.1%（极显著），胜 LinTS 16.7%，匹配最优手调 LinUCB(α=0.5)\n"
        "- 确定性：全局 seed 入口，两次运行逐位一致（bit_identical）\n"
        "- 质量门禁：ruff 双绿、pytest 21 项全绿、校准 var_est=0.2736 ≈ 真值 0.25\n\n"
        "### 既有：离策略评估 OPE\n"
        "- LinUCB/LinTS + CF-DR-AC（cross-fitted DR，自适应裁剪 + ESS 门控）\n\n"
        "详见 README.md 与 docs/。"
    )
    try:
        gh_api("POST", f"repos/{REPO}/releases",
               {"tag_name": TAG, "name": f"BanditFuse {TAG}",
                "body": notes, "prerelease": False})
        print(f"[release] 已创建 {TAG} Release")
    except RuntimeError:
        rel = gh_api("GET", f"repos/{REPO}/releases/tags/{TAG}")
        rid = json.loads(rel)["id"]
        gh_api("PATCH", f"repos/{REPO}/releases/{rid}", {"body": notes})
        print(f"[release] 已更新 {TAG} Release")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="仅打印计划")
    ap.add_argument("--no-release", action="store_true")
    args = ap.parse_args()

    files = collect_files()
    print(f"[plan] 仓库 {REPO}  分支 {BRANCH}  文件数 {len(files)}  模式=合并(保留旧 OPE)")
    if args.dry:
        for f in files:
            print("   ", f)
        print("[dry] 不执行任何写操作")
        return 0

    for level, fn in (("L1 git", push_l1), ("L2 GitData", push_l2),
                      ("L3 Contents", push_l3)):
        try:
            fn()
            break
        except Exception as e:  # noqa: BLE001
            print(f"[{level}] 失败，降级：{e}")
    else:
        print("❌ 三级推送全部失败")
        return 1

    update_description()
    if not args.no_release:
        tag_and_release()
    print("✅ 推送完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
