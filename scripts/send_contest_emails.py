#!/usr/bin/env python3
"""
AgentMesh Dual-Contest Email Dispatcher
Supports creating drafts or directly sending submission emails via macOS Mail.app.
Strictly adheres to no-emoji constraint and official competition guidelines.
"""

import argparse
import os
import subprocess
import sys

BASE_DIR = "/Users/liuyukai/CREATE/OSCHINA"

SHANGHAI_CONFIG = {
    "name": "Shanghai",
    "to": "oscc@oschina.cn",
    "subject": "【智算云赛道】AgentMesh：面向企业生产级场景的高可靠开源智能体网格与运行时系统-作品材料提交",
    "pdf": os.path.join(BASE_DIR, "docs/shanghai/AgentMesh_Project_Proposal_Shanghai_2026.pdf"),
    "video": os.path.join(BASE_DIR, "docs/AgentMesh_Demo_Video_2026.mp4"),
    "body_template": """尊敬的 2026 上海开源软件应用创新大赛组委会专家：

您好！

由 PandaaX 申报、刘钰恺负责的参赛项目《AgentMesh：面向企业生产级场景的高可靠开源智能体网格与运行时系统》（参赛赛道：智算云赛道，企业赛题：未选择）已完成全部系统研发、基准压测与申报材料编制。现按照组委会提交指引，正式提交参赛材料：

1. 开源代码仓库：
https://github.com/aoright/AgentMesh
（项目遵循 Apache-2.0 商业友好开源协议，726 项自动化测试全部通过，附带完备架构设计与微服务网格文档）

2. 作品介绍文档 (PDF)：
详见附件《AgentMesh_Project_Proposal_Shanghai_2026.pdf》，全面阐述了持久化确定性执行运行时、MCP 2.0 / A2A 双协议网关、三维能力沙箱、高可靠金融风控与智算云 AIOps 架构方案。

3. 作品演示视频 (1080P MP4)：
详见附件《AgentMesh_Demo_Video_2026.mp4》（时长 3 分 40 秒）。
视频动态演示了系统初始化状态、长流程注入 OOM 崩溃后的秒级断点自愈（0.26ms）、已完成步骤 0.0% 重复计费、0.17ms 三态熔断切流及 8/8 类越权渗透物理防御全流程。
{video_link_section}
团队郑重承诺本项目所有技术方案与源码完全自主原创，恪守开源治理规范。期待参与后续技术评审与线下交流！

此致
敬礼！

申报单位：PandaaX
项目负责人：刘钰恺 (电话：15371017258)
申报邮箱：{sender_email}
日期：2026 年 10 月
"""
}

BEIJING_CONFIG = {
    "name": "Beijing",
    "to": "bjoscc@oschina.cn",
    "subject": "【开源基础软件与解决方案】AgentMesh：面向企业生产级场景的高可靠开源智能体网格与运行时系统-作品材料提交",
    "pdf": os.path.join(BASE_DIR, "docs/beijing/AgentMesh_Project_Proposal_Beijing_2026.pdf"),
    "video": os.path.join(BASE_DIR, "docs/AgentMesh_Demo_Video_2026.mp4"),
    "body_template": """尊敬的 2026 开源行业解决方案创新赛组委会专家：

您好！

由 PandaaX 申报、刘钰恺负责的参赛项目《AgentMesh：面向企业生产级场景的高可靠开源智能体网格与运行时系统》（参赛赛道：开源基础软件与解决方案，赛道一）已完成全部系统研发、基准压测与申报方案编制。现按照赛道一要求提交参赛材料：

1. 开源代码仓库：
https://github.com/aoright/AgentMesh
（基于 Apache-2.0 协议全量开源，包含完整可运行代码、726 项回归测试与自动化构建工具链）

2. 作品介绍文档 (PDF)：
详见附件《AgentMesh_Project_Proposal_Beijing_2026.pdf》，深入阐述了企业级智能体确定性运行时、SHA-256 密码学链事件溯源、微服务三态熔断保护、金融合规反洗钱审查方案及北京市“开源首方案”落地规划。

3. 作品演示视频 (1080P MP4)：
详见附件《AgentMesh_Demo_Video_2026.mp4》（时长 3 分 40 秒）。
视频完整记录了系统健康巡检、混沌工程破坏性注入崩溃后的自愈重放、Token 零重复消耗及三维能力沙箱防御的真实终端执行全过程。
{video_link_section}
团队郑重承诺本项目提交的所有技术方案与源码完全自主原创。期待在总决赛路演现场向专家评委做深度汇报！

此致
敬礼！

申报单位：PandaaX
项目负责人：刘钰恺 (电话：15371017258)
申报邮箱：{sender_email}
日期：2026 年 10 月
"""
}

def escape_applescript_string(text: str) -> str:
    return text.replace('\\', '\\\\').replace('"', '\\"').replace('\r\n', '\n').replace('\n', '\\n')

def generate_applescript(config: dict, sender: str, action: str, video_link: str = "") -> str:
    video_link_section = f"备用在线视频链接：{video_link}\n" if video_link else ""
    body = config["body_template"].format(
        video_link_section=video_link_section,
        sender_email=sender
    )

    pdf_escaped = config["pdf"].replace('"', '\\"')
    video_escaped = config["video"].replace('"', '\\"')
    subject_escaped = escape_applescript_string(config["subject"])
    body_escaped = escape_applescript_string(body)
    to_address = config["to"]

    is_send = (action == "send")
    visible_flag = "false" if is_send else "true"
    send_cmd = "send newMsg" if is_send else "-- draft created"

    script = f"""
    set pdfFile to POSIX file "{pdf_escaped}"
    set videoFile to POSIX file "{video_escaped}"
    tell application "Mail"
        set newMsg to make new outgoing message with properties {{subject:"{subject_escaped}", content:"{body_escaped}" & return & return, sender:"刘钰恺 <{sender}>", visible:{visible_flag}}}
        tell newMsg
            make new to recipient at end of to recipients with properties {{address:"{to_address}"}}
            tell content
                make new attachment with properties {{file name:pdfFile}} at after the last paragraph
                make new attachment with properties {{file name:videoFile}} at after the last paragraph
            end tell
        end tell
        save newMsg
        {send_cmd}
    end tell
    """
    return script

def run_mail_action(target: str, sender: str, action: str, video_link: str = ""):
    targets = []
    if target in ["all", "shanghai"]:
        targets.append(SHANGHAI_CONFIG)
    if target in ["all", "beijing"]:
        targets.append(BEIJING_CONFIG)

    for cfg in targets:
        print(f"Processing email for [{cfg['name']}] -> {cfg['to']} (action={action})...")
        script = generate_applescript(cfg, sender, action, video_link)
        proc = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"Error processing {cfg['name']}: {proc.stderr}", file=sys.stderr)
            return False
        else:
            verb = "Sent" if action == "send" else "Draft created"
            print(f"[SUCCESS] {verb} for {cfg['name']} to {cfg['to']}.")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Send or draft contest submission emails via macOS Mail.")
    parser.add_argument("--action", choices=["draft", "send"], default="draft", help="Action to take: draft (default) or send")
    parser.add_argument("--sender", default="aorightyan@gmail.com", help="Sender email address")
    parser.add_argument("--target", choices=["all", "shanghai", "beijing"], default="all", help="Target contest")
    parser.add_argument("--video-link", default="", help="Optional external video link (e.g. Netdisk / Bilibili)")
    args = parser.parse_args()

    success = run_mail_action(args.target, args.sender, args.action, args.video_link)
    sys.exit(0 if success else 1)
