#!/usr/bin/env python3
"""
Build the 知乎 AI 问答分析 Dify workflow YAML.
Generates → validates → saves.
"""

import copy
import json
import sys
import uuid

sys.path.insert(0, ".")
from dify_dsl_generator import validate_dify_yaml, save_yaml, DifyValidationError

_sid = lambda s: str(uuid.uuid5(uuid.NAMESPACE_DNS, s))

# ---------------------------------------------------------------------------
# Base template
# ---------------------------------------------------------------------------

_BASE = {
    "app": {
        "mode": "workflow",
        "name": "知乎AI问答分析器",
        "description": "从知乎JSONL回答中批量评分、筛选高价值回答，提炼分类AI技巧指南",
        "icon": "📊",
        "icon_background": "#1E64F0",
        "use_icon_as_answer_icon": False,
    },
    "kind": "app",
    "version": "0.1.5",
    "workflow": {
        "version": "0.1.0",
        "conversation_variables": [],
        "environment_variables": [],
        "features": {
            "file_upload": {"enabled": True},
            "opening_statement": "",
            "retriever_resource": {"enabled": False},
            "sensitive_word_avoidance": {"enabled": False},
            "speech_to_text": {"enabled": False},
            "suggested_questions": [],
            "suggested_questions_after_answer": {"enabled": False},
            "text_to_speech": {"enabled": False},
        },
        "graph": {"nodes": [], "edges": []},
    },
}


def node(pos, nid, data, h=54, w=244):
    x, y = pos
    return {
        "id": nid, "type": "custom", "width": w, "height": h,
        "position": {"x": x, "y": y},
        "positionAbsolute": {"x": x, "y": y},
        "data": dict(data, desc="", selected=False),
    }


def edge(eid, src, tgt, src_type="", tgt_type="", in_iter=False):
    return {
        "id": eid, "source": src, "target": tgt,
        "sourceHandle": "source", "targetHandle": "target",
        "type": "custom", "zIndex": 0,
        "data": {
            "isInIteration": in_iter,
            "sourceType": src_type or "",
            "targetType": tgt_type or "",
        },
    }


# ===================================================================
# Build the workflow
# ===================================================================


def build() -> dict:
    wf = copy.deepcopy(_BASE)
    nodes = wf["workflow"]["graph"]["nodes"]
    edges = wf["workflow"]["graph"]["edges"]

    # ── Node 1: Start ──
    nodes.append(node((80, 300), "start", {
        "type": "start", "title": "开始 - 输入JSONL",
        "variables": [
            {"label": "raw_jsonl", "variable": "raw_jsonl",
             "required": True, "type": "paragraph",
             "max_length": 500000, "options": []},
            {"label": "question_title", "variable": "question_title",
             "required": False, "type": "text",
             "max_length": 500, "options": []},
        ],
    }))

    # ── Node 2: Code — Parse & Pre-filter ──
    nodes.append(node((380, 300), "code-parse", {
        "type": "code", "title": "解析JSONL & 预过滤",
        "code_language": "python3",
        "code": (
            "def main(raw_jsonl: str, question_title: str = '') -> dict:\n"
            "    import json\n"
            "    answers = []\n"
            "    question = question_title\n"
            "    lines = raw_jsonl.strip().split('\\n')\n"
            "    for line in lines:\n"
            "        if not line.strip():\n"
            "            continue\n"
            "        try:\n"
            "            obj = json.loads(line)\n"
            "        except json.JSONDecodeError:\n"
            "            continue\n"
            "        content = obj.get('content', '') or obj.get('text', '') or obj.get('answer', '')\n"
            "        if len(content) < 80:\n"
            "            continue\n"
            "        if len(content) > 8000:\n"
            "            content = content[:8000]\n"
            "        answers.append({\n"
            "            'id': str(obj.get('id', f'a{len(answers)}')),\n"
            "            'author': str(obj.get('author', '') or obj.get('name', '匿名')),\n"
            "            'content': content,\n"
            "            'upvotes': int(obj.get('upvotes', 0) or obj.get('vote_count', 0) or obj.get('likes', 0)),\n"
            "        })\n"
            "        if not question and obj.get('question'):\n"
            "            question = str(obj['question'])\n"
            "    answers.sort(key=lambda x: x['upvotes'], reverse=True)\n"
            "    total = len(answers)\n"
            "    batch_size = 25\n"
            "    batches = []\n"
            "    for i in range(0, total, batch_size):\n"
            "        batch = answers[i:i + batch_size]\n"
            "        batches.append(json.dumps({\n"
            "            'batch_id': i // batch_size + 1,\n"
            "            'total': (total + batch_size - 1) // batch_size,\n"
            "            'answers': batch\n"
            "        }, ensure_ascii=False))\n"
            "    return {\n"
            "        'total_raw': len(lines),\n"
            "        'after_filter': total,\n"
            "        'batches': batches,\n"
            "        'answers_all': json.dumps(answers, ensure_ascii=False),\n"
            "        'question': question or '关于使用AI的技巧',\n"
            "    }\n"
        ),
        "variables": [
            {"variable": "raw_jsonl", "value_selector": ["start", "raw_jsonl"]},
            {"variable": "question_title", "value_selector": ["start", "question_title"]},
        ],
        "outputs": {
            "total_raw": {"type": "number", "children": None},
            "after_filter": {"type": "number", "children": None},
            "batches": {"type": "array[string]", "children": None},
            "answers_all": {"type": "string", "children": None},
            "question": {"type": "string", "children": None},
        },
    }))

    # ── Node 3: Iteration — Batch Scoring ──
    nodes.append(node((680, 300), "iter-batch-score", {
        "type": "iteration", "title": "批量评分 (每批25条)",
        "isInIteration": False,
        "startNodeType": "start",
        "start_node_id": "start-batch",
        "iterator_selector": ["code-parse", "batches"],
        "output_selector": ["llm-score-batch", "text"],
        "children": {
            "nodes": [
                node((80, 162), "start-batch", {
                    "type": "start", "title": "每批开始",
                    "variables": [
                        {"label": "batch_json", "variable": "batch_json",
                         "required": True, "type": "text",
                         "max_length": 200000, "options": []},
                    ],
                }),
                node((380, 162), "llm-score-batch", {
                    "type": "llm", "title": "LLM评分",
                    "model": {
                        "provider": "openai",
                        "name": "gpt-4o-mini",
                        "mode": "chat",
                        "completion_params": {"temperature": 0.3},
                    },
                    "prompt_template": [
                        {
                            "id": _sid("sys-score"),
                            "role": "system",
                            "text": (
                                "你是内容质量评估专家。对一批知乎回答逐个打分（1-10分）。\n\n"
                                "评分标准：\n"
                                "9-10分：原创深度经验，具体可操作，有独特视角\n"
                                "7-8分：有实质内容，经验真实，比较详细\n"
                                "5-6分：泛泛而谈，有一点信息量但不够深入\n"
                                "3-4分：空洞重复，几乎没有新信息\n"
                                "1-2分：垃圾广告、灌水、完全无关\n\n"
                                "输出严格JSON数组（不要markdown代码块，不要任何其他文字）：\n"
                                '[{"id":"回答ID","score":8,"summary":"15字以内要点概括"}]\n'
                                "每个回答必须有id、score、summary三个字段。只输出JSON数组本身。"
                            ),
                        },
                        {
                            "id": _sid("usr-score"),
                            "role": "user",
                            "text": "请为以下这批知乎回答逐条评分，输出JSON数组：\n\n{{#start-batch.batch_json#}}",
                        },
                    ],
                    "context": {"enabled": False, "variable_selector": []},
                    "memory": {
                        "role_prefix": {"user": "", "assistant": ""},
                        "window": {"enabled": False, "size": 10},
                    },
                    "variables": [],
                }, h=98),
            ],
            "edges": [
                edge("e-batch-inner", "start-batch", "llm-score-batch",
                     src_type="start", tgt_type="llm", in_iter=True),
            ],
        },
        "variables": [],
    }))

    # ── Node 4: Code — Aggregate Scores & Filter ──
    nodes.append(node((980, 100), "code-aggregate", {
        "type": "code", "title": "汇总评分 & 精筛Top80",
        "code_language": "python3",
        "code": (
            "def main(iter_output: list, answers_all: str) -> dict:\n"
            "    import json\n"
            "    all_scored = []\n"
            "    for text in (iter_output or []):\n"
            "        t = text.strip()\n"
            "        if t.startswith('```'):\n"
            "            t = t.strip('`').replace('json', '', 1).strip()\n"
            "        try:\n"
            "            batch = json.loads(t)\n"
            "        except:\n"
            "            continue\n"
            "        if isinstance(batch, list):\n"
            "            all_scored.extend(batch)\n"
            "    all_scored.sort(key=lambda x: x.get('score', 0), reverse=True)\n"
            "    top80 = all_scored[:80]\n"
            "    top_ids = {item['id'] for item in top80}\n"
            "    try:\n"
            "        answers = json.loads(answers_all)\n"
            "    except:\n"
            "        answers = []\n"
            "    top_answers = [a for a in answers if a.get('id') in top_ids]\n"
            "    id_order = {item['id']: i for i, item in enumerate(top80)}\n"
            "    top_answers.sort(key=lambda a: id_order.get(a.get('id'), 999))\n"
            "    dist = {}\n"
            "    for item in all_scored:\n"
            "        s = item.get('score', 0)\n"
            "        dist[str(s)] = dist.get(str(s), 0) + 1\n"
            "    ranked_md = '# 知乎AI技巧回答 — 价值排名\\n\\n'\n"
            "    ranked_md += f'共评估 {len(all_scored)} 条回答，原始 {len(answers)} 条\\n\\n'\n"
            "    ranked_md += '| 排名 | 回答ID | 评分 | 摘要 |\\n|------|--------|------|------|\\n'\n"
            "    for i, item in enumerate(top80[:50], 1):\n"
            "        ranked_md += f\"| {i} | {item['id']} | {item['score']}/10 | {item.get('summary', '')} |\\n\"\n"
            "    top_contents = []\n"
            "    for a in top_answers[:80]:\n"
            "        top_contents.append(\n"
            "            f\"### [{a.get('id')}] {a.get('author', '匿名')} (赞同{a.get('upvotes', 0)})\\n\"\n"
            "            f\"{a.get('content', '')}\\n\"\n"
            "        )\n"
            "    return {\n"
            "        'top80_ids': list(top_ids)[:80],\n"
            "        'ranked_markdown': ranked_md,\n"
            "        'top_contents': '\\n---\\n'.join(top_contents),\n"
            "        'score_distribution': json.dumps(dist, ensure_ascii=False),\n"
            "        'total_evaluated': len(all_scored),\n"
            "        'avg_score': round(sum(i.get('score', 0) for i in all_scored) / max(len(all_scored), 1), 1),\n"
            "    }\n"
        ),
        "variables": [
            {"variable": "iter_output",
             "value_selector": ["iter-batch-score", "output"]},
            {"variable": "answers_all",
             "value_selector": ["code-parse", "answers_all"]},
        ],
        "outputs": {
            "top80_ids": {"type": "array[string]", "children": None},
            "ranked_markdown": {"type": "string", "children": None},
            "top_contents": {"type": "string", "children": None},
            "score_distribution": {"type": "string", "children": None},
            "total_evaluated": {"type": "number", "children": None},
            "avg_score": {"type": "number", "children": None},
        },
    }))

    # ── Node 5: LLM — Deep Analysis & Categorization ──
    nodes.append(node((680, 500), "llm-deep-analysis", {
        "type": "llm", "title": "深度提炼 & 分类合成",
        "model": {
            "provider": "openai",
            "name": "gpt-4o",
            "mode": "chat",
            "completion_params": {"temperature": 0.5},
        },
        "prompt_template": [
            {
                "id": _sid("sys-deep"),
                "role": "system",
                "text": (
                    "你是AI使用技巧的研究员和内容策划。请分析以下知乎高评分回答，完成两个任务：\n\n"
                    "## 任务1：按类别整理AI使用技巧\n"
                    "将提取的技巧归入以下类别（同类合并去重）：\n"
                    "- 📝 写作与内容创作\n"
                    "- 💻 编程与开发\n"
                    "- 🎨 设计与创意\n"
                    "- 📊 数据分析与研究\n"
                    "- ⚡ 效率与自动化\n"
                    "- 🎯 提示词工程 (Prompt Engineering)\n"
                    "- 🧠 学习与教育\n"
                    "- 💼 职场与商业\n"
                    "- 🔧 工具与插件推荐\n"
                    "- 🏆 综合/其他\n\n"
                    "每个技巧格式：\n"
                    "- **技巧名称** (来源回答ID, 赞同数): 具体操作描述 | 适用场景 | 难度⭐~⭐⭐⭐\n\n"
                    "## 任务2：生成综合指南\n"
                    "- 提炼出现频率最高的 Top 10 核心技巧\n"
                    "- 不同水平用户的建议分布（新手/进阶/专家各给什么建议）\n"
                    "- 「如果只记3条，就记这3条」\n\n"
                    "输出Markdown格式，层级清晰，可直接阅读发布。"
                ),
            },
            {
                "id": _sid("usr-deep"),
                "role": "user",
                "text": (
                    "以下是知乎问题「{{#code-parse.question#}}」下的高评分回答原文：\n\n"
                    "{{#code-aggregate.top_contents#}}\n\n"
                    "请按系统指令中的要求，完成分类整理和综合指南。"
                ),
            },
        ],
        "context": {"enabled": False, "variable_selector": []},
        "memory": {
            "role_prefix": {"user": "", "assistant": ""},
            "window": {"enabled": False, "size": 10},
        },
        "variables": [],
    }, h=98))

    # ── Node 6: Template Transform — Merge Final Output ──
    nodes.append(node((980, 500), "template-merge", {
        "type": "template-transform", "title": "合并最终报告",
        "template": (
            "{{ranked}}\n\n"
            "---\n\n"
            "## 📊 统计概览\n\n"
            "- 原始回答数：**{{total_raw}}** 条\n"
            "- 有效筛选后：**{{after_filter}}** 条\n"
            "- 综合评估数：**{{evaluated}}** 条\n"
            "- 平均评分：**{{avg}}/10**\n"
            "- 分数分布：{{dist}}\n\n"
            "---\n\n"
            "{{deep_analysis}}"
        ),
        "variables": [
            {"variable": "ranked",
             "value_selector": ["code-aggregate", "ranked_markdown"]},
            {"variable": "deep_analysis",
             "value_selector": ["llm-deep-analysis", "text"]},
            {"variable": "total_raw",
             "value_selector": ["code-parse", "total_raw"]},
            {"variable": "after_filter",
             "value_selector": ["code-parse", "after_filter"]},
            {"variable": "evaluated",
             "value_selector": ["code-aggregate", "total_evaluated"]},
            {"variable": "avg",
             "value_selector": ["code-aggregate", "avg_score"]},
            {"variable": "dist",
             "value_selector": ["code-aggregate", "score_distribution"]},
        ],
    }))

    # ── Node 7: End ──
    nodes.append(node((1280, 400), "end", {
        "type": "end", "title": "输出最终报告",
        "outputs": [
            {"value_selector": ["template-merge", "output"],
             "variable": "final_report"},
            {"value_selector": ["code-aggregate", "ranked_markdown"],
             "variable": "ranked_list"},
            {"value_selector": ["llm-deep-analysis", "text"],
             "variable": "tips_guide"},
        ],
    }))

    # ── Edges (top-level) ──
    edges.extend([
        edge("e-start-parse", "start", "code-parse",
             src_type="start", tgt_type="code"),
        edge("e-parse-iter", "code-parse", "iter-batch-score",
             src_type="code", tgt_type="iteration"),
        edge("e-iter-aggr", "iter-batch-score", "code-aggregate",
             src_type="iteration", tgt_type="code"),
        edge("e-aggr-deep", "code-aggregate", "llm-deep-analysis",
             src_type="code", tgt_type="llm"),
        edge("e-deep-template", "llm-deep-analysis", "template-merge",
             src_type="llm", tgt_type="template-transform"),
        edge("e-template-end", "template-merge", "end",
             src_type="template-transform", tgt_type="end"),
    ])

    return wf


if __name__ == "__main__":
    wf = build()
    print("Validating...")
    try:
        warnings = validate_dify_yaml(wf)
        if warnings:
            for w in warnings:
                print(f"  [WARNING] {w}")
        print("[OK] Validation PASSED")
        save_yaml(wf, "zhihu_ai_analysis_workflow.yml")
        print(f"  Total nodes: {len(wf['workflow']['graph']['nodes'])}")
        print(f"  Total edges: {len(wf['workflow']['graph']['edges'])}")
    except DifyValidationError as e:
        print(f"[FAIL] Validation FAILED: {e}")
        sys.exit(1)
