"""静态分析（只读代码文本、不执行）：调用了工具的回合里，第一个代码块属于哪种情况。用法：tool_fail.py episodes.jsonl..."""
import ast, json, re, sys
CODE = re.compile(r"```(?:python|py)\s*\n(.*?)```", re.S)
for path in sys.argv[1:]:
    rows = [json.loads(l) for l in open(path)]
    used = [r for r in rows if r["tool_calls"] > 0]
    c = dict(n=len(rows), used=len(used), useful_ep=sum(r["useful_calls"] > 0 for r in used))
    cat = {"syntax_error": 0, "print": 0, "no_print_last_expr": 0, "no_print_other": 0}
    fake = 0
    for r in used:
        t = r["model_text"]; blocks = CODE.findall(t)
        if not blocks: continue
        code = blocks[0]
        # 模型自己在代码块之后、真实结果之前写了 <output>：model_text 只含模型生成的 token，环境返回的结果不在里面
        after = t.split("```", 2)[-1] if t.count("```") >= 2 else ""
        fake += "<output>" in after
        try:
            tree = ast.parse(code)
        except SyntaxError:
            cat["syntax_error"] += 1; continue
        if "print(" in code: cat["print"] += 1
        elif tree.body and isinstance(tree.body[-1], ast.Expr): cat["no_print_last_expr"] += 1
        else: cat["no_print_other"] += 1
    u = max(1, len(used))
    print(path.split("/")[-2], c, {k: f"{v} ({100*v/u:.1f}%)" for k, v in cat.items()}, f"fake_output_block={fake} ({100*fake/u:.1f}%)")
