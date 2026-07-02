import fitz
import os
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')

paper_dir = r"E:\graduation_paper\文献"
pdf_files = sorted([f for f in os.listdir(paper_dir) if f.endswith('.pdf')])

# ========== DOI prefix → Journal name mapping ==========
DOI_JOURNAL_MAP = {
    '10.3901/JME': ('机械工程学报', 'EI'),
    '10.3724/SP.J.1004': ('自动化学报', 'EI'),
    '10.16383/j.aas': ('自动化学报', 'EI'),
    '10.12086/oee': ('光电工程', 'EI'),
    '10.13195/j.kzyjc': ('控制与决策', 'EI'),
    '10.3785/j.issn': ('浙江大学学报(工学版)', '中文核心'),
    '10.3724/SP.J.1187': ('电子测量与仪器学报', '中文核心'),
    '10.19651/j.cnki.emt': ('电子测量技术', '中文核心'),
    '10.19781/j.issn.1673-9140': ('电力科学与技术学报', '中文核心'),
    '10.13433/j.cnki': ('机械科学与技术', '中文核心'),
    '10.19554/j.cnki': ('包装工程', '中文核心'),
    '10.19695/j.cnki': ('信息技术与网络安全', '普通期刊'),
    '10.19769/j.zdhy': ('自动化与仪器仪表', '普通期刊'),
    '10.14004/j.cnki.ckt': ('计算机技术与自动化', '普通期刊'),
    '10.19335/j.cnki.2095-6649': ('新型工业化', '普通期刊'),
    '10.3778/j.issn.1002-8331': ('计算机工程与应用', '中文核心'),
    '10.20137/j.qykjyfz': ('前沿科技与发展', '普通期刊'),
    '10.19768/j.cnki.dgjs': ('电工技术', '普通期刊'),
    '10.3969/j.issn.1006-4311': ('价值工程', '普通期刊'),
    '10.13462/j.cnki': ('机械设计与制造', '中文核心'),
    '10.13382/j.jemi': ('电子测量与仪器学报', '中文核心'),
    '10.15938/j.jhust': ('哈尔滨工业大学学报', 'EI'),
    '10.16080/j.issn': ('航空学报', 'EI'),
    '10.1187/j.issn': ('电子测量与仪器学报', '中文核心'),
    '10.16383/j.aas': ('自动化学报', 'EI'),
    '10.16183/j.cnki.jsjtu': ('上海交通大学学报', 'EI'),
}

# English journal name → Chinese
EN_JOURNAL_MAP = {
    'JOURNAL OF MECHANICAL ENGINEERING': ('机械工程学报', 'EI'),
    'ACTA AUTOMATICA SINICA': ('自动化学报', 'EI'),
    'ACTA SCIENTIARUM NATURALIUM UNIVERSITATIS SUNYATSENI': ('中山大学学报(自然科学版)', '中文核心'),
    'JOURNAL OF AIR FORCE ENGINEERING UNIVERSITY': ('空军工程大学学报', '中文核心'),
    'CONTROL AND DECISION': ('控制与决策', 'EI'),
    'OPTO ELECTRON ENG': ('光电工程', 'EI'),
    'JOURNAL OF COMPUTER APPLICATIONS': ('计算机应用', '中文核心'),
    'JOURNAL OF TEXTILE RESEARCH': ('纺织学报', '中文核心'),
    'CHINESE JOURNAL OF SCIENTIFIC INSTRUMENT': ('电子测量与仪器学报', '中文核心'),
}

# ========== ANALYSIS ==========
stats = {'期刊论文': 0, '学位论文': 0, '会议论文': 0, '未知': 0}
stats_level = {}
all_results = []

for pdf_file in pdf_files:
    path = os.path.join(paper_dir, pdf_file)
    try:
        doc = fitz.open(path)
        text = ""
        for page_num in range(min(2, len(doc))):
            text += doc[page_num].get_text()
        doc.close()
    except Exception as e:
        all_results.append({'file': pdf_file, 'type': '读取失败', 'journal': str(e),
                           'year': '', 'doi': '', 'domain': '', 'level': '', 'is_review': False})
        continue

    # ========== EXTRACT DOI ==========
    doi = ""
    doi_m = re.search(r'DOI[:\s]*\s*(10\.\d{4,}/[^\s\)\]】]+)', text[:1500], re.IGNORECASE)
    if doi_m:
        doi = doi_m.group(1).rstrip('.').rstrip(',').rstrip(';')

    # ========== DOI-BASED JOURNAL IDENTIFICATION ==========
    doi_journal = ""
    doi_level = ""
    for prefix, (jname, jlevel) in DOI_JOURNAL_MAP.items():
        if doi and prefix in doi:
            doi_journal = jname
            doi_level = jlevel
            break

    # ========== YEAR ==========
    year = ""
    year_m = re.search(r'(?:^|\D)(20\d{2})[年\-\s]', text[:600])
    if not year_m:
        year_m = re.search(r'收稿日期[：:\s]*(20\d{2})', text[:1500])
    if not year_m:
        year_m = re.search(r'录用日期[：:\s]*(20\d{2})', text[:1500])
    if year_m:
        year = year_m.group(1)

    # ========== JOURNAL NAME (from header) ==========
    header_journal = ""
    header_lines = text.split('\n')[:25]
    for line in header_lines:
        line = line.strip()
        # Match Chinese journal name
        if re.search(r'(大学学报|学报|期刊)', line):
            clean = re.sub(r'(?:第\d+卷|第\d+期|Vol\.?\s*\d+|20\d{2}).*', '', line).strip()
            if clean and 3 < len(clean) < 40:
                header_journal = clean
                break

    # Match English journal name
    if not header_journal:
        for line in header_lines:
            line = line.strip().upper()
            for en_name in EN_JOURNAL_MAP:
                if en_name in line:
                    header_journal, _ = EN_JOURNAL_MAP[en_name]
                    break
            if header_journal:
                break

    # ========== JOURNAL = choose best source ==========
    journal = doi_journal or header_journal or ""

    # If still no journal, try citation format
    if not journal:
        cite_m = re.search(r'引用格式.*?\[J\]\.\s*([^\s,，\d]+)', text)
        if cite_m:
            journal = cite_m.group(1).strip('.')

    # ========== JOURNAL LEVEL ==========
    level = doi_level  # DOI-based takes priority
    if not level and journal:
        for en_name, (cn_name, en_level) in EN_JOURNAL_MAP.items():
            if cn_name in journal or en_name.upper() in journal.upper():
                level = en_level
                break
    if not level:
        # Heuristic
        if '学报' in journal and ('大学' in journal or '学院' in journal):
            level = '大学学报'
        elif '学报' in journal:
            level = '中文核心(可能性高)'
        elif journal:
            level = '待查'
        else:
            level = ''

    # ========== PAPER TYPE ==========
    paper_type = "未知"

    # Strong markers
    has_J = bool(re.search(r'\[J\]|引用格式.*?\[J\]', text))
    has_D = bool(re.search(r'硕士学位论文|博士学位论文|Master\s*Thesis|Doctoral\s*Dissertation', text[:1500]))
    has_C = bool(re.search(r'会议论文|Proceedings of|Conference on|Symposium on', text[:1000]))

    # Journal indicators
    journal_indicators_count = sum(1 for pat in [
        r'Vol\.?\s*\d+', r'第\d+卷', r'第\d+期',
        r'收稿日期|录用日期|修回日期',
        r'DOI[:\s]', r'doi[:\s]',
        r'基金项目|Foundation',
        r'中图分类号', r'文献标志码',
        r'引用格式.*?\[J\]',
        r'文章编号|论文编号',
    ] if re.search(pat, text[:1500]))

    # Thesis indicators
    thesis_indicators_count = sum(1 for pat in [
        r'硕士学位论文', r'博士学位论文',
        r'答辩日期', r'论文答辩',
        r'分类号\s*[A-Z]', r'密级', r'U\.?D\.?C\.?',
        r'指导教师|导师.*教授',
        r'学位授予单位',
    ] if re.search(pat, text[:1500]))

    if has_J:
        paper_type = "期刊论文"
    elif has_D:
        paper_type = "学位论文"
    elif has_C:
        paper_type = "会议论文"
    elif thesis_indicators_count >= 3:
        paper_type = "学位论文"
    elif journal_indicators_count >= 3:
        paper_type = "期刊论文"
    elif journal_indicators_count >= 2 and doi:
        paper_type = "期刊论文"

    # Override: if file size > 10MB, likely a thesis
    if paper_type == "未知":
        fsize_mb = os.path.getsize(path) / (1024*1024)
        if fsize_mb > 12:
            paper_type = "学位论文(推测：文件较大)"

    # ========== REVIEW ==========
    is_review = "综述" in pdf_file or "综述" in text[:500] or "Review" in pdf_file

    # ========== DOMAIN ==========
    domain = ""
    domain_map = [
        (r'飞机|航空|机体', '航空/飞机'),
        (r'PCB|印刷电路', 'PCB'),
        (r'LCD|TFT|液晶|面板|Mura', 'TFT-LCD'),
        (r'钢材|钢[板带轨管]|精密管件', '钢铁/钢材'),
        (r'电芯|锂电池|锂电|极片|涂布', '锂电池'),
        (r'芯片|晶圆|半导体|激光器', '芯片/半导体'),
        (r'轴瓦|轴承|三叉轴|滚子', '机械零件'),
        (r'绝缘子|输电线路|电连接器', '电力/输电'),
        (r'锯链', '锯链'),
        (r'铸件', '铸件'),
        (r'不锈钢', '不锈钢'),
        (r'凸轮轴', '凸轮轴'),
        (r'接插件|插芯', '接插件/插芯'),
        (r'啤酒|瓶口', '包装/瓶口'),
        (r'碳纤维|预浸料', '碳纤维'),
        (r'钻杆|螺纹', '钻杆'),
        (r'铝板|带钢|金属', '金属表面'),
        (r'压气机|叶片', '叶片'),
        (r'建筑|立面|双目视觉', '建筑/立面'),
        (r'YOLO.*演进|YOLO十年', 'YOLO综述'),
        (r'缺陷检测.*综述|综述.*缺陷检测|表面缺陷检测方法研究进展', '缺陷检测综述'),
    ]
    for pat, dm in domain_map:
        if re.search(pat, pdf_file):
            domain = dm
            break

    # ========== STATS ==========
    stats[paper_type] = stats.get(paper_type, 0) + 1
    if paper_type == "期刊论文" and level:
        stats_level[level] = stats_level.get(level, 0) + 1

    all_results.append({
        'file': pdf_file, 'type': paper_type, 'journal': journal,
        'year': year, 'doi': doi, 'domain': domain, 'level': level,
        'is_review': is_review
    })

# ========== PRINT ==========
print("\n" + "="*95)
print("                         📚 论文文献分析报告")
print("="*95)
print(f"  总计: {len(pdf_files)} 篇\n")

for i, r in enumerate(all_results):
    icons = {'期刊论文': '📰', '学位论文': '🎓', '会议论文': '🏛️', '未知': '❓', '读取失败': '💥'}
    icon = icons.get(r['type'], '❓')
    review_tag = ' 🔍[综述]' if r['is_review'] else ''

    print(f"\n{'─'*95}")
    line = f"  [{i+1:2d}] {icon} {r['file']}{review_tag}"
    print(line)

    info_parts = [f"类型: {r['type']}"]
    if r['year']: info_parts.append(f"年份: {r['year']}")
    if r['journal']: info_parts.append(f"期刊: {r['journal']}")
    if r['level']: info_parts.append(f"⭐{r['level']}")
    if r['domain']: info_parts.append(f"🏭{r['domain']}")
    if r['doi']: info_parts.append(f"DOI: {r['doi']}")
    print(f"       {' | '.join(info_parts)}")

# ========== STATS ==========
print(f"\n\n{'='*95}")
print("                         📊 统计分析")
print(f"{'='*95}")
print(f"\n  📁 论文类型分布:")
for t in ['期刊论文', '学位论文', '会议论文', '未知']:
    cnt = stats.get(t, 0)
    bar = '█' * min(cnt, 60)
    print(f"     {t}: {cnt:3d} 篇  {bar}")

print(f"\n  🏆 期刊级别分布 (期刊论文):")
all_levels = ['EI', '中文核心', '大学学报', '中文核心(可能性高)', '普通期刊', '待查']
for lv in all_levels:
    cnt = stats_level.get(lv, 0)
    if cnt > 0:
        bar = '█' * cnt
        print(f"     {lv}: {cnt} 篇  {bar}")

review_count = sum(1 for r in all_results if r['is_review'])
print(f"\n  🔍 综述类论文: {review_count} 篇")

print(f"\n  🏭 应用领域分布:")
domain_stats = {}
for r in all_results:
    if r['domain']:
        domain_stats[r['domain']] = domain_stats.get(r['domain'], 0) + 1
for dm, cnt in sorted(domain_stats.items(), key=lambda x: -x[1]):
    print(f"     {dm}: {cnt} 篇")

# Year distribution
print(f"\n  📅 年份分布:")
year_stats = {}
for r in all_results:
    if r['year']:
        year_stats[r['year']] = year_stats.get(r['year'], 0) + 1
for yr in sorted(year_stats.keys()):
    print(f"     {yr}: {year_stats[yr]} 篇")

print(f"\n{'='*95}")
print("  ⚠️ 注: 期刊级别基于 DOI 前缀和刊名自动匹配。\"未知\"的论文需手动确认。")
print(f"{'='*95}\n")
