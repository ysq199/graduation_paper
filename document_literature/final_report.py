import fitz, os, re, sys
sys.stdout.reconfigure(encoding='utf-8')

paper_dir = r"E:\graduation_paper\文献"
pdf_files = sorted([f for f in os.listdir(paper_dir) if f.endswith('.pdf')])

DOI_MAP = {
    '10.3901/JME': ('机械工程学报', 'EI'),
    '10.3724/SP.J.1004': ('自动化学报', 'EI'),
    '10.16383/j.aas': ('自动化学报', 'EI'),
    '10.12086/oee': ('光电工程', 'EI'),
    '10.13195/j.kzyjc': ('控制与决策', 'EI'),
    '10.11834/jig': ('中国图象图形学报', 'EI'),
    '10.3785/j.issn.1008-973X': ('浙江大学学报(工学版)', '中文核心'),
    '10.3724/SP.J.1187': ('电子测量与仪器学报', '中文核心'),
    '10.19651/j.cnki.emt': ('电子测量技术', '中文核心'),
    '10.19781/j.issn.1673-9140': ('电力科学与技术学报', '中文核心'),
    '10.13433/j.cnki.1003-8728': ('机械科学与技术', '中文核心'),
    '10.13462/j.cnki': ('机械设计与制造', '中文核心'),
    '10.19554/j.cnki.1001-3563': ('包装工程', '中文核心'),
    '10.19695/j.cnki.cn12-1369': ('信息技术与网络安全', '普通期刊'),
    '10.19769/j.zdhy': ('自动化与仪器仪表', '普通期刊'),
    '10.14004/j.cnki.ckt': ('计算机技术与自动化', '普通期刊'),
    '10.19335/j.cnki.2095-6649': ('新型工业化', '普通期刊'),
    '10.3778/j.issn.1002-8331': ('计算机工程与应用', '中文核心'),
    '10.20137/j.qykjyfz': ('前沿科技与发展', '普通期刊'),
    '10.19768/j.cnki.dgjs': ('电工技术', '普通期刊'),
    '10.3969/j.issn.1006-4311': ('价值工程', '普通期刊'),
}

JOURNAL_LEVEL = {
    '机械工程学报': 'EI',
    '自动化学报': 'EI',
    '控制与决策': 'EI',
    '光电工程': 'EI',
    '中国图象图形学报': 'EI',
    '光学精密工程': 'EI',
    '哈尔滨工业大学学报': 'EI',
    '浙江大学学报': 'EI',
    '华南理工大学学报': 'EI',
    '武汉大学学报': '中文核心',
    '中山大学学报': '中文核心',
    '空军工程大学学报': '中文核心',
    '电力科学与技术学报': '中文核心',
    '电子测量与仪器学报': '中文核心',
    '激光与光电子学进展': '中文核心',
    '计算机应用': '中文核心',
    '计算机工程与应用': '中文核心',
    '机械科学与技术': '中文核心',
    '纺织学报': '中文核心',
    '新型工业化': '普通期刊',
    '自动化应用': '普通期刊',
    '价值工程': '普通期刊',
    '信息技术与网络安全': '普通期刊',
    '自动化与仪器仪表': '普通期刊',
    '电子测量技术': '普通期刊',
    '包装工程': '中文核心',
    '传感技术学报': '中文核心',
}

EN_MAP = {
    'JOURNAL OF MECHANICAL ENGINEERING': '机械工程学报',
    'ACTA AUTOMATICA SINICA': '自动化学报',
    'CONTROL AND DECISION': '控制与决策',
    'JOURNAL OF COMPUTER APPLICATIONS': '计算机应用',
    'COMPUTER ENGINEERING AND APPLICATIONS': '计算机工程与应用',
}

CHINESE_JOURNAL_NAMES = [
    '机械工程学报', '自动化学报', '控制与决策', '光电工程', '中国图象图形学报',
    '计算机工程与应用', '计算机应用', '激光与光电子学进展', '电子测量与仪器学报',
    '自动化应用', '价值工程', '新型工业化', '传感技术学报', '包装工程',
    '信息技术与网络安全', '自动化与仪器仪表', '机械科学与技术',
]

all_results = []

for pdf_file in pdf_files:
    path = os.path.join(paper_dir, pdf_file)
    try:
        doc = fitz.open(path)
        text = ""
        for page_num in range(min(2, len(doc))):
            text += doc[page_num].get_text()
        doc.close()
    except:
        all_results.append({
            'file': pdf_file, 'type': '读取失败', 'journal': '',
            'year': '', 'doi': '', 'domain': '', 'level': '', 'is_review': False
        })
        continue

    fsize_mb = os.path.getsize(path) / (1024*1024)

    # DOI
    doi = ""
    doi_m = re.search(r'DOI[:\s]*\s*(10\.\d{4,}/[^\s\)\]】\n]+)', text[:2000], re.IGNORECASE)
    if not doi_m:
        doi_m = re.search(r'doi[:\s]*\s*(10\.\d{4,}/[^\s\)\]】\n]+)', text[:2000], re.IGNORECASE)
    if doi_m:
        doi = doi_m.group(1).rstrip('.').rstrip(',').rstrip(';')

    # Year
    year = ""
    for pat in [r'(?:^|\D)(20\d{2})[年\-\s]', r'收稿日期[：:\s]*(20\d{2})',
                r'录用日期[：:\s]*(20\d{2})']:
        ym = re.search(pat, text[:1500])
        if ym:
            year = ym.group(1)
            break

    # Journal from DOI
    journal = ""
    level = ""
    for prefix, (jname, jlevel) in DOI_MAP.items():
        if doi and prefix in doi:
            journal = jname
            level = jlevel
            break

    # Journal from English headers
    if not journal:
        up_text = text[:1500].upper()
        for en_name, cn_name in EN_MAP.items():
            if en_name in up_text:
                journal = cn_name
                level = JOURNAL_LEVEL.get(cn_name, '')
                break

    # Journal from Chinese headers
    if not journal:
        for line in text.split('\n')[:20]:
            line = line.strip()
            for jname in CHINESE_JOURNAL_NAMES:
                if jname in line:
                    journal = jname
                    level = JOURNAL_LEVEL.get(jname, '')
                    break
            if journal:
                break

    # Journal from Vol header (大学学报 pattern)
    if not journal:
        for line in text.split('\n')[:10]:
            line = line.strip()
            m = re.search(r'([一-鿿]+学报)', line)
            if m and len(line) < 40:
                journal = m.group(1)
                level = JOURNAL_LEVEL.get(journal, '大学学报' if '大学' in journal else '中文核心(可能性高)')
                break

    # Paper type
    paper_type = "未知"

    has_J_marker = bool(re.search(r'\[J\]|引用格式.*?\[J\]', text))
    has_D_marker = bool(re.search(r'硕士学位论文|博士学位论文|专业学位.*硕士', text[:2000]))
    has_C_marker = bool(re.search(r'\[C\]|会议论文|Proceedings of|Conference on', text[:1000]))

    j_indicators = sum(1 for pat in [
        r'(?:Vol\.?|第)\s*\d+', r'(?:No\.?|第)\s*\d+\s*期', r'ISSN\s*\d{4}',
        r'收稿日期|录用日期|修回日期|Received|Accepted',
        r'基金项目|Foundation', r'中图分类号|文献标志码|文章编号',
        r'引用格式.*?\[J\]', r'DOI[:\s]',
    ] if re.search(pat, text[:1500]))

    t_indicators = sum(1 for pat in [
        r'硕士学位论文|博士学位论文', r'答辩日期|论文答辩',
        r'指导教师|导师.*教授', r'学位授予单位|学号',
        r'分类号\s*[A-Z]', r'密级',
    ] if re.search(pat, text[:2000]))

    if has_J_marker:
        paper_type = "期刊论文"
    elif has_D_marker:
        paper_type = "学位论文"
    elif has_C_marker:
        paper_type = "会议论文"
    elif t_indicators >= 3:
        paper_type = "学位论文"
    elif j_indicators >= 3:
        paper_type = "期刊论文"
    elif j_indicators >= 2 and doi:
        paper_type = "期刊论文"
    elif journal and j_indicators >= 1:
        paper_type = "期刊论文"
    elif fsize_mb > 12:
        paper_type = "学位论文(推测:文件较大)"

    # Review
    is_review = "综述" in pdf_file or "综述" in text[:600] or "Review" in pdf_file

    # Domain
    domain = ""
    domain_map = [
        (r'飞机|航空|机体', '航空/飞机'),
        (r'PCB|印刷电路', 'PCB'),
        (r'LCD|TFT|液晶|面板|Mura', 'TFT-LCD'),
        (r'钢[材板带轨管]|精密管件', '钢铁/钢材'),
        (r'电芯|锂电池|锂电|极片|涂布', '锂电池'),
        (r'芯片|晶圆|半导体|激光器', '芯片/半导体'),
        (r'轴瓦|轴承|三叉轴|滚子', '机械零件'),
        (r'绝缘子|输电线路|电连接器', '电力/输电'),
        (r'锯链', '锯链'), (r'铸件', '铸件'), (r'不锈钢', '不锈钢'),
        (r'凸轮轴', '凸轮轴'), (r'接插件|插芯', '接插件/插芯'),
        (r'啤酒|瓶口', '包装/瓶口'), (r'碳纤维|预浸料', '碳纤维'),
        (r'钻杆|螺纹', '钻杆'), (r'铝板|带钢|金属', '金属表面'),
        (r'压气机|叶片', '叶片'), (r'建筑|立面|双目视觉', '建筑/立面'),
        (r'YOLO.*演进|YOLO十年', 'YOLO综述'),
        (r'缺陷检测.*综述|综述.*缺陷检测|自动光学.*综述', '缺陷检测综述'),
    ]
    for pat, dm in domain_map:
        if re.search(pat, pdf_file):
            domain = dm
            break

    all_results.append({
        'file': pdf_file, 'type': paper_type, 'journal': journal,
        'year': year, 'doi': doi, 'domain': domain, 'level': level,
        'is_review': is_review
    })

# ========= OUTPUT =========
print("\n" + "="*100)
print("                    FINAL REPORT: 论文类型与期刊级别分析")
print("="*100)

type_stats = {}
journal_levels = {}
domain_stats = {}
year_stats = {}
review_count = 0

for r in all_results:
    type_stats[r['type']] = type_stats.get(r['type'], 0) + 1
    if r['type'] == '期刊论文' and r['level']:
        journal_levels[r['level']] = journal_levels.get(r['level'], 0) + 1
    if r['domain']:
        domain_stats[r['domain']] = domain_stats.get(r['domain'], 0) + 1
    if r['year']:
        year_stats[r['year']] = year_stats.get(r['year'], 0) + 1
    if r['is_review']:
        review_count += 1

print(f"\n总计: {len(pdf_files)} 篇")
print(f"\n  --- 论文类型分布 ---")
for t, c in sorted(type_stats.items(), key=lambda x: -x[1]):
    print(f"  {t}: {c} 篇")

print(f"\n  --- 期刊级别分布 (期刊论文中已识别的) ---")
for lv, c in sorted(journal_levels.items(), key=lambda x: -x[1]):
    print(f"  {lv}: {c} 篇")

print(f"\n  --- 综述类论文: {review_count} 篇 ---")

print(f"\n  --- 应用领域分布(TOP10) ---")
for dm, c in sorted(domain_stats.items(), key=lambda x: -x[1])[:10]:
    print(f"  {dm}: {c} 篇")

print(f"\n  --- 年份分布 ---")
for yr in sorted(year_stats.keys()):
    print(f"  {yr}: {year_stats[yr]} 篇")

# Detail by type
print(f"\n\n{'='*100}")
print("  逐篇详情 (按类型分组)")
print(f"{'='*100}")

for type_name in ['期刊论文', '学位论文', '未知']:
    subset = [r for r in all_results if r['type'] == type_name]
    if not subset:
        continue
    print(f"\n{'─'*100}")
    print(f"  [{type_name}] ({len(subset)}篇)")
    print(f"{'─'*100}")
    for i, r in enumerate(subset):
        info = f"  [{i+1:2d}] {r['file'][:75]}"
        if r['year']:
            info += f" | {r['year']}"
        if r['journal']:
            info += f" | {r['journal']}"
        if r['level']:
            info += f" | [{r['level']}]"
        if r['is_review']:
            info += " | [综述]"
        if r['domain']:
            info += f" | {r['domain']}"
        print(info)

print(f"\n\n{'='*100}")
print("  备注: 期刊级别基于DOI前缀和刊名自动匹配，部分未识别的期刊论文需手动查证。")
print(f"{'='*100}")
