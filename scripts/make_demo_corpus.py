"""Generate a SYNTHETIC demo corpus (30 fictional government-scheme PDFs) and an 80-question golden set.

Why synthetic? It lets `make eval` run anywhere with zero downloads, and every gold answer is
guaranteed correct because the facts were generated, then the gold *page* is verified by re-parsing the PDFs.
This is only the OFFLINE DEMO (data/demo_pdfs, data/demo_golden.jsonl). The real project uses data/pdfs + eval/golden_set.jsonl
(same schema, see eval/README.md).
"""
from __future__ import annotations
import json, random, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors

PDF_DIR = ROOT / "data" / "demo_pdfs"
GOLD = ROOT / "data" / "demo_golden.jsonl"
MANIFEST = ROOT / "data" / "demo_manifest.json"
SEED = 7

# name, sector, beneficiaries, ministry, descriptor, benefit label, benefit type
S = [
 ("Kisan Samriddhi Yojana","agriculture","farmers","Ministry of Agriculture and Farmers Welfare","provides income support to small and marginal farmers","annual income assistance","annual"),
 ("Swasthya Suraksha Bima Yojana","health","families","Ministry of Health and Family Welfare","gives hospital insurance cover to low-income families","hospitalisation cover","cover"),
 ("Gram Awas Nirman Yojana","housing","rural households","Ministry of Rural Development","helps rural households build pucca houses","construction grant","onetime"),
 ("Vidya Sahayata Scholarship Scheme","education","students","Ministry of Education","funds school and college scholarships for meritorious students","scholarship","annual"),
 ("Mahila Udyami Protsahan Yojana","women's enterprise","women entrepreneurs","Ministry of Women and Child Development","supports women who start small businesses","start-up grant","onetime"),
 ("Matsya Nauka Bima Yojana","fisheries","fishers","Ministry of Fisheries, Animal Husbandry and Dairying","insures fishing boats and the crew","boat and crew insurance cover","cover"),
 ("Laghu Udyog Rin Sahayata Yojana","MSME credit","micro-enterprises","Ministry of Micro, Small and Medium Enterprises","offers interest subvention on loans to micro-enterprises","interest subvention grant","annual"),
 ("Kaushal Vikas Pratishthan Scheme","skills","trainees","Ministry of Skill Development and Entrepreneurship","pays for short-term skill training courses","training stipend","annual"),
 ("Vriddha Jan Pension Yojana","social security","senior citizens","Ministry of Social Justice and Empowerment","pays a monthly-linked pension to elderly persons without other support","pension support","annual"),
 ("Divyang Sashaktikaran Yojana","disability inclusion","persons with disabilities","Ministry of Social Justice and Empowerment","provides assistive devices and livelihood grants to persons with disabilities","assistive-device grant","onetime"),
 ("Shramik Kalyan Suraksha Yojana","labour welfare","unorganised workers","Ministry of Labour and Employment","covers accident and death risk of unorganised workers","accident cover","cover"),
 ("Nirmal Jal Sanyojan Yojana","drinking water","rural households","Ministry of Jal Shakti","gives tap water connections to rural homes","connection subsidy","onetime"),
 ("Saur Urja Kisan Yojana","renewable energy","farmers","Ministry of Agriculture and Farmers Welfare","subsidises solar irrigation pumps for farmers","solar pump subsidy","onetime"),
 ("Poshan Aahar Sahayata Yojana","nutrition","pregnant women and children","Ministry of Women and Child Development","delivers supplementary nutrition to mothers and young children","nutrition support","annual"),
 ("Navjat Shishu Raksha Yojana","child health","newborns","Ministry of Health and Family Welfare","finances special care for sick newborn babies","newborn care cover","cover"),
 ("Yuva Rozgar Protsahan Yojana","employment","young jobseekers","Ministry of Labour and Employment","rewards employers who hire first-time young workers","first-job incentive","onetime"),
 ("Pashu Dhan Vikas Yojana","livestock","livestock owners","Ministry of Agriculture and Farmers Welfare","strengthens cattle and goat rearing through assets and fodder support","livestock asset grant","onetime"),
 ("Van Mitra Ajeevika Yojana","forest livelihoods","forest-dwelling communities","Ministry of Tribal Affairs","creates livelihoods from minor forest produce","livelihood grant","annual"),
 ("Shahri Sadak Vikreta Yojana","urban livelihoods","street vendors","Ministry of Housing and Urban Affairs","offers working-capital loans to street vendors","working-capital loan","onetime"),
 ("Bunkar Sahayata Yojana","handlooms","handloom weavers","Ministry of Textiles","supports weavers with looms, yarn and marketing","weaver support grant","annual"),
 ("Gramin Shauchalaya Anudan Yojana","sanitation","rural households","Ministry of Jal Shakti","pays for household toilets in villages","toilet construction incentive","onetime"),
 ("Digital Saksharta Abhiyan","digital literacy","rural citizens","Ministry of Electronics and Information Technology","teaches basic digital skills to rural citizens","training voucher","onetime"),
 ("Anusuchit Jati Chhatravas Yojana","hostels","scheduled caste students","Ministry of Social Justice and Empowerment","builds hostels for scheduled caste students","hostel allowance","annual"),
 ("Krishi Yantra Anudan Yojana","farm mechanisation","farmers","Ministry of Agriculture and Farmers Welfare","subsidises tractors and other farm machinery","machinery subsidy","onetime"),
 ("Sagar Mitra Tatiya Yojana","coastal communities","coastal households","Ministry of Fisheries, Animal Husbandry and Dairying","improves coastal livelihoods through cold-chain and landing centres","coastal livelihood grant","annual"),
 ("Shilp Guru Kalakar Yojana","handicrafts","artisans","Ministry of Textiles","supports master craftspersons and their apprentices","artisan grant","annual"),
 ("Bal Vikas Anganwadi Poshan Yojana","early childhood","anganwadi children","Ministry of Women and Child Development","strengthens anganwadi centres for early childhood care","centre support grant","annual"),
 ("Gramin Rojgar Guarantee Sahayata Yojana","rural wages","rural workers","Ministry of Rural Development","guarantees paid work to rural workers","wage support","annual"),
 ("Vayu Shuddh Chulha Yojana","clean cooking","rural households","Ministry of Petroleum and Natural Gas","provides clean cooking stoves to poor households","stove subsidy","onetime"),
 ("Antyodaya Anna Suraksha Yojana","food security","poorest families","Ministry of Consumer Affairs, Food and Public Distribution","delivers subsidised foodgrains to the poorest families","foodgrain subsidy","annual"),
]

FILLER = {
 "Background and Objectives": [
  "The scheme was designed after consultations with state governments, civil society groups and representatives of {who}.",
  "Its objectives are to improve outcomes in {sector}, to reduce leakages through direct transfers, and to build transparent systems for delivery.",
  "The guidelines in this document apply uniformly to all states and union territories unless a state-specific addendum has been notified.",
  "Implementation follows the principle of cooperative federalism, with the central government setting norms and the states running field operations.",
  "Special emphasis is placed on reaching {who} in remote, hilly and tribal districts where service coverage has historically been thin.",
  "The scheme is demand driven and open-ended, so every eligible applicant who completes verification receives support.",
 ],
 "Eligibility": [
  "Eligibility is determined at the time of application and re-verified once every year through a simple digital process.",
  "A beneficiary may receive support under {name} only once, and duplicate registrations are cancelled after verification.",
  "Members of {who} who are already covered by an equivalent state programme must choose one of the two schemes.",
  "Persons who are serving or retired employees of central or state governments are not eligible, except for those in the lowest grade of service.",
  "Applicants must hold a valid identity document and an active bank account linked to the national identity database.",
  "Claims of eligibility are verified by the local body and any false declaration leads to recovery of the amount paid.",
 ],
 "Benefits": [
  "Benefits are credited directly to the bank account of the beneficiary through the public financial management system.",
  "The scheme does not replace other entitlements, and {who} may continue to receive benefits under unrelated programmes.",
  "Where a transfer fails because of an account error, the local officer corrects the record and the payment is retried.",
  "Beneficiaries receive an electronic acknowledgement on their registered mobile number after every transfer.",
  "Benefit rates are reviewed by the empowered committee and any revision is notified in the official gazette.",
  "Unspent balances at the end of a financial year are returned to the common pool and are not carried forward to individuals.",
 ],
 "Application Procedure": [
  "Applicants must keep their identity proof, bank passbook and a recent photograph ready before starting the application.",
  "After submission the application is routed to the local verification officer, who completes field checks within the prescribed time.",
  "Common service centres help {who} who have no internet access to fill and submit their applications.",
  "An application number is generated on submission and can be used to track the status at every stage.",
  "Rejected applications carry written reasons, and the applicant may re-apply after correcting the shortcoming.",
  "Applications can be submitted at any time of the year; there is no cut-off date for registration.",
 ],
 "Funding and Implementation": [
  "Funds are released to state implementing agencies in two tranches, subject to utilisation certificates for earlier releases.",
  "The central share and the state share are in the ratio notified in the annual operational guidelines.",
  "A dedicated project management unit supports states with training, data analysis and technical assistance.",
  "The scheme was first reviewed in {review}, and the findings led to simplified documentation and faster payments.",
  "Third-party evaluations are commissioned periodically and their reports are placed in the public domain.",
  "States must maintain separate accounts for the scheme and submit quarterly progress reports in the prescribed format.",
 ],
 "Monitoring and Grievance Redressal": [
  "Progress is monitored through a dashboard that shows enrolment, payments and pending verifications for every district.",
  "Social audits are conducted in each district every year, and the audit reports are discussed in open village meetings.",
  "Complaints may be lodged on the portal, through a written letter to the district office or in person at the block office.",
  "Each grievance is assigned a unique reference number so that the complainant can follow it until closure.",
  "Officers who fail to act on grievances are subject to departmental action as per the service rules.",
  "Unresolved complaints can be escalated to the state nodal officer and then to the central programme division.",
 ],
}

DOMAIN_STYLE = ParagraphStyle("b", fontName="Helvetica", fontSize=10.5, leading=14.5, spaceAfter=7)
H_STYLE = ParagraphStyle("h", fontName="Helvetica-Bold", fontSize=14, leading=18, spaceBefore=10, spaceAfter=6)
T_STYLE = ParagraphStyle("t", fontName="Helvetica-Bold", fontSize=20, leading=25, spaceAfter=4)
SUB_STYLE = ParagraphStyle("s", fontName="Helvetica", fontSize=10.5, leading=14, spaceAfter=10, textColor=colors.grey)


def inr(n: int) -> str:
    s = str(int(n))
    if len(s) <= 3:
        return s
    head, tail, parts = s[:-3], s[-3:], []
    while len(head) > 2:
        parts.insert(0, head[-2:]); head = head[:-2]
    if head:
        parts.insert(0, head)
    return ",".join(parts + [tail])


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def make_facts(i: int, rnd: random.Random) -> dict:
    name, sector, who, ministry, desc, blabel, btype = S[i]
    f = {"name": name, "doc_id": slug(name), "sector": sector, "who": who, "ministry": ministry,
         "descriptor": desc, "benefit_label": blabel, "benefit_type": btype}
    f["launch_year"] = rnd.randint(2005, 2023)
    if btype == "annual":
        f["benefit_amount"] = rnd.choice(range(3000, 36001, 3000)); f["benefit_freq"] = "per year"
        f["installments"] = rnd.choice([None, 2, 3, 4, 6]) if rnd.random() < 0.75 else None
    elif btype == "cover":
        f["benefit_amount"] = rnd.choice(range(100000, 500001, 50000)); f["benefit_freq"] = "per family per year"
        f["installments"] = None
    else:
        f["benefit_amount"] = rnd.choice(range(10000, 250001, 5000)); f["benefit_freq"] = "one-time"
        f["installments"] = None
    f["min_age"] = rnd.choice([5, 15, 18, 21, 25, 40, 60]) if rnd.random() < 0.7 else None
    f["max_age"] = f["min_age"] + rnd.choice([10, 15, 20, 30, 40]) if (f["min_age"] and rnd.random() < 0.45) else None
    f["income_ceiling"] = rnd.choice(range(100000, 800001, 50000)) if rnd.random() < 0.6 else None
    lakh = rnd.choice([2, 3, 5, 8, 12, 15, 20, 25, 40, 48, 60, 75, 90, 120])
    f["beneficiaries_lakh"] = lakh
    f["outlay_crore"] = lakh * rnd.choice(range(20, 401, 10))
    f["helpline"] = f"1800-{rnd.randint(100,999)}-{rnd.randint(1000,9999)}" if rnd.random() < 0.6 else None
    f["grievance_days"] = rnd.choice([7, 15, 21, 30, 45, 60]) if rnd.random() < 0.7 else None
    f["app_fee"] = rnd.choice([0, 0, 50, 100, 200]) if rnd.random() < 0.55 else None
    f["portal"] = slug(name).replace("_", "")[:18] + ".example.gov.in"
    f["review_year"] = f["launch_year"] + rnd.randint(2, 5)
    return f


def V(rnd, *opts):
    return rnd.choice(opts)


def build_sections(f: dict, rnd: random.Random):
    n, who = f["name"], f["who"]
    fill = lambda sec, k: [t.format(name=n, who=who, sector=f["sector"], review=f["review_year"])
                           for t in rnd.sample(FILLER[sec], k)]
    sec = {}
    bg = [V(rnd, f"{n} was launched in {f['launch_year']} to support {who} across the country.",
            f"The scheme came into force in {f['launch_year']} and has since been extended to {who} in all states."),
          f"{n} is the scheme that {f['descriptor']}."]
    sec["Background and Objectives"] = bg + fill("Background and Objectives", 6)
    el = []
    if f["min_age"]:
        el.append(V(rnd, f"Applicants must be at least {f['min_age']} years of age.", f"The minimum age for eligibility is {f['min_age']} years."))
    if f["max_age"]:
        el.append(V(rnd, f"The upper age limit is {f['max_age']} years.", f"Persons above {f['max_age']} years of age are not eligible."))
    if f["income_ceiling"]:
        x = inr(f["income_ceiling"])
        el.append(V(rnd, f"The annual household income of the applicant must not exceed Rs. {x}.", f"Households with an annual income above Rs. {x} are excluded."))
    sec["Eligibility"] = el + fill("Eligibility", 6)
    amt = inr(f["benefit_amount"]); bl = f["benefit_label"]
    if f["benefit_freq"] == "per year":
        b = V(rnd, f"Each eligible beneficiary receives {bl} of Rs. {amt} per year.", f"{n} provides {bl} amounting to Rs. {amt} per annum to every selected beneficiary.")
    elif f["benefit_freq"] == "one-time":
        b = V(rnd, f"Each eligible beneficiary receives a one-time {bl} of Rs. {amt}.", f"{n} provides a single {bl} of Rs. {amt} to every selected beneficiary.")
    else:
        b = V(rnd, f"Each enrolled family receives a {bl} of Rs. {amt} per family per year.", f"{n} provides a {bl} of Rs. {amt} per family per year.")
    be = [b]
    if f["installments"]:
        k = f["installments"]
        be.append(V(rnd, f"The assistance is released in {k} equal instalments over the year.", f"Payment is made in {k} instalments directly to the bank account of the beneficiary."))
    sec["Benefits"] = be + fill("Benefits", 6)
    ap = [V(rnd, f"Applications are submitted online through the portal {f['portal']}.", f"The official portal for registration is {f['portal']}.")]
    if f["app_fee"] is not None:
        ap.append(V(rnd, "No application fee is charged.", "Applications are accepted free of cost; no fee is payable.") if f["app_fee"] == 0
                  else V(rnd, f"An application fee of Rs. {f['app_fee']} is payable at the time of submission.", f"Applicants pay a one-time fee of Rs. {f['app_fee']}."))
    sec["Application Procedure"] = ap + fill("Application Procedure", 6)
    fu = [V(rnd, f"The total outlay approved for {n} is Rs. {inr(f['outlay_crore'])} crore.", f"A budget of Rs. {inr(f['outlay_crore'])} crore has been earmarked for {n}."),
          V(rnd, f"The scheme targets {f['beneficiaries_lakh']} lakh beneficiaries.", f"About {f['beneficiaries_lakh']} lakh {who} are expected to be covered."),
          V(rnd, f"The nodal ministry for {n} is the {f['ministry']}.", f"{n} is administered at the central level by the {f['ministry']}.")]
    sec["Funding and Implementation"] = fu + fill("Funding and Implementation", 6)
    gr = []
    if f["helpline"]:
        gr.append(V(rnd, f"Beneficiaries may call the toll-free helpline {f['helpline']} for assistance.", f"A helpline, {f['helpline']}, is available on all working days."))
    if f["grievance_days"]:
        gr.append(V(rnd, f"Grievances are resolved within {f['grievance_days']} days of registration.", f"Every complaint must be closed within {f['grievance_days']} days."))
    sec["Monitoring and Grievance Redressal"] = gr + fill("Monitoring and Grievance Redressal", 6)
    for k in sec:  # shuffle so the key sentence isn't always first
        rnd.shuffle(sec[k])
    return sec


def glance(f):
    r = [["Parameter", "Value"], ["Launch year", str(f["launch_year"])], ["Nodal ministry", f["ministry"]],
         ["Benefit amount", f"Rs. {inr(f['benefit_amount'])} {f['benefit_freq']}"]]
    if f["installments"]: r.append(["Number of instalments", str(f["installments"])])
    if f["min_age"]: r.append(["Minimum age", f"{f['min_age']} years"])
    if f["max_age"]: r.append(["Maximum age", f"{f['max_age']} years"])
    if f["income_ceiling"]: r.append(["Household income ceiling", f"Rs. {inr(f['income_ceiling'])}"])
    r += [["Programme outlay", f"Rs. {inr(f['outlay_crore'])} crore"], ["Target beneficiaries", f"{f['beneficiaries_lakh']} lakh"]]
    if f["helpline"]: r.append(["Helpline", f["helpline"]])
    if f["grievance_days"]: r.append(["Grievance resolution time", f"{f['grievance_days']} days"])
    if f["app_fee"] is not None: r.append(["Application fee", "Nil" if f["app_fee"] == 0 else f"Rs. {f['app_fee']}"])
    r.append(["Portal", f["portal"]])
    return r


def write_pdf(f, rnd):
    path = PDF_DIR / f"{f['doc_id']}.pdf"
    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=2.2*cm, rightMargin=2.2*cm, topMargin=2*cm, bottomMargin=2*cm,
                            title=f["name"], author="Synthetic demo corpus")
    story = [Paragraph(f["name"], T_STYLE), Paragraph(f"Government of India - {f['ministry']} - Scheme Guidelines (synthetic demo document)", SUB_STYLE)]
    sections = build_sections(f, rnd)
    for i, (h, paras) in enumerate(sections.items(), 1):
        story.append(Paragraph(f"{i}. {h}", H_STYLE))
        for p in paras:
            story.append(Paragraph(p, DOMAIN_STYLE))
    story.append(Paragraph(f"{len(sections)+1}. Scheme at a Glance", H_STYLE))
    t = Table(glance(f), colWidths=[6.5*cm, 9.5*cm])
    t.setStyle(TableStyle([("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                           ("FONTSIZE", (0, 0), (-1, -1), 10), ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                           ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke)]))
    story.append(t)
    doc.build(story)
    return path


# ----------------------------------------------------------------------------- golden set
def npat(s: str) -> str:
    return re.sub(r"(?<=\d),(?=\d)", "", s).lower()


def find_pages(parsed, keys):
    out = []
    for p in sorted(parsed.pages):
        txt = npat(parsed.page_text(p))
        if all(re.search(r"(?<![\w.])" + re.escape(npat(k)) + r"(?![\w])", txt) for k in keys):
            out.append(p)
    return out


def main():
    from ingest.parse import parse_pdf
    rnd = random.Random(SEED)
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    for old in PDF_DIR.glob("*.pdf"):
        old.unlink()
    facts = [make_facts(i, rnd) for i in range(len(S))]
    for f in facts:
        write_pdf(f, random.Random(f["doc_id"]))
    parsed = {f["doc_id"]: parse_pdf(PDF_DIR / f"{f['doc_id']}.pdf") for f in facts}
    MANIFEST.write_text(json.dumps(facts, indent=1))

    qs, qid = [], [0]

    def add(qtype, question, gold, keys, docs, src_keys=None, **extra):
        """docs: list of facts; src_keys: per-doc list of strings that must co-occur on the gold page."""
        qid[0] += 1
        src = []
        for d, k in zip(docs, src_keys or [keys] * len(docs)):
            pages = find_pages(parsed[d["doc_id"]], k) if k else []
            if k and not pages:
                raise RuntimeError(f"no gold page for {question} / {d['doc_id']} / {k}")
            src.append({"doc": d["doc_id"], "pages": pages})
        qs.append({"id": f"q{qid[0]:03d}", "type": qtype, "question": question, "gold_answer": gold,
                   "answer_keys": [npat(k) for k in keys], "gold_sources": src if qtype != "unanswerable" else [], **extra})

    def with_(key): return [f for f in facts if f.get(key) is not None]
    def pick(lst, k=1): return rnd.sample(lst, k)

    # --- single fact (28)
    singles = [
     ("launch_year", ["In which year was {n} launched?", "What is the launch year of {n}?"], lambda f: (str(f["launch_year"]), f"{f['name']} was launched in {f['launch_year']}.")),
     ("ministry", ["Which ministry is the nodal ministry for {n}?", "Which ministry administers {n} at the central level?"], lambda f: (f["ministry"].lower(), f"The nodal ministry is the {f['ministry']}.")),
     ("benefit_amount", ["What is the benefit amount under {n}?", "How much financial support does {n} provide to a beneficiary?"], lambda f: (str(f["benefit_amount"]), f"Rs. {inr(f['benefit_amount'])} {f['benefit_freq']}.")),
     ("installments", ["In how many instalments is the benefit under {n} paid?", "How many instalments does {n} use to release assistance?"], lambda f: (str(f["installments"]), f"{f['installments']} instalments.")),
     ("min_age", ["What is the minimum age to be eligible for {n}?", "From what age can a person apply to {n}?"], lambda f: (str(f["min_age"]), f"{f['min_age']} years.")),
     ("max_age", ["What is the upper age limit under {n}?", "Beyond what age is a person ineligible for {n}?"], lambda f: (str(f["max_age"]), f"{f['max_age']} years.")),
     ("income_ceiling", ["What is the household income ceiling for {n}?", "What annual income limit applies to applicants of {n}?"], lambda f: (str(f["income_ceiling"]), f"Rs. {inr(f['income_ceiling'])}.")),
     ("outlay_crore", ["What is the total outlay of {n}?", "How much budget has been earmarked for {n}?"], lambda f: (str(f["outlay_crore"]), f"Rs. {inr(f['outlay_crore'])} crore.")),
     ("beneficiaries_lakh", ["How many beneficiaries does {n} target?", "How many lakh people is {n} expected to cover?"], lambda f: (str(f["beneficiaries_lakh"]), f"{f['beneficiaries_lakh']} lakh.")),
     ("helpline", ["What is the helpline number for {n}?"], lambda f: (f["helpline"], f["helpline"])),
     ("grievance_days", ["Within how many days are grievances resolved under {n}?"], lambda f: (str(f["grievance_days"]), f"{f['grievance_days']} days.")),
     ("portal", ["Which portal is used to apply for {n}?"], lambda f: (f["portal"], f["portal"])),
    ]
    plan = [0, 1, 2, 2, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 10, 11, 0, 2, 4, 7, 6, 8, 1, 3, 5, 10]
    used = set()
    for si in plan[:28]:
        key, templs, fn = singles[si]
        cands = [f for f in with_(key) if (key, f["doc_id"]) not in used]
        f = rnd.choice(cands); used.add((key, f["doc_id"]))
        val, ans = fn(f)
        add("single", rnd.choice(templs).format(n=f["name"]), ans, [val], [f])

    # --- ambiguous / tricky wording (8): descriptors, paraphrase without keywords
    amb = []
    for key, tmpl, fn in [
        ("helpline", "What is the helpline for the scheme that {d}?", lambda f: f["helpline"]),
        ("outlay_crore", "How big is the budget of the programme that {d}?", lambda f: str(f["outlay_crore"])),
        ("launch_year", "In what year did the scheme that {d} begin?", lambda f: str(f["launch_year"])),
        ("benefit_amount", "How much money does the scheme that {d} hand out?", lambda f: str(f["benefit_amount"])),
        ("min_age", "How young can an applicant be for the scheme that {d}?", lambda f: str(f["min_age"])),
        ("beneficiaries_lakh", "Roughly how many lakh people will the scheme that {d} reach?", lambda f: str(f["beneficiaries_lakh"])),
        ("grievance_days", "What is the time limit for closing complaints in the programme that {d}?", lambda f: str(f["grievance_days"])),
        ("max_age", "After which birthday can nobody enrol in the scheme that {d}?", lambda f: str(f["max_age"])),
    ]:
        f = rnd.choice([x for x in with_(key) if (key, x["doc_id"]) not in used]); used.add((key, f["doc_id"]))
        add("ambiguous", tmpl.format(d=f["descriptor"]), f"{fn(f)} ({f['name']})", [fn(f)], [f])

    # --- numerical (12)
    def two(k1, k2=None, same_freq=False):
        c = [f for f in with_(k1) if k2 is None or f.get(k2) is not None]
        while True:
            a, b = rnd.sample(c, 2)
            if a[k1] != b[k1] and (not same_freq or a["benefit_freq"] == b["benefit_freq"]):
                return a, b
    for _ in range(3):
        f = rnd.choice([x for x in with_("installments")])
        while ("inst", f["doc_id"]) in used: f = rnd.choice(with_("installments"))
        used.add(("inst", f["doc_id"]))
        v = f["benefit_amount"] // f["installments"]
        add("numerical", f"What is the amount of each instalment under {f['name']}?", f"Rs. {inr(v)} ({inr(f['benefit_amount'])} / {f['installments']})", [str(v)], [f],
            src_keys=[[str(f["benefit_amount"])]], calc=True)
    for _ in range(2):
        f = rnd.choice([x for x in facts if x["benefit_freq"] == "per year" and ("yr", x["doc_id"]) not in used]); used.add(("yr", f["doc_id"]))
        yrs = rnd.choice([3, 5, 10]); v = f["benefit_amount"] * yrs
        add("numerical", f"If a person receives the benefit under {f['name']} for {yrs} years, what total amount will they have received?", f"Rs. {inr(v)}", [str(v)], [f], src_keys=[[str(f['benefit_amount'])]], calc=True)
    for _ in range(2):
        a, b = two("outlay_crore"); v = a["outlay_crore"] + b["outlay_crore"]
        add("numerical", f"What is the combined outlay of {a['name']} and {b['name']}?", f"Rs. {inr(v)} crore", [str(v)], [a, b], src_keys=[[str(a["outlay_crore"])], [str(b["outlay_crore"])]], calc=True)
    for _ in range(2):
        f = rnd.choice([x for x in facts if ("pb", x["doc_id"]) not in used]); used.add(("pb", f["doc_id"]))
        v = f["outlay_crore"] * 100 // f["beneficiaries_lakh"]
        add("numerical", f"What is the outlay per beneficiary, in rupees, for {f['name']}?", f"Rs. {inr(v)} ({inr(f['outlay_crore'])} crore / {f['beneficiaries_lakh']} lakh)", [str(v)], [f],
            src_keys=[[str(f["outlay_crore"])]], calc=True)
    for _ in range(1):
        f = rnd.choice([x for x in with_("max_age") if ("rng", x["doc_id"]) not in used]); used.add(("rng", f["doc_id"]))
        v = f["max_age"] - f["min_age"]
        add("numerical", f"What is the difference between the maximum and minimum eligible age under {f['name']}?", f"{v} years", [str(v)], [f], src_keys=[[str(f["max_age"])]], calc=True)
    for _ in range(1):
        f = rnd.choice([x for x in facts if ("old", x["doc_id"]) not in used]); used.add(("old", f["doc_id"]))
        v = 2026 - f["launch_year"]
        add("numerical", f"How many years have passed since {f['name']} was launched, as of 2026?", f"{v} years", [str(v)], [f], src_keys=[[str(f["launch_year"])]], calc=True)
    a, b = two("benefit_amount", same_freq=True); hi, lo = (a, b) if a["benefit_amount"] > b["benefit_amount"] else (b, a)
    v = hi["benefit_amount"] - lo["benefit_amount"]
    add("numerical", f"By how much does the benefit amount of {hi['name']} exceed that of {lo['name']}?", f"Rs. {inr(v)}", [str(v)], [hi, lo],
        src_keys=[[str(hi["benefit_amount"])], [str(lo["benefit_amount"])]], calc=True)

    # --- multi-hop / comparison (20)
    def which(label, key, hi_word, fmt, cmp="max", n=1, same_freq=False):
        for _ in range(n):
            while True:
                a, b = two(key, same_freq=same_freq)
                if ((label, a["doc_id"], b["doc_id"]) not in used) and ((label, b["doc_id"], a["doc_id"]) not in used): break
            used.add((label, a["doc_id"], b["doc_id"]))
            w, l = (a, b) if (a[key] > b[key]) == (cmp == "max") else (b, a)
            add("multihop", f"Which has a {hi_word}, {a['name']} or {b['name']}?", f"{w['name']} ({fmt(w)} vs {fmt(l)})", [w["name"].lower()], [a, b],
                src_keys=[[str(a[key])], [str(b[key])]], winner=w["name"].lower(), loser=l["name"].lower())
    which("ben", "benefit_amount", "higher benefit amount", lambda f: f"Rs. {inr(f['benefit_amount'])}", n=3, same_freq=True)
    which("out", "outlay_crore", "larger programme outlay", lambda f: f"Rs. {inr(f['outlay_crore'])} crore", n=3)
    which("early", "launch_year", "earlier launch", lambda f: str(f["launch_year"]), cmp="min", n=2)
    which("inc", "income_ceiling", "higher household income ceiling", lambda f: f"Rs. {inr(f['income_ceiling'])}", n=2)
    for _ in range(3):
        a, b = two("min_age")
        add("multihop", f"Compare the minimum eligible age under {a['name']} and {b['name']}.", f"{a['name']}: {a['min_age']} years; {b['name']}: {b['min_age']} years",
            [str(a["min_age"]), str(b["min_age"])], [a, b], src_keys=[[str(a["min_age"]) + " years"], [str(b["min_age"]) + " years"]])
    for _ in range(2):
        a, b = two("beneficiaries_lakh")
        add("multihop", f"How do the target beneficiary numbers of {a['name']} and {b['name']} compare?", f"{a['name']}: {a['beneficiaries_lakh']} lakh; {b['name']}: {b['beneficiaries_lakh']} lakh",
            [str(a["beneficiaries_lakh"]), str(b["beneficiaries_lakh"])], [a, b], src_keys=[[str(a["beneficiaries_lakh"]) + " lakh"], [str(b["beneficiaries_lakh"]) + " lakh"]])
    by_m = {}
    for f in facts: by_m.setdefault(f["ministry"], []).append(f)
    same_pairs = [rnd.sample(v, 2) for v in by_m.values() if len(v) >= 2]
    rnd.shuffle(same_pairs)
    for a, b in same_pairs[:2]:
        add("multihop", f"Are {a['name']} and {b['name']} run by the same ministry?", f"Yes, both are under the {a['ministry']}.", [a["ministry"].lower()], [a, b],
            src_keys=[[a["ministry"].lower()], [b["ministry"].lower()]], answer_prefix="yes")
    for _ in range(3):
        while True:
            a, b = rnd.sample(facts, 2)
            if a["ministry"] != b["ministry"]: break
        add("multihop", f"Are {a['name']} and {b['name']} run by the same ministry?", f"No. {a['name']} is under the {a['ministry']}; {b['name']} is under the {b['ministry']}.",
            [a["ministry"].lower(), b["ministry"].lower()], [a, b], src_keys=[[a["ministry"].lower()], [b["ministry"].lower()]], answer_prefix="no")

    # --- unanswerable (12)
    for key, tmpl in [("helpline", "What is the helpline number for {n}?"), ("grievance_days", "Within how many days are grievances resolved under {n}?"),
                      ("app_fee", "What application fee is charged under {n}?"), ("max_age", "What is the upper age limit under {n}?"),
                      ("income_ceiling", "What is the household income ceiling for {n}?"), ("installments", "In how many instalments is the benefit under {n} paid?")]:
        f = rnd.choice([x for x in facts if x.get(key) is None and (key, x["doc_id"]) not in used]); used.add((key, f["doc_id"]))
        add("unanswerable", tmpl.format(n=f["name"]), "Not stated in the documents.", [], [f], subtype="missing_attribute")
    for nm in ["Pradhan Sagar Rakshak Yojana", "Rashtriya Vayo Seva Mission", "Nav Bharat Udyam Yojana"]:
        add("unanswerable", f"What is the benefit amount under {nm}?", "Scheme is not in the corpus.", [], [], subtype="out_of_corpus")
    f1, f2, f3 = rnd.sample(facts, 3)
    add("unanswerable", f"How many grievances were received under {f1['name']} in 2024?", "Not stated.", [], [], subtype="not_in_docs")
    add("unanswerable", f"What was the farmer-level success rate of {f2['name']} in the most recent evaluation?", "Not stated.", [], [], subtype="not_in_docs")
    add("unanswerable", f"Who is the current secretary in charge of {f3['name']}?", "Not stated.", [], [], subtype="not_in_docs")

    counts = {}
    for q in qs: counts[q["type"]] = counts.get(q["type"], 0) + 1
    rnd.shuffle(qs)
    for i, q in enumerate(qs, 1): q["id"] = f"q{i:03d}"
    GOLD.write_text("\n".join(json.dumps(q, ensure_ascii=False) for q in qs) + "\n")
    print("PDFs:", len(facts), "| questions:", len(qs), counts)


if __name__ == "__main__":
    main()
