"""Silver-standard evaluation questions -- self-generated from the actual
indexed corpus (not from the brief), kept entirely separate from the 12
brief questions in questions.py. Each question is grounded in one real
chunk: the reference answer is written directly from that chunk's content,
not invented, and every entry records exactly which chunk it came from so
the pairing can be re-verified against the corpus at any time.

Three categories, to exercise the bilingual retrieval/generation pipeline
(guideline_router language bonus, hybrid_search dedup, generation/prompt.py's
two-prompt-set design) along a dimension the 12 brief questions don't touch
at all (those are all German-source, German-query):

- "de_de":  source content in German, question asked in German.
- "en_en":  source content in English (the 032-033OL English translation
            document, or 015-027OL's English evidence-review document),
            question asked in English.
- "de_en":  source content in German, question asked in English -- tests
            cross-lingual retrieval specifically: the query language differs
            from the language the supporting passage is actually written in,
            which is exactly the scenario guideline_router's language bonus
            and the reranker's language-agnostic scoring need to handle
            without a hard language filter excluding the right answer.

Not randomly re-sampled on every run -- these were hand-picked once from a
random sample of real chunks (see dev_logs.md) and the (question, answer)
pairs hand-written against the actual chunk text, so re-running the sampling
script would NOT reproduce this exact file. Treat this module as the fixed
artifact, not a generator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SilverCategory = Literal["de_de", "en_en", "de_en"]


@dataclass
class SilverQuestion:
    id: int
    category: SilverCategory
    question: str
    reference_answer: str
    source_guideline_id: str
    source_section_number: str
    source_section_title: str
    source_chunk_id: str
    source_language: str  # language the underlying chunk text is actually written in


SILVER_QUESTIONS: list[SilverQuestion] = [
    # ---- Category de_de: German source, German question ----
    SilverQuestion(
        id=1, category="de_de",
        question="Welches Dosierungsschema gilt für Gardasil® bei Jugendlichen ab 14 Jahren?",
        reference_answer="Bei Jugendlichen ab 14 Jahren wird Gardasil® nach einem 3-Dosen-Schema verabreicht (Monate 0-2-6).",
        source_guideline_id="015-027OL", source_section_number="18.3.",
        source_section_title="18.3. Aufklärung Impfung",
        source_chunk_id="015-027OL_langfassung::sec-156::text-2", source_language="de",
    ),
    SilverQuestion(
        id=2, category="de_de",
        question="Wie hoch ist das 10-Jahres-Risiko eines CIN3-Rezidivs, wenn der Kombinationstest (hrHPV + Zytologie) 6 Monate nach Therapie einer CIN2/3-Läsion negativ ist?",
        reference_answer="Das 10-Jahresrisiko eines CIN3-Rezidivs liegt bei 1,4 %, wenn der Kombinationstest (hrHPV-negativ und Zytologie-negativ) 6 Monate nach Therapie negativ ausfällt.",
        source_guideline_id="015-027OL", source_section_number="16.1.1.",
        source_section_title="16.1.1. Zeitpunkt und Dauer der Nachbetreuung",
        source_chunk_id="015-027OL_langfassung::sec-143::text-0", source_language="de",
    ),
    SilverQuestion(
        id=3, category="de_de",
        question="Welche Therapieoptionen sollten bei isolierter ossärer Metastasierung des Zervixkarzinoms geprüft werden, insbesondere bei Frakturgefährdung?",
        reference_answer="Es sollte die Möglichkeit einer lokalen Bestrahlung und/oder einer osteoonkologischen Therapie (z. B. Bisphosphonattherapie, Denosumab) geprüft werden. Liegt die Metastase in einem vorbestrahlten Bereich, muss zuvor eine ossäre Radionekrose ausgeschlossen werden.",
        source_guideline_id="032-033OL", source_section_number="18.3.3.",
        source_section_title="18.3.3. Ossäre Metastasen",
        source_chunk_id="032-033OL_langfassung::sec-176::text-0", source_language="de",
    ),
    SilverQuestion(
        id=4, category="de_de",
        question="Wie viele Frauen erhielten laut der Jahresstatistik 2016 eine histologische Abklärung im Rahmen der zytologischen Früherkennung in Deutschland, und welchem Anteil aller untersuchten Frauen entspricht das?",
        reference_answer="51.195 Frauen erhielten eine histologische Abklärung, was 0,32 % aller 15.839.847 untersuchten Frauen entspricht.",
        source_guideline_id="032-033OL", source_section_number="4.2.1.",
        source_section_title="4.2.1. Zervixkarzinomfrüherkennung in Deutschland",
        source_chunk_id="032-033OL_langfassung::sec-41::text-1", source_language="de",
    ),
    SilverQuestion(
        id=5, category="de_de",
        question="Nach welcher Klassifikation richtet sich die stadienabhängige Durchführung der radikalen Hysterektomie beim Zervixkarzinom?",
        reference_answer="Die radikale Hysterektomie wird stadienabhängig nach der Klassifikation von Piver-Ruthledge et al. (1974) durchgeführt, in Anlehnung an die Empfehlungen von Wertheim, Meigs, Latzko und Okabayashi.",
        source_guideline_id="032-033OL", source_section_number="9.2.",
        source_section_title="9.2. Operatives Vorgehen",
        source_chunk_id="032-033OL_langfassung::sec-87::text-0", source_language="de",
    ),
    SilverQuestion(
        id=6, category="de_de",
        question="Darf im Rahmen der Behandlung eines Beckenwandrezidivs nach primärer Radiotherapie im vorbestrahlten Volumen erneut eine Radiotherapie mit kurativer Dosis verabreicht werden?",
        reference_answer="Nein. Gemäß konsensbasierter Empfehlung (starker Konsens) soll im vorbestrahlten Volumen keine erneute Radiotherapie mit kurativer Dosis verabreicht werden.",
        source_guideline_id="032-033OL", source_section_number="17.3.4.",
        source_section_title="17.3.4. Behandlung des Beckenwandrezidivs nach primärer oder adjuvanter Radio-/Radiochemotherapie",
        source_chunk_id="032-033OL_langfassung::sec-163::text-0", source_language="de",
    ),

    # ---- Category en_en: English source, English question ----
    SilverQuestion(
        id=7, category="en_en",
        question="What is the recommended symptomatic treatment approach for acute radiotherapy-induced (radiogenic) cystitis?",
        reference_answer="Treatment focuses on symptom control with analgesia and spasmolysis (e.g. metamizole, centrally acting analgesics, butylscopolamine, oxybutynin). Urine alkalization and iron substitution (or transfusion in recurrent hematuria) can supplement therapy, and bacterial superinfections require appropriate antibiotics. Amifostine may be considered per the ASCO guideline to reduce radiotherapy-induced toxicity, though it is not approved for this indication in Germany.",
        source_guideline_id="032-033OL", source_section_number="12.2.1.",
        source_section_title="12.2.1. Radiogenic cystitis",
        source_chunk_id="032-033OL_english::sec-114::text-0", source_language="en",
    ),
    SilverQuestion(
        id=8, category="en_en",
        question="In the GOG 240 trial, what overall survival benefit was observed when bevacizumab was added to palliative chemotherapy for cervical cancer?",
        reference_answer="Adding bevacizumab to palliative chemotherapy (cisplatin/paclitaxel or topotecan/paclitaxel) resulted in a 3.5-month overall survival benefit (16.8 months vs. 13.3 months; HR 0.77; P = 0.007), along with an improved progression-free interval (8.2 vs. 6 months) and a higher response rate (49% vs. 36%).",
        source_guideline_id="032-033OL", source_section_number="18.3.5.1.",
        source_section_title="18.3.5.1. Targeted therapy",
        source_chunk_id="032-033OL_english::sec-174::text-0", source_language="en",
    ),
    SilverQuestion(
        id=9, category="en_en",
        question="What was the overall response rate of pembrolizumab in PD-L1-positive cervical cancer patients in the Keynote 158 trial?",
        reference_answer="The overall response rate was 14.6% (95% CI, 7.8% to 24.2%), including two complete responders among the PD-L1-positive patients who had progressed after first-line treatment.",
        source_guideline_id="032-033OL", source_section_number="18.3.5.2.",
        source_section_title="18.3.5.2. Second-line therapies in cervical carcinoma",
        source_chunk_id="032-033OL_english::sec-175::text-1", source_language="en",
    ),
    SilverQuestion(
        id=10, category="en_en",
        question="How has the ranking of cervical cancer among cancer types in German women changed between 1971 and 2016, and what factor is credited for this decline?",
        reference_answer="Cervical cancer went from being the most frequent cancer type among women in Germany in 1971 to the thirteenth most frequent in 2016 (1.9% of total malignancy incidence among women), a decline partly attributed to the introduction of cytology-based early detection screening in 1971.",
        source_guideline_id="032-033OL", source_section_number="3.1.",
        source_section_title="3.1. Incidence and mortality",
        source_chunk_id="032-033OL_english::sec-28::text-0", source_language="en",
    ),
    SilverQuestion(
        id=11, category="en_en",
        question="What does the available evidence say about the effectiveness of mistletoe therapy in cervical carcinoma?",
        reference_answer="The only available study on mistletoe therapy in cervical carcinoma reported a survival advantage, but the study had substantial methodological deficiencies (e.g. the patient-selection method for its matched-pair analysis was not reported). No survival benefit for mistletoe therapy has been shown in other tumor types, and although a quality-of-life benefit is claimed, high-quality confirmatory studies are still needed.",
        source_guideline_id="032-033OL", source_section_number="14.6.2.1.",
        source_section_title="14.6.2.1. Mistletoe therapy",
        source_chunk_id="032-033OL_english::sec-127::text-0", source_language="en",
    ),
    SilverQuestion(
        id=12, category="en_en",
        question="Which patient self-help organization was directly involved in developing this guideline, and what role did its representatives play?",
        reference_answer="The patient self-help group Frauenselbsthilfe Krebs e.V. (FSH), represented by Ms. Heidemarie Haase (with Ms. Marion Gebhardt as deputy), was directly involved. Patient representatives helped prepare guideline chapters, participated actively in the Patient Information Working Group, and took part in the consensus conferences with full voting rights.",
        source_guideline_id="032-033OL", source_section_number="1.10.3.",
        source_section_title="1.10.3. Patient Involvement",
        source_chunk_id="032-033OL_english::sec-14::text-0", source_language="en",
    ),

    # ---- Category de_en: German source, English-phrased question ----
    SilverQuestion(
        id=13, category="de_en",
        question="Approximately how many new cervical cancer cases occur per year in Germany, and how many conizations were performed in 2009 according to health insurance data?",
        reference_answer="According to the guideline, there are approximately 4,700 new cervical cancer cases per year in Germany. Based on health insurance data, about 90,600 conizations were performed in Germany in 2009, corresponding to 217 conizations per 100,000 women per year.",
        source_guideline_id="015-027OL", source_section_number="2.1.1.",
        source_section_title="2.1.1. Zielsetzung und Fragestellung",
        source_chunk_id="015-027OL_langfassung::sec-8::text-0", source_language="de",
    ),
    SilverQuestion(
        id=14, category="de_en",
        question="According to a Cochrane review cited in the guideline, does sending invitation letters increase participation in cervical cancer screening, and by how much?",
        reference_answer="Yes. A Cochrane review of twelve studies (99,651 participants total) found significantly higher participation rates among women who received an invitation letter compared to no intervention, with effect sizes ranging from a relative risk of 1.44 to 1.74.",
        source_guideline_id="015-027OL", source_section_number="13.2.",
        source_section_title="13.2. Einladungsschreiben",
        source_chunk_id="015-027OL_langfassung::sec-123::text-0", source_language="de",
    ),
    SilverQuestion(
        id=15, category="de_en",
        question="What proportion of false-negative cytological diagnoses in cervical cancer screening is attributed to screening errors, and how many computer-assisted diagnostic systems are currently available worldwide according to the guideline?",
        reference_answer="About one-third of false-negative cytological diagnoses result from screening errors (overlooked or misinterpreted, usually less conspicuous, cells). The guideline states that currently only two systems for computer-assisted diagnostics in cervical cytology are available worldwide.",
        source_guideline_id="015-027OL", source_section_number="6.3.1.",
        source_section_title="6.3.1. Einführung",
        source_chunk_id="015-027OL_langfassung::sec-55::text-0", source_language="de",
    ),
    SilverQuestion(
        id=16, category="de_en",
        question="According to the guideline, what established prognostic factors exist for cervical cancer, and can an unfavorable prognosis from a positive resection margin after radical hysterectomy be improved?",
        reference_answer="Established prognostic factors include tumor stage, the presence of pelvic or para-aortic lymph node metastases, and tumor size. A positive resection margin after radical hysterectomy is associated with an unfavorable prognosis (recurrence-free and overall survival), but this unfavorable effect can be positively influenced by adjuvant radiotherapy or radiochemotherapy.",
        source_guideline_id="032-033OL", source_section_number="7.3.",
        source_section_title="7.3. Morphologische Prognosefaktoren",
        source_chunk_id="032-033OL_langfassung::sec-59::text-0", source_language="de",
    ),
    SilverQuestion(
        id=17, category="de_en",
        question="What did a systematic review with meta-analysis conclude about complementary phytotherapy in cervical cancer treatment?",
        reference_answer="The review concluded that complementary phytotherapy may both improve the effectiveness of conventional therapy and alleviate its side effects, but further methodologically robust studies are needed before a formal recommendation can be made.",
        source_guideline_id="032-033OL", source_section_number="14.6.2.5.",
        source_section_title="14.6.2.5. Phytotherapie",
        source_chunk_id="032-033OL_langfassung::sec-134::text-0", source_language="de",
    ),
    SilverQuestion(
        id=18, category="de_en",
        question="For a pregnant patient diagnosed with FIGO stage II-IV cervical cancer, what is described as the gold-standard treatment approach when the diagnosis is made at term?",
        reference_answer="The gold-standard approach is a primary cesarean section (Sectio caesarea) followed by platinum-based radiochemotherapy (with cisplatin as radiosensitizer), since radiotherapy is not compatible with a continuing pregnancy. Maternal treatment takes priority, though the patient's wishes and the gestational week must be considered.",
        source_guideline_id="032-033OL", source_section_number="21.2.2.",
        source_section_title="21.2.2. FIGO-Stadien IIB, III und IV",
        source_chunk_id="032-033OL_langfassung::sec-197::text-0", source_language="de",
    ),
]
