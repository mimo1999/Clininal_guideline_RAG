"""Silver-standard evaluation questions -- self-generated from the actual
indexed corpus, covering ALL 7 guidelines in the corpus (at least 5 questions per guideline).

Each question is grounded in one real chunk: the reference answer is written
directly from that chunk's content, and every entry records exactly which
chunk it came from so the pairing can be re-verified against the corpus at any time.

Two categories, to exercise the bilingual retrieval/generation pipeline:
- "de_de": source content in German, question asked in German.
- "de_en": source content in German, question asked in English (cross-lingual retrieval).
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
    # =========================================================================
    # GUIDELINE 1: 015-027OL (S3-Leitlinie Prävention des Zervixkarzinoms)
    # =========================================================================
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
        id=3, category="de_en",
        question="Approximately how many new cervical cancer cases occur per year in Germany, and how many conizations were performed in 2009 according to health insurance data?",
        reference_answer="According to the guideline, there are approximately 4,700 new cervical cancer cases per year in Germany. Based on health insurance data, about 90,600 conizations were performed in Germany in 2009, corresponding to 217 conizations per 100,000 women per year.",
        source_guideline_id="015-027OL", source_section_number="2.1.1.",
        source_section_title="2.1.1. Zielsetzung und Fragestellung",
        source_chunk_id="015-027OL_langfassung::sec-8::text-0", source_language="de",
    ),
    SilverQuestion(
        id=4, category="de_en",
        question="According to a Cochrane review cited in the guideline, does sending invitation letters increase participation in cervical cancer screening, and by how much?",
        reference_answer="Yes. A Cochrane review of twelve studies (99,651 participants total) found significantly higher participation rates among women who received an invitation letter compared to no intervention, with effect sizes ranging from a relative risk of 1.44 to 1.74.",
        source_guideline_id="015-027OL", source_section_number="13.2.",
        source_section_title="13.2. Einladungsschreiben",
        source_chunk_id="015-027OL_langfassung::sec-123::text-0", source_language="de",
    ),
    SilverQuestion(
        id=5, category="de_en",
        question="What proportion of false-negative cytological diagnoses in cervical cancer screening is attributed to screening errors, and how many computer-assisted diagnostic systems are currently available worldwide according to the guideline?",
        reference_answer="About one-third of false-negative cytological diagnoses result from screening errors (overlooked or misinterpreted, usually less conspicuous, cells). The guideline states that currently only two systems for computer-assisted diagnostics in cervical cytology are available worldwide.",
        source_guideline_id="015-027OL", source_section_number="6.3.1.",
        source_section_title="6.3.1. Einführung",
        source_chunk_id="015-027OL_langfassung::sec-55::text-0", source_language="de",
    ),

    # =========================================================================
    # GUIDELINE 2: 032-033OL (S3-Leitlinie Diagnostik, Therapie und Nachsorge Zervixkarzinom)
    # =========================================================================
    SilverQuestion(
        id=6, category="de_de",
        question="Welche Therapieoptionen sollten bei isolierter ossärer Metastasierung des Zervixkarzinoms geprüft werden, insbesondere bei Frakturgefährdung?",
        reference_answer="Es sollte die Möglichkeit einer lokalen Bestrahlung und/oder einer osteoonkologischen Therapie (z. B. Bisphosphonattherapie, Denosumab) geprüft werden. Liegt die Metastase in einem vorbestrahlten Bereich, muss zuvor eine ossäre Radionekrose ausgeschlossen werden.",
        source_guideline_id="032-033OL", source_section_number="18.3.3.",
        source_section_title="18.3.3. Ossäre Metastasen",
        source_chunk_id="032-033OL_langfassung::sec-176::text-0", source_language="de",
    ),
    SilverQuestion(
        id=7, category="de_de",
        question="Wie viele Frauen erhielten laut der Jahresstatistik 2016 eine histologische Abklärung im Rahmen der zytologischen Früherkennung in Deutschland, und welchem Anteil aller untersuchten Frauen entspricht das?",
        reference_answer="51.195 Frauen erhielten eine histologische Abklärung, was 0,32 % aller 15.839.847 untersuchten Frauen entspricht.",
        source_guideline_id="032-033OL", source_section_number="4.2.1.",
        source_section_title="4.2.1. Zervixkarzinomfrüherkennung in Deutschland",
        source_chunk_id="032-033OL_langfassung::sec-41::text-1", source_language="de",
    ),
    SilverQuestion(
        id=8, category="de_de",
        question="Nach welcher Klassifikation richtet sich die stadienabhängige Durchführung der radikalen Hysterektomie beim Zervixkarzinom?",
        reference_answer="Die radikale Hysterektomie wird stadienabhängig nach der Klassifikation von Piver-Ruthledge et al. (1974) durchgeführt, in Anlehnung an die Empfehlungen von Wertheim, Meigs, Latzko und Okabayashi.",
        source_guideline_id="032-033OL", source_section_number="9.2.",
        source_section_title="9.2. Operatives Vorgehen",
        source_chunk_id="032-033OL_langfassung::sec-87::text-0", source_language="de",
    ),
    SilverQuestion(
        id=9, category="de_de",
        question="Darf im Rahmen der Behandlung eines Beckenwandrezidivs nach primärer Radiotherapie im vorbestrahlten Volumen erneut eine Radiotherapie mit kurativer Dosis verabreicht werden?",
        reference_answer="Nein. Gemäß konsensbasierter Empfehlung (starker Konsens) soll im vorbestrahlten Volumen keine erneute Radiotherapie mit kurativer Dosis verabreicht werden.",
        source_guideline_id="032-033OL", source_section_number="17.3.4.",
        source_section_title="17.3.4. Behandlung des Beckenwandrezidivs nach primärer oder adjuvanter Radio-/Radiochemotherapie",
        source_chunk_id="032-033OL_langfassung::sec-163::text-0", source_language="de",
    ),
    SilverQuestion(
        id=10, category="de_en",
        question="According to the guideline, what established prognostic factors exist for cervical cancer, and can an unfavorable prognosis from a positive resection margin after radical hysterectomy be improved?",
        reference_answer="Established prognostic factors include tumor stage, the presence of pelvic or para-aortic lymph node metastases, and tumor size. A positive resection margin after radical hysterectomy is associated with an unfavorable prognosis (recurrence-free and overall survival), but this unfavorable effect can be positively influenced by adjuvant radiotherapy or radiochemotherapy.",
        source_guideline_id="032-033OL", source_section_number="7.3.",
        source_section_title="7.3. Morphologische Prognosefaktoren",
        source_chunk_id="032-033OL_langfassung::sec-59::text-0", source_language="de",
    ),
    SilverQuestion(
        id=11, category="de_en",
        question="What did a systematic review with meta-analysis conclude about complementary phytotherapy in cervical cancer treatment?",
        reference_answer="The review concluded that complementary phytotherapy may both improve the effectiveness of conventional therapy and alleviate its side effects, but further methodologically robust studies are needed before a formal recommendation can be made.",
        source_guideline_id="032-033OL", source_section_number="14.6.2.5.",
        source_section_title="14.6.2.5. Phytotherapie",
        source_chunk_id="032-033OL_langfassung::sec-134::text-0", source_language="de",
    ),
    SilverQuestion(
        id=12, category="de_en",
        question="For a pregnant patient diagnosed with FIGO stage II-IV cervical cancer, what is described as the gold-standard treatment approach when the diagnosis is made at term?",
        reference_answer="The gold-standard approach is a primary cesarean section (Sectio caesarea) followed by platinum-based radiochemotherapy (with cisplatin as radiosensitizer), since radiotherapy is not compatible with a continuing pregnancy. Maternal treatment takes priority, though the patient's wishes and the gestational week must be considered.",
        source_guideline_id="032-033OL", source_section_number="21.2.2.",
        source_section_title="21.2.2. FIGO-Stadien IIB, III und IV",
        source_chunk_id="032-033OL_langfassung::sec-197::text-0", source_language="de",
    ),

    # =========================================================================
    # GUIDELINE 3: 015-059 (S2k-Leitlinie Vulvakarzinom und seine Vorstufen)
    # =========================================================================
    SilverQuestion(
        id=13, category="de_de",
        question="Wie lange dürfen Patientendaten im Krankenhaus gespeichert werden und nach welcher Zeit dürfen Krankenhäuser diese vernichten?",
        reference_answer="Patientendaten dürfen im Krankenhaus bis zu 30 Jahre gespeichert werden. Die Krankenhäuser dürfen die Daten allerdings nach 10 Jahren vernichten, weshalb Patientinnen sich Unterlagen wie Arztbriefe oder OP-Berichte als Kopie geben lassen können.",
        source_guideline_id="015-059", source_section_number="16.2",
        source_section_title="16.2 Datenschutz im Krankenhaus",
        source_chunk_id="015-059_patientenversion-p::sec-128::text-0", source_language="de",
    ),
    SilverQuestion(
        id=14, category="de_de",
        question="Warum verlangt die Durchführung großer operativer Eingriffe und Chemotherapien beim Vulvakarzinom spezialisierte Versorgungsstrukturen?",
        reference_answer="Große operative Eingriffe (Schwerpunktweiterbildung Gynäkologische Onkologie) und Chemotherapien (Zusatzweiterbildung Medikamentöse Tumortherapie) erfordern Ärztinnen und Ärzte mit entsprechenden Qualifikationen und ausgewiesener onkologischer Erfahrung in der interdisziplinären Versorgung. Da die Zahl spezialisierter Weiterbildungsberechtigungen stagniert, ist eine spezialisierte Versorgung erforderlich.",
        source_guideline_id="015-059", source_section_number="",
        source_section_title="Diagnostik, Therapie und Nachsorge des Vulvakarzinoms und seiner Vorstufen",
        source_chunk_id="015-059_langfassung-l::sec-52::text-0", source_language="de",
    ),
    SilverQuestion(
        id=15, category="de_de",
        question="Welche Empfehlung gibt die Leitlinie bezüglich alternativmedizinischer Behandlungsoptionen beim Vulvakarzinom, die zum Verzicht auf konventionelle Medizin führen?",
        reference_answer="Alternativmedizinische Behandlungsoptionen, die zum Verzicht auf konventionelle Medizin führen, sollen bei Patientinnen mit Vulvakarzinom abgelehnt werden, da keine Studien vorliegen und sie bezüglich Nebenwirkungen, Interaktionen und Prognoseverschlechterung als bedenklich eingestuft werden.",
        source_guideline_id="015-059", source_section_number="15.5.",
        source_section_title="Bedeutung alternativmedizinischer Methoden",
        source_chunk_id="015-059_langfassung-l::sec-151::text-0", source_language="de",
    ),
    SilverQuestion(
        id=16, category="de_en",
        question="According to the patient guideline for vulvar cancer, what medical procedures are included under the term radiology?",
        reference_answer="Radiology encompasses all medical imaging modalities (such as X-ray, CT, ultrasound, and MRI) as well as radiological procedures used for treatment, such as radiation therapy.",
        source_guideline_id="015-059", source_section_number="",
        source_section_title="Radiologie",
        source_chunk_id="015-059_patientenversion-p::sec-251::text-0", source_language="de",
    ),
    SilverQuestion(
        id=17, category="de_en",
        question="Who serves as the AWMF guideline representative (AWMF-Leitlinienbeauftragter) for the DGGG guideline commission?",
        reference_answer="Prof. Dr. Erich-Franz Solomayer serves as the AWMF guideline representative (AWMF-Leitlinienbeauftragter).",
        source_guideline_id="015-059", source_section_number="",
        source_section_title="Leitlinienkommission der DGGG",
        source_chunk_id="015-059_langfassung-l::sec-10::table-0", source_language="de",
    ),

    # =========================================================================
    # GUIDELINE 4: 032-034OL (S3-Leitlinie Endometriumkarzinom)
    # =========================================================================
    SilverQuestion(
        id=18, category="de_de",
        question="Welches Therapieschema wird als Qualitätsindikator für Patientinnen mit High-Risk-Endometriumkarzinom nach ESGO/ESTRO-Kriterien gefordert?",
        reference_answer="Patientinnen mit High-Risk-Endometriumkarzinom sollen eine adjuvante Chemotherapie mit 6 Zyklen Carboplatin (AUC 5 oder 6) und Paclitaxel (175 mg/m²) oder eine kombinierte Strahlen-Chemotherapie nach dem PORTEC-III-Schema erhalten.",
        source_guideline_id="032-034OL", source_section_number="12.",
        source_section_title="12 Qualitätsindikatoren",
        source_chunk_id="032-034OL_langfassung-l::sec-147::table-8", source_language="de",
    ),
    SilverQuestion(
        id=19, category="de_de",
        question="Wie sollen Therapieentscheidungen für ältere oder fragile Patientinnen mit Endometriumkarzinom getroffen werden?",
        reference_answer="Therapieentscheidungen für ältere Patientinnen sollen von den aktuellen Standardempfehlungen ausgehen und durch den Allgemeinstatus, die Lebenserwartung, die Patientenpräferenz sowie eine individuelle Nutzen-Risiko-Abwägung modifiziert werden.",
        source_guideline_id="032-034OL", source_section_number="11.1.",
        source_section_title="11 Fragile Patientinnen/Geriatrisches Assessment",
        source_chunk_id="032-034OL_langfassung-l::sec-146::table-0", source_language="de",
    ),
    SilverQuestion(
        id=20, category="de_de",
        question="Welche Telefonnummer bietet der Beratungsdienst INFONETZ KREBS für kostenlose Beratungen von Krebskranken und Angehörigen an?",
        reference_answer="Der Beratungsdienst INFONETZ KREBS bietet unter der kostenlosen Telefonnummer 0800 80708877 (Montag bis Freitag 8:00 - 17:00 Uhr) Unterstützung und Beratung an.",
        source_guideline_id="032-034OL", source_section_number="",
        source_section_title="INFONETZ KREBS",
        source_chunk_id="032-034OL_patientenversion-p1::sec-253::text-0", source_language="de",
    ),
    SilverQuestion(
        id=21, category="de_en",
        question="What alternative adjuvant treatment options exist for patients with endometrioid endometrial cancer and positive lymph nodes (Stage III-IVA) besides systemic chemotherapy alone?",
        reference_answer="Alternatively to systemic chemotherapy or simultaneous radiochemotherapy followed by systemic chemotherapy according to the PORTEC-III scheme, systemic chemotherapy can be combined with vaginal brachytherapy or sequential percutaneous radiotherapy.",
        source_guideline_id="032-034OL", source_section_number="7.3.",
        source_section_title="7.3 Strahlentherapie in fortgeschrittenen Stadien",
        source_chunk_id="032-034OL_langfassung-l::sec-103::table-0", source_language="de",
    ),
    SilverQuestion(
        id=22, category="de_en",
        question="What proportion of endometrial cancers belong to the POLE mutant molecular subtype, and what is the mutation frequency characteristic of this subtype?",
        reference_answer="The POLE mutant subtype accounts for approximately 9% of endometrial cancers, and it is characterized by a very high number of mutations (ultra-mutated).",
        source_guideline_id="032-034OL", source_section_number="4.5.15.",
        source_section_title="4.5.15 Molekulare Klassifikation des Endometriumkarzinoms",
        source_chunk_id="032-034OL_langfassung-l::sec-74::table-2", source_language="de",
    ),

    # =========================================================================
    # GUIDELINE 5: 032-035OL (S3-Leitlinie Ovarialkarzinom)
    # =========================================================================
    SilverQuestion(
        id=23, category="de_de",
        question="Was unterscheidet G1 (low grade) Tumorgewebe von G3 (high grade) Tumorgewebe beim Eierstockkrebs?",
        reference_answer="G1-Gewebe (low grade) ist dem normalen Eierstockgewebe noch ähnlich, gut differenziert und gilt als weniger aggressiv mit niedriger Wachstumsrate. G3-Gewebe (high grade) unterscheidet sich stark oder vollständig vom normalen Gewebe, ist schlecht/undifferenziert und wächst aggressiv mit hoher Wachstumsrate.",
        source_guideline_id="032-035OL", source_section_number="3.",
        source_section_title="Differenzierung der Krebszellen (Grading)",
        source_chunk_id="032-035OL_patientenversion-p::sec-30::text-0", source_language="de",
    ),
    SilverQuestion(
        id=24, category="de_de",
        question="Welche sehr häufigen Nebenwirkungen treten bei der Behandlung mit Cisplatin auf?",
        reference_answer="Sehr häufig treten Nierenfunktionsstörungen, Nerven- und Hörschädigungen sowie Übelkeit, Erbrechen und Veränderungen des Blutbildes auf.",
        source_guideline_id="032-035OL", source_section_number="",
        source_section_title="Cisplatin",
        source_chunk_id="032-035OL_patientenversion-p::sec-52::text-0", source_language="de",
    ),
    SilverQuestion(
        id=25, category="de_de",
        question="Warum suchen sich viele Krebserkrankte trotz des Wunsches nach Alltagskontinuität Unterstützung im Alltag?",
        reference_answer="Nicht nur die Erkrankung selbst, sondern auch die Behandlungen und ihre Folgen haben Auswirkungen auf das gewohnte Leben. Professionelle oder persönliche Unterstützung hilft Betroffenen und Angehörigen, mit schwierigen Situationen und Gegebenheiten umzugehen.",
        source_guideline_id="032-035OL", source_section_number="14.",
        source_section_title="14. Leben mit Krebs - den Alltag bewältigen",
        source_chunk_id="032-035OL_patientenversion-p::sec-94::text-0", source_language="de",
    ),
    SilverQuestion(
        id=26, category="de_en",
        question="In asymptomatic patients with ovarian cancer, does initiating recurrence treatment early based solely on an elevated CA 125 level improve overall survival?",
        reference_answer="No. An earlier presymptomatic initiation of recurrence treatment based solely on an elevated CA 125 level in asymptomatic patients is not associated with improved overall survival.",
        source_guideline_id="032-035OL", source_section_number="3.4.",
        source_section_title="3.4 Rezidivdiagnostik",
        source_chunk_id="032-035OL_langfassung-l::sec-42::rec-0", source_language="de",
    ),
    SilverQuestion(
        id=27, category="de_en",
        question="What is the main role of the German Cancer Society (Deutsche Krebsgesellschaft) as described in the patient guideline?",
        reference_answer="The German Cancer Society is the largest scientific-oncological society in Germany and provides up-to-date information on the diagnosis and treatment of cancer on its website.",
        source_guideline_id="032-035OL", source_section_number="",
        source_section_title="Deutsche Krebsgesellschaft",
        source_chunk_id="032-035OL_patientenversion-p::sec-157::text-0", source_language="de",
    ),

    # =========================================================================
    # GUIDELINE 6: 032-042 (S2k-Leitlinie Vaginalkarzinom)
    # =========================================================================
    SilverQuestion(
        id=28, category="de_de",
        question="Welchen Vorteil bietet die funktionelle Becken-MRT gegenüber der PET/CT bei der Beurteilung von intrapelvinen Vaginalkarzinomrezidiven?",
        reference_answer="Die funktionelle Becken-MRT kann gut zwischen Narbengewebe und Rezidivtumor unterscheiden, liefert eine bessere Einschätzung der lokalen Tumorinfiltration und des Tumorvolumens und geht im Gegensatz zur PET/CT nicht mit Strahlenexposition oder Radiotracergabe einher.",
        source_guideline_id="032-042", source_section_number="5.E20",
        source_section_title="Rezidivdiagnostik",
        source_chunk_id="032-042_langfassung::sec-137::text-0", source_language="de",
    ),
    SilverQuestion(
        id=29, category="de_de",
        question="Wer bleibt laut § 2 UrhG stets der eigentliche Urheber einer medizinisch-wissenschaftlichen S2k-Leitlinie?",
        reference_answer="Urheber im Sinne des § 2 UrhG bleibt immer die Miturhebergemeinschaft aller natürlichen Personen (Autoren), die an der Erstellung des Werkes beteiligt waren. Fachgesellschaften erhalten lediglich repräsentative Nutzungs- und Verwertungsrechte.",
        source_guideline_id="032-042", source_section_number="",
        source_section_title="Urheberrecht",
        source_chunk_id="032-042_langfassung::sec-30::text-0", source_language="de",
    ),
    SilverQuestion(
        id=30, category="de_de",
        question="Welche AWMF-Fachgesellschaften vertrat Prof. Dr. med. Lars Christian Horn in der Leitliniengruppe zum Vaginalkarzinom?",
        reference_answer="Prof. Dr. med. Lars Christian Horn vertrat die Fachgesellschaften DGP (Deutsche Gesellschaft für Pathologie), DKG und AGO.",
        source_guideline_id="032-042", source_section_number="",
        source_section_title="Leitliniengruppe",
        source_chunk_id="032-042_langfassung::sec-10::table-1", source_language="de",
    ),
    SilverQuestion(
        id=31, category="de_en",
        question="How many scientific societies appointed representatives for the consensus conference of the S2k vaginal cancer guideline?",
        reference_answer="A total of 30 scientific societies appointed representatives for the consensus conference after reminders were sent out.",
        source_guideline_id="032-042", source_section_number="",
        source_section_title="Leitlinienreport",
        source_chunk_id="032-042_langfassung::sec-61::text-3", source_language="de",
    ),
    SilverQuestion(
        id=32, category="de_en",
        question="What was the purpose of appointing representatives from various medical societies during the development of the vaginal cancer guideline?",
        reference_answer="The purpose was to ensure broad representation of the user target group (Anwenderzielgruppe) across all relevant clinical specialties during the text creation and consensus conference.",
        source_guideline_id="032-042", source_section_number="",
        source_section_title="Leitlinieninformationen",
        source_chunk_id="032-042_langfassung::sec-11::text-0", source_language="de",
    ),

    # =========================================================================
    # GUIDELINE 7: 032-055OL (S3-Leitlinie Komplementärmedizin)
    # =========================================================================
    SilverQuestion(
        id=33, category="de_de",
        question="Was ist der Hauptunterschied in der Zielsetzung zwischen Alternativmedizin und Komplementärmedizin bei Krebserkrankungen?",
        reference_answer="Alternativmedizin versteht sich als Ersatz für konventionelle Behandlungen und zielt darauf ab, die Erkrankung direkt zu heilen. Komplementärmedizin versteht sich als Ergänzung zur Schulmedizin und zielt darauf ab, Symptome und Nebenwirkungen zu kontrollieren und die Lebensqualität zu stärken.",
        source_guideline_id="032-055OL", source_section_number="1.",
        source_section_title="Was versteht man unter Komplementärmedizin?",
        source_chunk_id="032-055OL_patientenversion-p1::sec-21::text-0", source_language="de",
    ),
    SilverQuestion(
        id=34, category="de_de",
        question="Warum rät die S3-Leitlinie Komplementärmedizin ausdrücklich von der Anwendung von Amygdalin (Laetrile, 'Vitamin B17') ab?",
        reference_answer="Weil erhebliche, möglicherweise lebensbedrohliche Nebenwirkungen durch eine Blausäurevergiftung (wie Übelkeit, Erbrechen, Schwindel, Kopfschmerzen und in Einzelfällen Todesfälle) auftreten können.",
        source_guideline_id="032-055OL", source_section_number="4.",
        source_section_title="Nebenwirkungen von Amygdalin",
        source_chunk_id="032-055OL_patientenversion-p1::sec-152::text-0", source_language="de",
    ),
    SilverQuestion(
        id=35, category="de_de",
        question="In welcher Stadt hat die Stiftung Deutsche Krebshilfe ihren Sitz laut der Patientenleitlinie?",
        reference_answer="Die Stiftung Deutsche Krebshilfe hat ihren Sitz in Bonn (Buschstraße 32, 53113 Bonn).",
        source_guideline_id="032-055OL", source_section_number="",
        source_section_title="Stiftung Deutsche Krebshilfe",
        source_chunk_id="032-055OL_patientenversion-p1::sec-255::text-0", source_language="de",
    ),
    SilverQuestion(
        id=36, category="de_en",
        question="What effect did real acupuncture have on post-radiation xerostomia compared to sham control in the Cochrane review by Furness et al. (2013)?",
        reference_answer="Real acupuncture achieved a statistically significant group difference in favor of the intervention compared to sham control for both evoked (SMD = 0.72) and spontaneous saliva flow (SMD = 0.76).",
        source_guideline_id="032-055OL", source_section_number="4.11",
        source_section_title="Xerostomie",
        source_chunk_id="032-055OL_langfassung-l::sec-39::text-1", source_language="de",
    ),
    SilverQuestion(
        id=37, category="de_en",
        question="Did systematic reviews of randomized controlled trials find a significant effect of Vitamin D supplementation on PSA response in prostate cancer patients?",
        reference_answer="No. Across the evaluated RCTs (such as Attia et al. and Beer et al.), no significant differences in PSA response were found between patients receiving Vitamin D and those receiving placebo.",
        source_guideline_id="032-055OL", source_section_number="4.",
        source_section_title="Studien ohne Spiegelmessung",
        source_chunk_id="032-055OL_langfassung-l::sec-263::text-3", source_language="de",
    ),
]
