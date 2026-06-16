"""
NOTE_TYPE-aware text cleaning for clinical notes.

Usage:
    from shared.utils import clean_note

    cleaned = clean_note(text, note_type='Clinician')

Rules are organized into:
  - UNIVERSAL_RULES: applied to all notes regardless of type
  - TYPE_SPECIFIC_RULES: keyed by NOTE_TYPE, applied only to matching notes
"""

import re

# Universal rules: These apply across all note types and are deduplicated from the analyses.
UNIVERSAL_RULES = [
    {
        'name': 'signature_block',
        'pattern': r'(?:Staff Surgeon|MD|PhD|NP|RN|DMD|MSN|Instructor in Medicine|Physician).*?(?:Dana Farber Cancer Institute|DFCI|Harvard Medical School|450 Brookline Avenue).*|(?:Signed by|This report was electronically signed by).*|By his/her signature.*?Electronically signed.*?\d{2}:\d{2}:\d{2}(?:AM|PM)',
        'replacement': '',
        'flags': re.MULTILINE | re.IGNORECASE,
        'confidence': 'high',
        'description': 'Removes signature blocks and institutional affiliations.'
    },
    {
        'name': 'confidentiality_disclaimer',
        'pattern': r'(?:DANA FARBER CANCER INSTITUTE|LANK CENTER FOR GENITOURINARY ONCOLOGY).*?(?:Boston, MA|Brookline Avenue).*|This report is limited to the body part and modality requested.*|Massachusetts General Physicians Organization.*?www\.mydermpath\.org',
        'replacement': '',
        'flags': re.MULTILINE | re.IGNORECASE,
        'confidence': 'high',
        'description': 'Removes confidentiality disclaimers and institutional headers.'
    },
    {
        'name': 'pagination_marker',
        'pattern': r'Page \d+ of \d+|\[Length: \d+ chars\]',
        'replacement': '',
        'flags': re.MULTILINE | re.IGNORECASE,
        'confidence': 'high',
        'description': 'Removes pagination markers.'
    },
    {
        'name': 'decorative_separator',
        'pattern': r'[=*-]{3,}',
        'replacement': '',
        'flags': re.MULTILINE,
        'confidence': 'high',
        'description': 'Removes decorative separators.'
    },
    {
        'name': 'collapse_blank_lines',
        'pattern': r'\n\s*\n+',
        'replacement': '\n\n',
        'flags': 0,
        'confidence': 'high',
        'description': 'Collapses multiple blank lines into one, preserving paragraph structure.'
    },
    {
        'name': 'collapse_horizontal_whitespace',
        'pattern': r'[ \t]{2,}',
        'replacement': ' ',
        'flags': 0,
        'confidence': 'high',
        'description': 'Collapses repeated spaces/tabs on the same line.'
    }
]

# Clinician-specific rules: These apply only to clinician notes.
CLINICIAN_RULES = [
    {
        'name': 'vitals_block',
        # Anchor to a line that starts with a vitals label and require a word-boundaried,
        # multi-character unit. The old pattern allowed bare single-letter units (m/C/F),
        # so a stray "BP"/"Temp" in prose would match up to the next c/f/m and delete text.
        'pattern': r'^[ \t]*(?:BP|Pulse|Temp|Resp|Ht|Wt|BMI)\b.*?\b(?:kg|lbs?|cm|mmHg|bpm)\b',
        'replacement': '',
        'flags': re.MULTILINE | re.IGNORECASE,
        'confidence': 'high',
        'description': 'Removes vitals blocks.'
    },
    {
        'name': 'medication_list',
        'pattern': r'(?:Current Outpatient Prescriptions|Medications Reviewed).*?(?:tablet|capsule|injection|spray).*',
        'replacement': '',
        'flags': re.MULTILINE | re.IGNORECASE,
        'confidence': 'high',
        'description': 'Removes medication list dumps.'
    },
    {
        'name': 'allergy_list',
        'pattern': r'(?:Allergies).*?(?:No Known Allergies|NKDA).*',
        'replacement': '',
        'flags': re.MULTILINE | re.IGNORECASE,
        'confidence': 'high',
        'description': 'Removes allergy lists.'
    },
    {
        'name': 'review_of_systems',
        'pattern': r'(?:Review of Systems).*?(?:negative|denies).*',
        'replacement': '',
        'flags': re.MULTILINE | re.IGNORECASE,
        'confidence': 'high',
        'description': 'Removes review of systems template text.'
    },
    {
        'name': 'problem_list',
        'pattern': r'(?:Problem List Items Addressed This Visit|Active Problem List).*',
        'replacement': '',
        'flags': re.MULTILINE | re.IGNORECASE,
        'confidence': 'high',
        'description': 'Removes problem list headers.'
    }
]

# Imaging-specific rules: These apply only to imaging notes.
IMAGING_RULES = [
    {
        'name': 'system_generated_headers',
        'pattern': r'(?:Exam Number|Report Status|Ordering Provider|Accession number)[:\s].*',
        'replacement': '',
        'flags': re.MULTILINE | re.IGNORECASE,
        'confidence': 'high',
        'description': 'Removes system-generated headers with metadata.'
    },
    {
        'name': 'technical_parameters',
        'pattern': r'(?:TECHNIQUE|CTDIvol|DLP|Dose|MRI COIL CHARGE).*',
        'replacement': '',
        'flags': re.MULTILINE | re.IGNORECASE,
        'confidence': 'high',
        'description': 'Removes technical parameters related to imaging techniques.'
    },
    {
        'name': 'standardized_report_header_labels',
        'pattern': r'^[ \t]*(INDICATION|COMPARISON|FINDINGS|IMPRESSION|TECHNIQUE|EXAM)[ \t]*:[ \t]*(?=\S)',
        'replacement': '',
        'flags': re.MULTILINE | re.IGNORECASE,
        'confidence': 'high',
        'description': 'Removes section header labels when inline with content (e.g. "FINDINGS: ..."), preserving the content itself.'
    }
]

# Pathology-specific rules: These apply only to pathology notes.
PATHOLOGY_RULES = [
    {
        'name': 'gross_description_boilerplate',
        'pattern': r'(GROSS DESCRIPTION.*?submitted in toto.*?Dictated by.*?Physician)',
        'replacement': '',
        'flags': re.MULTILINE | re.DOTALL | re.IGNORECASE,
        'confidence': 'high',
        'description': 'Removes gross description templates.'
    },
    {
        'name': 'staining_protocol_boilerplate',
        'pattern': r'(Immunohistochemistry performed.*?FDA has determined.*?not necessary)',
        'replacement': '',
        'flags': re.MULTILINE | re.DOTALL | re.IGNORECASE,
        'confidence': 'high',
        'description': 'Removes immunohistochemistry method boilerplate.'
    },
]

TYPE_SPECIFIC_RULES = {
    'Clinician': CLINICIAN_RULES,
    'Imaging': IMAGING_RULES,
    'Pathology': PATHOLOGY_RULES,
}

def clean_note(text, note_type=None):
    """Clean a clinical note by applying universal and type-specific regex rules.

    Args:
        text: Raw clinical note text.
        note_type: One of 'Clinician', 'Imaging', 'Pathology', or None.
            If None, only universal rules are applied.

    Returns:
        Cleaned text string.
    """
    text = str(text)

    # Apply universal rules
    for rule in UNIVERSAL_RULES:
        text = re.sub(rule['pattern'], rule['replacement'], text, flags=rule['flags'])

    # Apply type-specific rules
    if note_type and note_type in TYPE_SPECIFIC_RULES:
        for rule in TYPE_SPECIFIC_RULES[note_type]:
            text = re.sub(rule['pattern'], rule['replacement'], text, flags=rule['flags'])

    # Final whitespace cleanup
    text = re.sub(r'\n\s*\n+', '\n\n', text)
    return text.strip()
