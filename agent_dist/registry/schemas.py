DEFAULT_INPUT_TYPES = {
    "dicom",
    "pdf",
    "text",
    "image",
    "hl7",
    "json"
}

INTENTS = {
    "anonymization": {
        "description": "Remove or mask sensitive patient information",
        "capabilities": {
            "dicom_anonymization": {
                "description": "Anonymize DICOM PHI"
            },
            "text_anonymization": {
                "description": "Anonymize clinical text"
            }
        }
    },
    "clinical_consultation": {
        "description": "Medical consultation and guidance",
        "capabilities": {
            "general_practice": {
                "description": "Primary care advice"
            },
            "cardiology": {
                "description": "Heart-related consultation"
            }
        }
    }
}
