import os
from io import BytesIO

def to_markdown(research_data: dict) -> str:
    summary = research_data.get("summary", "")
    key_findings = research_data.get("key_findings", [])
    sources = research_data.get("sources", [])
    
    md = f"# Research Report\n\n## Executive Summary\n{summary}\n\n## Detailed Findings\n"
    for finding in key_findings:
        md += f"- {finding}\n"
        
    if sources:
        md += "\n## Source References\n"
        for idx, s in enumerate(sources, 1):
            md += f"[{idx}] {s.get('filename')} (Page {s.get('page_number', '?')})\n"
            
    return md

def to_pdf(research_data: dict) -> BytesIO:
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet
        
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        story.append(Paragraph("Research Report", styles['Title']))
        story.append(Spacer(1, 12))
        
        # Summary
        story.append(Paragraph("Executive Summary", styles['Heading2']))
        story.append(Paragraph(research_data.get("summary", "").replace('\n', '<br/>'), styles['BodyText']))
        story.append(Spacer(1, 12))
        
        # Findings
        story.append(Paragraph("Detailed Findings", styles['Heading2']))
        for finding in research_data.get("key_findings", []):
            story.append(Paragraph(f"- {finding}", styles['BodyText']))
        story.append(Spacer(1, 12))
        
        # Sources
        sources = research_data.get("sources", [])
        if sources:
            story.append(Paragraph("Source References", styles['Heading2']))
            for idx, s in enumerate(sources, 1):
                story.append(Paragraph(f"[{idx}] {s.get('filename')} (Page {s.get('page_number', '?')})", styles['BodyText']))
                
        doc.build(story)
        buffer.seek(0)
        return buffer
    except ImportError:
        # Fallback to plain text bytes if reportlab fails
        buffer = BytesIO()
        buffer.write(to_markdown(research_data).encode("utf-8"))
        buffer.seek(0)
        return buffer
