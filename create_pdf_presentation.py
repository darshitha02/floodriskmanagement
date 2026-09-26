"""
Script to generate a professional PDF presentation deck for FloodRiskAI Platform
"""

import os
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, HRFlowable

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(BASE_DIR, "FloodRiskAI_Platform_Presentation.pdf")

def build_pdf():
    # Landscape letter size: 11 x 8.5 inches (792 x 612 pt)
    doc = SimpleDocTemplate(
        PDF_PATH,
        pagesize=landscape(letter),
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    # Custom Palette
    c_primary = colors.HexColor("#0b2e4f")
    c_secondary = colors.HexColor("#0d6efd")
    c_accent = colors.HexColor("#00d2ff")
    c_dark = colors.HexColor("#071927")
    c_text = colors.HexColor("#1e293b")

    # Custom Paragraph Styles
    title_style = ParagraphStyle(
        'CoverTitle',
        parent=styles['Title'],
        fontName='Helvetica-Bold',
        fontSize=28,
        leading=34,
        textColor=colors.white,
        alignment=0
    )
    
    subtitle_style = ParagraphStyle(
        'CoverSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=14,
        leading=18,
        textColor=c_accent,
        alignment=0
    )

    slide_title_style = ParagraphStyle(
        'SlideTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=c_primary,
        spaceAfter=8
    )

    body_style = ParagraphStyle(
        'SlideBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=11,
        leading=16,
        textColor=c_text,
        spaceAfter=6
    )

    story = []

    def make_header(title_text, category_text="FLOODRISKAI PLATFORM PRESENTATION"):
        return [
            Paragraph(f"<font color='#0d6efd'><b>{category_text.upper()}</b></font>", body_style),
            Paragraph(title_text, slide_title_style),
            HRFlowable(width="100%", thickness=2, color=c_secondary, spaceBefore=4, spaceAfter=12)
        ]

    # -------------------------------------------------------------------------
    # SLIDE 1: COVER SLIDE
    # -------------------------------------------------------------------------
    cover_table_data = [
        [
            Paragraph("FloodRiskAI Platform Presentation", title_style),
        ],
        [
            Spacer(1, 10),
        ],
        [
            Paragraph("Next-Generation AI Disaster Intelligence & Early Warning System", title_style),
        ],
        [
            Spacer(1, 15),
        ],
        [
            Paragraph("DataQuest 2026 · Disaster Management Analytics (Level 1 + Level 2 + Level 3)", subtitle_style),
        ],
        [
            Paragraph("Vignan's Foundation for Science, Technology & Research", subtitle_style),
        ]
    ]
    cover_table = Table(cover_table_data, colWidths=[720])
    cover_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), c_dark),
        ('PADDING', (0,0), (-1,-1), 30),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,-1), (-1,-1), 40),
    ]))
    story.append(cover_table)
    story.append(PageBreak())

    # -------------------------------------------------------------------------
    # SLIDE 2: EXECUTIVE SUMMARY & PROBLEM STATEMENT
    # -------------------------------------------------------------------------
    story.extend(make_header("Executive Summary & Problem Statement"))
    story.append(Paragraph("<b>The Disaster Challenge:</b> Annual monsoon flooding affects millions across Indian river basins (Assam, Kerala, Bihar, Odisha, Maharashtra). Existing tools lack integrated AI risk scoring, real-time river telemetry, climate stress simulation, and automated emergency playbook dispatch.", body_style))
    story.append(Spacer(1, 10))

    exec_data = [
        [
            Paragraph("<b>Core Objectives</b>", ParagraphStyle('H', parent=body_style, fontName='Helvetica-Bold', textColor=c_primary)),
            Paragraph("<b>Platform Innovation</b>", ParagraphStyle('H', parent=body_style, fontName='Helvetica-Bold', textColor=c_primary)),
        ],
        [
            Paragraph("• Transition from static historical reporting to <b>real-time predictive AI intelligence</b>.<br/>• Provide local <b>SHAP explainability</b> for every risk assessment.<br/>• Automate <b>Disaster Response Playbooks</b> for District Disaster Management Authorities (DDMA).", body_style),
            Paragraph("• <b>3-Tiered Platform:</b> Level 1 Analytics + Level 2 Multi-Model AI + Level 3 Climate Simulation.<br/>• <b>Location Search Engine:</b> Instant flood risk prediction for 50+ Indian cities.<br/>• <b>Interactive GIS Mapping:</b> River telemetry stream + state heatmaps.", body_style),
        ]
    ]
    t_exec = Table(exec_data, colWidths=[350, 350])
    t_exec.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#e2e8f0")),
        ('BACKGROUND', (0,1), (-1,1), colors.HexColor("#f1f5f9")),
        ('PADDING', (0,0), (-1,-1), 12),
        ('GRID', (0,0), (-1,-1), 1, colors.HexColor("#cbd5e1")),
    ]))
    story.append(t_exec)
    story.append(PageBreak())

    # -------------------------------------------------------------------------
    # SLIDE 3: 3-TIERED ARCHITECTURE
    # -------------------------------------------------------------------------
    story.extend(make_header("System Architecture: 3-Tiered Intelligence Platform"))
    
    arch_data = [
        [
            Paragraph("<b>LEVEL 1 · ANALYTICS</b>", ParagraphStyle('A1', parent=body_style, fontName='Helvetica-Bold', textColor=colors.HexColor("#0284c7"))),
            Paragraph("<b>LEVEL 2 · AI PREDICTION</b>", ParagraphStyle('A2', parent=body_style, fontName='Helvetica-Bold', textColor=colors.HexColor("#0d6efd"))),
            Paragraph("<b>LEVEL 3 · INTELLIGENCE</b>", ParagraphStyle('A3', parent=body_style, fontName='Helvetica-Bold', textColor=colors.HexColor("#7c3aed"))),
        ],
        [
            Paragraph("• Historical timeline trends (1967-2023)<br/>• Severity class ratio analysis<br/>• Primary meteorological trigger breakdown<br/>• Real-time river telemetry monitoring stream", body_style),
            Paragraph("• Multi-Model Suite (Random Forest, SVM, Logistic Regression)<br/>• Local SHAP feature waterfall attributions<br/>• Animated SVG circular risk gauge<br/>• Historical event quick-loaders", body_style),
            Paragraph("• Climate scenario stress simulator<br/>• Peak discharge sensitivity curves<br/>• Location search geocoding engine<br/>• Automated DDMA response playbook dispatch", body_style),
        ]
    ]
    t_arch = Table(arch_data, colWidths=[230, 230, 240])
    t_arch.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#e2e8f0")),
        ('BACKGROUND', (0,1), (-1,1), colors.HexColor("#ffffff")),
        ('PADDING', (0,0), (-1,-1), 12),
        ('GRID', (0,0), (-1,-1), 1, colors.HexColor("#cbd5e1")),
    ]))
    story.append(t_arch)
    story.append(PageBreak())

    # -------------------------------------------------------------------------
    # SLIDE 4: DATASETS & DATA ENGINE
    # -------------------------------------------------------------------------
    story.extend(make_header("Data Infrastructure & Benchmark Datasets"))
    
    data_table_data = [
        [
            Paragraph("<b>Dataset Component</b>", ParagraphStyle('DH', parent=body_style, fontName='Helvetica-Bold', textColor=c_primary)),
            Paragraph("<b>Volume & Scope</b>", ParagraphStyle('DH', parent=body_style, fontName='Helvetica-Bold', textColor=c_primary)),
            Paragraph("<b>Primary Features & Purpose</b>", ParagraphStyle('DH', parent=body_style, fontName='Helvetica-Bold', textColor=c_primary)),
        ],
        [
            Paragraph("<b>Gauge Flood Events Dataset</b><br/>(`floodevents_indofloods.csv`)", body_style),
            Paragraph("<b>4,548 Events</b><br/>Indo-Floods Benchmark", body_style),
            Paragraph("Peak Discharge ($m^3/s$), Flood Volume, Event Duration, Time to Peak, Recession Time, Start Month, Flood Type (Moderate vs. Severe). Used for ML model training.", body_style),
        ],
        [
            Paragraph("<b>India Flood Inventory Dataset</b><br/>(`India_Flood_Inventory_v3.csv`)", body_style),
            Paragraph("<b>6,876 Records</b><br/>IMD & NDMA Historical", body_style),
            Paragraph("State & District boundaries, Main Causes (Heavy Rain, Cyclone, Cloudburst), Fatalities, Displaced Persons, Area Affected. Used for Level 1 GIS Heatmaps.", body_style),
        ],
        [
            Paragraph("<b>River Telemetry Stream</b><br/>(Live JSON Stream)", body_style),
            Paragraph("<b>Major River Basins</b><br/>Brahmaputra, Ganga, etc.", body_style),
            Paragraph("Real-time river stage levels vs. danger thresholds, discharge rates, telemetry trend (rising/receding), and status tags (CRITICAL, WARNING, NORMAL).", body_style),
        ]
    ]
    t_data = Table(data_table_data, colWidths=[220, 150, 350])
    t_data.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0b2e4f")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('PADDING', (0,0), (-1,-1), 10),
        ('GRID', (0,0), (-1,-1), 1, colors.HexColor("#cbd5e1")),
    ]))
    story.append(t_data)
    story.append(PageBreak())

    # -------------------------------------------------------------------------
    # SLIDE 5: MACHINE LEARNING & BENCHMARKS
    # -------------------------------------------------------------------------
    story.extend(make_header("Machine Learning Suite & Performance Benchmarks"))
    story.append(Paragraph("Evaluated multiple machine learning algorithms on test data split (25% holdout) with standard scaling:", body_style))
    story.append(Spacer(1, 6))

    ml_data = [
        [
            Paragraph("<b>Model Architecture</b>", ParagraphStyle('MH', parent=body_style, fontName='Helvetica-Bold', textColor=c_primary)),
            Paragraph("<b>Accuracy (%)</b>", ParagraphStyle('MH', parent=body_style, fontName='Helvetica-Bold', textColor=c_primary)),
            Paragraph("<b>Precision (%)</b>", ParagraphStyle('MH', parent=body_style, fontName='Helvetica-Bold', textColor=c_primary)),
            Paragraph("<b>Recall (%)</b>", ParagraphStyle('MH', parent=body_style, fontName='Helvetica-Bold', textColor=c_primary)),
            Paragraph("<b>F1-Score (%)</b>", ParagraphStyle('MH', parent=body_style, fontName='Helvetica-Bold', textColor=c_primary)),
            Paragraph("<b>ROC-AUC Score</b>", ParagraphStyle('MH', parent=body_style, fontName='Helvetica-Bold', textColor=c_primary)),
        ],
        [
            Paragraph("<b>Random Forest Classifier</b> (Primary)", body_style),
            Paragraph("<b>82.5%</b>", body_style),
            Paragraph("75.4%", body_style),
            Paragraph("75.9%", body_style),
            Paragraph("75.6%", body_style),
            Paragraph("<b>0.879</b>", body_style),
        ],
        [
            Paragraph("<b>Support Vector Machine (SVM)</b>", body_style),
            Paragraph("71.3%", body_style),
            Paragraph("77.9%", body_style),
            Paragraph("27.8%", body_style),
            Paragraph("40.9%", body_style),
            Paragraph("0.763", body_style),
        ],
        [
            Paragraph("<b>Logistic Regression Classifier</b>", body_style),
            Paragraph("69.8%", body_style),
            Paragraph("62.1%", body_style),
            Paragraph("34.5%", body_style),
            Paragraph("44.3%", body_style),
            Paragraph("0.742", body_style),
        ]
    ]
    t_ml = Table(ml_data, colWidths=[200, 100, 100, 100, 100, 120])
    t_ml.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0d6efd")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('PADDING', (0,0), (-1,-1), 10),
        ('GRID', (0,0), (-1,-1), 1, colors.HexColor("#cbd5e1")),
    ]))
    story.append(t_ml)
    story.append(Spacer(1, 10))
    story.append(Paragraph("<b>SHAP Feature Contribution Decomposition:</b> Local feature impacts compute how parameters like <i>Peak Discharge (+18.5%)</i> and <i>Flood Volume (+12.1%)</i> push risk scores up relative to population baseline means.", body_style))
    story.append(PageBreak())

    # -------------------------------------------------------------------------
    # SLIDE 6: LOCATION SEARCH ENGINE
    # -------------------------------------------------------------------------
    story.extend(make_header("Location Intelligence & Place Risk Search"))
    story.append(Paragraph("Users can search <b>any Indian City, District, or State</b> (e.g. <i>Guwahati, Wayanad, Patna, Kochi, Cuttack, Mumbai, Delhi, Varanasi</i>) to instantly predict location-specific flood risk.", body_style))
    story.append(Spacer(1, 10))

    loc_data = [
        [
            Paragraph("<b>Searched Location</b>", ParagraphStyle('LH', parent=body_style, fontName='Helvetica-Bold', textColor=c_primary)),
            Paragraph("<b>River Basin System</b>", ParagraphStyle('LH', parent=body_style, fontName='Helvetica-Bold', textColor=c_primary)),
            Paragraph("<b>Typical Peak Discharge</b>", ParagraphStyle('LH', parent=body_style, fontName='Helvetica-Bold', textColor=c_primary)),
            Paragraph("<b>Predicted Flood Risk Score</b>", ParagraphStyle('LH', parent=body_style, fontName='Helvetica-Bold', textColor=c_primary)),
        ],
        [
            Paragraph("<b>Guwahati, Assam</b>", body_style),
            Paragraph("Brahmaputra Basin", body_style),
            Paragraph("14,200 cumec", body_style),
            Paragraph("<font color='#ef4444'><b>82.0% (HIGH RISK)</b></font>", body_style),
        ],
        [
            Paragraph("<b>Wayanad, Kerala</b>", body_style),
            Paragraph("Kabini / Chaliyar Rivers", body_style),
            Paragraph("7,200 cumec", body_style),
            Paragraph("<font color='#f59e0b'><b>63.5% (MEDIUM RISK)</b></font>", body_style),
        ],
        [
            Paragraph("<b>Patna, Bihar</b>", body_style),
            Paragraph("Ganga Basin", body_style),
            Paragraph("8,900 cumec", body_style),
            Paragraph("<font color='#ef4444'><b>74.2% (HIGH RISK)</b></font>", body_style),
        ],
        [
            Paragraph("<b>Mumbai, Maharashtra</b>", body_style),
            Paragraph("Mithi / Coastal Drainage", body_style),
            Paragraph("5,200 cumec", body_style),
            Paragraph("<font color='#f59e0b'><b>52.1% (MEDIUM RISK)</b></font>", body_style),
        ]
    ]
    t_loc = Table(loc_data, colWidths=[180, 180, 180, 180])
    t_loc.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0b2e4f")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('PADDING', (0,0), (-1,-1), 10),
        ('GRID', (0,0), (-1,-1), 1, colors.HexColor("#cbd5e1")),
    ]))
    story.append(t_loc)
    story.append(PageBreak())

    # -------------------------------------------------------------------------
    # SLIDE 7: EMERGENCY ALERT CENTER & PLAYBOOKS
    # -------------------------------------------------------------------------
    story.extend(make_header("Emergency Alert Center & Automated Response Playbooks"))
    story.append(Paragraph("Automated threat level classification triggers instant localized operational action plans for District Disaster Management Authorities (DDMA):", body_style))
    story.append(Spacer(1, 10))

    alert_data = [
        [
            Paragraph("<b>Threat Level</b>", ParagraphStyle('AH', parent=body_style, fontName='Helvetica-Bold', textColor=c_primary)),
            Paragraph("<b>Trigger Condition</b>", ParagraphStyle('AH', parent=body_style, fontName='Helvetica-Bold', textColor=c_primary)),
            Paragraph("<b>Automated Operational Response Playbook</b>", ParagraphStyle('AH', parent=body_style, fontName='Helvetica-Bold', textColor=c_primary)),
        ],
        [
            Paragraph("<font color='#ef4444'><b>RED ALERT (CRITICAL)</b></font>", body_style),
            Paragraph("Water Level > Danger Level<br/>Risk Score ≥ 66%", body_style),
            Paragraph("• Mandatory floodplain evacuation within 2km radius.<br/>• Dispatch 2 NDRF search-and-rescue teams with motorboats.<br/>• Ready 12 government school shelters (capacity 15,000 displaced persons).", body_style),
        ],
        [
            Paragraph("<font color='#f59e0b'><b>YELLOW ALERT (WARNING)</b></font>", body_style),
            Paragraph("Water Level near Danger<br/>Risk Score 33% - 66%", body_style),
            Paragraph("• Issue preliminary warning advisories to low-lying villages.<br/>• Standby SDRF units alerted.<br/>• Increase telemetry monitoring frequency to hourly interval readings.", body_style),
        ],
        [
            Paragraph("<font color='#10b981'><b>NORMAL</b></font>", body_style),
            Paragraph("Water Level Normal<br/>Risk Score < 33%", body_style),
            Paragraph("• Routine gauge recording and standard operational telemetry data logging.", body_style),
        ]
    ]
    t_alert = Table(alert_data, colWidths=[160, 180, 380])
    t_alert.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0b2e4f")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('PADDING', (0,0), (-1,-1), 10),
        ('GRID', (0,0), (-1,-1), 1, colors.HexColor("#cbd5e1")),
    ]))
    story.append(t_alert)
    story.append(PageBreak())

    # -------------------------------------------------------------------------
    # SLIDE 8: DEPLOYMENT & LIVE LINKS
    # -------------------------------------------------------------------------
    story.extend(make_header("Deployment Infrastructure & Live Working Links"))
    
    dep_data = [
        [
            Paragraph("<b>Deployment Target</b>", ParagraphStyle('DH2', parent=body_style, fontName='Helvetica-Bold', textColor=c_primary)),
            Paragraph("<b>URL / Location</b>", ParagraphStyle('DH2', parent=body_style, fontName='Helvetica-Bold', textColor=c_primary)),
            Paragraph("<b>Description</b>", ParagraphStyle('DH2', parent=body_style, fontName='Helvetica-Bold', textColor=c_primary)),
        ],
        [
            Paragraph("<b>Localhost Web Server</b>", body_style),
            Paragraph("`http://127.0.0.1:5000`", body_style),
            Paragraph("Flask WSGI Server with full Glassmorphism UI, GIS Leaflet maps, and live API endpoints.", body_style),
        ],
        [
            Paragraph("<b>GitHub Repository</b>", body_style),
            Paragraph("`https://github.com/darshitha02/floodriskmanagement`", body_style),
            Paragraph("Main source repository with commit history, ML models, and deployment configurations.", body_style),
        ],
        [
            Paragraph("<b>Render Cloud Web App</b>", body_style),
            Paragraph("`https://floodriskmanagement.onrender.com`", body_style),
            Paragraph("Production Flask web server deployment via Gunicorn (`Procfile` & `render.yaml`).", body_style),
        ],
        [
            Paragraph("<b>Streamlit Cloud App</b>", body_style),
            Paragraph("`https://share.streamlit.io`", body_style),
            Paragraph("Native Streamlit interactive cloud app (`streamlit_app.py`) with Plotly & Folium integration.", body_style),
        ]
    ]
    t_dep = Table(dep_data, colWidths=[180, 250, 290])
    t_dep.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0d6efd")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('PADDING', (0,0), (-1,-1), 10),
        ('GRID', (0,0), (-1,-1), 1, colors.HexColor("#cbd5e1")),
    ]))
    story.append(t_dep)
    story.append(Spacer(1, 20))
    story.append(Paragraph("<b>Conclusion:</b> FloodRiskAI delivers an extraordinary end-to-end disaster intelligence system empowering disaster management authorities, researchers, and citizens with accurate, explainable, and actionable flood risk insights.", body_style))

    # Build document
    doc.build(story)
    print("PDF build complete:", PDF_PATH)

if __name__ == "__main__":
    build_pdf()
