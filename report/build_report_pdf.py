from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "concrete_strength_ml_report.pdf"
FIGURE = ROOT / "images" / "model_evaluation.png"


def register_fonts() -> tuple[str, str, str]:
    font_dir = Path(r"C:\Windows\Fonts")
    regular, bold, italic = "Times-Roman", "Times-Bold", "Times-Italic"
    candidates = {
        "TimesCustom": font_dir / "times.ttf",
        "TimesCustom-Bold": font_dir / "timesbd.ttf",
        "TimesCustom-Italic": font_dir / "timesi.ttf",
    }
    if all(path.exists() for path in candidates.values()):
        for name, path in candidates.items():
            pdfmetrics.registerFont(TTFont(name, str(path)))
        regular, bold, italic = "TimesCustom", "TimesCustom-Bold", "TimesCustom-Italic"
    return regular, bold, italic


REGULAR, BOLD, ITALIC = register_fonts()


def add_page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont(REGULAR, 8)
    canvas.drawCentredString(A4[0] / 2, 10 * mm, str(doc.page))
    canvas.restoreState()


styles = getSampleStyleSheet()
body = ParagraphStyle(
    "Body",
    parent=styles["BodyText"],
    fontName=REGULAR,
    fontSize=9.2,
    leading=11.0,
    spaceAfter=3.5,
    textColor=colors.black,
)
section = ParagraphStyle(
    "Section",
    parent=body,
    fontName=BOLD,
    fontSize=13.5,
    leading=15,
    spaceBefore=7,
    spaceAfter=4,
)
title = ParagraphStyle(
    "Title",
    parent=body,
    fontName=BOLD,
    fontSize=17,
    leading=19,
    alignment=TA_CENTER,
    spaceAfter=5,
)
center = ParagraphStyle(
    "Center",
    parent=body,
    alignment=TA_CENTER,
    fontSize=9,
    leading=10.5,
)
caption = ParagraphStyle(
    "Caption",
    parent=body,
    fontName=ITALIC,
    fontSize=8.2,
    leading=9.5,
    alignment=TA_CENTER,
    spaceBefore=2,
    spaceAfter=4,
)
small = ParagraphStyle(
    "Small",
    parent=body,
    fontSize=8.1,
    leading=9.5,
    spaceAfter=2.5,
)


story = [
    Paragraph("Predicting Concrete Compressive Strength from Mix Design and Curing Age", title),
    Paragraph("Siyu Lu (G2509922G)", center),
    Paragraph(
        'Course: Machine Learning Mini-Project &nbsp;&nbsp; | &nbsp;&nbsp; '
        'GitHub: <link href="https://github.com/lucylululu721/concrete-strength-ml" color="#204a87">'
        'https://github.com/lucylululu721/concrete-strength-ml</link>',
        center,
    ),
    Spacer(1, 3),
    Paragraph("1&nbsp;&nbsp; Problem and Motivation", section),
    Paragraph(
        "Compressive-strength tests take time and destroy the tested specimens. I therefore treated early "
        "strength estimation as a supervised regression problem. The inputs are cement, blast-furnace "
        "slag, fly ash, water, superplasticizer, coarse aggregate, fine aggregate (all kg/m<super>3</super>), "
        "and curing age (days); the output is compressive strength in MPa. Such a model could help screen "
        "candidate mixes, although it cannot replace laboratory tests or code checks.",
        body,
    ),
    Paragraph("2&nbsp;&nbsp; Dataset and Method", section),
    Paragraph(
        "I used the UCI Concrete Compressive Strength dataset: 1,030 observations, eight numerical inputs, "
        "one target, and no missing values [1,2]. Using random seed 42, I split it into 720 training, 155 "
        "validation, and 155 test observations. The test set was kept aside until model choices had been fixed.",
        body,
    ),
    Paragraph(
        "I compared three models. Standardized Ridge regression was the linear baseline. Random forest "
        "represented bagging, while gradient boosting added trees sequentially to correct earlier errors. "
        "I tuned each model using validation RMSE, refitted it on the combined training and validation data, "
        "and evaluated it once on the test set using MAE, RMSE, and R<super>2</super>. For the validation-selected "
        "model, permutation importance with 30 repeats measured sensitivity to each input. The implementation "
        "uses pandas and scikit-learn [3].",
        body,
    ),
    Paragraph("3&nbsp;&nbsp; Results and Discussion", section),
]

table_data = [
    ["Model", "Val. RMSE", "Test MAE", "Test RMSE", "Test R2"],
    ["Ridge regression", "11.731", "7.825", "9.698", "0.641"],
    ["Random forest", "5.222", "3.488", "4.937", "0.907"],
    ["Gradient boosting", "5.034", "2.778", "4.161", "0.934"],
]
result_table = Table(table_data, colWidths=[54 * mm, 27 * mm, 27 * mm, 29 * mm, 25 * mm])
result_table.setStyle(
    TableStyle(
        [
            ("FONTNAME", (0, 0), (-1, 0), BOLD),
            ("FONTNAME", (0, 1), (-1, -1), REGULAR),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("ALIGN", (1, 0), (-1, -1), "CENTER"),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e6e6e6")),
            ("LINEABOVE", (0, 0), (-1, 0), 0.8, colors.black),
            ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.black),
            ("LINEBELOW", (0, -1), (-1, -1), 0.8, colors.black),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]
    )
)
parameter_table = Table(
    [
        ["Model", "Selected configuration"],
        ["Ridge regression", "alpha = 10"],
        [
            "Random forest",
            "400 trees; unrestricted depth; minimum leaf = 1; all features per split",
        ],
        [
            "Gradient boosting",
            "400 trees; learning rate = 0.1; depth = 3; minimum leaf = 1",
        ],
    ],
    colWidths=[42 * mm, 120 * mm],
)
parameter_table.setStyle(
    TableStyle(
        [
            ("FONTNAME", (0, 0), (-1, 0), BOLD),
            ("FONTNAME", (0, 1), (-1, -1), REGULAR),
            ("FONTSIZE", (0, 0), (-1, -1), 8.4),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e6e6e6")),
            ("LINEABOVE", (0, 0), (-1, 0), 0.8, colors.black),
            ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.black),
            ("LINEBELOW", (0, -1), (-1, -1), 0.8, colors.black),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]
    )
)
story.extend(
    [
        Paragraph("Table 1. Validation and held-out test performance. Errors are in MPa.", caption),
        result_table,
        Spacer(1, 4),
        Paragraph(
            "Gradient boosting had the lowest validation RMSE, so I chose it before looking at test performance. "
            "On the test set it reduced MAE by 64% and RMSE by 57% relative to Ridge, and it also improved on "
            "the random forest. Its test R<super>2</super> was 0.934.",
            body,
        ),
        Image(str(FIGURE), width=172 * mm, height=66 * mm),
        Paragraph(
            "Figure 1. Held-out predictions and gradient-boosting permutation importance. Error bars show one "
            "standard deviation over 30 permutations.",
            caption,
        ),
        Paragraph(
            "The gradient-boosting predictions follow the one-to-one line more closely, but several high-strength "
            "mixes are still underestimated. Permutation importance ranked age and cement highest, followed by "
            "water and slag. These rankings describe model sensitivity, not causal effects.",
            body,
        ),
        PageBreak(),
        Paragraph("4&nbsp;&nbsp; Conclusions and Reflection", section),
        Paragraph(
            "Gradient boosting gave the best result, with test MAE 2.78 MPa, RMSE 4.16 MPa, and R<super>2</super> "
            "of 0.934. The nonlinear models captured mix interactions that Ridge missed. The selected settings "
            "are listed below so the comparison can be repeated.",
            body,
        ),
        Paragraph("Table 2. Hyperparameters selected by validation RMSE.", caption),
        parameter_table,
        Spacer(1, 4),
        Paragraph(
            "The scatter plot also shows a weakness: predictions move toward the centre of the observed range, "
            "and several high-strength mixtures are underestimated. Extreme combinations are sparse in the 875 "
            "development observations, and tree models do not extrapolate well. MAE describes the typical error, "
            "whereas RMSE gives more weight to these less frequent large misses.",
            body,
        ),
        Paragraph(
            "A more important limitation is transfer to new data. The dataset does not record curing temperature, "
            "specimen geometry, testing standard, or material source, and a random split mainly measures "
            "interpolation. A useful next step would be an external or laboratory-grouped test with prediction "
            "intervals. Until then, the model is suitable for preliminary screening, not acceptance decisions.",
            body,
        ),
        Paragraph("Reproducibility", section),
        Paragraph(
            "The repository includes the data download and processing code, pinned dependencies, split seed, "
            "validation records, tests, and a JSON prediction example. The dataset is CC BY 4.0, and no pretrained "
            "model was used.",
            small,
        ),
        Paragraph("References", section),
        Paragraph(
            "[1] I.-C. Yeh. <i>Concrete Compressive Strength</i> [Dataset]. UCI Machine Learning "
            "Repository, 1998. https://doi.org/10.24432/C5PK67.<br/>"
            "[2] I.-C. Yeh. Modeling of strength of high-performance concrete using artificial neural "
            "networks. <i>Cement and Concrete Research</i>, 28(12):1797-1808, 1998.<br/>"
            "[3] F. Pedregosa et al. Scikit-learn: Machine learning in Python. <i>Journal of Machine "
            "Learning Research</i>, 12:2825-2830, 2011.",
            small,
        ),
    ]
)

doc = SimpleDocTemplate(
    str(OUTPUT),
    pagesize=A4,
    rightMargin=18 * mm,
    leftMargin=18 * mm,
    topMargin=14 * mm,
    bottomMargin=15 * mm,
    title="Predicting Concrete Compressive Strength",
    author="Siyu Lu",
)
doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
print(OUTPUT)

