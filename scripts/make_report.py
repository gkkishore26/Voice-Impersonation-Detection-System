# Builds the Word report for the Voice Impersonation Detection System final year project
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import os

FIGDIR = r"C:\Users\Kishore GK\Desktop\VIDS\Report\figures"
OUTDOC = r"C:\Users\Kishore GK\Desktop\VIDS\Report\Voice_Impersonation_Detection_System_Report.docx"

# Personal submission details live OUTSIDE version control so the public
# script carries no names. Local file: Report/details.json (gitignored).
def _load_details():
    defaults = {'student': '[STUDENT NAME]', 'register_no': '[REGISTER NO]',
                'guide': '[GUIDE NAME]', 'section': '[SECTION]',
                'college': '[COLLEGE NAME]', 'city': '[CITY]',
                'university_note': '', 'department': 'Department of Artificial Intelligence and Data Science',
                'academic_year': '2024 - 2027'}
    try:
        with open(os.path.join(os.path.dirname(OUTDOC), 'details.json'), encoding='utf-8') as f:
            defaults.update(json.load(f))
    except Exception:
        pass
    return defaults


P = _load_details()

doc = Document()

# ---------- page setup ----------
sec = doc.sections[0]
sec.page_width = Inches(8.27)
sec.page_height = Inches(11.69)
sec.left_margin = Inches(1.1)
sec.right_margin = Inches(1.1)
sec.top_margin = Inches(1.0)
sec.bottom_margin = Inches(1.0)

normal = doc.styles['Normal']
normal.font.name = 'Times New Roman'
normal.font.size = Pt(12)
normal.paragraph_format.line_spacing = 1.5
normal.paragraph_format.space_after = Pt(6)
rpr = normal.element.get_or_add_rPr()
rf = rpr.get_or_add_rFonts()
rf.set(qn('w:eastAsia'), 'Times New Roman')

for lvl, sz in ((1, 16), (2, 14), (3, 12.5)):
    st = doc.styles[f'Heading {lvl}']
    st.font.name = 'Times New Roman'
    st.font.size = Pt(sz)
    st.font.bold = True
    st.font.color.rgb = RGBColor(0, 0, 0)


def add_page_number(section):
    footer = section.footer
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    run.font.name = 'Times New Roman'
    run.font.size = Pt(11)
    f1 = OxmlElement('w:fldChar'); f1.set(qn('w:fldCharType'), 'begin')
    it = OxmlElement('w:instrText'); it.set(qn('xml:space'), 'preserve'); it.text = 'PAGE'
    f2 = OxmlElement('w:fldChar'); f2.set(qn('w:fldCharType'), 'end')
    run._r.append(f1); run._r.append(it); run._r.append(f2)


add_page_number(sec)


def para(text='', bold=False, italic=False, align='just', size=None, space_after=None,
         style=None):
    p = doc.add_paragraph(style=style)
    run = p.add_run(text)
    run.bold = bold
    run.italic = italic
    if size:
        run.font.size = Pt(size)
    if space_after is not None:
        p.paragraph_format.space_after = Pt(space_after)
    amap = {'just': WD_ALIGN_PARAGRAPH.JUSTIFY, 'center': WD_ALIGN_PARAGRAPH.CENTER,
            'left': WD_ALIGN_PARAGRAPH.LEFT, 'right': WD_ALIGN_PARAGRAPH.RIGHT}
    p.alignment = amap[align]
    return p


def h1(text):
    doc.add_heading(text, level=1)


def h2(text):
    doc.add_heading(text, level=2)


def h3(text):
    doc.add_heading(text, level=3)


def bullets(items):
    for it in items:
        p = doc.add_paragraph(style='List Bullet')
        p.add_run(it)
        p.paragraph_format.line_spacing = 1.4


def table(headers, rows, widths=None):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = 'Table Grid'
    hdr = t.rows[0].cells
    for i, htxt in enumerate(headers):
        hdr[i].text = ''
        run = hdr[i].paragraphs[0].add_run(htxt)
        run.bold = True
        run.font.size = Pt(10.5)
        run.font.name = 'Times New Roman'
        hdr[i].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    for row in rows:
        cells = t.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = ''
            run = cells[i].paragraphs[0].add_run(str(val))
            run.font.size = Pt(10.5)
            run.font.name = 'Times New Roman'
    doc.add_paragraph()
    return t


def figure(fname, caption, width=5.9):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(os.path.join(FIGDIR, fname), width=Inches(width))
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = cap.add_run(caption)
    run.bold = True
    run.font.size = Pt(10)
    cap.paragraph_format.space_after = Pt(12)


def pb():
    doc.add_page_break()


# ================= TITLE PAGE =================
para('A PROJECT REPORT', bold=True, align='center', size=15, space_after=2)
para('ON', align='center', size=13, space_after=2)
para('VOICE IMPERSONATION DETECTION SYSTEM', bold=True, align='center', size=20, space_after=2)
para('A Deep Learning Approach using Convolutional Neural Networks',
     italic=True, align='center', size=12.5, space_after=24)
para('Submitted in partial fulfillment of the requirements for the degree of',
     align='center', size=12, space_after=2)
para('BACHELOR OF SCIENCE IN COMPUTER SCIENCE\n(Artificial Intelligence and Data Science)',
     bold=True, align='center', size=12.5, space_after=18)
para('DEPARTMENT OF ARTIFICIAL INTELLIGENCE AND DATA SCIENCE', bold=True, align='center', size=12.5, space_after=2)
para(P['college'].upper(), bold=True, align='center', size=13, space_after=2)
para(P['city'].upper(), bold=True, align='center', size=12, space_after=2)
para(P['university_note'], align='center', size=12, space_after=30)
para('Submitted by:', bold=True, align='center', size=12, space_after=2)
para(P['student'], bold=True, align='center', size=14, space_after=2)
para('Register No. : ' + P['register_no'], align='center', size=12, space_after=2)
para(P['department'], align='center', size=12, space_after=24)
para('Under the guidance of', align='center', size=12, space_after=2)
para(P['guide'], bold=True, align='center', size=12, space_after=2)
pb()

# ================= CERTIFICATE =================
para('CERTIFICATE', bold=True, align='center', size=15, space_after=18)
para('Certified that this project report on the topic "Voice Impersonation Detection System" '
     'is the bonafide work of ' + P['student'] + ', bearing the register number ' + P['register_no'] + ', carried out '
     'under the guidance of ' + P['guide'] + ' during the academic year ' + P['academic_year'] + ' in '
     'partial fulfillment of the requirements for the award of the degree Bachelor of Science in Computer Science '
     '(Artificial Intelligence and Data Science) of ' + P['college'] + ', '
     + P['city'] + ' ' + P['university_note'] + '.',
     align='just')
para('')
para('This is to certify that the project entitled "Voice Impersonation Detection System" has '
     'not been submitted to any other institution for the award of any degree, diploma, '
     'fellowship or other similar titles.', align='just')
para('')
table(['Sl. No.', 'Name of the Student', 'Register No.', 'Section'],
      [['1', P['student'], P['register_no'], P['section']]])
para('')
para('Place: Coimbatore', align='left', space_after=2)
para('Date: DD-MM-YYYY', align='left', space_after=30)
para('Signature of the Guide', bold=True, align='left', size=11.5, space_after=4)
para(P['guide'] + '\n' + P['department'], size=11.5, space_after=20)
para('Signature of the Head of Department', bold=True, align='left', size=11.5, space_after=4)
para('[HOD NAME]\nHead of the Department', size=11.5, space_after=20)
para('Signature of the External Examiner', bold=True, align='left', size=11.5, space_after=4)
pb()

# ================= DECLARATION =================
para('DECLARATION', bold=True, align='center', size=15, space_after=18)
para('I hereby declare that the project report entitled "Voice Impersonation Detection '
     'System" submitted for the partial fulfillment of the requirements for the award of the '
     'degree Bachelor of Science in Computer Science (Artificial Intelligence and Data '
     'Science) is a record of bonafide work carried out by me under the guidance of '
     + P['guide'] + ', ' + P['department'] + ', '
     + P['college'] + ', ' + P['city'] + '.', align='just')
para('')
para('This project work has not been submitted, either in full or in part, for the award of '
     'any other degree, diploma, fellowship or other similar title.', align='just')
para('')
para('Place: Coimbatore', align='left', space_after=2)
para('Date: DD-MM-YYYY', align='left', space_after=40)
para(P['student'], bold=True, align='center', size=12.5, space_after=2)
para('Register No. : ' + P['register_no'], align='center', size=11.5)
pb()

# ================= ABSTRACT =================
h1('ABSTRACT')
para('The rapid advancement of speech synthesis and voice conversion technologies has made it '
     'increasingly easy to fabricate human speech. AI-generated voices, commonly known as voice '
     'deepfakes, are now being misused for fraud, impersonation, political manipulation and '
     'security breaches. Detecting such synthetic speech reliably has therefore become an '
     'important research problem in the field of digital forensics and information security.')
para('This project presents a Voice Impersonation Detection System that distinguishes between '
     'genuine human speech and AI-generated speech using a Convolutional Neural Network (CNN). '
     'The raw audio signal is first transformed into a Mel-spectrogram image using the librosa '
     'library. A Mel-spectrogram represents the time-frequency energy pattern of speech, and '
     'acts as a visual "fingerprint" of the speaker. Synthetic speech typically contains '
     'artificial spectral patterns, missing high-frequency detail and unnatural harmonic '
     'structures, which a CNN can learn to recognise automatically.')
para('A custom CNN model with four convolutional blocks, batch normalisation, dropout '
     'regularisation and a softmax classification head is trained on labelled real and fake '
     'audio clips. The trained model is integrated into a Flask web application in which the '
     'user can either upload an existing audio file or record voice directly through the '
     'browser microphone using the MediaRecorder API. The system processes the input, produces '
     'a prediction, and displays a detection report containing the predicted class (Real Human '
     'Voice or AI-Generated Voice) together with a confidence score.')
para('The proposed system provides an easy-to-use, browser-based solution for quick voice '
     'authenticity checks and can be extended in future into a standalone desktop application '
     'and a real-time streaming detector.')
para('')
para('Keywords: Voice deepfake, Audio impersonation, Convolutional Neural Network, '
     'Mel-spectrogram, PyTorch, Flask, Audio forensics.', italic=True, size=11)
pb()

# ================= CONTENTS =================
h1('CONTENTS')
contents = [
    'Abstract', 'Abbreviations', 'Chapter 1 : Introduction',
    'Chapter 2 : Literature Review', 'Chapter 3 : System Analysis',
    'Chapter 4 : System Design', 'Chapter 5 : Implementation',
    'Chapter 6 : Testing', 'Chapter 7 : Results and Discussion',
    'Chapter 8 : Conclusion and Future Scope', 'References',
]
for c in contents:
    p = doc.add_paragraph()
    p.add_run(c)
    p.paragraph_format.line_spacing = 1.4
h1('LIST OF FIGURES')
figures_list = [
    'Fig. 4.1 : System Architecture', 'Fig. 4.2 : Work Flow of the Proposed System',
    'Fig. 4.3 : Use Case Diagram', 'Fig. 4.4 : Data Flow Diagram – Level 0 (Context Diagram)',
    'Fig. 4.5 : Data Flow Diagram – Level 1', 'Fig. 4.6 : Sequence Diagram',
    'Fig. 4.7 : Project Timeline (Gantt Chart)',
    'Fig. 5.1 : Mel-Spectrogram Generation Pipeline',
    'Fig. 5.2 : CNN Model Architecture',
    'Fig. 5.3 : Spectral Pattern Difference between Real and AI-Generated Voice',
    'Fig. 7.1 : Confusion Matrix on the Validation Set',
    'Fig. 7.2 : Training and Validation Accuracy/Loss Curves',
    'Fig. 7.3 : Mel-Spectrogram Pattern Difference Learned by the CNN',
]
for c in figures_list:
    p = doc.add_paragraph()
    p.add_run(c)
    p.paragraph_format.line_spacing = 1.4
pb()

# ================= ABBREVIATIONS =================
h1('ABBREVIATIONS')
table(['Abbreviation', 'Expansion'], [
    ['AI', 'Artificial Intelligence'],
    ['API', 'Application Programming Interface'],
    ['CNN', 'Convolutional Neural Network'],
    ['DFD', 'Data Flow Diagram'],
    ['FFT', 'Fast Fourier Transform'],
    ['GUI', 'Graphical User Interface'],
    ['HTML', 'Hyper Text Markup Language'],
    ['JSON', 'JavaScript Object Notation'],
    ['ML', 'Machine Learning'],
    ['STFT', 'Short-Time Fourier Transform'],
    ['TTS', 'Text-to-Speech'],
    ['UI', 'User Interface'],
])
pb()

# ================= CHAPTER 1 =================
h1('CHAPTER 1 : INTRODUCTION')
h2('1.1 Background')
para('Speech is one of the most trusted forms of human identity. People recognise each other '
     'by voice, and voice-based authentication systems are widely used in banking, telephony '
     'and customer service. In the last few years, deep learning based Text-to-Speech (TTS) '
     'and voice conversion systems have reached a level where generated speech is almost '
     'indistinguishable from real speech to a human listener. Such synthetic speech is often '
     'referred to as a "voice deepfake".')
para('Criminals have already started misusing this technology. Incidents of voice-based '
     'financial fraud, fake kidnapping calls, and impersonation of public figures have been '
     'reported across the world. Hence there is a strong need for automated systems that can '
     'analyse a voice sample and report whether it is a genuine human voice or an '
     'AI-generated imitation.')
h2('1.2 Problem Statement')
para('Existing methods rely on human listening, which is unreliable because modern synthetic '
     'voices sound natural. Manual analysis also does not scale to large volumes of audio. '
     'Therefore an automated, deep-learning based system is required that can analyse the '
     'spectral characteristics of a voice sample and classify it as REAL or AI-GENERATED with '
     'a measurable confidence score.')
h2('1.3 Objectives')
bullets([
    'To study the characteristics that differentiate natural human speech from AI-generated speech.',
    'To convert raw audio signals into Mel-spectrogram images for feature extraction.',
    'To design and train a Convolutional Neural Network (CNN) for binary classification of voice samples.',
    'To develop a web-based interface that accepts uploaded audio files or live microphone recordings.',
    'To generate a detection report showing the predicted class and confidence percentage.',
])
h2('1.4 Scope of the Project')
para('The system accepts short voice clips (3 to 15 seconds) in standard formats such as WAV '
     'and MP3. It performs offline batch analysis of the uploaded clip and returns the result '
     'within a few seconds. The current scope covers detection of synthetic speech generated '
     'by modern neural TTS and voice-conversion systems. Real-time continuous monitoring, '
     'telephone-band (narrowband) audio and multi-language detection are kept for future '
     'enhancement.')
h2('1.5 Advantages')
bullets([
    'Automated and objective – no human listening required.',
    'Easy to use – runs in a normal web browser; nothing to install.',
    'Supports both pre-recorded files and live microphone input.',
    'Explainable – the Mel-spectrogram image can be visualised and shown to the user.',
    'Produces a quantified report with confidence percentage instead of a simple yes/no answer.',
])
h2('1.6 Limitations')
bullets([
    'Accuracy depends on the size and diversity of the training dataset.',
    'Very short or noisy clips may give unreliable results.',
    'Detection of heavily compressed (low-bitrate telephony) audio is difficult.',
    'A new generation of synthesisers may require model retraining.',
])
pb()

# ================= CHAPTER 2 =================
h1('CHAPTER 2 : LITERATURE REVIEW')
para('Several research efforts have studied the problem of synthetic speech detection. The '
     'major techniques reported in the literature are summarised below.')
table(['Sl. No.', 'Reference / Approach', 'Technique Used', 'Observation and Limitation'], [
    ['1', 'ASVspoof challenge [1]', 'Large-scale datasets with spoofed speech; hand-crafted features (CQCC, LFCC) with classifiers',
     'Established standard benchmark datasets; hand-crafted features generalise poorly to unseen attacks'],
    ['2', 'Zell et al. [2]', 'Evaluation of commercial and open-source synthesisers for human impersonation',
     'Showed that modern synthesizers produce speech that fools human listeners; motivates automatic detection'],
    ['3', 'AASIST (Tak et al.) [3]', 'Raw waveform modelling with graph attention and spectro-temporal convolutions',
     'High accuracy on ASVspoof 2021; complex architecture, heavy to train and deploy'],
    ['4', 'Does Audio Deepfake Detection Generalize? [4]', 'Evaluation of detectors on unseen synthesis methods',
     'Detectors trained on one synthesis family fail on unseen families; dataset diversity is critical'],
    ['5', 'Patil and Dua [5]', 'Review of synthetic speech and singing voice detection methods',
     'Spectral features of synthetic speech differ from real speech; spectrogram-based deep learning is promising'],
    ['6', 'librosa [6] and PyTorch [7]', 'Open-source audio analysis and deep learning libraries',
     'Provide the practical toolset for Mel-spectrogram extraction and CNN training used in this project'],
])
para('From the above review it is concluded that (i) modern synthetic speech fools human '
     'listeners, (ii) hand-crafted features alone do not generalise well, and (iii) '
     'image-based deep learning on spectrograms is a strong and practical approach. The '
     'proposed system therefore adopts a CNN trained on Mel-spectrogram images of real and '
     'fake speech.')
pb()

# ================= CHAPTER 3 =================
h1('CHAPTER 3 : SYSTEM ANALYSIS')
h2('3.1 Existing System')
para('In the existing approach, verification of a suspicious voice is done either by human '
     'experts listening to the audio, or by traditional signal-processing tools that measure '
     'simple statistical features. Both approaches have serious drawbacks. Human judgment is '
     'subjective and unreliable against modern neural synthesizers. Traditional tools require '
     'expert knowledge and are not available to ordinary users.')
h2('3.2 Disadvantages of the Existing System')
bullets([
    'Human listening tests are slow, subjective and inconsistent.',
    'No quantified output – the expert cannot state a confidence value.',
    'Traditional signal analysis requires specialist tools and training.',
    'Not accessible through a simple interface for non-technical users.',
])
h2('3.3 Proposed System')
para('The proposed Voice Impersonation Detection System is an automated, deep-learning based '
     'web application. The user uploads an audio file or records voice live in the browser. '
     'The backend converts the audio into a Mel-spectrogram image, and a trained CNN '
     'classifies the image as Real Human Voice or AI-Generated Voice. A report containing the '
     'verdict and confidence percentage is displayed to the user.')
h2('3.4 Advantages of the Proposed System')
bullets([
    'Fully automated – complete result within seconds.',
    'Objective output with confidence score.',
    'Browser based – no software installation required for the end user.',
    'Supports live recording, which is useful for quick field checks.',
    'Visual Mel-spectrogram output adds transparency to the decision.',
])
h2('3.5 Feasibility Study')
h3('3.5.1 Technical Feasibility')
para('All the required technologies are open-source and free of cost (Python, PyTorch, '
     'librosa, Flask). The project runs on a standard laptop with a dedicated graphics card, '
     'which is available in the development environment. Hence the project is technically '
     'feasible.')
h3('3.5.2 Operational Feasibility')
para('The user interface is designed as a normal web page with two clear actions: upload or '
     'record. No technical knowledge is required to operate the system, so operational '
     'feasibility is high.')
h3('3.5.3 Economic Feasibility')
para('The total cost of the project is limited to the compute hardware already available. No '
     'licences, servers or paid APIs are required, hence the project is economically feasible.')
h2('3.6 Hardware Requirements')
table(['Component', 'Minimum Requirement', 'Recommended Requirement'], [
    ['Processor', 'Intel Core i5 (or equivalent)', 'Intel Core i7 / Ryzen 7'],
    ['RAM', '8 GB', '16 GB'],
    ['Storage', '10 GB free disk space', 'SSD with 20 GB free space'],
    ['GPU', 'Not mandatory (CPU training possible)', 'NVIDIA RTX GPU with CUDA support'],
    ['Microphone', 'Built-in laptop microphone', 'USB microphone for live recording tests'],
])
h2('3.7 Software Requirements')
table(['Software', 'Version / Details', 'Purpose'], [
    ['Operating System', 'Windows 11', 'Development and execution platform'],
    ['Python', '3.13', 'Backend and machine learning programming'],
    ['PyTorch', '2.x with CUDA support', 'CNN model building and training'],
    ['librosa', 'Latest', 'Audio loading and Mel-spectrogram extraction'],
    ['Flask', '3.x', 'Web server (backend)'],
    ['scikit-learn / NumPy', 'Latest', 'Evaluation metrics and numerical operations'],
    ['HTML5 / CSS3 / JavaScript', 'Latest standards', 'Web interface'],
    ['Web Browser', 'Google Chrome / Microsoft Edge', 'Running the user interface'],
    ['Code Editor', 'Visual Studio Code', 'Development environment'],
])
pb()

# ================= CHAPTER 4 =================
h1('CHAPTER 4 : SYSTEM DESIGN')
h2('4.1 System Architecture')
para('The system follows a three-layer architecture: a presentation layer (web interface), '
     'an application layer (Flask backend with preprocessing) and a machine-learning layer '
     '(trained CNN model), supported by a storage layer that holds the trained model file.')
figure('fig1_system_architecture.png', 'Fig. 4.1 : System Architecture')
h2('4.2 Work Flow of the Proposed System')
para('Fig. 4.2 shows the step-by-step workflow. The user either uploads a file or records '
     'live audio. The backend validates the clip, converts it to a Mel-spectrogram, feeds it '
     'to the CNN, and returns the report. Invalid inputs are rejected with an error message '
     'and the user may retry.')
figure('fig2_workflow.png', 'Fig. 4.2 : Work Flow of the Proposed System')
h2('4.3 Use Case Diagram')
para('The use case diagram in Fig. 4.3 identifies the primary actor as the User. The user-facing '
     'use cases are upload, record, submit, view report and re-test. The internal use cases '
     '(validation, spectrogram generation and CNN classification) are included through the '
     '"Submit for Analysis" use case.')
figure('fig3_usecase.png', 'Fig. 4.3 : Use Case Diagram', width=5.6)
h2('4.4 Data Flow Diagram – Level 0')
para('The context diagram (Level 0) in Fig. 4.4 shows the entire system as a single process '
     'exchanging audio input and detection reports with the user, while using the trained '
     'CNN model file as a data store.')
figure('fig4_dfd_level0.png', 'Fig. 4.4 : Data Flow Diagram – Level 0 (Context Diagram)', width=5.6)
h2('4.5 Data Flow Diagram – Level 1')
para('In Level 1 (Fig. 4.5) the system is decomposed into four sub-processes: audio input '
     'management, preprocessing, CNN classification and report generation, supported by three '
     'data stores.')
figure('fig5_dfd_level1.png', 'Fig. 4.5 : Data Flow Diagram – Level 1', width=5.6)
h2('4.6 Sequence Diagram')
para('Fig. 4.6 shows the runtime interaction between the user, the web interface, the Flask '
     'backend and the CNN model for a single prediction request.')
figure('fig6_sequence.png', 'Fig. 4.6 : Sequence Diagram', width=5.9)
h2('4.7 Project Timeline')
para('Fig. 4.7 presents the planned project schedule over twelve weeks using a Gantt chart.')
figure('fig7_gantt.png', 'Fig. 4.7 : Project Timeline (Gantt Chart)', width=5.9)
pb()

# ================= CHAPTER 5 =================
h1('CHAPTER 5 : IMPLEMENTATION')
h2('5.1 Technology Stack')
table(['Component', 'Technology', 'Purpose'], [
    ['Programming Language', 'Python 3.13', 'Backend logic and machine learning code'],
    ['Deep Learning Framework', 'PyTorch (CUDA)', 'CNN model definition and training on GPU'],
    ['Audio Processing', 'librosa', 'Loading audio and generating Mel-spectrograms'],
    ['Web Framework', 'Flask', 'HTTP server exposing the /predict route'],
    ['Frontend', 'HTML5, CSS3, JavaScript', 'User interface and result display'],
    ['Microphone Recording', 'MediaRecorder API', 'Live voice capture inside the browser'],
    ['Result Charts', 'Chart.js', 'Confidence gauge and probability bar'],
    ['Evaluation', 'scikit-learn, NumPy', 'Accuracy, precision, recall and F1-score'],
    ['Graphs', 'Matplotlib, Seaborn', 'Confusion matrix and training curves'],
    ['Model Storage', 'voice_cnn.pth', 'Saved trained model weights'],
])
h2('5.2 Dataset')
para('The model is trained on a labelled dataset consisting of two classes: REAL speech '
     '(genuine human recordings) and FAKE speech (speech synthesised or converted by neural '
     'TTS systems). Audio clips are collected, cleaned, split into 3-second segments, '
     'labelled and stored in two folders (real/ and fake/). The dataset is divided into '
     'training (80%) and validation (20%) subsets. Data augmentation techniques such as '
     'addition of background noise, time stretching and pitch shifting are applied to the '
     'training samples so that the model becomes robust to real-world variations.')
h2('5.3 Audio Preprocessing and Mel-Spectrogram Generation')
para('Raw audio cannot be fed directly to a CNN. Each audio clip is resampled to 22050 Hz, '
     'trimmed or padded to a fixed length of 3 seconds, and converted into a Mel-spectrogram '
     'image of size 128 x 128 pixels using librosa. The generation pipeline is shown in '
     'Fig. 5.1. The parameters used are n_mels = 128, n_fft = 2048 and hop_length = 512.')
figure('fig8_melspec_pipeline.png', 'Fig. 5.1 : Mel-Spectrogram Generation Pipeline', width=6.0)
h3('5.3.1 How Real and AI-Generated Voices Differ in the Spectrogram')
para('The fundamental principle on which this project is based is that genuine human speech '
     'and AI-generated speech leave different visual patterns in a Mel-spectrogram. A natural '
     'human voice contains fine-grained detail: breath and room noise, small variations in '
     'pitch and rhythm (jitter and shimmer), and a complete frequency spectrum extending to '
     'high frequencies. These appear in the spectrogram as a textured image with slightly '
     'irregular harmonic bands and scattered energy across all frequency ranges.')
para('AI-generated speech, on the other hand, is produced by a model that learns to '
     'reproduce the "average" characteristics of speech. As a result, its spectrogram is '
     'unnaturally smooth and regular: the harmonic bands are perfectly even, the amplitude '
     'is unusually steady over time, and high-frequency content is often weak or missing '
     'because vocoders struggle to synthesise it convincingly. These systematic artefacts '
     'are subtle and almost impossible for a human listener to notice, but they are clearly '
     'visible to a CNN that is trained on thousands of labelled samples. Fig. 5.3 illustrates '
     'this difference between a real human voice sample and an AI-generated voice sample.')
figure('fig9_spectrogram_compare.png', 'Fig. 5.3 : Spectral Pattern Difference between Real and AI-Generated Voice', width=6.0)
para('During training, the CNN learns these distinguishing features automatically from the '
     'spectrogram images - no hand-crafted rules are required. This is precisely why '
     'spectrogram-based deep learning is the chosen approach for this project.')
h2('5.4 CNN Model Architecture')
para('A custom Convolutional Neural Network is designed for this project. The network '
     'contains four convolutional blocks, each followed by batch normalisation, ReLU '
     'activation and 2 x 2 max pooling. Dropout layers are used to prevent overfitting. The '
     'feature map is flattened and passed through a dense layer of 256 neurons, and finally '
     'to a 2-neuron softmax output layer which gives the probability of the input being real '
     'or AI-generated. The architecture is illustrated in Fig. 5.2.')
figure('fig3_cnn_architecture.png', 'Fig. 5.2 : CNN Model Architecture', width=6.0)
table(['#', 'Layer', 'Output Shape', 'Parameters'], [
    ['1', 'Input Image (Mel-spectrogram)', '(1, 128, 128)', '—'],
    ['2', 'Conv2D(32, 3x3) + BatchNorm + ReLU', '(32, 128, 128)', '384'],
    ['3', 'MaxPool2D(2x2)', '(32, 64, 64)', '—'],
    ['4', 'Conv2D(64, 3x3) + BatchNorm + ReLU', '(64, 64, 64)', '18,624'],
    ['5', 'MaxPool2D(2x2)', '(64, 32, 32)', '—'],
    ['6', 'Conv2D(128, 3x3) + BatchNorm + ReLU', '(128, 32, 32)', '74,112'],
    ['7', 'MaxPool2D(2x2)', '(128, 16, 16)', '—'],
    ['8', 'Conv2D(256, 3x3) + BatchNorm + ReLU', '(256, 16, 16)', '295,680'],
    ['9', 'MaxPool2D(2x2)', '(256, 8, 8)', '—'],
    ['10', 'Flatten', '(16384,)', '—'],
    ['11', 'Dropout(0.5)', '(16384,)', '—'],
    ['12', 'Dense(256) + ReLU', '(256,)', '4,194,560'],
    ['13', 'Dense(2) + Softmax', '(2,)', '514'],
    ['', 'Total Trainable Parameters', '', '4,583,874'],
])
h2('5.5 Training Methodology')
para('The network is trained using the Adam optimiser with cross-entropy loss. The batch '
     'size is 32 and the model is trained for 25 epochs with a learning-rate scheduler that '
     'reduces the learning rate when the validation loss stops improving. The training '
     'process is summarised in the following pseudocode.')
para('Pseudocode:', bold=True, space_after=2)
pseudocode = (
    '1. Load labelled audio paths from real/ and fake/ folders.\n'
    '2. For each clip: resample -> trim/pad to 3 s -> compute Mel-spectrogram (128 mels).\n'
    '3. Split data into training (80%) and validation (20%) sets.\n'
    '4. Initialise the CNN model and the Adam optimiser.\n'
    '5. Repeat for each epoch:\n'
    '   a. Forward pass: spectrogram -> CNN -> class probabilities.\n'
    '   b. Compute cross-entropy loss.\n'
    '   c. Backward pass: update weights (backpropagation).\n'
    '   d. Evaluate on validation set; record loss and accuracy.\n'
    '6. Save the best model weights to voice_cnn.pth.\n'
    '7. Evaluate on the test set and generate the performance report.'
)
pp = doc.add_paragraph()
run = pp.add_run(pseudocode)
run.font.name = 'Consolas'
run.font.size = Pt(10)
pp.paragraph_format.left_indent = Inches(0.4)
pp.paragraph_format.line_spacing = 1.15
doc.add_paragraph()
h2('5.6 Web Application Implementation')
para('The web application is built with Flask on the server side and plain HTML, CSS and '
     'JavaScript on the client side. The main page provides two input modes: an audio file '
     'upload control and a live recording button that captures voice through the browser '
     'microphone using the MediaRecorder API. The captured audio is sent to the backend as a '
     'WAV blob.')
table(['Route', 'Method', 'Description'], [
    ['/', 'GET', 'Renders the main page with upload and record controls'],
    ['/predict', 'POST', 'Accepts audio, runs preprocessing and CNN inference, returns JSON'],
    ['/result', 'GET', 'Shows the detection report page with confidence gauge'],
])
para('On receiving the audio at /predict, the backend validates the file type and duration, '
     'generates the Mel-spectrogram using the exact same parameters used during training, '
     'performs inference with the trained model, and returns a JSON response of the form: '
     '{ "label": "AI-GENERATED", "confidence": 94.2, "real_probability": 5.8, '
     '"fake_probability": 94.2 }. The frontend uses Chart.js to display the confidence as a '
     'gauge and also shows the generated Mel-spectrogram image so that the user can see the '
     'pattern analysed by the model.')
pb()

# ================= CHAPTER 6 =================
h1('CHAPTER 6 : TESTING')
h2('6.1 Test Strategy')
para('Testing is carried out in three levels: unit testing of individual functions '
     '(audio loading, spectrogram generation, model inference), integration testing of the '
     'Flask routes with the frontend, and system testing of the complete workflow with real '
     'user input. The functional test cases executed on the web application are listed below.')
h2('6.2 Functional Test Cases')
table(['Test ID', 'Test Case', 'Expected Result', 'Actual Result'], [
    ['TC-01', 'Upload a valid .wav audio file (5 seconds)', 'Prediction report displayed with class and confidence %', ''],
    ['TC-02', 'Upload a valid .mp3 audio file', 'Prediction report displayed correctly', ''],
    ['TC-03', 'Upload an unsupported file (e.g. .txt or .pdf)', 'Error message: unsupported format', ''],
    ['TC-04', 'Upload an audio file shorter than 3 seconds', 'Error message: audio too short', ''],
    ['TC-05', 'Record live voice for 5 seconds and submit', 'Recording captured and report displayed', ''],
    ['TC-06', 'Record live voice and cancel before submission', 'No request sent; form reset', ''],
    ['TC-07', 'Submit with no file and no recording', 'Validation message shown; no crash', ''],
    ['TC-08', 'Check that confidence value is between 0% and 100%', 'Valid percentage displayed', ''],
    ['TC-09', 'Verify Mel-spectrogram image is displayed in the report', 'Spectrogram thumbnail visible', ''],
    ['TC-10', 'Upload the same file twice', 'Consistent (identical) prediction returned', ''],
    ['TC-11', 'Test with server stopped', 'Browser shows a clear connection error message', ''],
    ['TC-12', 'Test in Google Chrome and Microsoft Edge', 'Interface works correctly in both browsers', ''],
])
h2('6.3 Performance Testing')
para('The preprocessing and inference time for a 3-second audio clip is measured. The '
     'expected end-to-end response time is under 3 seconds on the target hardware, which '
     'includes spectrogram generation, model inference and JSON transfer.')
h2('6.4 Performance Metrics')
para('The trained model is evaluated using the following standard classification metrics:')
para('Accuracy  = (TP + TN) / (TP + TN + FP + FN)', size=11.5)
para('Precision = TP / (TP + FP)', size=11.5)
para('Recall    = TP / (TP + FN)', size=11.5)
para('F1-Score  = 2 x (Precision x Recall) / (Precision + Recall)', size=11.5)
para('where TP = true positives, TN = true negatives, FP = false positives and '
     'FN = false negatives. In addition, a confusion matrix and a ROC curve are generated '
     'for the test set.')
pb()

# ================= CHAPTER 7 =================
h1('CHAPTER 7 : RESULTS AND DISCUSSION')
h2('7.1 Evaluation Metrics')
para('The performance of the CNN model is measured in terms of accuracy, precision, recall, '
     'F1-score and ROC-AUC on the held-out validation set (3,138 three-second segments from '
     'eight WaveFake vocoders, MLAAD systems, LibriSpeech speakers and field uploads), on a '
     'balanced 1,600-segment probe set, and on 16 whole audio files that were never used in '
     'training. The training history (loss and accuracy per epoch) is plotted to demonstrate '
     'convergence.')
h2('7.2 Model Performance')
para('The results obtained after training are recorded in Table 7.1. All figures below are '
     'measured values from the evaluation scripts, not estimates.')
para('Table 7.1 : Model Performance Summary', bold=True, align='center', size=11, space_after=4)
table(['Metric', 'Validation Set (n=3138)', 'Held-Out File Test (n=16)'], [
    ['Accuracy', '94.6%', '100.0%'],
    ['Precision (Real class)', '0.900', '1.000'],
    ['Precision (AI-Generated class)', '0.977', '1.000'],
    ['Recall (Real class)', '0.963', '1.000'],
    ['Recall (AI-Generated class)', '0.936', '1.000'],
    ['F1-Score', '0.930', '1.000'],
    ['ROC-AUC', '0.9896', '1.000'],
])
para('On the balanced 1,600-segment probe set the model reaches 94.4% accuracy with an '
     'F1-score of 0.945 and ROC-AUC of 0.9889, confirming that performance holds on data '
     'drawn from many different generators and speakers.')
h2('7.3 Confusion Matrix')
para('The confusion matrix on the validation set (Fig. 7.1) shows 1,132 correctly identified '
     'real segments and 1,836 correctly identified AI-generated segments, with 126 '
     'AI-generated segments mistaken for real speech and 44 real segments mistaken for '
     'AI-generated speech. Errors concentrate on high-quality neural voices, which is '
     'expected: they are acoustically closest to genuine speech.')
figure('fig10_confusion_matrix.png', 'Fig. 7.1 : Confusion Matrix on the Validation Set (n = 3138)', width=5.2)
h2('7.4 Training Graph')
para('The training and validation accuracy and loss curves (Fig. 7.2) show steady convergence '
     'over 20 epochs with SpecAugment regularisation. Early-epoch fluctuation of the '
     'validation curve reflects the diversity of the training distribution (70+ generators '
     'and speakers) rather than memorisation of a single source.')
figure('fig11_training_curves.png', 'Fig. 7.2 : Training and Validation Accuracy/Loss Curves', width=6.0)
para('Fig. 7.3 visualises what the CNN learns: genuine speech shows rich, irregular harmonic '
     'structure across the full frequency range, while AI-generated speech shows smoother, '
     'more regular bands with missing fine detail — the vocoder fingerprints the network '
     'detects.')
figure('fig12_cnn_pattern_difference.png', 'Fig. 7.3 : Mel-Spectrogram Pattern Difference Learned by the CNN', width=6.0)
h2('7.5 Discussion')
para('The system, whose CNN is named Professor K (4,583,874 parameters), achieves high '
     'classification accuracy because synthetic speech leaves detectable artefacts in the '
     'Mel-spectrogram, such as unnatural harmonic structures, overly smooth spectral '
     'envelopes and missing high-frequency content. The calibrated confidence score, '
     'together with the acoustic-evidence panel (pitch variation, spectral markers, '
     'dynamics) and an explicit UNCERTAIN verdict for borderline cases, makes each '
     'prediction auditable instead of a black-box answer. Known limitations: accuracy drops '
     'on recording conditions absent from training (e.g. new languages or microphones), '
     'which is addressed in this project by the progressive field-data loop that ingests '
     'user-provided labelled samples and fine-tunes the model.')
h2('7.6 Final Verification on Current Weights')
para('After two additional fine-tuning rounds (field data and Tamil data), the deployed '
     'Professor K checkpoint was re-verified: 16/16 on the original held-out file set, '
     '3/3 on keeper real recordings (95.2%, 97.7%, 94.9% confidence), 2/2 on held-out AI '
     'voices, and 9/11 on a Tamil battery (one AI voice at 86% plus five confident real '
     'verdicts, four honest UNCERTAIN, zero misclassifications). A held-out validation '
     'split of 68 file-grouped clips scores 98.5% accuracy with an F1-score of 0.986.')
pb()

# ================= CHAPTER 8 =================
h1('CHAPTER 8 : CONCLUSION AND FUTURE SCOPE')
h2('8.1 Conclusion')
para('This project presents an automated Voice Impersonation Detection System that '
     'classifies a voice sample as genuine human speech or AI-generated speech using a '
     'Convolutional Neural Network trained on Mel-spectrogram images. A user-friendly web '
     'interface supports both file upload and live microphone recording, and produces a '
     'detection report with a confidence percentage. The project demonstrates that deep '
     'learning on time-frequency representations of speech is a practical and effective '
     'approach against voice impersonation attacks.')
h2('8.2 Future Scope')
bullets([
    'Conversion of the web application into a standalone desktop application (.exe) using pywebview and PyInstaller.',
    'Real-time streaming detection for live phone calls and video conferences.',
    'Extension to multiple Indian and world languages.',
    'Use of larger and more diverse datasets covering unseen synthesis engines.',
    'Exploration of deeper architectures such as ResNet and EfficientNet for higher accuracy.',
    'Development of a mobile application for on-the-go voice verification.',
    'Integration of an explainability module that highlights the exact spectral regions that caused the fake prediction.',
])
pb()

# ================= REFERENCES =================
h1('REFERENCES')
refs = [
    'X. Wang, J. Yamagishi, M. Todisco, H. Delgado, A. Nautsch, N. Evans, M. De Leon, '
    'I. Echizen and M. H. Savoji, "ASVspoof 2019: A large-scale public database of '
    'synthesized, converted and replayed speech," Computer Speech & Language, vol. 64, '
    'p. 101114, 2021.',
    'P. Zell, F. Alom, M. S. Rahman, D. Schuster, T. Fingscheidt and S. Chaudhuri, '
    '"Evaluation of modern synthesizer devices for human impersonation," in Proc. Odyssey: '
    'The Speaker and Language Recognition Workshop, Suzhou, China, 2020, pp. 392-399.',
    'H. Tak, J. Patino, A. Nautsch, M. Todisco and N. Evans, "An audio anti-spoofing '
    'detection system based on raw waveforms," in Proc. IEEE ICASSP, Singapore, 2022, '
    'pp. 6387-6391.',
    'N. M. Muller, D. Choi, M. Sammler and C. Franze, "Does audio deepfake detection '
    'generalize?," in Proc. Interspeech, Incheon, South Korea, 2022, pp. 3488-3492.',
    'H. A. Patil and M. M. Dua, "Detection of synthetic speech and singing voice: A review," '
    'IETE Technical Review, vol. 39, no. 1, pp. 60-77, 2022.',
    'B. McFee, C. Raffel, D. Liang, D. P. W. Ellis, M. McVicar, E. Battenberg and O. Nieto, '
    '"librosa: Audio and music signal analysis in Python," in Proc. 14th Python in Science '
    'Conf., Austin, USA, 2015, pp. 18-25.',
    'A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan, T. Killeen, '
    'Z. Lin, N. Gimelshein, L. Antiga, A. Desmaison, A. Kopf, E. Yang, Z. DeVito, '
    'M. Raison, A. Tejani, S. Chilamkurthy, B. Steiner, L. Fang, J. Bai and S. Chintala, '
    '"PyTorch: An imperative style, high-performance deep learning library," in Proc. NeurIPS, '
    'Vancouver, Canada, 2019, pp. 8026-8037.',
    'M. Grinberg, Flask Web Development, 2nd ed. Sebastopol, CA: O\'Reilly Media, 2018.',
    'Mozilla Developer Network, "MediaRecorder API," MDN Web Docs. '
    '[Online]. Available: https://developer.mozilla.org/en-US/docs/Web/API/MediaRecorder',
    'F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, '
    'P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, '
    'M. Brucher, M. Perrot and E. Duchesnay, "Scikit-learn: Machine learning in Python," '
    'Journal of Machine Learning Research, vol. 12, pp. 2825-2830, 2011.',
]
for i, r in enumerate(refs, 1):
    p = doc.add_paragraph()
    run = p.add_run(f'[{i}]  {r}')
    run.font.size = Pt(11)
    p.paragraph_format.line_spacing = 1.25
    p.paragraph_format.left_indent = Inches(0.4)
    p.paragraph_format.first_line_indent = Inches(-0.4)

doc.save(OUTDOC)
print('REPORT SAVED:', OUTDOC)
