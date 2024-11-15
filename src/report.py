from pylatex import Document, Section, Math

doc = Document()
with doc.create(Section("Mathematical Formula")):
    doc.append(Math(data=["E = mc^2"]))

# This command automatically compiles the LaTeX code into a PDF
doc.generate_pdf("math_report", clean_tex=False, compiler="pdflatex")