import PyPDF2

def extractTextfromPdf(filename):
    # pdfFileObject = open(filename, 'rb')
    pdfReader = PyPDF2.PdfFileReader(filename)
    count=pdfReader.numPages
    output = []
    for i in range(count):
        page = pdfReader.getPage(i)
        output.append(page.extractText())

    list2 = [x.replace('\n', '') for x in output]

    #pdfFileObject.close()

    return list2






