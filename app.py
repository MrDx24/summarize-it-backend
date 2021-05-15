from flask import Flask, request, jsonify, make_response
from flask_cors import CORS, cross_origin
import tfidf_textrank as tftr
import webcrawl as wb
import pdf_reader as pr
import timeit
import io

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})
#app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/summary', methods=['POST'])
def summarize():
    if request.method == 'POST':

        start = timeit.default_timer()
        text_data = request.get_json()
        summary_data = tftr.generate_summary(text_data['ogstr'],7)  
        stop = timeit.default_timer()
        tot_time = stop - start
        total_time = round(tot_time, 4)
        return jsonify({'data':summary_data,'tot_time':total_time}), 201 
    #else:
        #return make_response(jsonify({'error':"not successfull"}) , 500 )  

@app.route('/summaryUrl', methods=['POST'])
def summarizeUrl():
    if request.method == 'POST':

        start = timeit.default_timer()
        text_data = request.get_json()
        extraxtedData = wb.extractTextFromUrl(text_data['dataUrl'])
        summary_data = tftr.generate_summary(extraxtedData,7)  
        stop = timeit.default_timer()
        tot_time = stop - start
        total_time = round(tot_time, 4)
        return jsonify({'data':summary_data,'tot_time':total_time}), 201 
    #else:
        #return make_response(jsonify({'error':"not successfull"}) , 500 )  


@app.route('/summaryFile', methods=['POST'])
def summarizeFile():
    if request.method == 'POST':

        start = timeit.default_timer() 
        text_data = request.files['file'].read() 
        summary=""
        extention=request.files['file'].filename.lower().split(".") 
        extLen = len(extention)

        if extention[extLen-1] == "txt":
            f = text_data.decode()
            summary = tftr.generate_summary(f,7)
            
        if extention[extLen-1] == "pdf":
            file = pr.extractTextfromPdf(io.BytesIO(text_data)) 
            summary = tftr.generate_summary(str(file),7)

        stop = timeit.default_timer()
        tot_time = stop - start
        total_time = round(tot_time, 4)

        return jsonify({'data':summary,'tot_time':total_time}), 201 
    #else:
        #return make_response(jsonify({'error':"not successfull"}) , 500 )  

if __name__ == '__main__':
    app.run()