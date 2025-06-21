import os
from PIL import Image
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, redirect, send_file, url_for
from googletrans import Translator
from twilio.rest import Client
from reportlab.pdfgen import canvas
from io import BytesIO
import cv2
from werkzeug.utils import secure_filename
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.graphics.barcode import qr
from reportlab.platypus import Frame, Paragraph
from reportlab.lib.styles import getSampleStyleSheet


from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)
translator = Translator()

# Configuration
app.config['UPLOAD_FOLDER'] = 'static'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Load the model
model = tf.keras.models.load_model("Plant_Village_Detection_Model.h5")

# Class labels
classes = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_r', 'Apple___healthy',
           'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
           'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
           'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
           'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
           'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
           'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
           'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
           'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
           'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
           'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
           'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']

# Suggestions
suggestions = {
 "Apple___Apple_scab": "Remove fallen leaves. Apply sulfur fungicide weekly. Plant resistant varieties like 'Liberty'.",
"Apple___Black_rot": "Prune infected branches. Apply captan fungicide. Remove all mummified fruit.",
"Apple___Cedar_apple_r": "Remove nearby junipers. Apply myclobutanil at bud break. Plant resistant varieties.",
"Apple___healthy": "Maintain proper pruning. Monitor regularly. Keep area weed-free.","Blueberry___healthy": "Maintain acidic soil (pH 4.5-5.5). Use pine bark mulch. Irrigate at base.",
"Cherry_(including_sour)___Powdery_mildew": "Apply potassium bicarbonate. Improve air circulation. Avoid wetting leaves.",
"Cherry_(including_sour)___healthy": "Prune for sunlight penetration. Monitor for pests. Maintain consistent moisture.",
"Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Rotate with soybeans. Apply azoxystrobin. Space plants properly.",
"Corn_(maize)___Common_rust_": "Plant early-maturing varieties. Spray chlorothalonil. Remove volunteer plants.",
"Corn_(maize)___Northern_Leaf_Blight": "Till crop residues. Use resistant hybrids. Apply fungicide at tasseling.",
"Corn_(maize)___healthy": "Ensure proper nitrogen levels. Monitor for pests. Rotate crops annually.","Grape___Black_rot": "Remove mummified berries. Apply mancozeb weekly. Train vines vertically.",
"Grape___Esca_(Black_Measles)": "Disinfect pruning tools. Avoid severe pruning. Apply borax to cuts.",
"Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Apply copper fungicide. Remove infected leaves. Keep vineyard clean.",
"Grape___healthy": "Prune properly. Monitor for pests. Maintain balanced nutrition.","Orange___Haunglongbing_(Citrus_greening)": "Remove infected trees. Control psyllids with imidacloprid. Use reflective mulch.","Peach___Bacterial_spot": "Apply copper sprays in dormancy. Avoid overhead irrigation. Plant resistant varieties.",
"Peach___healthy": "Prune for airflow. Monitor for borers. Maintain consistent watering.","Pepper,_bell___Bacterial_spot": "Use disease-free seeds. Apply copper sprays. Space plants adequately.",
"Pepper,_bell___healthy": "Rotate crops. Avoid wetting foliage. Monitor for aphids.","Potato___Early_blight": "Remove infected leaves. Apply chlorothalonil. Maintain nitrogen levels.",
"Potato___Late_blight": "Destroy infected plants. Apply mancozeb preventatively. Hill potatoes properly.",
"Potato___healthy": "Use certified seed potatoes. Monitor soil moisture. Rotate crops.","Raspberry___healthy": "Prune old canes. Maintain weed-free rows. Provide trellising support.","Soybean___healthy": "Rotate with corn. Monitor for aphids. Maintain proper pH (6.0-6.8).","Squash___Powdery_mildew": "Apply neem oil. Plant resistant varieties. Water at soil level.","Strawberry___Leaf_scorch": "Remove old leaves after harvest. Apply copper fungicide. Improve drainage.",
"Strawberry___healthy": "Renovate beds annually. Use straw mulch. Monitor for spider mites.","Tomato___Bacterial_spot": "Treat seeds with hot water. Apply copper+mancozeb. Avoid working with wet plants.",
"Tomato___Leaf_Mold": "Reduce humidity. Apply potassium bicarbonate. Space plants properly.",
"Tomato___Septoria_leaf_spot": "Mulch heavily. Apply copper fungicide. Remove lower leaves.",
"Tomato___Spider_mites Two-spotted_spider_mite": "Spray water to dislodge. Apply insecticidal soap. Encourage predatory mites.",
"Tomato___Target_Spot": "Apply azoxystrobin. Improve air circulation. Remove infected leaves.",
"Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Control whiteflies with yellow sticky traps. Use reflective mulch. Remove infected plants.",
"Tomato___Tomato_mosaic_virus": "Disinfect tools. Control aphids. Remove infected plants immediately."
}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.
    return img_array

def draw_bounding_boxes(image_path, boxes, output_path):
    img = cv2.imread(image_path)
    for box in boxes:
        x_min, y_min, x_max, y_max = box
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(img, 'Detected Area', (x_min, y_min - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imwrite(output_path, img)

def predict_image_class_and_boxes(image_path):
    img = load_and_preprocess_image(image_path)
    pred = model.predict(img)
    class_index = np.argmax(pred)
    label = classes[class_index]
    boxes = [[50, 50, 100, 100], [150, 150, 200, 200]]  # Example boxes
    return label, boxes

def send_sms(to_number, disease, suggestion):
    try:
        account_sid = os.getenv("TWILIO_SID")
        auth_token = os.getenv("TWILIO_AUTH_TOKEN")
        from_number = os.getenv("TWILIO_NUMBER")

        if not all([account_sid, auth_token, from_number]):
            print("❌ Missing Twilio credentials. Check environment variables.")
            return

        client = Client(account_sid, auth_token)
        message_body = f"🌿 CropProtect Report\nDisease: {disease}\nSuggestion: {suggestion}"
        message = client.messages.create(
            body=message_body,
            from_=from_number,
            to=to_number
        )
        print(f"✅ SMS sent successfully. SID: {message.sid}")
    except Exception as e:
        print("❌ Error sending SMS:", str(e))

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
            
        if file and allowed_file(file.filename):
            # Ensure upload directory exists
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            
            # Save original file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Generate highlighted version
            highlighted_filename = f"highlighted_{filename}"
            highlighted_path = os.path.join(app.config['UPLOAD_FOLDER'], highlighted_filename)
            
            # Predict and draw boxes
            label, boxes = predict_image_class_and_boxes(filepath)
            draw_bounding_boxes(filepath, boxes, highlighted_path)
            
            # Get suggestions and translations
            suggestion = suggestions.get(label, "No specific suggestion available.")
            language = request.form['language']
            phone = request.form['phone']
            
            readable_label = label.replace("___", " - ")
            translated_label = translator.translate(readable_label, dest=language).text
            translated_suggestion = translator.translate(suggestion, dest=language).text
            
            # Send SMS
            try:
                send_sms(phone, readable_label, suggestion)
            except Exception as e:
                print("❌ Failed to send SMS:", e)

            return render_template("app.html",
                               label=readable_label,
                               suggestion=suggestion,
                               translated_label=translated_label,
                               translated_suggestion=translated_suggestion,
                               file_path=filename,
                               highlighted_image_path=highlighted_filename)
    
    return render_template("app.html")

@app.route('/generate_report', methods=['POST'])
def generate_report():
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib import colors
    from reportlab.lib.units import inch

    # Get form data
    label = request.form['label']
    suggestion = request.form['suggestion']
    translated_label = request.form.get('translated_label', '')
    translated_suggestion = request.form.get('translated_suggestion', '')
    file_path = request.form.get('file_path', '')

    buffer = BytesIO()
    p = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    margin = 50
    y = height - margin

    # Title & Date
    p.setFont("Helvetica-Bold", 16)
    p.drawString(margin, y, "🌿 CropVista - Disease Detection Report")
    y -= 25
    p.setFont("Helvetica", 10)
    p.drawString(margin, y, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    y -= 40

    # Function to wrap and draw text
    def draw_wrapped_text(canvas, title, text, x, y_start, width=480, font_size=11, space_after=20):
        from reportlab.platypus import Paragraph
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import Frame

        style = getSampleStyleSheet()["Normal"]
        style.fontSize = font_size
        style.leading = font_size + 2
        style.alignment = TA_LEFT

        para = Paragraph(f"<b>{title}</b><br/>{text}", style)
        f = Frame(x, y_start - 60, width, 60, showBoundary=0)
        f.addFromList([para], canvas)
        return y_start - 70

    # Text sections
    y = draw_wrapped_text(p, "🔍 Predicted Disease (English):", label, margin, y)
    y = draw_wrapped_text(p, "💊 Suggestion (English):", suggestion, margin, y)
   

    # Image
    if file_path:
        highlighted_path = os.path.join(app.config['UPLOAD_FOLDER'], f"highlighted_{file_path}")
        if os.path.exists(highlighted_path):
            try:
                img = Image.open(highlighted_path)
                max_width = 4 * inch
                max_height = 4 * inch
                img.thumbnail((max_width, max_height))
                img_io = BytesIO()
                img.save(img_io, format='PNG')
                img_io.seek(0)
                y -= int(max_height) + 30
                p.drawImage(ImageReader(img_io), margin, y, width=max_width, height=max_height)
            except Exception as e:
                print("❌ Error adding image to PDF:", e)

    p.showPage()
    p.save()
    buffer.seek(0)

    return send_file(buffer, as_attachment=True, download_name="CropProtect_Report.pdf", mimetype='application/pdf')


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(host='0.0.0.0', port=3000, debug=True)