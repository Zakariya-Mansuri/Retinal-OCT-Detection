from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image


app = Flask(__name__)

classes = {0 : 'CNV', 1 : 'DME', 2 :"Drusen", 3:'Normal'}


model = load_model('my_model.h5')

def predict_label(img_path):
	i = image.load_img(img_path, target_size=(224,224))
	i = image.img_to_array(i)
	i = i.reshape(1, 224,224,3)
	p = model.predict(i)
	return classes[p.argmax()].upper()


# routes
@app.route("/", methods=['GET', 'POST'])
def home():
	return render_template("index.html")

@app.route("/about")
def about_page():
	return "About You..!!!"


@app.route("/submit", methods = ['GET', 'POST'])
def submit():
	if request.method == 'POST':
		img = request.files['my_image']
		print(img)
		img_path = "static/" + img.filename	
		print(img_path)
		img.save(img_path)

		p = predict_label(img_path)



	return render_template("index.html", prediction = p, img_path = img_path)


if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True, port=8000)	